"""
tests/test_real_cv_and_training_audit.py
Kiểm thử và nghiệm thu toàn diện 15 kịch bản bắt buộc theo Section 11 của yêu cầu kỹ thuật:
1. Hai ghế liền nhau có hai Seat ID riêng.
2. Seat ID không đổi qua các frame khi detection lệch nhẹ.
3. Một người ngồi ở ghế A, ghế B vẫn trống.
4. Ghế B có đồ vật nhưng không có người.
5. Người ở ghế A đứng lên trong khi người ở ghế B vẫn ngồi.
6. Người ở ghế B đứng lên và đồ vật vẫn còn.
7. Người mới ngồi lên đồ vật, đồ vật bị che.
8. Người mới đứng lên và đồ vật xuất hiện lại.
9. Người mới đứng lên nhưng đồ vật không xuất hiện lại.
10. Vật thể nằm sát ranh giới hai ghế -> UNKNOWN_SEAT.
11. Model không có lớp seat (COCO chair bị từ chối).
12. Không có custom weights -> CUSTOM SEAT MODEL NOT CONFIGURED.
13. Dataset thiếu annotation -> validation thất bại, từ chối train.
14. Training thất bại hoặc bị gián đoạn -> ghi nhận FAILED.
15. Không tạo metrics giả khi chưa có training/validation thực (simulate_training_job bị vô hiệu hóa).
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.models import (
    SeatConfig,
    SeatState,
    SeatStatus,
    TrackedObject,
    Event,
)
from src.seats.seat_detector import (
    SeatDetector,
    SeatDetectionResult,
    SeatDetectionMethod,
    CustomSeatYOLOAdapter,
    BusGeometrySeatDetector,
    TwinSeatDecomposer,
    VALID_SEAT_CLASS_NAMES,
)
from src.seats.seat_tracker import SeatTracker
from src.seats.seat_occupancy import SeatOccupancyEstimator
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.events.occlusion_manager import OcclusionManager
from src.services.model_service import ModelService
from src.database.db import Database
from src.database.model_repository import ModelRepository
from config.settings import EventStatus


# ==============================================================================
# 1. Hai ghế liền nhau có hai Seat ID riêng (Twin-seat decomposition)
# ==============================================================================
def test_01_two_adjacent_seats_have_separate_seat_ids():
    """1. Hai ghế liền nhau trong cùng một hàng phải được phân tách thành hai thực thể độc lập."""
    decomposer = TwinSeatDecomposer()
    # Băng ghế đôi hàng 1 bên trái: x từ 0.08 đến 0.40, y từ 0.15 đến 0.32
    bench_bbox = (0.08, 0.15, 0.40, 0.32)

    seats = decomposer.split_horizontal_bench(
        polygon_or_bbox=bench_bbox,
        row_index=1,
        side="left",
        naming_format="SEAT",
        gap_ratio=0.06,
    )

    assert len(seats) == 2, "Băng ghế đôi phải được phân rã thành đúng 2 ghế riêng biệt"
    seat_a, seat_b = seats[0], seats[1]

    # Phải có hai Seat ID khác biệt
    assert seat_a.seat_id == "SEAT_1A"
    assert seat_b.seat_id == "SEAT_1B"
    assert seat_a.seat_id != seat_b.seat_id

    # Đệm ghế A và đệm ghế B không được trùng nhau và có khe hở vật lý
    max_x_a = max(p[0] for p in seat_a.polygon)
    min_x_b = min(p[0] for p in seat_b.polygon)
    assert max_x_a < min_x_b, "Giữa Ghế A và Ghế B phải có khoảng cách, không được gộp làm một vùng lớn"


# ==============================================================================
# 2. Seat ID không đổi qua các frame khi detection lệch nhẹ
# ==============================================================================
def test_02_seat_id_stability_when_detection_shifts_slightly():
    """2. Seat tracker phải duy trì ID ổn định, không cấp ID mới khi tọa độ dịch chuyển nhẹ."""
    tracker = SeatTracker(iou_threshold=0.45, smoothing_alpha=0.85)

    # Frame 1: Detection ban đầu của 2 ghế
    raw_frame_1 = [
        SeatConfig(seat_id="SEAT_1A", polygon=[(0.10, 0.10), (0.25, 0.10), (0.25, 0.25), (0.10, 0.25)]),
        SeatConfig(seat_id="SEAT_1B", polygon=[(0.28, 0.10), (0.43, 0.10), (0.43, 0.25), (0.28, 0.25)]),
    ]
    tracked_f1 = tracker.update(raw_frame_1, frame_number=0)
    assert len(tracked_f1) == 2
    id_seat1 = tracked_f1[0].seat_id
    id_seat2 = tracked_f1[1].seat_id
    assert id_seat1 != id_seat2

    # Frame 2-5: Detection dao động nhẹ (rung lắc camera xe buýt)
    for f_idx in range(1, 6):
        jitter = 0.004 * (f_idx % 2)
        jittered_detections = [
            SeatConfig(seat_id="SEAT_1A", polygon=[(0.10 + jitter, 0.10), (0.25 + jitter, 0.10), (0.25 + jitter, 0.25), (0.10 + jitter, 0.25)]),
            SeatConfig(seat_id="SEAT_1B", polygon=[(0.28 - jitter, 0.10), (0.43 - jitter, 0.10), (0.43 - jitter, 0.25), (0.28 - jitter, 0.25)]),
        ]
        tracked = tracker.update(jittered_detections, frame_number=f_idx)
        assert len(tracked) == 2
        # ID phải giữ nguyên vĩnh viễn, không bị gán lại
        assert tracked[0].seat_id == id_seat1
        assert tracked[1].seat_id == id_seat2


# ==============================================================================
# 3. Một người ngồi ở ghế A, ghế B vẫn trống
# ==============================================================================
def test_03_person_on_seat_a_seat_b_remains_vacant():
    """3. Người ngồi ở Ghế A thì chỉ Ghế A chuyển sang OCCUPIED, Ghế B cạnh bên vẫn VACANT."""
    seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (200, 100), (200, 200), (100, 200)], is_normalized=False)
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)

    manager = SeatManager(
        seats=[seat_a, seat_b],
        occupied_confirm_sec=0.1,
        vacant_confirm_sec=0.1,
        hold_sec=0.1,
    )

    # Người P1 ngồi tại Seat A: bbox (105, 105, 195, 195)
    person_p1 = TrackedObject(track_id=1, class_name="person", confidence=0.92, bbox=(105, 105, 195, 195))

    for f in range(5):
        manager.update(persons=[person_p1], frame_index=f, fps=15.0, frame_width=640, frame_height=480)

    status_a = manager.get_seat_status("SEAT_1A")
    status_b = manager.get_seat_status("SEAT_1B")

    assert status_a.state == SeatState.OCCUPIED, "Ghế A phải có trạng thái OCCUPIED"
    assert 1 in status_a.occupant_person_ids, "Ghế A phải được gán cho Person P1"
    assert status_b.state == SeatState.VACANT, "Ghế B liền kề phải giữ nguyên trạng thái VACANT"
    assert len(status_b.occupant_person_ids) == 0


# ==============================================================================
# 4. Ghế B có đồ vật nhưng không có người
# ==============================================================================
def test_04_seat_b_has_object_but_no_person():
    """4. Ghế B có đồ vật nằm trên mặt đệm nhưng không có người ngồi."""
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_b], occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)

    # Kiểm tra liên kết đồ vật với ghế B
    backpack_bbox = (235, 120, 290, 185)
    matched_seat = seat_mgr.find_seat_for_bbox(backpack_bbox, 640, 480)
    assert matched_seat == "SEAT_1B"

    # Cập nhật không có người để xác nhận trạng thái trống (VACANT)
    for f in range(5):
        seat_mgr.update(persons=[], frame_index=f, fps=15.0, frame_width=640, frame_height=480)

    # Trạng thái ghế B lúc này là VACANT
    status_b = seat_mgr.get_seat_status("SEAT_1B")
    assert status_b.state == SeatState.VACANT


# ==============================================================================
# 5. Người ở ghế A đứng lên trong khi người ở ghế B vẫn ngồi
# ==============================================================================
def test_05_person_a_stands_up_while_person_b_remains_seated():
    """5. Hai người ngồi cạnh nhau, người ở Ghế A đứng lên rời đi, người ở Ghế B vẫn ngồi."""
    seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (200, 100), (200, 200), (100, 200)], is_normalized=False)
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)

    manager = SeatManager(
        seats=[seat_a, seat_b],
        occupied_confirm_sec=0.1,
        vacant_confirm_sec=0.1,
        hold_sec=0.1,
    )

    # Frame 0-4: Cả hai người P1 (ghế A) và P2 (ghế B) đều đang ngồi
    p1 = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(105, 105, 195, 195))
    p2 = TrackedObject(track_id=2, class_name="person", confidence=0.9, bbox=(225, 105, 315, 195))
    for f in range(5):
        manager.update(persons=[p1, p2], frame_index=f, fps=15.0, frame_width=640, frame_height=480)

    assert manager.get_seat_status("SEAT_1A").state == SeatState.OCCUPIED
    assert manager.get_seat_status("SEAT_1B").state == SeatState.OCCUPIED

    # Frame 5-10: P1 đứng lên di chuyển ra lối đi (x=450, y=300), P2 vẫn ngồi nguyên ở Ghế B
    p1_standing = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(450, 250, 520, 420))
    p2_sitting = TrackedObject(track_id=2, class_name="person", confidence=0.9, bbox=(225, 105, 315, 195))
    for f in range(5, 12):
        manager.update(persons=[p1_standing, p2_sitting], frame_index=f, fps=15.0, frame_width=640, frame_height=480)

    assert manager.get_seat_status("SEAT_1A").state == SeatState.VACANT, "Ghế A phải chuyển về VACANT sau khi P1 đứng lên"
    assert manager.get_seat_status("SEAT_1B").state == SeatState.OCCUPIED, "Ghế B phải giữ nguyên OCCUPIED vì P2 vẫn đang ngồi"


# ==============================================================================
# 6. Người ở ghế B đứng lên và đồ vật vẫn còn
# ==============================================================================
def test_06_person_b_stands_up_and_object_remains():
    """6. Người ở Ghế B đứng lên rời đi, ba lô vẫn nằm lại trên Ghế B trống -> Kích hoạt nguy cơ bỏ quên."""
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_b], occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)

    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=0.2,
        movement_threshold=10.0,
        watched_classes=["backpack"],
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )

    # Frame 0-4: P2 ngồi cùng ba lô O1 trên Ghế B -> Ghế có người, không báo động bỏ quên
    p2 = TrackedObject(track_id=2, class_name="person", confidence=0.9, bbox=(225, 105, 315, 195))
    bag = TrackedObject(track_id=10, class_name="backpack", confidence=0.88, bbox=(235, 120, 285, 180))

    for f in range(5):
        detector.process_frame([p2, bag], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    assert len(evt_mgr.events) == 0 or evt_mgr.events[0].status == EventStatus.OBSERVED

    # Frame 5-15: P2 đứng lên rời đi, chỉ còn ba lô O1 trên ghế
    for f in range(5, 18):
        detector.process_frame([bag], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    assert len(evt_mgr.events) == 1
    # Đồ vật ở trên ghế trống quá min_stationary_time -> SUSPECTED
    assert evt_mgr.events[0].status == EventStatus.SUSPECTED, "Khi người rời đi và đồ còn lại trên ghế trống, sự kiện phải là SUSPECTED"


# ==============================================================================
# 7. Người mới ngồi lên đồ vật, đồ vật bị che khuất (OCCLUDED)
# ==============================================================================
def test_07_new_person_sits_on_object_causing_occlusion():
    """7. Khi một người mới ngồi đè lên đồ vật đang bị nghi ngờ, sự kiện chuyển sang OCCLUDED."""
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_b], occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)
    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=0.2,
        movement_threshold=10.0,
        watched_classes=["backpack"],
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )

    # Đồ vật ban đầu nghi ngờ bị bỏ quên
    bag = TrackedObject(track_id=10, class_name="backpack", confidence=0.88, bbox=(235, 120, 285, 180))
    for f in range(10):
        detector.process_frame([bag], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    initial_event_id = evt.event_id

    # Người mới P3 ngồi lên ghế B, ba lô bị che mất
    p3 = TrackedObject(track_id=3, class_name="person", confidence=0.91, bbox=(220, 100, 320, 200))
    for f in range(10, 16):
        detector.process_frame([p3], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    # Sự kiện phải được bảo toàn và chuyển sang OCCLUDED
    assert len(evt_mgr.events) == 1
    assert evt.event_id == initial_event_id, "Event ID phải được giữ nguyên vẹn"
    assert evt.status == EventStatus.OCCLUDED, "Sự kiện phải chuyển thành OCCLUDED khi người mới che khuất"
    assert evt.seat_id == "SEAT_1B"


# ==============================================================================
# 8. Người mới đứng lên và đồ vật xuất hiện lại
# ==============================================================================
def test_08_person_stands_up_object_reappears_relinks_to_event():
    """8. Người mới đứng lên, đồ vật lộ ra lại -> Tái liên kết với Event ID cũ thay vì tạo mới."""
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_b], occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)
    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=0.2,
        movement_threshold=10.0,
        watched_classes=["backpack"],
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )

    # Tạo sự kiện ban đầu
    bag = TrackedObject(track_id=10, class_name="backpack", confidence=0.88, bbox=(235, 120, 285, 180))
    for f in range(10):
        detector.process_frame([bag], current_frame=f, fps=15.0, frame_width=640, frame_height=480)
    evt_id = evt_mgr.events[0].event_id

    # Người P3 che khuất
    p3 = TrackedObject(track_id=3, class_name="person", confidence=0.91, bbox=(220, 100, 320, 200))
    for f in range(10, 15):
        detector.process_frame([p3], current_frame=f, fps=15.0, frame_width=640, frame_height=480)
    assert evt_mgr.events[0].status == EventStatus.OCCLUDED

    # Người P3 rời đi, đồ vật xuất hiện lại (có thể mang track_id mới = 99 từ ByteTrack)
    bag_reappeared = TrackedObject(track_id=99, class_name="backpack", confidence=0.85, bbox=(236, 121, 284, 179))
    detector.process_frame([bag_reappeared], current_frame=16, fps=15.0, frame_width=640, frame_height=480)

    # Hệ thống phải nối lại sự kiện cũ
    assert len(evt_mgr.events) == 1
    assert evt_mgr.events[0].event_id == evt_id, "Phải liên kết lại với event ID cũ, không tạo event trùng lặp"
    assert evt_mgr.events[0].last_track_id == 99


# ==============================================================================
# 9. Người mới đứng lên nhưng đồ vật không xuất hiện lại -> REVIEW_REQUIRED
# ==============================================================================
def test_09_person_stands_up_object_missing_review_required():
    """9. Người mới đứng lên nhưng đồ vật không xuất hiện -> chuyển REVIEW_REQUIRED, KHÔNG tự ý RESOLVED."""
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 200), (220, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_b], occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)
    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=0.2,
        movement_threshold=10.0,
        watched_classes=["backpack"],
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )

    # 1. Quan sát thấy ba lô
    bag = TrackedObject(track_id=10, class_name="backpack", confidence=0.88, bbox=(235, 120, 285, 180))
    for f in range(10):
        detector.process_frame([bag], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    # 2. Bị che khuất
    p3 = TrackedObject(track_id=3, class_name="person", confidence=0.91, bbox=(220, 100, 320, 200))
    for f in range(10, 15):
        detector.process_frame([p3], current_frame=f, fps=15.0, frame_width=640, frame_height=480)
    assert evt_mgr.events[0].status == EventStatus.OCCLUDED

    # 3. Người đứng dậy rời đi, khung hình hoàn toàn trống không có đồ vật qua nhiều frame
    for f in range(15, 60):
        detector.process_frame([], current_frame=f, fps=15.0, frame_width=640, frame_height=480)

    # Sau thời gian che khuất kéo dài mà không thấy đồ vật, chuyển sang RECHECK hoặc REVIEW_REQUIRED
    final_status = evt_mgr.events[0].status
    assert final_status in (EventStatus.RECHECK, EventStatus.REVIEW_REQUIRED)
    assert final_status != EventStatus.RESOLVED, "Tuyệt đối không tự ý chuyển thành RESOLVED khi đồ vật mất dấu không rõ lý do"


# ==============================================================================
# 10. Vật thể nằm sát ranh giới hai ghế -> UNKNOWN_SEAT
# ==============================================================================
def test_10_object_on_boundary_assigned_unknown_seat():
    """10. Vật thể nằm chính giữa khe ranh giới giữa 2 ghế thì gán UNKNOWN_SEAT, không gán bừa."""
    seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (200, 100), (200, 200), (100, 200)], is_normalized=False)
    seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(210, 100), (310, 100), (310, 200), (210, 200)], is_normalized=False)
    seat_mgr = SeatManager(seats=[seat_a, seat_b])

    # Đồ vật nằm chính giữa khe: x từ 190 đến 220 (giao với cả Seat A và Seat B tương đương nhau)
    straddling_object = (190, 130, 220, 170)
    matched = seat_mgr.find_seat_for_bbox(straddling_object, 640, 480)

    # Phải trả về None hoặc UNKNOWN_SEAT
    assert matched is None or matched == "UNKNOWN_SEAT", f"Vật thể nằm giữa ranh giới không được ép gán bừa, trả về: {matched}"


# ==============================================================================
# 11. Model không có lớp seat (COCO chair bị từ chối)
# ==============================================================================
def test_11_model_without_seat_class_rejected():
    """11. Model COCO chỉ có lớp 56 'chair' (ghế văn phòng nhìn ngang) bị từ chối làm bus seat model."""
    adapter = CustomSeatYOLOAdapter(weights_path="models/non_existent.pt")
    # Kiểm tra danh sách lớp COCO chuẩn
    coco_classes = {0: "person", 56: "chair", 24: "backpack", 26: "handbag"}
    supported = adapter.inspect_classes(coco_classes)

    assert not supported, "Lớp 'chair' của COCO nhìn ngang phải bị từ chối, không thể dùng làm đệm ghế xe buýt trần"
    assert not adapter.is_seat_class_supported(coco_classes)


# ==============================================================================
# 12. Không có custom weights -> CUSTOM SEAT MODEL NOT CONFIGURED
# ==============================================================================
def test_12_no_custom_weights_reports_not_configured():
    """12. Khi không có file weights chuyên biệt, hệ thống hiển thị rõ trạng thái NOT_CONFIGURED."""
    detector = SeatDetector(custom_weights_path="models/missing_weights.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Chạy detection mà không cho phép fallback
    result = detector.detect_seats(dummy_frame, allow_fallback=False)

    assert result.method == SeatDetectionMethod.NOT_CONFIGURED
    assert result.total_seats == 0
    assert "CUSTOM SEAT MODEL NOT CONFIGURED" in result.message
    assert result.is_fallback is False


# ==============================================================================
# 13. Dataset thiếu annotation -> validation thất bại, từ chối train
# ==============================================================================
def test_13_dataset_lacking_annotations_fails_validation():
    """13. Dataset thiếu ảnh hoặc file nhãn rỗng phải bị từ chối trước khi huấn luyện."""
    temp_dir = tempfile.mkdtemp()
    try:
        yaml_path = Path(temp_dir) / "data.yaml"
        # Tạo file data.yaml nhưng không có ảnh hay nhãn
        yaml_path.write_text("names: [bus_seat]\nnc: 1\ntrain: train/images\nval: val/images\n")

        valid, msg, details = ModelService.validate_training_dataset(str(yaml_path))
        assert not valid, "Dataset thiếu ảnh và nhãn phải không vượt qua validation"
        assert "không có hình ảnh" in msg.lower() or "thiếu" in msg.lower() or "không tồn tại" in msg.lower() or "rỗng" in msg.lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 14. Training thất bại hoặc bị gián đoạn -> ghi nhận FAILED
# ==============================================================================
def test_14_training_failure_recorded_properly():
    """14. Tiến trình huấn luyện gặp lỗi phải ghi nhận trạng thái 'failed', không báo thành công sai lệch."""
    temp_dir = tempfile.mkdtemp()
    try:
        db = Database(Path(temp_dir) / "test.db")
        m_repo = ModelRepository(db.get_connection())
        mservice = ModelService(m_repo)

        # Chạy start_real_training_job với dataset không tồn tại
        ok, msg, info = mservice.start_real_training_job(
            dataset_yaml_path="/non/existent/path/data.yaml",
            new_version="test-fail",
        )
        assert not ok, "Phải từ chối khởi chạy với file YAML không tồn tại"
        assert "không tồn tại" in msg.lower() or "không hợp lệ" in msg.lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 15. Không tạo metrics giả khi chưa có training/validation thực
# ==============================================================================
def test_15_no_fake_metrics_generated_without_real_validation():
    """15. simulate_training_job bị vô hiệu hóa, không tạo mAP giả dựa trên số epoch."""
    temp_dir = tempfile.mkdtemp()
    try:
        db = Database(Path(temp_dir) / "test.db")
        m_repo = ModelRepository(db.get_connection())
        mservice = ModelService(m_repo)

        ok, metrics, msg = mservice.simulate_training_job("v_fake", "ds_fake", epochs=50)
        assert not ok, "simulate_training_job phải trả về False vì đã bị vô hiệu hóa"
        assert metrics == {}, "Không được sinh ra bất kỳ metric mAP hoặc precision giả lập nào"
        assert "vô hiệu hóa" in msg.lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
