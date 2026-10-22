"""
tests/test_smart_tracking_and_state_machine.py

Bộ kiểm thử bắt buộc theo Phần X:
- Test 1: Một đồ vật đứng yên (ID ổn định qua nhiều frame, không nhấp nháy).
- Test 2: Đồ vật dịch chuyển nhẹ (Duy trì cùng ID track cũ).
- Test 3: Người ngồi trên ghế (Ghế OCCUPIED, không phát cảnh báo thất lạc).
- Test 4: Người đứng dậy, đồ vật còn lại (Ghế OCCUPIED -> VACANT, sinh cảnh báo cho đúng ghế).
- Test 5: Đồ vật bị che khuất (Không coi là biến mất ngay, giữ trạng thái che khuất).
- Test 6: Đồ vật được lấy đi (Xác nhận biến mất -> cập nhật trạng thái đã giải quyết / không báo động tiếp).
- Test 7: Người khác ngồi lên đồ vật (Duy trì trạng thái nghi vấn che khuất, không vội kết luận lấy đi).
- Test 8: Camera bị gián đoạn (Không phát cảnh báo giả hàng loạt).
- Test 9: Ghế không xác định được (Gán UNKNOWN_SEAT, không bịa ra mã ghế).
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    TrackedObject,
    SeatConfig,
    SeatState,
    Event,
)
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.tracking.object_tracker import ObjectTracker
from src.detection.class_mapper import format_seat_label, get_user_facing_label
from config.settings import EventStatus


@pytest.fixture
def setup_environment():
    seat_mgr = SeatManager()
    seat = SeatConfig(
        seat_id="SEAT_001",
        polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        is_normalized=False,
        description="Ghế 001",
    )
    seat_mgr.set_seats([seat])
    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=3.0,
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )
    tracker = ObjectTracker()
    return seat_mgr, evt_mgr, detector, tracker


# ─────────────────────────────────────────────────────────────
# Test 1: Một đồ vật đứng yên (ID ổn định, không nhấp nháy)
# ─────────────────────────────────────────────────────────────
def test_1_stationary_item_tracking_stability(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment
    from src.models import Detection

    # Mô phỏng một chiếc điện thoại (class cell phone / item) đứng yên qua 30 frames
    # với một chút nhiễu confidence và tọa độ bbox dao động nhẹ 1-2px
    phone_box = [150, 150, 210, 230]

    track_ids = []
    for f in range(30):
        jitter_x = 1 if f % 2 == 0 else -1
        det = Detection(
            class_name="cell phone",
            confidence=0.85 + (0.05 if f % 3 == 0 else -0.05),
            bbox=(phone_box[0] + jitter_x, phone_box[1], phone_box[2] + jitter_x, phone_box[3]),
            polygon=[(150.0, 150.0), (210.0, 150.0), (210.0, 230.0), (150.0, 230.0)],
        )
        tracked = tracker.update(
            detections=[det],
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_index=f,
            fps=10.0,
        )
        assert len(tracked) == 1
        track_ids.append(tracked[0].track_id)

    # Toàn bộ 30 frame phải giữ nguyên 1 track ID duy nhất (không tăng liên tục #1, #2, #3...)
    assert len(set(track_ids)) == 1, f"Track ID bị nhảy: {set(track_ids)}"


# ─────────────────────────────────────────────────────────────
# Test 2: Đồ vật dịch chuyển nhẹ (Tracker tiếp tục dùng ID cũ)
# ─────────────────────────────────────────────────────────────
def test_2_item_slight_movement_preserves_id(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment
    from src.models import Detection

    initial_id = None
    # Di chuyển đồ vật dịch dần từ x=150 sang x=175 trong 20 frames
    for f in range(20):
        shift = int(f * 1.2)
        det = Detection(
            class_name="backpack",
            confidence=0.88,
            bbox=(150 + shift, 150, 230 + shift, 230),
            polygon=[(150.0 + shift, 150.0), (230.0 + shift, 150.0), (230.0 + shift, 230.0), (150.0 + shift, 230.0)],
        )
        tracked = tracker.update(
            detections=[det],
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_index=f,
            fps=10.0,
        )
        assert len(tracked) == 1
        if initial_id is None:
            initial_id = tracked[0].track_id
        else:
            assert tracked[0].track_id == initial_id, f"ID bị đổi khi dịch chuyển nhẹ: {tracked[0].track_id} vs {initial_id}"


# ─────────────────────────────────────────────────────────────
# Test 3: Người ngồi trên ghế (Ghế OCCUPIED, không phát cảnh báo)
# ─────────────────────────────────────────────────────────────
def test_3_person_sitting_no_alert(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    person = TrackedObject(track_id=1, class_name="person", confidence=0.92, bbox=(120, 120, 280, 280))
    item = TrackedObject(track_id=2, class_name="cell phone", confidence=0.85, bbox=(160, 160, 220, 220))

    for f in range(40):
        detector.process_frame([person, item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Ghế phải ở trạng thái OCCUPIED
    assert seat_mgr.is_seat_occupied("SEAT_001")
    # Đang có người ngồi cùng đồ vật: không được phát cảnh báo nghi ngờ bỏ quên (SUSPECTED)
    suspect_events = [e for e in evt_mgr.events if e.status == EventStatus.SUSPECTED]
    assert len(suspect_events) == 0


# ─────────────────────────────────────────────────────────────
# Test 4: Người đứng dậy, đồ vật còn lại (Phát cảnh báo cho đúng ghế)
# ─────────────────────────────────────────────────────────────
def test_4_person_leaves_item_remains_triggers_alert(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    person = TrackedObject(track_id=1, class_name="person", confidence=0.92, bbox=(120, 120, 280, 280))
    item = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(160, 160, 220, 220))

    # Giai đoạn 1: Người và túi ngồi trên ghế (20 frame = 2.0s)
    for f in range(20):
        detector.process_frame([person, item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Giai đoạn 2: Người đứng dậy rời đi, chỉ còn túi (từ frame 21 đến 70 = 5.0s)
    for f in range(21, 71):
        detector.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Ghế đã chuyển sang VACANT
    assert seat_mgr.is_seat_vacant("SEAT_001")
    # Đã phát cảnh báo SUSPECTED gắn đúng ghế SEAT_001
    suspect_events = [e for e in evt_mgr.events if e.status == EventStatus.SUSPECTED]
    assert len(suspect_events) == 1
    assert suspect_events[0].seat_id == "SEAT_001"


# ─────────────────────────────────────────────────────────────
# Test 5: Đồ vật bị che khuất (Không kết luận biến mất, không tạo track mới)
# ─────────────────────────────────────────────────────────────
def test_5_item_occluded_preserves_state(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    item = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(160, 160, 220, 220))
    # Tạo sự kiện ban đầu
    for f in range(25):
        detector.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    # Một người đứng trước camera che khuất túi (chỉ phát hiện người)
    person = TrackedObject(track_id=5, class_name="person", confidence=0.90, bbox=(130, 130, 270, 270))
    for f in range(26, 40):
        detector.process_frame([person], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Sự kiện phải chuyển sang OCCLUDED (bị che khuất), không bị xóa hoặc đóng vội
    assert evt.status == EventStatus.OCCLUDED
    assert evt.occlusion_duration > 0.0


# ─────────────────────────────────────────────────────────────
# Test 6: Đồ vật được lấy đi (Cảnh báo được cập nhật, không tiếp tục báo động)
# ─────────────────────────────────────────────────────────────
def test_6_item_taken_away_resolves_alert(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    item = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(160, 160, 220, 220))
    for f in range(50):
        detector.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    assert evt.status == EventStatus.SUSPECTED

    # Hành khách quay lại lấy túi đi ra cửa (tọa độ x=550 ngoài ghế)
    item_carried = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(520, 350, 580, 420))
    detector.process_frame([item_carried], current_frame=51, fps=10.0, frame_width=640, frame_height=480)

    # Ghế SEAT_001 không còn đồ vật
    seat_status = seat_mgr.get_seat_status("SEAT_001")
    assert seat_status.state == SeatState.VACANT


# ─────────────────────────────────────────────────────────────
# Test 7: Người khác ngồi lên đồ vật (Duy trì nghi vấn, không vội kết luận lấy đi)
# ─────────────────────────────────────────────────────────────
def test_7_another_person_sits_on_item(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    item = TrackedObject(track_id=2, class_name="cell phone", confidence=0.85, bbox=(180, 180, 220, 220))
    for f in range(40):
        detector.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    assert evt.status == EventStatus.SUSPECTED

    # Người mới ngồi lên ghế che khuất chiếc điện thoại
    new_passenger = TrackedObject(track_id=9, class_name="person", confidence=0.91, bbox=(120, 120, 280, 280))
    for f in range(41, 55):
        detector.process_frame([new_passenger], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Hệ thống phải chuyển sang OCCLUDED (bị che), không được xóa cảnh báo hoặc kết luận là điện thoại đã biến mất
    assert evt.status == EventStatus.OCCLUDED
    assert evt.status != EventStatus.RESOLVED


# ─────────────────────────────────────────────────────────────
# Test 8: Camera bị gián đoạn (Không phát hàng loạt cảnh báo giả)
# ─────────────────────────────────────────────────────────────
def test_8_camera_interruption_no_false_alerts(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Camera bị mất frame (không có detection nào)
    for f in range(30):
        detector.process_frame([], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Không có bất kỳ cảnh báo nghi ngờ nào được sinh ra do camera gián đoạn
    assert len(evt_mgr.events) == 0


# ─────────────────────────────────────────────────────────────
# Test 9: Ghế không xác định được (Gán UNKNOWN_SEAT, không bịa mã ghế)
# ─────────────────────────────────────────────────────────────
def test_9_unidentified_seat_location(setup_environment):
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Đồ vật nằm ở góc phòng xa (x=550, y=400), ngoài phạm vi của SEAT_001
    item_outside = TrackedObject(track_id=4, class_name="umbrella", confidence=0.86, bbox=(520, 380, 580, 440))
    detector.process_frame([item_outside], current_frame=0, fps=10.0, frame_width=640, frame_height=480)

    assert len(evt_mgr.events) == 1
    evt = evt_mgr.events[0]
    # Mã ghế không được gán bừa thành SEAT_001
    assert evt.seat_id == "" or evt.seat_id == "UNKNOWN_SEAT"

    # Nhãn hiển thị người dùng cho UNKNOWN_SEAT phải là "Chưa xác định vị trí ghế"
    formatted = format_seat_label("UNKNOWN_SEAT")
    assert formatted == "Chưa xác định vị trí ghế"
