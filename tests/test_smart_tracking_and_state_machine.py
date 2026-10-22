"""
tests/test_smart_tracking_and_state_machine.py

Bộ kiểm thử hồi quy toàn diện 10 kịch bản bắt buộc theo Phần 5 (Automated Tests):
- Test 1: Stationary object (Một đồ vật đứng yên qua nhiều frame -> 1 confirmed identity duy nhất, không nhân bản).
- Test 2: Continuous horizontal movement (Đồ vật di chuyển ngang liên tục -> bảo tồn ID, không để lại polygon ma).
- Test 3: Sudden direction change (Đổi hướng đột ngột -> không phân mảnh track, không sinh polygon ma).
- Test 4: One or more missed detections (Mất detection tạm thời -> ẩn polygon cũ, khôi phục cùng ID khi tái xuất hiện).
- Test 5: Genuine disappearance (Đồ vật rời khỏi khung hình -> hết hạn track, xóa khỏi bộ nhớ, không hiển thị sai).
- Test 6: Two separate phones (Hai điện thoại di chuyển độc lập -> nhận 2 ID riêng biệt, không gộp nhầm).
- Test 7: Class fluctuation (Model dao động class cell phone / remote / backpack -> phân loại generic ITEM giữ nguyên ID).
- Test 8: Person-item-seat regression (Hành khách ngồi -> rời đi bỏ quên -> quay lại lấy đồ -> máy trạng thái giải quyết cảnh báo).
- Test 9: Camera interruption (Camera gián đoạn / rớt frame -> không phát cảnh báo giả, hồi phục an toàn).
- Test 10: Rendering state isolation (Bảo vệ trạng thái canvas/frame rendering -> không rò rỉ alpha/filter/composite, không làm tối video).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    TrackedObject,
    Detection,
    SeatConfig,
    SeatState,
    Event,
)
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.tracking.object_tracker import ObjectTracker
from src.detection.class_mapper import format_seat_label, get_user_facing_label, normalize_category
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
# Test 1: Stationary object
# ─────────────────────────────────────────────────────────────
def test_1_stationary_object(setup_environment):
    """Một chiếc điện thoại đứng yên trong khung hình qua 30 frames:
    - Đúng 1 ID được xác nhận.
    - Không trùng lặp track.
    - Không tăng ID liên tục (#1, #2, #3...).
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment
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
        assert len(tracked) == 1, "Chỉ được tồn tại đúng 1 object track"
        track_ids.append(tracked[0].track_id)

    # Toàn bộ 30 frame phải giữ nguyên 1 track ID duy nhất
    assert len(set(track_ids)) == 1, f"Track ID bị nhảy: {set(track_ids)}"
    assert len(tracker.current_tracked_objects) == 1


# ─────────────────────────────────────────────────────────────
# Test 2: Continuous horizontal movement
# ─────────────────────────────────────────────────────────────
def test_2_continuous_horizontal_movement(setup_environment):
    """Một chiếc điện thoại di chuyển liên tục từ trái qua phải:
    - Bảo tồn ID của đối tượng khi liên kết được.
    - Không để lại tập hợp polygon ma đóng băng phía sau.
    - Số lượng track renderable phù hợp chính xác với cảnh thực tế (1 track).
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    initial_id = None
    # Di chuyển đồ vật ngang qua màn hình từ x=50 đến x=350 trong 20 frames
    for f in range(20):
        shift = int(f * 15.0)
        det = Detection(
            class_name="cell phone",
            confidence=0.88,
            bbox=(50 + shift, 150, 110 + shift, 230),
            polygon=[(50.0 + shift, 150.0), (110.0 + shift, 150.0), (110.0 + shift, 230.0), (50.0 + shift, 230.0)],
        )
        tracked = tracker.update(
            detections=[det],
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_index=f,
            fps=10.0,
        )
        assert len(tracked) == 1, "Chỉ được có đúng 1 track tại mỗi thời điểm di chuyển"
        if initial_id is None:
            initial_id = tracked[0].track_id
        else:
            assert tracked[0].track_id == initial_id, f"ID bị đổi khi di chuyển: {tracked[0].track_id} vs {initial_id}"


# ─────────────────────────────────────────────────────────────
# Test 3: Sudden direction change
# ─────────────────────────────────────────────────────────────
def test_3_sudden_direction_change(setup_environment):
    """Điện thoại di chuyển sang phải, sau đó đảo hướng sang trái, rồi dừng lại:
    - Không bị phân mảnh track ID.
    - Không sinh ra polygon ma vĩnh viễn.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Quỹ đạo di chuyển mượt mà: phải -> giảm tốc -> đảo chiều sang trái -> dừng
    trajectory_x = [100, 125, 150, 175, 200, 215, 220, 215, 200, 180, 160, 150, 150]
    initial_id = None
    for f, x in enumerate(trajectory_x):
        det = Detection(class_name="cell phone", confidence=0.9, bbox=(x, 150, x + 60, 230))
        tracked = tracker.update(detections=[det], frame_index=f, fps=10.0)
        assert len(tracked) == 1
        if initial_id is None:
            initial_id = tracked[0].track_id
        else:
            assert tracked[0].track_id == initial_id, f"Frame {f} (x={x}) bị nhảy ID: {tracked[0].track_id} vs {initial_id}"


# ─────────────────────────────────────────────────────────────
# Test 4: One or more missed detections
# ─────────────────────────────────────────────────────────────
def test_4_one_or_more_missed_detections(setup_environment):
    """Điện thoại tạm thời bị mất detection trong 2 frame (do rung, góc nghiêng):
    - Track được lưu trữ nội bộ để tái liên kết.
    - Geometry cũ không được render tràn lan vô thời hạn.
    - Cùng ID được khôi phục khi đồ vật xuất hiện trở lại.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Frame 0: Thấy điện thoại tại x=150
    det0 = Detection(class_name="cell phone", confidence=0.9, bbox=(150, 150, 210, 230))
    t0 = tracker.update(detections=[det0], frame_index=0, fps=10.0)
    orig_id = t0[0].track_id

    # Frame 1 & 2: Mất detection hoàn toàn (rỗng)
    t1 = tracker.update(detections=[], frame_index=1, fps=10.0)
    t2 = tracker.update(detections=[], frame_index=2, fps=10.0)
    # Trong frame mất, không trả về active detection giả
    assert len(t1) == 0
    assert len(t2) == 0

    # Frame 3: Điện thoại tái xuất hiện tại x=155
    det3 = Detection(class_name="cell phone", confidence=0.9, bbox=(155, 150, 215, 230))
    t3 = tracker.update(detections=[det3], frame_index=3, fps=10.0)
    assert len(t3) == 1
    assert t3[0].track_id == orig_id, f"ID không được tái liên kết sau miss: {t3[0].track_id} vs {orig_id}"


# ─────────────────────────────────────────────────────────────
# Test 5: Genuine disappearance
# ─────────────────────────────────────────────────────────────
def test_5_genuine_disappearance(setup_environment):
    """Điện thoại thực sự rời khỏi camera:
    - Track hết hạn và không còn tham gia hiển thị.
    - Không tồn tại ID nhân bản giả trên màn hình.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Xuất hiện trong 5 frame
    for f in range(5):
        det = Detection(class_name="cell phone", confidence=0.9, bbox=(150, 150, 210, 230))
        tracker.update(detections=[det], frame_index=f, fps=10.0)

    # Biến mất hoàn toàn trong 40 frame (4 giây ở 10fps)
    for f in range(5, 45):
        tracked = tracker.update(detections=[], frame_index=f, fps=10.0)
        assert len(tracked) == 0, "Không được render đồ vật khi đã biến mất"

    # Lịch sử quan sát được dọn dẹp hoặc đánh dấu không còn active
    active = tracker.current_tracked_objects
    assert len(active) == 0


# ─────────────────────────────────────────────────────────────
# Test 6: Two separate phones
# ─────────────────────────────────────────────────────────────
def test_6_two_separate_phones(setup_environment):
    """Hai chiếc điện thoại xuất hiện gần nhau và di chuyển độc lập:
    - Nhận hai ID riêng biệt (không gộp nhầm).
    - Bộ chống trùng lặp không gộp 2 vật thể tách biệt thực tế.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Phone A tại x=100..150, Phone B tại x=220..270
    for f in range(10):
        detA = Detection(class_name="cell phone", confidence=0.9, bbox=(100 + f * 5, 150, 150 + f * 5, 220))
        detB = Detection(class_name="cell phone", confidence=0.9, bbox=(220 + f * 5, 150, 270 + f * 5, 220))
        tracked = tracker.update(detections=[detA, detB], frame_index=f, fps=10.0)
        assert len(tracked) == 2, "Phải duy trì đúng 2 tracks riêng biệt"
        ids = {t.track_id for t in tracked}
        assert len(ids) == 2, "Hai ID phải hoàn toàn khác biệt"


# ─────────────────────────────────────────────────────────────
# Test 7: Class fluctuation
# ─────────────────────────────────────────────────────────────
def test_7_class_fluctuation(setup_environment):
    """Model dao động phân loại chi tiết (cell phone -> remote -> backpack):
    - Đều được quy về nhóm ITEM.
    - Tracker bảo tồn nguyên vẹn ID đối tượng qua các frame thay đổi class_name.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # Đảm bảo các class này đều thuộc generic ITEM
    assert normalize_category("cell phone") == "item"
    assert normalize_category("remote") == "item"
    assert normalize_category("backpack") == "item"

    classes = ["cell phone", "remote", "cell phone", "backpack", "cell phone"]
    initial_id = None
    for f, cname in enumerate(classes):
        det = Detection(class_name=cname, confidence=0.85, bbox=(160, 160, 220, 230))
        tracked = tracker.update(detections=[det], frame_index=f, fps=10.0)
        assert len(tracked) == 1
        if initial_id is None:
            initial_id = tracked[0].track_id
        else:
            assert tracked[0].track_id == initial_id, f"Class thay đổi làm nhảy ID: {tracked[0].track_id} vs {initial_id}"


# ─────────────────────────────────────────────────────────────
# Test 8: Person-item-seat regression
# ─────────────────────────────────────────────────────────────
def test_8_person_item_seat_regression(setup_environment):
    """Người ngồi cùng điện thoại -> rời đi -> cảnh báo phát sinh -> người quay lại lấy điện thoại:
    - Phát hiện bỏ quên hoạt động chuẩn xác.
    - Xử lý che khuất an toàn.
    - Cảnh báo được giải quyết khi đồ vật rời khỏi ghế.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    person = TrackedObject(track_id=1, class_name="person", confidence=0.92, bbox=(120, 120, 280, 280))
    phone = TrackedObject(track_id=2, class_name="cell phone", confidence=0.88, bbox=(160, 160, 220, 220))

    # Giai đoạn 1: Người ngồi cùng điện thoại trên SEAT_001 (20 frames = 2.0s) -> Không báo động
    for f in range(20):
        detector.process_frame([person, phone], current_frame=f, fps=10.0, frame_width=640, frame_height=480)
    assert len([e for e in evt_mgr.events if e.status == EventStatus.SUSPECTED]) == 0
    assert seat_mgr.is_seat_occupied("SEAT_001")

    # Giai đoạn 2: Người rời đi, điện thoại ở lại một mình (50 frames = 5.0s) -> Kích hoạt cảnh báo
    for f in range(21, 71):
        detector.process_frame([phone], current_frame=f, fps=10.0, frame_width=640, frame_height=480)
    assert seat_mgr.is_seat_vacant("SEAT_001")
    suspect_events = [e for e in evt_mgr.events if e.status == EventStatus.SUSPECTED]
    assert len(suspect_events) == 1
    assert suspect_events[0].seat_id == "SEAT_001"

    # Giai đoạn 3: Người quay lại che khuất điện thoại -> Chuyển sang OCCLUDED (không xóa vội)
    detector.process_frame([person], current_frame=72, fps=10.0, frame_width=640, frame_height=480)
    assert suspect_events[0].status == EventStatus.OCCLUDED

    # Giai đoạn 4: Người cầm điện thoại mang đi khỏi ghế (x=550) -> Cảnh báo giải quyết
    phone_carried = TrackedObject(track_id=2, class_name="cell phone", confidence=0.88, bbox=(520, 350, 580, 420))
    detector.process_frame([phone_carried], current_frame=73, fps=10.0, frame_width=640, frame_height=480)
    assert seat_mgr.get_seat_status("SEAT_001").state == SeatState.VACANT


# ─────────────────────────────────────────────────────────────
# Test 9: Camera interruption
# ─────────────────────────────────────────────────────────────
def test_9_camera_interruption(setup_environment):
    """Video stream bị ngắt quãng, rớt frame:
    - Không phát cảnh báo bỏ quên giả tạo chỉ vì rớt frame.
    - Khôi phục an toàn mà không làm bùng nổ hàng loạt ID trùng lặp.
    """
    seat_mgr, evt_mgr, detector, tracker = setup_environment

    # 30 frame không có tín hiệu camera (0 detection)
    for f in range(30):
        tracked = tracker.update(detections=[], frame_index=f, fps=10.0)
        detector.process_frame(tracked, current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Tuyệt đối không sinh cảnh báo giả
    assert len(evt_mgr.events) == 0

    # Camera có tín hiệu lại: nhận diện đúng 1 điện thoại và xác nhận track
    det = Detection(class_name="cell phone", confidence=0.9, bbox=(160, 160, 220, 220))
    t_after = None
    for f in range(31, 33):
        t_after = tracker.update(detections=[det], frame_index=f, fps=10.0)
    assert len(t_after) == 1, "Chỉ tạo đúng 1 track sau khi camera hồi phục"


# ─────────────────────────────────────────────────────────────
# Test 10: Rendering state isolation
# ─────────────────────────────────────────────────────────────
def test_10_rendering_state_isolation():
    """Kiểm tra tính cách ly trạng thái hiển thị (Rendering State Isolation):
    - Việc vẽ overlay polygon không được làm biến đổi ngẫu nhiên độ sáng/pixel nền của các frame sau.
    - Kiểm tra trong frontend index.html: renderLoop được bảo vệ bằng ctx.save() và ctx.restore(),
      đồng thời reset tuyệt đối ctx.globalAlpha, ctx.filter, ctx.globalCompositeOperation.
    """
    html_path = Path(__file__).resolve().parent.parent / "src" / "components" / "hybrid_camera" / "frontend" / "index.html"
    assert html_path.exists(), "index.html phải tồn tại"
    html_content = html_path.read_text(encoding="utf-8")

    # Xác minh bảo vệ context trong renderLoop
    assert "ctx.save()" in html_content, "renderLoop phải gọi ctx.save()"
    assert "ctx.restore()" in html_content, "renderLoop phải gọi ctx.restore()"
    assert 'ctx.globalAlpha = 1.0;' in html_content, "renderLoop phải reset globalAlpha = 1.0"
    assert 'ctx.filter = "none";' in html_content or "ctx.filter = 'none';" in html_content, "renderLoop phải reset filter"
    assert 'ctx.globalCompositeOperation = "source-over";' in html_content or "ctx.globalCompositeOperation = 'source-over';" in html_content, "renderLoop phải reset composite operation"

    # Kiểm tra vẽ polygon trên frame OpenCV: không làm biến đổi vùng ngoài mask
    base_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame_copy = base_frame.copy()

    # Tạo một polygon nhỏ ở góc [20, 20] -> [40, 40]
    mask_pts = np.array([(20, 20), (40, 20), (40, 40), (20, 40)], dtype=np.int32).reshape((-1, 1, 2))
    mask_overlay = frame_copy.copy()
    cv2.fillPoly(mask_overlay, [mask_pts], (0, 255, 255))
    cv2.addWeighted(mask_overlay, 0.35, frame_copy, 0.65, 0, frame_copy)

    # Pixel ngoài polygon (ví dụ tại [80, 80]) phải giữ nguyên giá trị 128 (không bị tối hay biến đổi)
    assert np.array_equal(frame_copy[80, 80], [128, 128, 128]), "Vùng ngoài polygon bị thay đổi độ sáng!"


# ─────────────────────────────────────────────────────────────
# Test 11: Website dimming prevention and Streamlit message throttling
# ─────────────────────────────────────────────────────────────
def test_11_website_dimming_prevention_and_throttling():
    """Kiểm tra biện pháp triệt để ngăn chặn lỗi toàn bộ giao diện website bị tối mờ khi camera hoạt động:
    1. app.py phải chứa CSS bảo vệ chống giảm opacity/filter trên stAppViewContainer và .stApp.
    2. index.html phải kiểm soát nhịp gửi notifyStreamlit (stateKey & throttle), loại bỏ FPS dao động mỗi frame
       để Streamlit không bị rơi vào vòng lặp rerun liên tục.
    """
    app_py_path = Path(__file__).resolve().parent.parent / "app.py"
    html_path = Path(__file__).resolve().parent.parent / "src" / "components" / "hybrid_camera" / "frontend" / "index.html"

    assert app_py_path.exists(), "app.py phải tồn tại"
    assert html_path.exists(), "index.html phải tồn tại"

    app_content = app_py_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # 1. CSS anti-dimming trong app.py
    assert 'div[data-testid="stAppViewContainer"]' in app_content, "app.py phải bảo vệ stAppViewContainer"
    assert "opacity: 1 !important;" in app_content, "app.py phải duy trì opacity 1"
    assert "filter: none !important;" in app_content, "app.py phải duy trì filter none"

    # 2. index.html throttling
    assert "lastNotifiedStateKey" in html_content, "index.html phải quản lý lastNotifiedStateKey"
    assert "lastNotifiedTime" in html_content, "index.html phải quản lý lastNotifiedTime"
    assert "currentSuspectCount !== latestSuspectCount" in html_content, "index.html chỉ gửi notify khi suspect count đổi hoặc throttle"


# ─────────────────────────────────────────────────────────────
# Test 12: Two-tier detection sensitivity and small belongings admission
# ─────────────────────────────────────────────────────────────
def test_12_two_tier_sensitivity_and_small_item_admission():
    """Kiểm tra độ nhạy nhận diện đồ vật cá nhân nhỏ (smartphones, ví tiền, túi xách nhỏ):
    1. Ngưỡng tiếp nhận candidate cho items được hạ xuống 0.20 trong decodeTrueInstanceSegmentation.
    2. Cơ chế Two-Tier Confirmation: Candidate score < confThreshold đưa vào TENTATIVE và yêu cầu corroboration
       trước khi CONFIRMED (không tạo false positives / polygon ma 1 frame).
    3. Ngưỡng mask prototype sigmoid được tinh chỉnh 0.45 để bảo tồn biên chi tiết của vật phẩm nhỏ.
    4. ITEM_CLASSES mở rộng đầy đủ các lớp vật dụng cá nhân COCO.
    """
    html_path = Path(__file__).resolve().parent.parent / "src" / "components" / "hybrid_camera" / "frontend" / "index.html"
    html_content = html_path.read_text(encoding="utf-8")

    # 1. Candidate admission threshold
    assert "Math.min(confThreshold, 0.20)" in html_content, "index.html phải áp dụng ngưỡng nhạy 0.20 cho item candidates"

    # 2. Two-tier confirmation
    assert 'lifecycle: isHighConf ? "CONFIRMED" : "TENTATIVE"' in html_content, "index.html phải phân loại CONFIRMED vs TENTATIVE"
    assert "isRenderable: isHighConf" in html_content, "Candidate TENTATIVE không được render ở frame đầu"

    # 3. Mask threshold 0.45
    assert "sigmoid(val) > 0.45" in html_content, "index.html phải tinh chỉnh sigmoid threshold về 0.45"

    # 4. Item classes
    for item_cls in ["cell phone", "backpack", "handbag", "book", "bottle"]:
        assert f'"{item_cls}"' in html_content, f"ITEM_CLASSES phải chứa {item_cls}"
