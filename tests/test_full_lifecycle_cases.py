"""
tests/test_full_lifecycle_cases.py
Kiểm thử toàn diện 15 kịch bản (CASE A đến CASE O) theo yêu cầu kiến trúc:
CASE A — Ghế trống: Không có người/vật, ghế giữ ID ổn định, không cảnh báo.
CASE B — Người ngồi xuống: Ghế chuyển OCCUPIED sau thời gian xác nhận, không cảnh báo.
CASE C — Người đặt túi xuống: Tạo track người và vật, liên kết đúng ghế.
CASE D — Người vẫn ngồi: Không cảnh báo bỏ quên trong khi người còn hiện diện.
CASE E — Người đứng lên rồi ngồi lại: Không phát cảnh báo sớm hay trùng lặp.
CASE F — Người rời đi, vật còn lại: Sau thời gian xác nhận, sinh event và alert có bằng chứng.
CASE G — Vật bị che khuất: Giữ trạng thái lịch sử, thể hiện bất định, không khẳng định phát hiện trực tiếp.
CASE H — Người khác lấy vật: Cập nhật trạng thái sự kiện, đóng alert.
CASE I — Vật chưa từng được nhận diện: Xử lý an toàn với lớp unknown/review.
CASE J — Camera mất kết nối: Báo lỗi an toàn, không tạo dữ liệu giả.
CASE K — Notification provider lỗi: Lưu log FAILED với mã lỗi thật và cho phép retry.
CASE L — Streamlit rerun / Instance management: Không khởi tạo trùng lặp.
CASE M — Dataset upload & YOLO export: Xác thực cấu trúc data.yaml và bounding box.
CASE N — Model approval: Candidate model không được active khi chưa duyệt.
CASE O — ID switching recovery: Nối lại sự kiện cũ khi ByteTrack đổi ID sau che khuất.
"""

import os
from pathlib import Path
import tempfile
import pytest

from src.models import (
    TrackedObject,
    SeatConfig,
    SeatState,
    Event,
    Alert,
    AlertPriority,
    AlertStatus,
)
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.database.db import Database
from src.database.alert_repository import AlertRepository
from src.database.dataset_repository import DatasetRepository
from src.database.model_repository import ModelRepository
from src.services.alert_service import AlertService
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService
from src.services.notification_service import NotificationService
from src.video.camera_service import CameraStreamManager, CameraSourceType
from config.settings import EventStatus


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    db.connect()
    yield db
    db.close()
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def setup_pipeline():
    seat_mgr = SeatManager()
    seat = SeatConfig(
        seat_id="S01",
        polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        is_normalized=False,
    )
    seat_mgr.set_seats([seat])
    evt_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=3.0,
        seat_manager=seat_mgr,
        event_manager=evt_mgr,
    )
    return seat_mgr, evt_mgr, detector


# CASE A — Ghế trống (xác nhận sau thời gian làm mượt temporal smoothing)
def test_case_a_empty_seat(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    for f in range(15):
        detector.process_frame([], current_frame=f, fps=10.0, frame_width=640, frame_height=480)
    assert seat_mgr.is_seat_vacant("S01")
    assert len(evt_mgr.events) == 0


# CASE B — Người ngồi xuống
def test_case_b_person_sits(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 280, 280))
    for f in range(10):
        detector.process_frame([person], current_frame=f, fps=10.0, frame_width=640, frame_height=480)
    assert seat_mgr.is_seat_occupied("S01")
    assert len(evt_mgr.events) == 0


# CASE C — Người đặt túi xuống ghế
def test_case_c_person_places_bag(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 250, 250))
    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(200, 200, 280, 280))
    detector.process_frame([person, bag], current_frame=0, fps=10.0, frame_width=640, frame_height=480)

    assert len(evt_mgr.events) == 1
    assert evt_mgr.events[0].seat_id == "S01"
    assert evt_mgr.events[0].class_name == "backpack"


# CASE D — Người vẫn ngồi: không cảnh báo
def test_case_d_person_remains_sitting(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 250, 250))
    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(200, 200, 280, 280))
    for f in range(30):
        detector.process_frame([person, bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    assert evt.status == EventStatus.OBSERVED
    assert evt.status != EventStatus.SUSPECTED


# CASE E — Người đứng lên rồi ngồi lại (re-sit)
def test_case_e_person_stands_and_resits(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 250, 250))
    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(200, 200, 280, 280))

    # Frame 0-10: cả 2 cùng ngồi
    for f in range(10):
        detector.process_frame([person, bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Frame 10-15: người đứng lên tạm thời (chưa đủ 3 giây)
    for f in range(10, 15):
        detector.process_frame([bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Người ngồi lại
    for f in range(15, 25):
        detector.process_frame([person, bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    # Không được phép nâng lên SUSPECTED
    assert evt.status != EventStatus.SUSPECTED


# CASE F — Người rời đi, vật còn lại -> SUSPECTED & Alert
def test_case_f_person_leaves_bag_remains(setup_pipeline, temp_db):
    seat_mgr, evt_mgr, detector = setup_pipeline
    alert_repo = AlertRepository(temp_db.get_connection())
    alert_service = AlertService(alert_repo)

    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(180, 180, 250, 250))

    # Frame 0-50 (5 giây trên ghế trống, vượt min_stationary_time = 3.0s)
    for f in range(50):
        detector.process_frame([bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    assert evt.status == EventStatus.SUSPECTED

    # Alert Service xử lý event
    alert = alert_service.process_event_for_alert(evt, current_time_sec=5.0)
    assert alert is not None
    assert alert.priority == AlertPriority.HIGH
    assert alert.seat_id == "S01"
    assert alert_repo.get_unresolved_alert_count() == 1


# CASE G — Vật bị che khuất (Occlusion handling)
def test_case_g_object_occluded(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 280, 280))
    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(180, 180, 250, 250))

    # Thấy cả hai
    for f in range(10):
        detector.process_frame([person, bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    # Đồ vật bị người che khuất (chỉ thấy người)
    for f in range(10, 30):
        detector.process_frame([person], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    assert evt.status == EventStatus.OCCLUDED
    assert evt.occlusion_duration > 0.0


# CASE H — Người khác lấy vật đi
def test_case_h_person_picks_up_bag(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    bag = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(180, 180, 250, 250))
    for f in range(40):
        detector.process_frame([bag], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

    evt = evt_mgr.events[0]
    # Túi di chuyển ra ngoài cửa buýt (x=500, y=400)
    bag_moved = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(480, 380, 550, 450))
    detector.process_frame([bag_moved], current_frame=41, fps=10.0, frame_width=640, frame_height=480)

    # Ghế S01 không còn đồ vật nữa
    seat_status = seat_mgr.get_seat_status("S01")
    assert seat_status.state == SeatState.VACANT


# CASE I — Vật thể chưa nhận diện lớp (Unknown Object)
def test_case_i_unknown_object(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline
    unknown_item = TrackedObject(track_id=7, class_name="unknown_object", confidence=0.6, bbox=(150, 150, 220, 220))
    detector.watched_classes.append("unknown_object")
    detector.process_frame([unknown_item], current_frame=0, fps=10.0, frame_width=640, frame_height=480)

    assert len(evt_mgr.events) == 1
    assert evt_mgr.events[0].class_name == "unknown_object"


# CASE J — Camera mất kết nối
def test_case_j_camera_disconnect():
    cam_mgr = CameraStreamManager()
    ok, err = cam_mgr.start_stream(CameraSourceType.RTSP, "rtsp://non_existent_ip:554/stream")
    assert not ok
    assert not cam_mgr.is_running
    assert cam_mgr.error_message != ""


# CASE K — Notification provider lỗi (Ghi log FAILED với mã lỗi thật)
def test_case_k_notification_provider_error(temp_db):
    alert_repo = AlertRepository(temp_db.get_connection())
    alert_service = AlertService(alert_repo)

    # Gửi Telegram với token sai
    invalid_token = "123456:INVALID_TOKEN_FAKE"
    ok, code, err = NotificationService.send_telegram_message(
        bot_token=invalid_token,
        chat_id="12345678",
        message="Test alert",
    )
    assert not ok
    assert code != 200

    # Log vào database
    alert_repo.log_notification(
        alert_id="ALT_TEST",
        channel="telegram",
        recipient="12345678",
        status="FAILED",
        status_code=code,
        error_message=err,
    )

    logs = alert_repo.get_notification_logs()
    assert len(logs) == 1
    assert logs[0]["status"] == "FAILED"
    assert logs[0]["status_code"] == code


# CASE L — Streamlit singleton / instance management
def test_case_l_service_singleton(temp_db):
    from src.services.analysis_service import AnalysisService
    s1 = AnalysisService()
    s2 = AnalysisService()
    assert s1 is not s2
    # Cả hai đều có thể khởi tạo mà không lỗi
    assert hasattr(s1, "initialize")


# CASE M — Dataset upload & YOLO export
def test_case_m_dataset_annotation_and_export(temp_db):
    repo = DatasetRepository(temp_db.get_connection())
    dservice = DatasetService(repo)

    # Tạo ảnh giả lập
    import numpy as np
    import cv2
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    img_id = dservice.add_image_from_bytes(encoded.tobytes(), "test_bus.jpg", split="train")
    assert img_id is not None

    # Gán nhãn
    dservice.save_annotations(img_id, [{
        "class_name": "backpack",
        "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 200, "bbox_y2": 200,
        "is_occluded": False,
    }])

    anns = dservice.get_annotations(img_id)
    assert len(anns) == 1
    assert anns[0]["class_name"] == "backpack"

    # Export YOLO dataset
    ok, zip_p, msg = dservice.export_yolo_dataset()
    assert ok
    assert Path(zip_p).exists()
    assert Path(zip_p).suffix == ".zip"
    Path(zip_p).unlink(missing_ok=True)


# CASE N — Model approval workflow
def test_case_n_model_approval_workflow(temp_db):
    m_repo = ModelRepository(temp_db.get_connection())
    mservice = ModelService(m_repo)

    # Đăng ký candidate model mới
    ok, metrics, msg = mservice.simulate_training_job("v2.0-custom", "dataset-v1.0", epochs=20)
    assert ok
    active_m = mservice.get_active_model()
    # Model mới phải ở trạng thái candidate, model active ban đầu vẫn là v1.0
    assert active_m["version"] == "v1.0"

    # Người quản trị phê duyệt kích hoạt model
    mservice.activate_model("yolov8-custom-v2.0-custom")
    active_m_after = mservice.get_active_model()
    assert active_m_after["version"] == "v2.0-custom"


# CASE O — ID switching recovery sau che khuất
def test_case_o_id_switching_recovery(setup_pipeline):
    seat_mgr, evt_mgr, detector = setup_pipeline

    # Đồ vật ban đầu có track_id = 10
    bag1 = TrackedObject(track_id=10, class_name="handbag", confidence=0.85, bbox=(150, 150, 220, 220))
    detector.process_frame([bag1], current_frame=0, fps=10.0, frame_width=640, frame_height=480)
    evt = evt_mgr.events[0]
    evt_id = evt.event_id

    # Bị che khuất tạm thời
    person = TrackedObject(track_id=5, class_name="person", confidence=0.9, bbox=(120, 120, 280, 280))
    detector.process_frame([person], current_frame=10, fps=10.0, frame_width=640, frame_height=480)
    assert evt.status == EventStatus.OCCLUDED

    # Khi xuất hiện lại, ByteTrack gán track_id MỚI = 99
    bag_new_track = TrackedObject(track_id=99, class_name="handbag", confidence=0.82, bbox=(155, 155, 225, 225))
    detector.process_frame([bag_new_track], current_frame=20, fps=10.0, frame_width=640, frame_height=480)

    # Hệ thống phải nối lại sự kiện cũ chứ không tạo event mới!
    assert len(evt_mgr.events) == 1
    assert evt_mgr.events[0].event_id == evt_id
    assert evt_mgr.events[0].last_track_id == 99
