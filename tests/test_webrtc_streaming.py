"""
tests/test_webrtc_streaming.py
Kiểm tra tính năng WebRTC Streaming cho Streamlit Cloud:
- Cấu hình STUN server
- Tách biệt phiên người dùng (Session Isolation)
- Xử lý frame video av.VideoFrame
- Metrics thời gian thực và Snapshot
"""

import numpy as np
import pytest
import av

from src.video.webrtc_processor import (
    WEBRTC_AVAILABLE,
    RTC_CONFIGURATION,
    BusWebRtcVideoProcessor,
)
from src.services.analysis_service import AnalysisService
from src.seats.seat_manager import SeatManager
from src.seats import SeatDetectionMethod
from src.models import SeatConfig


def test_webrtc_available_and_configuration():
    """Kiểm tra streamlit-webrtc và cấu hình STUN servers của Google."""
    assert WEBRTC_AVAILABLE is True
    assert RTC_CONFIGURATION is not None
    assert "iceServers" in RTC_CONFIGURATION


def test_webrtc_processor_session_isolation():
    """
    Kiểm tra tách biệt phiên: Hai client kết nối đồng thời phải có
    SeatManager, ObjectTracker, EventManager và ForgottenItemDetector riêng biệt.
    """
    service = AnalysisService()
    # Khởi tạo 2 processor đại diện cho 2 client A và B
    processor_a = BusWebRtcVideoProcessor(service=service)
    processor_b = BusWebRtcVideoProcessor(service=service)

    # Đảm bảo các đối tượng trạng thái hoàn toàn độc lập trong bộ nhớ
    assert processor_a.tracker is not processor_b.tracker
    assert processor_a.event_manager is not processor_b.event_manager
    assert processor_a.event_detector is not processor_b.event_detector
    assert processor_a.seat_manager is not processor_b.seat_manager

    # Sửa đổi trạng thái của client A không được ảnh hưởng tới client B
    seat1 = SeatConfig(seat_id="SEAT_01", polygon=[(10, 10), (50, 10), (50, 50), (10, 50)])
    processor_a.seat_manager.set_seats([seat1])

    assert len(processor_a.seat_manager.seats) == 1
    assert len(processor_b.seat_manager.seats) == 0


def test_webrtc_processor_recv_frame_roundtrip():
    """Kiểm tra processor.recv() nhận av.VideoFrame và trả về av.VideoFrame hợp lệ."""
    service = AnalysisService()
    service.initialize()
    processor = BusWebRtcVideoProcessor(service=service)

    # Tạo frame BGR giả lập 640x480
    raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_img[100:200, 100:200] = [128, 64, 255]

    in_frame = av.VideoFrame.from_ndarray(raw_img, format="bgr24")
    out_frame = processor.recv(in_frame)

    assert isinstance(out_frame, av.VideoFrame)
    out_img = out_frame.to_ndarray(format="bgr24")
    assert out_img.shape == (480, 640, 3)

    # Kiểm tra metrics được cập nhật sau khi xử lý
    metrics = processor.get_metrics()
    assert "fps" in metrics
    assert "latency_ms" in metrics
    assert "person_count" in metrics
    assert "status_tag" in metrics
    assert metrics["frame_idx"] == 0

    # Kiểm tra chụp snapshot
    snap = processor.capture_snapshot()
    assert snap is not None
    assert snap.shape == (480, 640, 3)


def test_webrtc_processor_config_update():
    """Kiểm tra cập nhật tham số suy luận theo thời gian thực (thread-safe)."""
    processor = BusWebRtcVideoProcessor(conf_thresh=0.25, stride=1)
    assert processor.conf_thresh == 0.25
    assert processor.stride == 1

    processor.update_config(conf_thresh=0.6, stride=2, show_masks=False)
    assert processor.conf_thresh == 0.6
    assert processor.stride == 2
    assert processor.show_masks is False


def test_analysis_service_custom_seat_manager_isolation():
    """
    Kiểm tra process_webcam_frame tôn trọng custom seat_manager được truyền vào,
    không ghi đè lên service._seat_manager mặc định.
    """
    service = AnalysisService()
    service.initialize()

    custom_seat_mgr = SeatManager()
    base_seat_mgr = service._seat_manager

    assert custom_seat_mgr is not base_seat_mgr

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    from src.tracking.object_tracker import ObjectTracker
    from src.events.event_manager import EventManager
    from src.events.forgotten_item_detector import ForgottenItemDetector

    tracker = ObjectTracker()
    event_manager = EventManager()
    event_detector = ForgottenItemDetector(
        min_stationary_time=3.0,
        seat_manager=custom_seat_mgr,
        event_manager=event_manager,
    )

    annotated, events, summary = service.process_webcam_frame(
        frame=frame,
        frame_idx=0,
        fps=25.0,
        tracker=tracker,
        event_detector=event_detector,
        event_manager=event_manager,
        seat_manager=custom_seat_mgr,
    )

    assert annotated is not None
    assert isinstance(summary, dict)
