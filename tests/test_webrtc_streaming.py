"""
tests/test_webrtc_streaming.py
Kiểm tra tính năng WebRTC Streaming cho Streamlit Cloud:
- Cấu hình STUN server
- Tách biệt phiên người dùng (Session Isolation)
- Xử lý frame video av.VideoFrame và Decoupled Queue
- Bỏ qua frame cũ khi quá tải (Frame Dropping)
- Metrics thời gian thực (Input FPS, Infer FPS, Output FPS, Latency Avg & P95, Dropped)
- Snapshot và dừng luồng an toàn (Lifecycle & on_ended)
"""

import time
import numpy as np
import pytest
import av

from src.video.webrtc_processor import (
    WEBRTC_AVAILABLE,
    RTC_CONFIGURATION,
    BusWebRtcVideoProcessor,
    ActiveSessionTracker,
    get_system_telemetry,
)
from src.services.analysis_service import AnalysisService
from src.seats.seat_manager import SeatManager
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
    processor_a = BusWebRtcVideoProcessor(service=service)
    processor_b = BusWebRtcVideoProcessor(service=service)

    try:
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
    finally:
        processor_a.stop()
        processor_b.stop()


def test_webrtc_processor_recv_frame_roundtrip_and_metrics():
    """Kiểm tra processor.recv() không chặn luồng, trả về VideoFrame hợp lệ và đo lường metrics."""
    service = AnalysisService()
    service.initialize()
    processor = BusWebRtcVideoProcessor(service=service, infer_imgsz=480)

    try:
        # Tạo frame BGR giả lập 640x480
        raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_img[100:200, 100:200] = [128, 64, 255]

        in_frame = av.VideoFrame.from_ndarray(raw_img, format="bgr24")
        
        t0 = time.perf_counter()
        out_frame = processor.recv(in_frame)
        recv_duration_ms = (time.perf_counter() - t0) * 1000

        # recv() phải chạy cực nhanh (dưới 10ms trên CPU để không nghẽn WebRTC)
        assert recv_duration_ms < 20.0
        assert isinstance(out_frame, av.VideoFrame)
        out_img = out_frame.to_ndarray(format="bgr24")
        assert out_img.shape == (480, 640, 3)

        # Chờ worker thread xử lý frame
        time.sleep(0.15)

        # Kiểm tra đầy đủ các metrics chi tiết theo yêu cầu 8
        metrics = processor.get_metrics()
        assert "input_fps" in metrics
        assert "infer_fps" in metrics
        assert "output_fps" in metrics
        assert "latency_avg_ms" in metrics
        assert "latency_p95_ms" in metrics
        assert "dropped_frames" in metrics
        assert "active_sessions" in metrics
        assert "cpu_percent" in metrics
        assert "process_rss_mb" in metrics
        assert "status_tag" in metrics
        assert metrics["active_sessions"] >= 1

        # Kiểm tra chụp snapshot
        snap = processor.capture_snapshot()
        assert snap is not None
        assert snap.shape == (480, 640, 3)
    finally:
        processor.stop()
        assert processor.is_online is False
        assert "OFFLINE" in processor.get_metrics()["status_tag"]


def test_webrtc_processor_frame_dropping():
    """Kiểm tra cơ chế drop frame cũ khi hàng đợi đầy (Queue maxsize=1)."""
    service = AnalysisService()
    service.initialize()
    processor = BusWebRtcVideoProcessor(service=service)

    try:
        raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
        in_frame = av.VideoFrame.from_ndarray(raw_img, format="bgr24")

        # Gửi liên tiếp 5 frame với khoảng cách cực ngắn mà không sleep
        for _ in range(5):
            processor.recv(in_frame)

        # Vì worker AI tốn khoảng 30-50ms/frame, một số frame trong 5 frame phải bị drop
        time.sleep(0.1)
        metrics = processor.get_metrics()
        assert metrics["dropped_frames"] >= 1
    finally:
        processor.stop()


def test_webrtc_processor_config_update():
    """Kiểm tra cập nhật tham số suy luận theo thời gian thực (thread-safe)."""
    processor = BusWebRtcVideoProcessor(conf_thresh=0.25, stride=1)
    try:
        assert processor.conf_thresh == 0.25
        assert processor.stride == 1

        processor.update_config(conf_thresh=0.6, stride=2, show_masks=False, infer_imgsz=384)
        assert processor.conf_thresh == 0.6
        assert processor.stride == 2
        assert processor.show_masks is False
        assert processor.infer_imgsz == 384
    finally:
        processor.stop()


def test_system_telemetry():
    """Kiểm tra module đọc CPU, RAM và GPU."""
    telemetry = get_system_telemetry()
    assert "cpu_percent" in telemetry
    assert "process_rss_mb" in telemetry
    assert "system_mem_percent" in telemetry
    assert "device" in telemetry
    assert telemetry["process_rss_mb"] > 0
