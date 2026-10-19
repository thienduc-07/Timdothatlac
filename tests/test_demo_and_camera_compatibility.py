"""
tests/test_demo_and_camera_compatibility.py
Unit tests verifying:
- Universal camera compatibility & safe caller
- Demo video generator robustness & multi-codec support
- Demo ground truth lifecycle correctness (passenger seats, leaves item, item suspected)
- Prevention of TypeError on Streamlit Cloud
"""

import pytest
import numpy as np
from pathlib import Path

from src.video.camera_service import (
    ensure_demo_video_exists,
    generate_bus_simulation_frame,
    get_demo_ground_truth,
)
from src.services.analysis_service import AnalysisService
from src.tracking.object_tracker import ObjectTracker
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from app import safe_process_webcam_frame


def test_simulation_frame_generation():
    """Kiểm tra frame mô phỏng xe buýt được sinh ra chuẩn kích thước và màu sắc."""
    for f_idx in [0, 30, 80, 120, 180]:
        frame = generate_bus_simulation_frame(f_idx, w=640, h=360)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8
        # Kiểm tra không phải frame đen hoàn toàn
        assert np.mean(frame) > 10


def test_demo_ground_truth_lifecycle():
    """Kiểm tra kịch bản mô phỏng đầy đủ: người lên xe -> ngồi -> rời đi -> để quên ba lô."""
    # Frame 10: Người đang đi vào
    gt_enter = get_demo_ground_truth(10)
    assert gt_enter["has_person"] is True
    assert gt_enter["has_bag"] is True

    # Frame 60: Người đang ngồi ở Ghế 1A
    gt_seated = get_demo_ground_truth(60)
    assert gt_seated["has_person"] is True
    assert gt_seated["seat_id"] == "Ghế 1A"

    # Frame 120: Người đang đứng dậy đi ra
    gt_leaving = get_demo_ground_truth(120)
    assert gt_leaving["has_person"] is True
    assert gt_leaving["has_bag"] is True

    # Frame 160: Người đã đi hẳn, ba lô bị bỏ quên ở Ghế 1A
    gt_abandoned = get_demo_ground_truth(160)
    assert gt_abandoned["has_person"] is False
    assert gt_abandoned["has_bag"] is True
    assert gt_abandoned["seat_id"] == "Ghế 1A"


def test_demo_video_file_created():
    """Kiểm tra file video demo được tạo thành công và có kích thước hợp lệ."""
    video_path = ensure_demo_video_exists(force_recreate=False)
    p = Path(video_path)
    assert p.exists()
    assert p.stat().st_size > 20000


def test_safe_process_webcam_frame_immunity():
    """Kiểm tra safe_process_webcam_frame không bị văng TypeError với tham số lạ."""
    service = AnalysisService()
    service.initialize()
    tracker = ObjectTracker()
    event_manager = EventManager()
    event_detector = ForgottenItemDetector(seat_manager=service.seat_manager, event_manager=event_manager)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Truyền tham số chuẩn cùng tham số tùy ý (giả lập cache phiên bản cũ/mới trên Streamlit Cloud)
    annotated, events, summary = safe_process_webcam_frame(
        service=service,
        frame=dummy_frame,
        frame_idx=0,
        fps=20.0,
        tracker=tracker,
        event_detector=event_detector,
        event_manager=event_manager,
        confidence_threshold=0.3,
        show_seats=True,
        show_persons=True,
        show_objects=True,
        show_masks=True,
        unexpected_cloud_cache_arg="should_not_crash",
        another_unrecognized_kwarg=999,
    )
    assert annotated.shape == (360, 640, 3)
    assert isinstance(events, list)
    assert isinstance(summary, dict)
