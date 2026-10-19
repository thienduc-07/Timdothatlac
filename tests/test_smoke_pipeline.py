"""
tests/test_smoke_pipeline.py
Smoke test chạy thử toàn bộ pipeline VideoProcessor:
- Tạo video giả lập ngắn (synthetic video) bằng OpenCV.
- Chạy VideoProcessor với SeatManager và mock/pretrained detection.
- Kiểm tra video đầu ra được tạo ra thành công và có thể đọc lại.
"""

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SeatConfig, TrackedObject
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.video.video_processor import VideoProcessor
from src.video.video_reader import VideoReader


def create_synthetic_bus_video(filepath: Path, width: int = 640, height: int = 480, num_frames: int = 30):
    """Tạo file video MP4 đơn giản bằng OpenCV để kiểm thử pipeline."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, 10.0, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Vẽ một chiếc ghế giả lập
        cv2.rectangle(frame, (100, 100), (300, 300), (80, 80, 80), -1)
        cv2.putText(frame, f"Frame {i}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)

    out.release()


class DummyDetector:
    """Mock detector để test pipeline mà không cần GPU/tải trọng mạng."""
    def __init__(self):
        self.confidence_threshold = 0.35
        self._target_class_ids = [0, 24, 26]
        self.model = None


class DummyTracker:
    """Mock tracker tạo các vật thể giả lập qua các frame."""
    def __init__(self):
        self.frame_idx = 0

    def reset(self):
        self.frame_idx = 0

    def update(self, model, frame, frame_index, fps, confidence_threshold, target_class_ids):
        self.frame_idx = frame_index
        # Frame 0-10: người và balo trên ghế
        # Frame 11-29: chỉ còn balo trên ghế
        objects = []
        if frame_index <= 10:
            objects.append(TrackedObject(
                track_id=1,
                class_name="person",
                confidence=0.9,
                bbox=(110, 110, 290, 290),
            ))
        objects.append(TrackedObject(
            track_id=2,
            class_name="backpack",
            confidence=0.85,
            bbox=(150, 150, 220, 220),
        ))
        return objects


def test_video_processor_smoke(tmp_path):
    """Kiểm tra VideoProcessor hoàn thành việc xử lý video và xuất ra file hợp lệ."""
    video_input = tmp_path / "test_bus_cam.mp4"
    create_synthetic_bus_video(video_input, width=640, height=480, num_frames=20)

    seat = SeatConfig(
        seat_id="SEAT_01",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        is_normalized=False,
    )
    seat_mgr = SeatManager(seats=[seat], occupied_confirm_sec=0.2, vacant_confirm_sec=0.2, hold_sec=0.2)
    event_mgr = EventManager()
    detector = ForgottenItemDetector(
        min_stationary_time=0.5,
        seat_manager=seat_mgr,
        event_manager=event_mgr,
    )

    processor = VideoProcessor(
        detector=DummyDetector(),
        tracker=DummyTracker(),
        event_detector=detector,
        event_manager=event_mgr,
        seat_manager=seat_mgr,
    )

    result = processor.process_video(str(video_input))

    assert result.success is True
    assert result.total_frames_processed > 0
    assert Path(result.output_video_path).exists()
    assert Path(result.output_video_path).stat().st_size > 0

    # Kiểm tra video đầu ra đọc lại được
    reader = VideoReader(result.output_video_path)
    meta = reader.open()
    assert meta.is_valid is True
    assert meta.total_frames > 0
    reader.close()
