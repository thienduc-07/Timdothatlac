"""
tests/test_seat_manager.py
Kiểm thử module quản lý vùng ghế (Seat ROI) và occupancy tracking:
- Chuyển đổi tọa độ chuẩn hóa sang pixel.
- Tính toán diện tích đa giác và tỷ lệ giao cắt (overlap ratio).
- Gán đồ vật vào đúng ghế (tránh gán nhiều ghế).
- Làm mượt theo thời gian (temporal smoothing) cho trạng thái có người / trống.
- Cơ chế hold time chống rung lắc khi mất detection 1 frame.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SeatConfig, SeatState, TrackedObject
from src.seats.seat_config import (
    SeatConfigManager,
    calculate_polygon_area,
    calculate_overlap_ratio,
    is_point_in_seat,
)
from src.seats.seat_manager import SeatManager


class TestSeatGeometry:
    """Kiểm thử các hàm hình học cho Seat ROI."""

    def test_calculate_polygon_area(self):
        """Tính diện tích hình vuông 100x100 = 10,000."""
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        area = calculate_polygon_area(poly)
        assert area == pytest.approx(10000.0, rel=1e-2)

    def test_point_in_seat(self):
        """Kiểm tra điểm nằm trong và ngoài đa giác ghế."""
        poly = [(100, 100), (300, 100), (300, 300), (100, 300)]
        assert is_point_in_seat((200, 200), poly) is True  # Ở giữa
        assert is_point_in_seat((50, 50), poly) is False   # Bên ngoài
        assert is_point_in_seat((100, 100), poly) is True  # Trên đỉnh

    def test_calculate_overlap_ratio_full_overlap(self):
        """Bbox trùng khớp hoàn toàn với vùng ghế -> tỷ lệ 1.0."""
        poly = [(100, 100), (200, 100), (200, 200), (100, 200)]
        bbox = (100, 100, 200, 200)
        overlap_seat, overlap_bbox = calculate_overlap_ratio(bbox, poly)
        assert overlap_seat == pytest.approx(1.0, rel=1e-2)
        assert overlap_bbox == pytest.approx(1.0, rel=1e-2)

    def test_calculate_overlap_ratio_no_overlap(self):
        """Bbox hoàn toàn nằm ngoài vùng ghế -> tỷ lệ 0.0."""
        poly = [(100, 100), (200, 100), (200, 200), (100, 200)]
        bbox = (300, 300, 400, 400)
        overlap_seat, overlap_bbox = calculate_overlap_ratio(bbox, poly)
        assert overlap_seat == 0.0
        assert overlap_bbox == 0.0

    def test_normalized_coordinates_conversion(self):
        """Chuyển đổi tọa độ [0.0 - 1.0] thành tọa độ pixel chính xác."""
        seat = SeatConfig(
            seat_id="SEAT_01",
            polygon=[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)],
            is_normalized=True,
        )
        pixels = seat.get_pixel_polygon(frame_width=1000, frame_height=500)
        assert pixels == [(100, 100), (500, 100), (500, 400), (100, 400)]


class TestSeatOccupancySmoothing:
    """Kiểm thử logic xác định có người và làm mượt theo thời gian."""

    def test_seat_becomes_occupied_after_threshold_frames(self):
        """Ghế chuyển sang OCCUPIED khi có người liên tục đủ số frame."""
        seat = SeatConfig(
            seat_id="SEAT_01",
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
            is_normalized=False,
        )
        manager = SeatManager(
            seats=[seat],
            overlap_threshold=0.2,
            occupied_confirm_sec=0.2,  # 0.2s * 10fps = 2 frames
            vacant_confirm_sec=0.5,
            hold_sec=0.5,
        )

        person = TrackedObject(
            track_id=10,
            class_name="person",
            confidence=0.9,
            bbox=(120, 120, 280, 280),  # Nằm trọn trên ghế
        )

        # Frame 0 (t = 0.0s) -> 1 frame có người, chưa đủ 2 frame
        manager.update([person], frame_index=0, fps=10.0, frame_width=1000, frame_height=1000)
        assert manager.get_seat_status("SEAT_01").state != SeatState.OCCUPIED

        # Frame 1 (t = 0.1s) -> 2 frame có người, đủ ngưỡng
        manager.update([person], frame_index=1, fps=10.0, frame_width=1000, frame_height=1000)
        assert manager.get_seat_status("SEAT_01").state == SeatState.OCCUPIED
        assert manager.is_seat_occupied("SEAT_01") is True

    def test_seat_does_not_become_vacant_on_single_missing_frame(self):
        """Mất detection người 1 frame (nhiễu) -> không được chuyển ngay sang VACANT."""
        seat = SeatConfig(
            seat_id="SEAT_01",
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        )
        manager = SeatManager(
            seats=[seat],
            occupied_confirm_sec=0.1,
            vacant_confirm_sec=1.0,
            hold_sec=1.0,
        )

        person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 280, 280))

        # Cho người ngồi để xác nhận OCCUPIED
        for f in range(5):
            manager.update([person], frame_index=f, fps=10.0, frame_width=1000, frame_height=1000)
        assert manager.is_seat_occupied("SEAT_01") is True

        # Frame 5: Người bị mất detection trong 1 frame
        manager.update([], frame_index=5, fps=10.0, frame_width=1000, frame_height=1000)
        # Vẫn giữ OCCUPIED do chưa đủ vacant_confirm_sec và chưa hết hold_sec
        assert manager.get_seat_status("SEAT_01").state == SeatState.OCCUPIED

    def test_seat_becomes_vacant_after_person_leaves(self):
        """Người rời đi đủ thời gian -> ghế chuyển sang VACANT."""
        seat = SeatConfig(
            seat_id="SEAT_01",
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        )
        manager = SeatManager(
            seats=[seat],
            occupied_confirm_sec=0.1,
            vacant_confirm_sec=0.5,
            hold_sec=0.5,
        )

        person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 120, 280, 280))

        # Cho người ngồi
        for f in range(5):
            manager.update([person], frame_index=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Người rời đi trong 1.5s (15 frames)
        for f in range(5, 20):
            manager.update([], frame_index=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert manager.is_seat_vacant("SEAT_01") is True
        assert manager.is_seat_occupied("SEAT_01") is False

    def test_single_seat_assignment(self):
        """Vật thể chỉ được gán cho ghế phù hợp nhất, không bị gán trùng cho 2 ghế."""
        seat1 = SeatConfig(
            seat_id="SEAT_01",
            polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        )
        seat2 = SeatConfig(
            seat_id="SEAT_02",
            polygon=[(200, 100), (300, 100), (300, 200), (200, 200)],
        )
        manager = SeatManager(seats=[seat1, seat2])

        # Vật thể nằm chủ yếu trên SEAT_01
        item_bbox = (110, 110, 190, 190)
        assigned = manager.find_seat_for_bbox(item_bbox, frame_width=1000, frame_height=1000)
        assert assigned == "SEAT_01"

        # Vật thể nằm giữa ranh giới nhưng tâm lệch về SEAT_02
        item_bbox2 = (190, 120, 250, 180)  # Tâm là (220, 150) thuộc SEAT_02
        assigned2 = manager.find_seat_for_bbox(item_bbox2, frame_width=1000, frame_height=1000)
        assert assigned2 == "SEAT_02"

    def test_generate_bus_grid_flexible_layouts(self):
        """Kiểm tra bộ sinh sơ đồ ghế linh hoạt cho xe buýt."""
        # 1. Xe buýt 4 hàng x 2 dãy (1L - 1R) = 8 ghế
        grid8 = SeatConfigManager.generate_bus_grid(rows=4, left_cols=1, right_cols=1, naming_style="GHE")
        assert len(grid8) == 8
        assert grid8[0].seat_id == "Ghế 1A"
        assert grid8[1].seat_id == "Ghế 1B"
        assert grid8[-1].seat_id == "Ghế 4B"

        # 2. Xe buýt 4 hàng x 4 dãy (2L - 2R) = 16 ghế với ký hiệu SEAT
        grid16 = SeatConfigManager.generate_bus_grid(rows=4, left_cols=2, right_cols=2, naming_style="SEAT")
        assert len(grid16) == 16
        assert grid16[0].seat_id == "SEAT_01"
        assert grid16[-1].seat_id == "SEAT_16"

