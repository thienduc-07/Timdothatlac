"""
src/seats/seat_tracker.py
Module theo dõi và duy trì tính ổn định của các chiếc ghế trên xe buýt:
- Gán ID ổn định xuyên suốt (SEAT_1A, SEAT_1B... hoặc SEAT_001, SEAT_002...).
- Tuyệt đối không gán ID mới cho cùng một chiếc ghế ở mỗi frame.
- Xử lý sai lệch tọa độ nhỏ giữa các frame bằng bộ lọc làm mượt không gian (Spatial EMA Smoothing).
- Giữ vững cấu trúc ghế khi có người ngồi lên hoặc đứng dậy (cố định trên sàn xe buýt).
- Quản lý đồng thời nhiều hàng ghế trong khoang xe.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.models import SeatConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedSeat:
    """Đại diện cho một chiếc ghế đang được hệ thống theo dõi duy trì ổn định."""
    seat_id: str
    polygon: List[Tuple[float, float]]
    is_normalized: bool = True
    description: str = ""
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 1
    is_active: bool = True

    def to_seat_config(self) -> SeatConfig:
        return SeatConfig(
            seat_id=self.seat_id,
            polygon=self.polygon,
            is_normalized=self.is_normalized,
            description=self.description,
        )


def polygon_centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Tính trọng tâm (centroid) của polygon."""
    if not pts:
        return (0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def polygon_aabb(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Tính Axis-Aligned Bounding Box (x1, y1, x2, y2) của polygon."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def calculate_polygon_iou(
    poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]
) -> float:
    """Tính chỉ số IoU giữa hai đa giác ghế."""
    b1 = polygon_aabb(poly1)
    b2 = polygon_aabb(poly2)

    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area1 = max(0.0001, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0001, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / max(0.0001, union_area)


class SeatTracker:
    """
    Theo dõi và làm mượt vị trí từng ghế qua các frame video:
    - Duy trì ánh xạ seat_id ổn định.
    - Áp dụng Exponential Moving Average (EMA) để triệt tiêu rung lắc camera.
    - Giữ nguyên cấu trúc ghế khi bị người che khuất (Seat Permanence Guarantee).
    """

    def __init__(
        self,
        iou_threshold: float = 0.45,
        smoothing_alpha: float = 0.85,  # 85% vị trí đã ổn định, 15% tọa độ mới (chống giật)
        dist_threshold: float = 0.15,   # Ngưỡng khoảng cách trọng tâm (trong không gian normalized)
    ):
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.dist_threshold = dist_threshold
        self._tracked_seats: Dict[str, TrackedSeat] = {}
        self._seat_counter = 1

    @property
    def total_seats(self) -> int:
        return len(self._tracked_seats)

    def reset(self):
        """Reset toàn bộ tracker."""
        self._tracked_seats.clear()
        self._seat_counter = 1

    def register_seats(self, seats: List[SeatConfig], frame_number: int = 0):
        """Khởi tạo hoặc cập nhật danh sách ghế được cấu hình ban đầu."""
        for s in seats:
            self._tracked_seats[s.seat_id] = TrackedSeat(
                seat_id=s.seat_id,
                polygon=list(s.polygon),
                is_normalized=s.is_normalized,
                description=s.description,
                first_seen_frame=frame_number,
                last_seen_frame=frame_number,
                total_detections=1,
                is_active=True,
            )
        logger.info(f"SeatTracker đã ghi nhận {len(seats)} ghế cố định")

    def update(
        self, detected_seats: List[SeatConfig], frame_number: int
    ) -> List[SeatConfig]:
        """
        Cập nhật danh sách ghế phát hiện được ở frame hiện tại:
        - So khớp ghế phát hiện mới với ghế đã lưu (bằng IoU & khoảng cách trọng tâm).
        - Nếu khớp: Làm mượt tọa độ để chống rung lắc (spatial smoothing), giữ nguyên ID.
        - Nếu không có phát hiện ở một ghế (ví dụ có người đang ngồi đè lên ghế):
          -> Cấu trúc ghế VẪN ĐƯỢC GIỮ NGUYÊN (không bị xóa, không sinh ID mới).
        """
        if not self._tracked_seats:
            # Nếu tracker chưa có ghế nào, đăng ký toàn bộ
            self.register_seats(detected_seats, frame_number)
            return self.get_active_seats()

        matched_existing_ids = set()

        for det in detected_seats:
            det_poly = det.polygon
            det_c = polygon_centroid(det_poly)

            best_match_id = None
            best_score = 0.0

            for sid, trk in self._tracked_seats.items():
                if sid in matched_existing_ids:
                    continue

                iou = calculate_polygon_iou(det_poly, trk.polygon)
                trk_c = polygon_centroid(trk.polygon)
                dist = ((det_c[0] - trk_c[0]) ** 2 + (det_c[1] - trk_c[1]) ** 2) ** 0.5

                if iou >= self.iou_threshold or dist <= self.dist_threshold:
                    score = iou + (1.0 - min(1.0, dist))
                    if score > best_score:
                        best_score = score
                        best_match_id = sid

            if best_match_id:
                # Ghép thành công: làm mượt tọa độ (EMA smoothing)
                trk = self._tracked_seats[best_match_id]
                old_poly = trk.polygon
                smoothed_poly = []

                if len(old_poly) == len(det_poly):
                    alpha = self.smoothing_alpha
                    for (ox, oy), (nx, ny) in zip(old_poly, det_poly):
                        sx = alpha * ox + (1.0 - alpha) * nx
                        sy = alpha * oy + (1.0 - alpha) * ny
                        smoothed_poly.append((round(sx, 4), round(sy, 4)))
                    trk.polygon = smoothed_poly
                else:
                    trk.polygon = det_poly

                trk.last_seen_frame = frame_number
                trk.total_detections += 1
                matched_existing_ids.add(best_match_id)
            else:
                # Ghế mới phát hiện chưa có trong cơ sở dữ liệu
                new_id = det.seat_id
                if not new_id or new_id in self._tracked_seats:
                    new_id = f"SEAT_{self._seat_counter:03d}"
                    self._seat_counter += 1

                self._tracked_seats[new_id] = TrackedSeat(
                    seat_id=new_id,
                    polygon=list(det_poly),
                    is_normalized=det.is_normalized,
                    description=det.description or f"Ghế {new_id}",
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    total_detections=1,
                    is_active=True,
                )
                matched_existing_ids.add(new_id)

        # Trả về toàn bộ danh sách ghế (kể cả ghế đang bị hành khách ngồi che khuất)
        return self.get_active_seats()

    def get_active_seats(self) -> List[SeatConfig]:
        """Lấy danh sách cấu hình của tất cả các ghế đang theo dõi."""
        return [trk.to_seat_config() for trk in self._tracked_seats.values() if trk.is_active]

    def get_seat(self, seat_id: str) -> Optional[SeatConfig]:
        """Lấy cấu hình của 1 ghế theo ID."""
        trk = self._tracked_seats.get(seat_id)
        return trk.to_seat_config() if trk else None
