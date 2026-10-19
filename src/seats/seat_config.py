"""
src/seats/seat_config.py
Quản lý cấu hình vùng ghế (Seat ROI) và các thuật toán hình học liên quan:
- Định nghĩa tọa độ đa giác (polygon) cho từng ghế (chuẩn hóa hoặc pixel).
- Đọc/ghi cấu hình từ file JSON.
- Tính toán giao cắt giữa Bounding Box và vùng ghế.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.models import SeatConfig
from src.utils.logger import get_logger
from config.settings import DEFAULT_SEATS_CONFIG_PATH

logger = get_logger(__name__)


def calculate_polygon_area(polygon_pts: List[Tuple[int, int]]) -> float:
    """
    Tính diện tích đa giác bằng thuật toán Shoelace (hoặc cv2.contourArea).
    """
    if len(polygon_pts) < 3:
        return 0.0
    pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
    return float(cv2.contourArea(pts))


def is_point_in_seat(point: Tuple[float, float], polygon_pts: List[Tuple[int, int]]) -> bool:
    """
    Kiểm tra một điểm (x, y) có nằm trong vùng ghế hay không.
    Dùng cv2.pointPolygonTest (>= 0 là nằm trong hoặc trên cạnh).
    """
    if len(polygon_pts) < 3:
        return False
    pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
    return result >= 0


def calculate_overlap_ratio(
    bbox: Tuple[int, int, int, int],
    polygon_pts: List[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Tính tỷ lệ giao thoa giữa một Bounding Box (x1, y1, x2, y2) và đa giác ghế.

    Returns:
        (overlap_wrt_seat, overlap_wrt_bbox):
        - overlap_wrt_seat: diện tích giao nhau / diện tích vùng ghế
        - overlap_wrt_bbox: diện tích giao nhau / diện tích bounding box
    """
    if len(polygon_pts) < 3:
        return 0.0, 0.0

    bx1, by1, bx2, by2 = bbox
    bbox_w = max(0, bx2 - bx1)
    bbox_h = max(0, by2 - by1)
    bbox_area = float(bbox_w * bbox_h)
    if bbox_area <= 0:
        return 0.0, 0.0

    seat_pts = np.array(polygon_pts, dtype=np.int32)
    min_sx, min_sy = np.min(seat_pts, axis=0)
    max_sx, max_sy = np.max(seat_pts, axis=0)

    # Kiểm tra nhanh xem AABB (Axis-Aligned Bounding Box) có giao nhau không
    inter_x1 = max(bx1, min_sx)
    inter_y1 = max(by1, min_sy)
    inter_x2 = min(bx2, max_sx)
    inter_y2 = min(by2, max_sy)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0, 0.0

    seat_area = calculate_polygon_area(polygon_pts)
    if seat_area <= 0:
        return 0.0, 0.0

    # Tạo sub-mask cục bộ chỉ trong vùng bao quanh cả 2 để tối ưu tốc độ
    union_x1 = min(bx1, min_sx)
    union_y1 = min(by1, min_sy)
    union_x2 = max(bx2, max_sx)
    union_y2 = max(by2, max_sy)

    mask_w = int(union_x2 - union_x1) + 1
    mask_h = int(union_y2 - union_y1) + 1

    if mask_w <= 0 or mask_h <= 0 or mask_w > 4000 or mask_h > 4000:
        return 0.0, 0.0

    mask_seat = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask_bbox = np.zeros((mask_h, mask_w), dtype=np.uint8)

    shifted_seat_pts = seat_pts - np.array([union_x1, union_y1])
    cv2.fillPoly(mask_seat, [shifted_seat_pts.reshape((-1, 1, 2))], 1)

    s_bx1 = max(0, bx1 - union_x1)
    s_by1 = max(0, by1 - union_y1)
    s_bx2 = min(mask_w - 1, bx2 - union_x1)
    s_by2 = min(mask_h - 1, by2 - union_y1)
    cv2.rectangle(mask_bbox, (s_bx1, s_by1), (s_bx2, s_by2), 1, -1)

    inter_mask = cv2.bitwise_and(mask_seat, mask_bbox)
    intersection_area = float(np.sum(inter_mask))

    overlap_wrt_seat = intersection_area / seat_area
    overlap_wrt_bbox = intersection_area / bbox_area

    return min(1.0, overlap_wrt_seat), min(1.0, overlap_wrt_bbox)


def calculate_polygon_overlap_ratio(
    polygon_a_pts: List[Tuple[int, int]],
    polygon_b_pts: List[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Tính tỷ lệ giao thoa giữa hai đa giác (ví dụ mask của vật thể và polygon của ghế).

    Returns:
        (overlap_wrt_a, overlap_wrt_b)
    """
    if len(polygon_a_pts) < 3 or len(polygon_b_pts) < 3:
        return 0.0, 0.0

    pts_a = np.array(polygon_a_pts, dtype=np.int32)
    pts_b = np.array(polygon_b_pts, dtype=np.int32)

    min_a = np.min(pts_a, axis=0)
    max_a = np.max(pts_a, axis=0)
    min_b = np.min(pts_b, axis=0)
    max_b = np.max(pts_b, axis=0)

    # AABB check
    if max_a[0] <= min_b[0] or min_a[0] >= max_b[0] or max_a[1] <= min_b[1] or min_a[1] >= max_b[1]:
        return 0.0, 0.0

    area_a = calculate_polygon_area(polygon_a_pts)
    area_b = calculate_polygon_area(polygon_b_pts)
    if area_a <= 0 or area_b <= 0:
        return 0.0, 0.0

    union_min = np.minimum(min_a, min_b)
    union_max = np.maximum(max_a, max_b)

    mask_w = int(union_max[0] - union_min[0]) + 1
    mask_h = int(union_max[1] - union_min[1]) + 1

    if mask_w <= 0 or mask_h <= 0 or mask_w > 4000 or mask_h > 4000:
        return 0.0, 0.0

    mask_a = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros((mask_h, mask_w), dtype=np.uint8)

    shifted_a = pts_a - union_min
    shifted_b = pts_b - union_min

    cv2.fillPoly(mask_a, [shifted_a.reshape((-1, 1, 2))], 1)
    cv2.fillPoly(mask_b, [shifted_b.reshape((-1, 1, 2))], 1)

    inter_mask = cv2.bitwise_and(mask_a, mask_b)
    intersection_area = float(np.sum(inter_mask))

    overlap_wrt_a = intersection_area / area_a
    overlap_wrt_b = intersection_area / area_b

    return min(1.0, overlap_wrt_a), min(1.0, overlap_wrt_b)


class SeatConfigManager:
    """
    Quản lý việc lưu, nạp và xác thực cấu hình vùng ghế từ file JSON.
    Hỗ trợ cả tọa độ chuẩn hóa [0.0, 1.0] và tọa độ pixel thực tế.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path or DEFAULT_SEATS_CONFIG_PATH)

    def load_seats(self) -> List[SeatConfig]:
        """
        Đọc danh sách SeatConfig từ file JSON.
        Trả về danh sách rỗng nếu file không tồn tại hoặc lỗi.
        """
        if not self.config_path.exists():
            logger.info(f"Chưa tìm thấy file cấu hình ghế tại: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            seats_data = data.get("seats", [])
            seats = []
            for item in seats_data:
                seat_id = str(item.get("seat_id", "")).strip()
                polygon_raw = item.get("polygon", [])
                if not seat_id or len(polygon_raw) < 3:
                    continue
                polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_raw]
                is_normalized = bool(item.get("is_normalized", False))
                description = str(item.get("description", ""))
                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=polygon,
                    is_normalized=is_normalized,
                    description=description,
                ))

            logger.info(f"Đã tải {len(seats)} vùng ghế từ: {self.config_path}")
            return seats
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình ghế từ {self.config_path}: {e}")
            return []

    def save_seats(self, seats: List[SeatConfig], description: str = "Bus Seat ROI") -> bool:
        """
        Lưu danh sách SeatConfig vào file JSON.
        """
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            seats_dict = [
                {
                    "seat_id": s.seat_id,
                    "description": s.description,
                    "is_normalized": s.is_normalized,
                    "polygon": [[round(pt[0], 4), round(pt[1], 4)] for pt in s.polygon],
                }
                for s in seats
            ]

            payload = {
                "version": "1.0",
                "description": description,
                "seats": seats_dict,
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info(f"Đã lưu {len(seats)} vùng ghế vào: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu cấu hình ghế: {e}")
            return False

    @staticmethod
    def create_preset_seats(frame_width: int = 1280, frame_height: int = 720) -> List[SeatConfig]:
        """Tạo mẫu 4 ghế xe buýt mặc định."""
        return SeatConfigManager.generate_bus_grid(rows=2, left_cols=1, right_cols=1, naming_style="SEAT")

    @staticmethod
    def generate_bus_grid(
        rows: int = 4,
        left_cols: int = 1,
        right_cols: int = 1,
        naming_style: str = "GHE",  # "GHE" (Ghế 1A, 1B...) hoặc "SEAT" (SEAT_01, SEAT_02...)
        start_y: float = 0.15,
        end_y: float = 0.92,
        aisle_ratio: float = 0.18,
    ) -> List[SeatConfig]:
        """
        Tự động tạo lưới ghế cho xe buýt với số lượng hàng và cột tùy chọn:
        - rows: Số hàng ghế (ví dụ: 3, 4, 5, 6, 8 hàng...).
        - left_cols: Số ghế bên trái lối đi (1 hoặc 2).
        - right_cols: Số ghế bên phải lối đi (1 hoặc 2).
        - naming_style: Kiểu đánh số ghế ("GHE" hoặc "SEAT").
        """
        seats = []
        seat_counter = 1

        total_cols = left_cols + right_cols
        # Tính toán chiều rộng và chiều cao mỗi ghế trong không gian normalized [0, 1]
        available_x = 1.0 - 0.10 - aisle_ratio  # Trừ lề 2 bên (0.10) và lối đi giữa
        seat_w = available_x / max(1, total_cols)

        row_h = (end_y - start_y) / max(1, rows)
        seat_h = row_h * 0.75  # Chiều cao ghế (chừa khoảng trống để chân)

        col_letters = ["A", "B", "C", "D"]

        for r in range(rows):
            y1 = start_y + r * row_h
            y2 = y1 + seat_h

            current_col_idx = 0

            # 1. Các ghế bên trái lối đi
            for c in range(left_cols):
                x1 = 0.05 + c * seat_w
                x2 = x1 + seat_w * 0.90

                if naming_style == "GHE":
                    seat_id = f"Ghế {r + 1}{col_letters[current_col_idx]}"
                else:
                    seat_id = f"SEAT_{seat_counter:02d}"

                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=[
                        (round(x1, 4), round(y1, 4)),
                        (round(x2, 4), round(y1, 4)),
                        (round(x2, 4), round(y2, 4)),
                        (round(x1, 4), round(y2, 4)),
                    ],
                    is_normalized=True,
                    description=f"Hàng {r + 1} - Bên trái",
                ))
                current_col_idx += 1
                seat_counter += 1

            # 2. Các ghế bên phải lối đi
            right_start_x = 0.05 + (left_cols * seat_w) + aisle_ratio
            for c in range(right_cols):
                x1 = right_start_x + c * seat_w
                x2 = x1 + seat_w * 0.90

                if naming_style == "GHE":
                    seat_id = f"Ghế {r + 1}{col_letters[current_col_idx]}"
                else:
                    seat_id = f"SEAT_{seat_counter:02d}"

                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=[
                        (round(x1, 4), round(y1, 4)),
                        (round(x2, 4), round(y2, 4)),
                        (round(x2, 4), round(y2, 4)),
                        (round(x1, 4), round(y2, 4)),
                    ],
                    is_normalized=True,
                    description=f"Hàng {r + 1} - Bên phải",
                ))
                current_col_idx += 1
                seat_counter += 1

        return seats

