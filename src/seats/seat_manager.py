"""
src/seats/seat_manager.py
Quản lý trạng thái các vùng ghế (Seat ROI) trên xe buýt:
- Tự động nhận diện và phân biệt từng chiếc ghế riêng biệt từ góc nhìn trần xe buýt.
- Phân rã và quản lý từng ghế kề nhau trong cùng một hàng (Ghế A & Ghế B).
- Xác định ghế đang có người ngồi (OCCUPIED) hay trống (VACANT) bằng vùng tiếp xúc cơ thể.
- Gán đồ vật cho ghế phù hợp nhất hoặc đánh dấu UNKNOWN_SEAT nếu nằm ở ranh giới giữa hai ghế.
- Áp dụng làm mượt theo thời gian (temporal hysteresis) để tránh rung lắc trạng thái do nhiễu detection.
- Không kết luận ghế trống chỉ vì 1 frame không phát hiện thấy người.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.models import SeatConfig, SeatState, SeatStatus, TrackedObject
from src.seats.seat_config import (
    SeatConfigManager,
    calculate_overlap_ratio,
    is_point_in_seat,
)
from src.seats.seat_detector import (
    SeatDetector,
    SeatDetectionResult,
    SeatDetectionMethod,
)
from src.seats.seat_tracker import SeatTracker
from src.seats.seat_occupancy import SeatOccupancyEstimator
from src.utils.logger import get_logger
from config.settings import (
    PERSON_SEAT_OVERLAP_THRESHOLD,
    SEAT_OCCUPIED_CONFIRM_SECONDS,
    SEAT_VACANT_CONFIRM_SECONDS,
    SEAT_HOLD_SECONDS,
)

logger = get_logger(__name__)


class SeatManager:
    """
    Quản lý danh sách ghế, tự động nhận diện và cập nhật trạng thái có người/trống qua từng frame.
    """

    def __init__(
        self,
        seats: Optional[List[SeatConfig]] = None,
        overlap_threshold: float = PERSON_SEAT_OVERLAP_THRESHOLD,
        occupied_confirm_sec: float = SEAT_OCCUPIED_CONFIRM_SECONDS,
        vacant_confirm_sec: float = SEAT_VACANT_CONFIRM_SECONDS,
        hold_sec: float = SEAT_HOLD_SECONDS,
        custom_weights_path: Optional[str] = "models/bus_seat_yolov8.pt",
    ):
        self.overlap_threshold = overlap_threshold
        self.occupied_confirm_sec = occupied_confirm_sec
        self.vacant_confirm_sec = vacant_confirm_sec
        self.hold_sec = hold_sec

        self._seats: Dict[str, SeatConfig] = {}
        self._statuses: Dict[str, SeatStatus] = {}

        # Khởi tạo các module con chuyên biệt
        self.detector = SeatDetector(custom_weights_path=custom_weights_path)
        self.tracker = SeatTracker()
        self.occupancy_estimator = SeatOccupancyEstimator(
            overlap_threshold=overlap_threshold,
            occupied_confirm_sec=occupied_confirm_sec,
            vacant_confirm_sec=vacant_confirm_sec,
            hold_sec=hold_sec,
        )

        self.active_detection_method: SeatDetectionMethod = (
            SeatDetectionMethod.MANUAL_CONFIGURATION if seats else SeatDetectionMethod.NOT_CONFIGURED
        )

        if seats is not None:
            self.set_seats(seats)
        else:
            # Thử nạp từ file cấu hình mặc định
            loaded = SeatConfigManager().load_seats()
            if loaded:
                self.set_seats(loaded)
                self.active_detection_method = SeatDetectionMethod.MANUAL_CONFIGURATION

    @property
    def is_configured(self) -> bool:
        """Kiểm tra xem hệ thống đã có cấu hình ghế hay chưa."""
        return len(self._seats) > 0

    @property
    def seats(self) -> List[SeatConfig]:
        """Trả về danh sách cấu hình ghế."""
        return list(self._seats.values())

    def set_seats(self, seats: List[SeatConfig]):
        """Cập nhật danh sách ghế và khởi tạo trạng thái."""
        self._seats.clear()
        self._statuses.clear()
        for s in seats:
            self._seats[s.seat_id] = s
            self._statuses[s.seat_id] = SeatStatus(
                seat_id=s.seat_id,
                state=SeatState.UNKNOWN,
                notes="Chưa bắt đầu phân tích",
            )
        self.tracker.register_seats(seats)
        logger.info(f"SeatManager đã cấu hình {len(self._seats)} ghế: {list(self._seats.keys())}")

    def reset(self):
        """Reset trạng thái tất cả các ghế về UNKNOWN cho video mới."""
        for seat_id in self._statuses:
            self._statuses[seat_id] = SeatStatus(
                seat_id=seat_id,
                state=SeatState.UNKNOWN,
                notes="Reset trạng thái",
            )
        self.tracker.reset()
        if self._seats:
            self.tracker.register_seats(list(self._seats.values()))

    @staticmethod
    def generate_bus_grid(
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_style: str = "SEAT",
        start_y: float = 0.12,
        end_y: float = 0.90,
        aisle_ratio: float = 0.18,
    ) -> List[SeatConfig]:
        """Tự động tạo sơ đồ lưới ghế xe buýt với từng ghế riêng biệt."""
        return SeatConfigManager.generate_bus_grid(
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_style="GHE" if naming_style in ("GHE", "alphanumeric") else "SEAT",
            start_y=start_y,
            end_y=end_y,
            aisle_ratio=aisle_ratio,
        )

    def auto_detect_and_configure_seats(
        self,
        frame: np.ndarray,
        allow_fallback: bool = False,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        save_to_disk: bool = True,
    ) -> SeatDetectionResult:
        """
        Tự động nhận diện từng ghế từ frame camera bằng Computer Vision Model:
        - Nếu có Custom Model: Trả về MODEL DETECTION.
        - Nếu không có Custom Model và allow_fallback=False: Báo CUSTOM SEAT MODEL NOT CONFIGURED.
        - Nếu allow_fallback=True: Sinh GEOMETRY FALLBACK được công khai rõ ràng.
        """
        result = self.detector.detect_seats(
            frame=frame,
            allow_fallback=allow_fallback,
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
        )

        if result.seats:
            self.set_seats(result.seats)
            self.active_detection_method = result.method
            if save_to_disk:
                SeatConfigManager().save_seats(
                    result.seats,
                    description=f"Seats configured via {result.method.value}",
                )
        elif result.method == SeatDetectionMethod.NOT_CONFIGURED:
            self.active_detection_method = SeatDetectionMethod.NOT_CONFIGURED

        return result

    def get_pixel_polygon(
        self, seat_id: str, frame_width: int, frame_height: int
    ) -> List[Tuple[int, int]]:
        """Lấy danh sách đỉnh polygon theo pixel thực tế trên frame."""
        cfg = self._seats.get(seat_id)
        if not cfg:
            return []
        return cfg.get_pixel_polygon(frame_width, frame_height)

    def update(
        self,
        persons: List[Union[TrackedObject, Tuple[int, int, int, int]]],
        frame_index: int,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, SeatStatus]:
        """
        Cập nhật trạng thái người ngồi trên từng ghế ở frame hiện tại:
        - Sử dụng SeatOccupancyEstimator phân tích vùng tiếp xúc cơ thể.
        - Phân xử tình huống ghế đôi kề nhau trong một hàng.
        - Áp dụng làm mượt theo thời gian.
        """
        if not self._seats:
            return self._statuses

        return self.occupancy_estimator.update_occupancy(
            seats=list(self._seats.values()),
            current_statuses=self._statuses,
            persons=persons,
            frame_index=frame_index,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def find_seat_for_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        min_overlap: float = 0.10,
        ambiguity_diff_threshold: float = 0.08,
    ) -> Optional[str]:
        """
        Xác định vùng ghế mà một vật thể (bbox) đang thuộc về:
        - Nếu vật thể nằm gần ranh giới giữa 2 ghế hoặc không đủ bằng chứng: Trả về 'UNKNOWN_SEAT'.
        - Nếu vật thể nằm trọn trong 1 ghế hoặc có tỷ lệ giao thoa vượt trội: Trả về seat_id đó.
        - Tránh gán nhầm lẫn giữa 2 ghế nằm cạnh nhau trong cùng hàng.
        - Trả về None nếu nằm ngoài tất cả các ghế.
        """
        if not self._seats:
            return None

        bx1, by1, bx2, by2 = bbox
        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

        seat_scores: List[Tuple[str, float, bool]] = []  # (seat_id, score, is_center_inside)

        for seat_id, config in self._seats.items():
            poly_pts = config.get_pixel_polygon(frame_width, frame_height)
            is_inside = is_point_in_seat(center, poly_pts)
            overlap_seat, overlap_bbox = calculate_overlap_ratio(bbox, poly_pts)
            score = max(overlap_seat, overlap_bbox)

            if is_inside or score >= min_overlap:
                effective_score = score + (0.35 if is_inside else 0.0)
                seat_scores.append((seat_id, effective_score, is_inside))

        if not seat_scores:
            return None

        # Sắp xếp theo điểm số giảm dần
        seat_scores.sort(key=lambda x: x[1], reverse=True)

        if len(seat_scores) == 1:
            return seat_scores[0][0]

        top1_id, top1_score, top1_center = seat_scores[0]
        top2_id, top2_score, top2_center = seat_scores[1]

        # Kiểm tra tính mơ hồ ở ranh giới (Boundary Ambiguity):
        # Nếu cả 2 ghế đều có điểm cao xấp xỉ nhau (chênh lệch nhỏ)
        diff = abs(top1_score - top2_score)
        if diff < ambiguity_diff_threshold and top2_score >= min_overlap:
            logger.debug(
                f"Đồ vật nằm ở ranh giới giữa {top1_id} và {top2_id} (diff {diff:.2f}) -> UNKNOWN_SEAT"
            )
            return "UNKNOWN_SEAT"

        # Nếu tâm nằm trọn trong ghế 1 và ghế 2 chỉ chạm nhẹ viền
        if top1_center and not top2_center:
            return top1_id

        if top1_score >= min_overlap:
            return top1_id

        return None

    def get_last_departed_person(self, seat_id: str) -> Optional[Tuple[int, float]]:
        """Lấy thông tin người vừa rời khỏi ghế (person_track_id, departure_time_sec)."""
        return self.occupancy_estimator.get_last_departed_person(seat_id)

    def is_seat_occupied(self, seat_id: str) -> bool:
        """Kiểm tra ghế có đang ở trạng thái OCCUPIED hay không."""
        status = self._statuses.get(seat_id)
        return status is not None and status.state == SeatState.OCCUPIED

    def is_seat_vacant(self, seat_id: str) -> bool:
        """Kiểm tra ghế có đang ở trạng thái VACANT hay không."""
        status = self._statuses.get(seat_id)
        return status is not None and status.state == SeatState.VACANT

    def get_seat_status(self, seat_id: str) -> Optional[SeatStatus]:
        """Lấy trạng thái chi tiết của ghế."""
        return self._statuses.get(seat_id)

    def get_all_statuses(self) -> Dict[str, SeatStatus]:
        """Lấy trạng thái của toàn bộ ghế."""
        return self._statuses

    def get_model_diagnostics(self) -> Dict[str, Union[str, bool, int, float, List[str]]]:
        """Lấy thông tin chẩn đoán về model nhận diện ghế."""
        return self.detector.inspect_coco_classes()
