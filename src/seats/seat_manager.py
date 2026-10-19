"""
src/seats/seat_manager.py
Quản lý trạng thái các vùng ghế (Seat ROI) trên xe buýt:
- Xác định ghế đang có người ngồi (OCCUPIED) hay trống (VACANT).
- Áp dụng làm mượt theo thời gian (temporal smoothing) để tránh rung lắc trạng thái do nhiễu detection.
- Không kết luận ghế trống chỉ vì 1 frame không phát hiện thấy người.
- Gán đồ vật cho ghế phù hợp nhất (tránh gán 1 vật cho nhiều ghế).
"""

from typing import Dict, List, Optional, Tuple, Union
from src.models import SeatConfig, SeatState, SeatStatus, TrackedObject
from src.seats.seat_config import (
    SeatConfigManager,
    calculate_overlap_ratio,
    is_point_in_seat,
)
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
    Quản lý danh sách ghế và cập nhật trạng thái có người/trống qua từng frame.
    """

    def __init__(
        self,
        seats: Optional[List[SeatConfig]] = None,
        overlap_threshold: float = PERSON_SEAT_OVERLAP_THRESHOLD,
        occupied_confirm_sec: float = SEAT_OCCUPIED_CONFIRM_SECONDS,
        vacant_confirm_sec: float = SEAT_VACANT_CONFIRM_SECONDS,
        hold_sec: float = SEAT_HOLD_SECONDS,
    ):
        self.overlap_threshold = overlap_threshold
        self.occupied_confirm_sec = occupied_confirm_sec
        self.vacant_confirm_sec = vacant_confirm_sec
        self.hold_sec = hold_sec

        self._seats: Dict[str, SeatConfig] = {}
        self._statuses: Dict[str, SeatStatus] = {}

        if seats is not None:
            self.set_seats(seats)
        else:
            # Thử nạp từ file cấu hình mặc định
            loaded = SeatConfigManager().load_seats()
            if loaded:
                self.set_seats(loaded)

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
        logger.info(f"SeatManager đã cấu hình {len(self._seats)} ghế: {list(self._seats.keys())}")

    def reset(self):
        """Reset trạng thái tất cả các ghế về UNKNOWN cho video mới."""
        for seat_id, s in self._statuses.items():
            self._statuses[seat_id] = SeatStatus(
                seat_id=seat_id,
                state=SeatState.UNKNOWN,
                notes="Reset trạng thái",
            )

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
        Cập nhật trạng thái người ngồi trên từng ghế ở frame hiện tại.

        Args:
            persons: Danh sách người (TrackedObject hoặc bbox dạng (x1, y1, x2, y2))
            frame_index: Chỉ số frame hiện tại
            fps: FPS của video
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame

        Returns:
            Dict[seat_id, SeatStatus] với trạng thái mới nhất
        """
        current_time = frame_index / fps if fps > 0 else 0.0
        occupied_frames_thresh = max(1, int(self.occupied_confirm_sec * (fps if fps > 0 else 15)))
        vacant_frames_thresh = max(1, int(self.vacant_confirm_sec * (fps if fps > 0 else 15)))

        for seat_id, config in self._seats.items():
            status = self._statuses[seat_id]
            poly_pts = config.get_pixel_polygon(frame_width, frame_height)

            max_overlap = 0.0
            occupants = []

            for p in persons:
                if isinstance(p, TrackedObject):
                    p_bbox = p.bbox
                    p_id = p.track_id
                elif isinstance(p, (tuple, list)) and len(p) == 4:
                    p_bbox = tuple(p)
                    p_id = -1
                else:
                    continue

                overlap_seat, overlap_bbox = calculate_overlap_ratio(p_bbox, poly_pts)
                # Dùng max của tỷ lệ trên diện tích ghế hoặc diện tích người
                effective_overlap = max(overlap_seat, overlap_bbox * 0.5)

                if effective_overlap >= self.overlap_threshold:
                    if effective_overlap > max_overlap:
                        max_overlap = effective_overlap
                    if p_id != -1 and p_id not in occupants:
                        occupants.append(p_id)

            status.overlap_ratio = max_overlap

            if max_overlap >= self.overlap_threshold:
                # Phát hiện người trên ghế ở frame này
                status.consecutive_occupied_frames += 1
                status.consecutive_vacant_frames = 0
                status.last_person_seen_time = current_time
                status.occupant_person_ids = occupants

                # Làm mượt: chỉ chuyển sang OCCUPIED khi phát hiện đủ số frame
                if status.consecutive_occupied_frames >= occupied_frames_thresh:
                    if status.state != SeatState.OCCUPIED:
                        status.state = SeatState.OCCUPIED
                        status.occupied_since_sec = current_time
                        status.notes = f"Có người ngồi (ước lượng overlap {max_overlap:.0%})"
                        logger.debug(f"{seat_id} chuyển sang OCCUPIED tại {current_time:.2f}s")
            else:
                # Không phát hiện người trên ghế ở frame này
                status.consecutive_vacant_frames += 1
                status.consecutive_occupied_frames = 0

                # Kiểm tra thời gian duy trì (hold time):
                time_since_person = (
                    current_time - status.last_person_seen_time
                    if status.last_person_seen_time >= 0
                    else 999.0
                )

                if (
                    status.consecutive_vacant_frames >= vacant_frames_thresh
                    and time_since_person >= self.hold_sec
                ):
                    if status.state != SeatState.VACANT:
                        status.state = SeatState.VACANT
                        status.vacant_since_sec = current_time
                        status.occupant_person_ids = []
                        status.notes = "Ghế trống (xác nhận sau thời gian làm mượt)"
                        logger.debug(f"{seat_id} chuyển sang VACANT tại {current_time:.2f}s")

        return self._statuses

    def find_seat_for_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        min_overlap: float = 0.10,
    ) -> Optional[str]:
        """
        Xác định vùng ghế mà một vật thể (bbox) đang thuộc về.
        - Ưu tiên: Tâm vật thể nằm trọn trong ghế.
        - Thứ nhì: Ghế có tỷ lệ giao thoa lớn nhất và vượt ngưỡng min_overlap.
        - Tránh gán một vật thể cho nhiều ghế cùng lúc.

        Returns:
            seat_id tương ứng, hoặc None nếu nằm ngoài tất cả các ghế.
        """
        if not self._seats:
            return None

        bx1, by1, bx2, by2 = bbox
        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

        best_seat_id = None
        best_overlap = 0.0

        for seat_id, config in self._seats.items():
            poly_pts = config.get_pixel_polygon(frame_width, frame_height)

            # Ưu tiên kiểm tra tâm nằm trong polygon
            if is_point_in_seat(center, poly_pts):
                return seat_id

            # Tính overlap ratio
            overlap_seat, overlap_bbox = calculate_overlap_ratio(bbox, poly_pts)
            score = max(overlap_seat, overlap_bbox)
            if score > best_overlap and score >= min_overlap:
                best_overlap = score
                best_seat_id = seat_id

        return best_seat_id

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
