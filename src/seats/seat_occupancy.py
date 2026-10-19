"""
src/seats/seat_occupancy.py
Module ước lượng trạng thái có người/trống của từng ghế (Seat Occupancy Estimator):
- Tách bạch: Seat ID, Person Track ID, Object Track ID, Event ID.
- Không phụ thuộc đơn thuần vào tâm bounding box (dễ sai lệch ở góc nhìn camera trần).
- Phân tích vùng tiếp xúc cơ thể (Lower body / pelvic contact zone: 50% dưới của bbox người).
- Tính toán giao thoa đa giác chính xác giữa người và từng ghế độc lập.
- Xử lý tình huống hai ghế liền nhau trong cùng một hàng (Ghế A và Ghế B):
    + Ghế A có người, Ghế B trống.
    + Một người ngồi giữa hai ghế hoặc lấn làn -> Đánh dấu AMBIGUOUS / UNKNOWN thay vì gán bừa.
    + Ghi nhận chính xác người vừa rời khỏi ghế nào (Departure Tracking).
- Cơ chế làm mượt theo thời gian (Temporal Hysteresis) chống rung lắc trạng thái.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.models import SeatConfig, SeatState, SeatStatus, TrackedObject
from src.seats.seat_config import calculate_overlap_ratio
from src.utils.logger import get_logger
from config.settings import (
    PERSON_SEAT_OVERLAP_THRESHOLD,
    SEAT_OCCUPIED_CONFIRM_SECONDS,
    SEAT_VACANT_CONFIRM_SECONDS,
    SEAT_HOLD_SECONDS,
)

logger = get_logger(__name__)


class SeatOccupancyEstimator:
    """
    Ước lượng trạng thái có người / trống cho từng chiếc ghế độc lập trên xe buýt.
    """

    def __init__(
        self,
        overlap_threshold: float = PERSON_SEAT_OVERLAP_THRESHOLD,
        occupied_confirm_sec: float = SEAT_OCCUPIED_CONFIRM_SECONDS,
        vacant_confirm_sec: float = SEAT_VACANT_CONFIRM_SECONDS,
        hold_sec: float = SEAT_HOLD_SECONDS,
        ambiguity_overlap_diff: float = 0.12,  # Độ chênh lệch tối thiểu giữa 2 ghế để gán chắc chắn
    ):
        self.overlap_threshold = overlap_threshold
        self.occupied_confirm_sec = occupied_confirm_sec
        self.vacant_confirm_sec = vacant_confirm_sec
        self.hold_sec = hold_sec
        self.ambiguity_overlap_diff = ambiguity_overlap_diff

        # Lưu lịch sử người vừa rời khỏi ghế: seat_id -> (person_id, departure_time_sec)
        self.departure_history: Dict[str, Tuple[int, float]] = {}

    def extract_person_contact_zone(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Trích xuất vùng tiếp xúc mặt ghế của người (Pelvic/Lower Body contact zone).
        Trong góc nhìn camera từ trần xuống, đầu/vai có thể nghiêng sang bên hoặc lối đi,
        nhưng phần thân dưới (tọa độ từ y1 + 0.45*h đến y2) mới là phần tiếp xúc đệm ghế.
        """
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        contact_y1 = y1 + int(h * 0.45)
        return (x1, contact_y1, x2, y2)

    def evaluate_twin_seat_distribution(
        self,
        person_bbox: Tuple[int, int, int, int],
        seat_a_poly: List[Tuple[int, int]],
        seat_b_poly: List[Tuple[int, int]],
    ) -> Tuple[float, float, str]:
        """
        Đánh giá phân bổ người trên 2 ghế kề nhau trong một hàng.

        Returns:
            (overlap_a, overlap_b, verdict):
            - verdict = "SEAT_A": Người ngồi rõ ràng trên Ghế A.
            - verdict = "SEAT_B": Người ngồi rõ ràng trên Ghế B.
            - verdict = "AMBIGUOUS": Người ngồi giữa 2 ghế, không thể gán chắc chắn.
            - verdict = "NONE": Người không ngồi trên cả 2 ghế.
        """
        contact_box = self.extract_person_contact_zone(person_bbox)

        # Tính tỷ lệ giao nhau của vùng tiếp xúc với từng ghế
        overlap_a_seat, _ = calculate_overlap_ratio(contact_box, seat_a_poly)
        overlap_b_seat, _ = calculate_overlap_ratio(contact_box, seat_b_poly)

        # Tính thêm full bbox để phòng trường hợp bounding box thấp
        full_a_seat, _ = calculate_overlap_ratio(person_bbox, seat_a_poly)
        full_b_seat, _ = calculate_overlap_ratio(person_bbox, seat_b_poly)

        effective_a = max(overlap_a_seat, full_a_seat * 0.6)
        effective_b = max(overlap_b_seat, full_b_seat * 0.6)

        max_overlap = max(effective_a, effective_b)
        if max_overlap < self.overlap_threshold:
            return effective_a, effective_b, "NONE"

        # Nếu một người đè lên cả 2 ghế với tỷ lệ xấp xỉ nhau (ví dụ: 35% trên A và 32% trên B)
        diff = abs(effective_a - effective_b)
        if min(effective_a, effective_b) >= (self.overlap_threshold * 0.7) and diff < self.ambiguity_overlap_diff:
            return effective_a, effective_b, "AMBIGUOUS"

        if effective_a > effective_b:
            return effective_a, effective_b, "SEAT_A"
        else:
            return effective_a, effective_b, "SEAT_B"

    def update_occupancy(
        self,
        seats: List[SeatConfig],
        current_statuses: Dict[str, SeatStatus],
        persons: List[Union[TrackedObject, Tuple[int, int, int, int]]],
        frame_index: int,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, SeatStatus]:
        """
        Cập nhật trạng thái từng ghế dựa trên danh sách người ở frame hiện tại:
        - Phân tích tọa độ tiếp xúc thân dưới.
        - Phân xử chính xác giữa 2 ghế liền kề trong cùng hàng.
        - Áp dụng làm mượt theo thời gian (Hysteresis) để trạng thái không bị nhấp nháy.
        - Ghi nhận người vừa rời khỏi ghế khi ghế chuyển sang trống.
        """
        current_time = frame_index / fps if fps > 0 else 0.0
        occupied_frames_thresh = max(1, int(self.occupied_confirm_sec * (fps if fps > 0 else 15)))
        vacant_frames_thresh = max(1, int(self.vacant_confirm_sec * (fps if fps > 0 else 15)))

        # Chuẩn bị danh sách người chuẩn hóa: [(track_id, bbox)]
        norm_persons: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for p in persons:
            if isinstance(p, TrackedObject):
                norm_persons.append((p.track_id, p.bbox))
            elif isinstance(p, (tuple, list)) and len(p) == 4:
                norm_persons.append((-1, tuple(p)))

        # Nhóm các ghế theo hàng để phân xử ghế đôi liền nhau (twin seats)
        # Giả sử tên ghế như SEAT_1A, SEAT_1B -> hàng 1
        row_groups: Dict[str, List[SeatConfig]] = {}
        for s in seats:
            row_prefix = s.seat_id.split("A")[0].split("B")[0].split("C")[0].split("D")[0]
            row_groups.setdefault(row_prefix, []).append(s)

        # Tính toán mức độ giao nhau cho từng ghế
        seat_overlaps: Dict[str, float] = {s.seat_id: 0.0 for s in seats}
        seat_occupants: Dict[str, List[int]] = {s.seat_id: [] for s in seats}
        seat_ambiguity: Dict[str, bool] = {s.seat_id: False for s in seats}

        for p_id, p_bbox in norm_persons:
            contact_box = self.extract_person_contact_zone(p_bbox)

            # Kiểm tra với từng ghế
            matched_seats_for_person = []
            for s in seats:
                poly_pts = s.get_pixel_polygon(frame_width, frame_height)
                overlap_contact, _ = calculate_overlap_ratio(contact_box, poly_pts)
                overlap_full, _ = calculate_overlap_ratio(p_bbox, poly_pts)
                effective = max(overlap_contact, overlap_full * 0.6)

                if effective >= (self.overlap_threshold * 0.7):
                    matched_seats_for_person.append((s.seat_id, effective, poly_pts))

            if not matched_seats_for_person:
                continue

            if len(matched_seats_for_person) == 1:
                sid, eff, _ = matched_seats_for_person[0]
                if eff > seat_overlaps[sid]:
                    seat_overlaps[sid] = eff
                if p_id != -1 and p_id not in seat_occupants[sid]:
                    seat_occupants[sid].append(p_id)
            else:
                # Người này giao thoa với nhiều hơn 1 ghế (khả năng là 2 ghế liền nhau)
                matched_seats_for_person.sort(key=lambda x: x[1], reverse=True)
                top1_id, top1_eff, _ = matched_seats_for_person[0]
                top2_id, top2_eff, _ = matched_seats_for_person[1]

                diff = top1_eff - top2_eff
                # Nếu độ chênh lệch nhỏ hơn ngưỡng -> người ngồi ở giữa 2 ghế (AMBIGUOUS)
                if diff < self.ambiguity_overlap_diff and top2_eff >= self.overlap_threshold:
                    seat_ambiguity[top1_id] = True
                    seat_ambiguity[top2_id] = True
                    seat_overlaps[top1_id] = max(seat_overlaps[top1_id], top1_eff)
                    seat_overlaps[top2_id] = max(seat_overlaps[top2_id], top2_eff)
                    if p_id != -1:
                        if p_id not in seat_occupants[top1_id]:
                            seat_occupants[top1_id].append(p_id)
                        if p_id not in seat_occupants[top2_id]:
                            seat_occupants[top2_id].append(p_id)
                else:
                    # Gán cho ghế có tỷ lệ vượt trội
                    if top1_eff > seat_overlaps[top1_id]:
                        seat_overlaps[top1_id] = top1_eff
                    if p_id != -1 and p_id not in seat_occupants[top1_id]:
                        seat_occupants[top1_id].append(p_id)

        # Cập nhật trạng thái từng ghế và áp dụng bộ lọc thời gian
        for s in seats:
            sid = s.seat_id
            status = current_statuses.get(sid)
            if not status:
                status = SeatStatus(seat_id=sid, state=SeatState.UNKNOWN)
                current_statuses[sid] = status

            overlap = seat_overlaps.get(sid, 0.0)
            status.overlap_ratio = overlap
            is_ambiguous = seat_ambiguity.get(sid, False)

            if is_ambiguous:
                # Trường hợp không thể phân định rõ ghế nào
                status.notes = f"Chưa rõ: Hành khách nằm/ngồi giữa 2 ghế liền nhau (overlap {overlap:.0%})"
                status.state = SeatState.UNKNOWN
                continue

            if overlap >= self.overlap_threshold:
                # Phát hiện người trên ghế
                status.consecutive_occupied_frames += 1
                status.consecutive_vacant_frames = 0
                status.last_person_seen_time = current_time
                status.occupant_person_ids = seat_occupants.get(sid, [])

                if status.consecutive_occupied_frames >= occupied_frames_thresh:
                    if status.state != SeatState.OCCUPIED:
                        status.state = SeatState.OCCUPIED
                        status.occupied_since_sec = current_time
                        p_labels = [f"P{pid}" for pid in status.occupant_person_ids if pid != -1]
                        p_str = f" ({', '.join(p_labels)})" if p_labels else ""
                        status.notes = f"Có người ngồi{p_str} (overlap {overlap:.0%})"
                        logger.debug(f"Ghế {sid} chuyển sang OCCUPIED tại {current_time:.2f}s")
            else:
                # Không phát hiện người trên ghế
                status.consecutive_vacant_frames += 1
                status.consecutive_occupied_frames = 0

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
                        # Ghi nhận người vừa rời ghế (Departure Tracking)
                        if status.occupant_person_ids:
                            last_p = status.occupant_person_ids[-1]
                            self.departure_history[sid] = (last_p, current_time)
                            dep_note = f"Người P{last_p} vừa rời ghế"
                        else:
                            dep_note = "Ghế trống (xác nhận sau thời gian làm mượt)"

                        status.state = SeatState.VACANT
                        status.vacant_since_sec = current_time
                        status.occupant_person_ids = []
                        status.notes = dep_note
                        logger.debug(f"Ghế {sid} chuyển sang VACANT tại {current_time:.2f}s: {dep_note}")

        return current_statuses

    def get_last_departed_person(self, seat_id: str) -> Optional[Tuple[int, float]]:
        """Lấy thông tin (person_track_id, departed_time_sec) của người gần nhất rời khỏi ghế này."""
        return self.departure_history.get(seat_id)
