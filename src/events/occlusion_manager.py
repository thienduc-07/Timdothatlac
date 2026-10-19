"""
src/events/occlusion_manager.py
Quản lý và đánh giá tình trạng che khuất (Occlusion) của các vật thể trên xe buýt:
- Đánh giá xem vật thể mất detection do bị người che khuất hay do ghế trống mà vật không còn.
- Phân biệt 4 trường hợp che khuất theo đặc tả kiến trúc:
    + Trường hợp A: Vùng ghế có người che khuất (OCCUPIED / person overlap).
    + Trường hợp B: Vùng ghế không bị che khuất (ghế VACANT nhưng vật thể mất detection).
    + Trường hợp C: Người mới ngồi lên rồi đứng dậy (OCCUPIED -> VACANT).
    + Trường hợp D: Vật thể biến mất nhưng ghế không có người.
- Hỗ trợ so sánh bằng chứng (Re-association evidence matching) khi vật thể tái xuất hiện.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.models import SeatStatus, SeatState, TrackedObject
from src.seats.seat_config import calculate_overlap_ratio
from src.utils.logger import get_logger
from config.settings import (
    MAX_REASSOCIATION_DISTANCE_PX,
    MAX_OCCLUSION_REASSOCIATION_TIME_SEC,
)

logger = get_logger(__name__)


class OcclusionAssessment:
    """Kết quả đánh giá che khuất cho một vật thể tại frame hiện tại."""

    def __init__(
        self,
        is_occluded: bool,
        case_type: str,
        reason: str,
        evidence: Optional[Dict] = None,
    ):
        self.is_occluded = is_occluded
        self.case_type = case_type      # "CASE_A", "CASE_B", "CASE_C", "CASE_D"
        self.reason = reason
        self.evidence = evidence or {}


class OcclusionManager:
    """
    Module phân tích bằng chứng che khuất và kiểm tra tái liên kết vật thể.
    """

    def __init__(
        self,
        max_reassoc_dist: float = MAX_REASSOCIATION_DISTANCE_PX,
        max_occlusion_time: float = MAX_OCCLUSION_REASSOCIATION_TIME_SEC,
    ):
        self.max_reassoc_dist = max_reassoc_dist
        self.max_occlusion_time = max_occlusion_time

    def assess_missing_item(
        self,
        seat_id: str,
        seat_status: Optional[SeatStatus],
        last_known_bbox: Tuple[int, int, int, int],
        persons: List[TrackedObject],
        current_time_sec: float,
        last_seen_time_sec: float,
    ) -> OcclusionAssessment:
        """
        Đánh giá lý do vật thể không còn xuất hiện trong detection.

        Args:
            seat_id: Mã ghế của vật thể
            seat_status: Trạng thái hiện tại của ghế
            last_known_bbox: Vị trí cuối cùng nhìn thấy vật thể
            persons: Danh sách người phát hiện trong frame
            current_time_sec: Thời điểm hiện tại
            last_seen_time_sec: Thời điểm cuối cùng nhìn thấy vật thể

        Returns:
            OcclusionAssessment chứa kết luận và bằng chứng
        """
        # Kiểm tra xem có người nào đang đè lên vị trí cũ của vật thể không
        person_covering_item = False
        covering_person_id = -1
        max_person_cover_overlap = 0.0

        for p in persons:
            p_bbox = p.bbox
            # Tính overlap giữa bbox người và bbox vật thể cũ
            bx1, by1, bx2, by2 = last_known_bbox
            px1, py1, px2, py2 = p_bbox

            ix1, iy1 = max(bx1, px1), max(by1, py1)
            ix2, iy2 = min(bx2, px2), min(by2, py2)

            if ix1 < ix2 and iy1 < iy2:
                inter_area = float((ix2 - ix1) * (iy2 - iy1))
                item_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
                ratio = inter_area / item_area
                if ratio > max_person_cover_overlap:
                    max_person_cover_overlap = ratio
                    covering_person_id = p.track_id
                    if ratio >= 0.20:
                        person_covering_item = True

        # Trường hợp A: Ghế có người ngồi HOẶC có người che khu vực vật thể
        is_seat_occupied = (
            seat_status is not None and seat_status.state == SeatState.OCCUPIED
        )
        if is_seat_occupied or person_covering_item:
            evidence = {
                "seat_state": seat_status.state.value if seat_status else "unknown",
                "seat_overlap": seat_status.overlap_ratio if seat_status else 0.0,
                "person_covering_item": person_covering_item,
                "covering_person_id": covering_person_id,
                "covering_overlap": max_person_cover_overlap,
                "missing_duration": current_time_sec - last_seen_time_sec,
            }
            reason = (
                f"Vùng ghế {seat_id} đang có người ngồi/che khuất. "
                f"Không có bằng chứng vật thể bị lấy đi."
            )
            return OcclusionAssessment(
                is_occluded=True,
                case_type="CASE_A",
                reason=reason,
                evidence=evidence,
            )

        # Trường hợp B & D: Ghế không có người che khuất
        is_seat_vacant = (
            seat_status is not None and seat_status.state == SeatState.VACANT
        )
        if is_seat_vacant:
            # Trường hợp B: Ghế nhìn thấy rõ và trống, nhưng vật thể không còn xuất hiện
            evidence = {
                "seat_state": "vacant",
                "missing_duration": current_time_sec - last_seen_time_sec,
            }
            reason = (
                f"Vùng ghế {seat_id} quan sát được rõ ràng nhưng vật thể không xuất hiện. "
                f"Cần xác minh (RECHECK) trước khi kết luận."
            )
            return OcclusionAssessment(
                is_occluded=False,
                case_type="CASE_B",
                reason=reason,
                evidence=evidence,
            )
        else:
            # Trường hợp D: Ghế ở trạng thái UNKNOWN hoặc trung gian
            evidence = {
                "seat_state": seat_status.state.value if seat_status else "unknown",
                "missing_duration": current_time_sec - last_seen_time_sec,
            }
            reason = (
                f"Vật thể mất detection nhưng chưa đủ bằng chứng vùng ghế bị che khuất hoàn toàn. "
                f"Đưa vào trạng thái xác minh (RECHECK)."
            )
            return OcclusionAssessment(
                is_occluded=False,
                case_type="CASE_D",
                reason=reason,
                evidence=evidence,
            )

    def evaluate_reassociation(
        self,
        event_class_name: str,
        event_seat_id: str,
        event_last_bbox: Tuple[int, int, int, int],
        event_occlusion_duration: float,
        candidate_class_name: str,
        candidate_seat_id: Optional[str],
        candidate_bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, float, str]:
        """
        So sánh bằng chứng quan sát được để quyết định có nối lại sự kiện cũ hay không.

        Tiêu chí:
        1. Cùng lớp vật thể (class_name).
        2. Cùng vùng ghế (seat_id).
        3. Khoảng cách tâm không vượt quá ngưỡng max_reassoc_dist.
        4. Thời gian che khuất chưa vượt quá max_occlusion_time.

        Returns:
            (is_match, confidence_score, explanation):
            - is_match: True nếu đủ bằng chứng kết luận là cùng vật thể.
            - confidence_score: 0.0 - 1.0.
            - explanation: giải thích chi tiết.
        """
        # 1. Kiểm tra lớp vật thể
        if event_class_name != candidate_class_name:
            return False, 0.0, f"Khác lớp: {event_class_name} != {candidate_class_name}"

        # 2. Kiểm tra vùng ghế
        if event_seat_id != candidate_seat_id:
            return False, 0.0, f"Khác ghế: {event_seat_id} != {candidate_seat_id}"

        # 3. Kiểm tra thời gian che khuất
        if event_occlusion_duration > self.max_occlusion_time:
            return (
                False,
                0.2,
                f"Thời gian che khuất quá dài ({event_occlusion_duration:.1f}s > {self.max_occlusion_time}s)",
            )

        # 4. Kiểm tra khoảng cách hình học giữa 2 vị trí
        old_cx = (event_last_bbox[0] + event_last_bbox[2]) / 2.0
        old_cy = (event_last_bbox[1] + event_last_bbox[3]) / 2.0
        new_cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
        new_cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0

        dist = float(np.hypot(old_cx - new_cx, old_cy - new_cy))

        if dist > self.max_reassoc_dist:
            return (
                False,
                0.3,
                f"Khoảng cách vị trí lớn ({dist:.1f}px > {self.max_reassoc_dist}px)",
            )

        # Tính điểm tin cậy dựa trên khoảng cách và thời gian
        dist_score = max(0.0, 1.0 - (dist / self.max_reassoc_dist))
        time_score = max(0.0, 1.0 - (event_occlusion_duration / self.max_occlusion_time))
        confidence = 0.6 * dist_score + 0.4 * time_score

        explanation = (
            f"Trùng khớp: Cùng ghế {event_seat_id}, cùng lớp {candidate_class_name}, "
            f"khoảng cách {dist:.1f}px (ngưỡng {self.max_reassoc_dist}px), "
            f"thời gian che {event_occlusion_duration:.1f}s. Điểm tin cậy: {confidence:.0%}"
        )

        return True, confidence, explanation
