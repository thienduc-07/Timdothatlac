"""
src/events/forgotten_item_detector.py
Module phân tích phát hiện đồ bỏ quên tích hợp nhận biết che khuất (Occlusion-aware):
- Tách bạch các khái niệm: Object tracking, Seat occupancy, Occlusion detection, Event lifecycle.
- Quản lý State Machine: OBSERVED, ITEM_LEFT, SUSPECTED, OCCLUDED, RECHECK, REVIEW_REQUIRED, RESOLVED, DISMISSED.
- Phân biệt 4 trường hợp che khuất:
    + Trường hợp A: Vùng ghế có người che khuất -> chuyển OCCLUDED, giữ nguyên Event ID.
    + Trường hợp B: Vùng ghế không bị che nhưng đồ biến mất -> RECHECK -> REVIEW_REQUIRED.
    + Trường hợp C: Người mới ngồi lên rồi đứng dậy -> kiểm tra lại, khôi phục sự kiện nếu đồ xuất hiện.
    + Trường hợp D: Biến mất khi ghế không có người -> không vội gán OCCLUDED, đưa vào xác minh.
- Chống tạo cảnh báo sớm khi hành khách vẫn đang ngồi trên ghế.
- Phân biệt rõ thời gian quan sát (observed), thời gian che (occluded), thời gian trên ghế trống (vacant).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.models import (
    TrackObservation,
    TrackedObject,
    Event,
    SeatState,
    SeatStatus,
)
from src.seats.seat_manager import SeatManager
from src.events.occlusion_manager import OcclusionManager
from src.events.event_manager import EventManager
from src.utils.logger import get_logger
from config.settings import (
    MIN_STATIONARY_TIME_SEC,
    STATIONARY_MOVEMENT_THRESHOLD,
    ALERT_DELAY_AFTER_PERSON_LEFT_SEC,
    RECHECK_DURATION_SECONDS,
    ITEM_CLASSES,
    EventStatus,
)

logger = get_logger(__name__)


class ForgottenItemDetector:
    """
    Phát hiện đồ vật nghi ngờ bị bỏ quên trên xe buýt.
    """

    def __init__(
        self,
        min_stationary_time: float = MIN_STATIONARY_TIME_SEC,
        movement_threshold: float = STATIONARY_MOVEMENT_THRESHOLD,
        alert_delay: float = ALERT_DELAY_AFTER_PERSON_LEFT_SEC,
        watched_classes: Optional[List[str]] = None,
        seat_manager: Optional[SeatManager] = None,
        occlusion_manager: Optional[OcclusionManager] = None,
        event_manager: Optional[EventManager] = None,
        recheck_duration: float = RECHECK_DURATION_SECONDS,
    ):
        self.min_stationary_time = min_stationary_time
        self.movement_threshold = movement_threshold
        self.alert_delay = alert_delay
        self.watched_classes = watched_classes or ITEM_CLASSES
        self.seat_manager = seat_manager
        self.occlusion_manager = occlusion_manager or OcclusionManager()
        self.event_manager = event_manager
        self.recheck_duration = recheck_duration

        # Quản lý trạng thái nội bộ
        self._track_status: Dict[int, str] = {}
        # Theo dõi thời gian vào RECHECK: event_id -> recheck_start_time
        self._recheck_start_times: Dict[str, float] = {}
        # Theo dõi frame trước đó để tính dt
        self._last_frame_time: float = -1.0

    def reset(self):
        """Reset trạng thái cho video mới."""
        self._track_status.clear()
        self._recheck_start_times.clear()
        self._last_frame_time = -1.0

    def process_frame(
        self,
        tracked_objects: List[TrackedObject],
        current_frame: int,
        fps: float,
        frame_width: int,
        frame_height: int,
        video_name: str = "",
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Xử lý toàn diện một frame trong pipeline VideoProcessor theo kiến trúc Occlusion-Aware:
        1. Phân loại người (person) và đồ vật (item).
        2. Cập nhật SeatManager để xác định trạng thái từng ghế.
        3. Phân tích từng đồ vật quan sát được:
           - Gán vùng ghế (Seat ROI).
           - Tái kết nối với sự kiện cũ nếu khớp bằng chứng (sau khi bị che khuất).
           - Cập nhật trạng thái (OBSERVED -> ITEM_LEFT -> SUSPECTED).
        4. Kiểm tra các sự kiện đang hoạt động nhưng không được phát hiện ở frame này (Occlusion Analysis):
           - Trường hợp A: Ghế có người che -> chuyển OCCLUDED.
           - Trường hợp B/D: Ghế trống/không che -> RECHECK -> REVIEW_REQUIRED.
        """
        current_time_sec = current_frame / fps if fps > 0 else 0.0
        dt = (
            current_time_sec - self._last_frame_time
            if self._last_frame_time >= 0
            else (1.0 / fps if fps > 0 else 0.067)
        )
        self._last_frame_time = current_time_sec

        # Tách người và đồ vật
        persons = [obj for obj in tracked_objects if obj.class_name == "person"]
        items = [
            obj for obj in tracked_objects if obj.class_name in self.watched_classes
        ]

        # Cập nhật trạng thái các ghế
        if self.seat_manager and self.seat_manager.is_configured:
            self.seat_manager.update(
                persons=persons,
                frame_index=current_frame,
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        events_to_return = []
        detected_event_ids_this_frame = set()

        # 3. Phân tích đồ vật quan sát được trong frame
        for item in items:
            track_id = item.track_id
            class_name = item.class_name
            bbox = item.bbox
            confidence = item.confidence

            # Xác định ghế cho đồ vật (ưu tiên dùng mask polygon nếu có)
            seat_id = ""
            if self.seat_manager and self.seat_manager.is_configured:
                if hasattr(self.seat_manager, "find_seat_for_object"):
                    seat_id = self.seat_manager.find_seat_for_object(
                        item, frame_width, frame_height
                    ) or ""
                else:
                    seat_id = self.seat_manager.find_seat_for_bbox(
                        bbox, frame_width, frame_height
                    ) or ""

            # Kiểm tra xem có sự kiện nào đã tồn tại gắn với track_id này không
            existing_event = None
            if self.event_manager:
                existing_event = self.event_manager.get_event_by_track_id(track_id)

                # Nếu chưa có theo track_id, thử tìm sự kiện trên ghế này đang OCCLUDED hoặc RECHECK
                if not existing_event and seat_id:
                    seat_event = self.event_manager.get_active_event_for_seat(seat_id)
                    if seat_event and seat_event.status in (
                        EventStatus.OCCLUDED,
                        EventStatus.RECHECK,
                    ):
                        # Đánh giá bằng chứng xem có phải cùng 1 vật không
                        is_match, score, explanation = self.occlusion_manager.evaluate_reassociation(
                            event_class_name=seat_event.class_name,
                            event_seat_id=seat_event.seat_id,
                            event_last_bbox=seat_event.last_known_bbox,
                            event_occlusion_duration=seat_event.occlusion_duration,
                            candidate_class_name=class_name,
                            candidate_seat_id=seat_id,
                            candidate_bbox=bbox,
                        )
                        if is_match:
                            logger.info(
                                f"Tái kết nối thành công: Vật thể {class_name} (track {track_id}) "
                                f"khớp với sự kiện {seat_event.event_id} trên ghế {seat_id}. "
                                f"Lý do: {explanation}"
                            )
                            self.event_manager.reassociate_detection_to_event(
                                event_id=seat_event.event_id,
                                new_track_id=track_id,
                                new_bbox=bbox,
                                confidence=confidence,
                                frame_number=current_frame,
                                video_time_sec=current_time_sec,
                                reason=explanation,
                            )
                            existing_event = seat_event

            # Lấy trạng thái ghế
            seat_is_occupied = False
            seat_is_vacant = False
            if self.seat_manager and seat_id:
                seat_is_occupied = self.seat_manager.is_seat_occupied(seat_id)
                seat_is_vacant = self.seat_manager.is_seat_vacant(seat_id)

            if existing_event:
                detected_event_ids_this_frame.add(existing_event.event_id)
                # Cập nhật thông tin quan sát
                existing_event.last_seen_frame = current_frame
                existing_event.last_seen_time_sec = current_time_sec
                existing_event.last_known_bbox = bbox
                existing_event.bbox_x1, existing_event.bbox_y1, existing_event.bbox_x2, existing_event.bbox_y2 = bbox
                existing_event.confidence = confidence
                existing_event.observed_duration += dt
                existing_event.duration_seconds = existing_event.observed_duration

                if self._recheck_start_times.get(existing_event.event_id):
                    del self._recheck_start_times[existing_event.event_id]

                # Chuyển đổi trạng thái nếu ghế thay đổi
                if seat_is_occupied:
                    # Hành khách đang ngồi cùng đồ vật -> chỉ ở mức OBSERVED (hoặc giữ SUSPECTED nếu đã lỡ nghi ngờ)
                    if existing_event.status in (EventStatus.ITEM_LEFT,):
                        self.event_manager.transition_event(
                            existing_event.event_id,
                            EventStatus.OBSERVED,
                            current_frame,
                            current_time_sec,
                            f"Hành khách đang ngồi tại ghế {seat_id}",
                            frame=frame,
                        )
                elif seat_is_vacant or not self.seat_manager or not self.seat_manager.is_configured:
                    # Ghế trống -> tăng thời gian đồ vật nằm trên ghế trống
                    existing_event.vacant_duration += dt

                    if existing_event.status == EventStatus.OBSERVED:
                        self.event_manager.transition_event(
                            existing_event.event_id,
                            EventStatus.ITEM_LEFT,
                            current_frame,
                            current_time_sec,
                            f"Ghế {seat_id} đã xác nhận trống, vật thể vẫn còn lại",
                            frame=frame,
                        )
                    elif existing_event.status == EventStatus.ITEM_LEFT:
                        # Đủ thời gian trên ghế trống -> nâng lên SUSPECTED
                        if existing_event.vacant_duration >= self.min_stationary_time:
                            self.event_manager.transition_event(
                                existing_event.event_id,
                                EventStatus.SUSPECTED,
                                current_frame,
                                current_time_sec,
                                f"Vật thể đứng yên trên ghế trống {existing_event.vacant_duration:.1f}s "
                                f"(vượt ngưỡng {self.min_stationary_time:.1f}s)",
                                frame=frame,
                            )
            else:
                # Tạo sự kiện mới
                initial_status = EventStatus.OBSERVED
                reason = f"Phát hiện vật thể {class_name}"
                if seat_is_vacant:
                    initial_status = EventStatus.ITEM_LEFT
                    reason = f"Vật thể {class_name} phát hiện trên ghế trống {seat_id}"

                if self.event_manager:
                    new_evt = self.event_manager.create_event(
                        class_name=class_name,
                        seat_id=seat_id,
                        initial_status=initial_status,
                        track_id=track_id,
                        bbox=bbox,
                        confidence=confidence,
                        frame_number=current_frame,
                        video_time_sec=current_time_sec,
                        video_name=video_name,
                        reason=reason,
                        frame=frame,
                    )
                    detected_event_ids_this_frame.add(new_evt.event_id)
                self._track_status[track_id] = initial_status

        # 4. Phân tích các sự kiện KHÔNG xuất hiện ở frame này (Occlusion Handling)
        if self.event_manager:
            for event in list(self.event_manager.events):
                if event.event_id in detected_event_ids_this_frame:
                    continue

                # Chỉ xét các sự kiện đang hoạt động
                if event.status in (
                    EventStatus.RESOLVED,
                    EventStatus.DISMISSED,
                ):
                    continue

                seat_status = (
                    self.seat_manager.get_seat_status(event.seat_id)
                    if (self.seat_manager and event.seat_id)
                    else None
                )

                assessment = self.occlusion_manager.assess_missing_item(
                    seat_id=event.seat_id,
                    seat_status=seat_status,
                    last_known_bbox=event.last_known_bbox,
                    persons=persons,
                    current_time_sec=current_time_sec,
                    last_seen_time_sec=event.last_seen_time_sec,
                )

                if assessment.is_occluded:
                    # Trường hợp A: Vùng ghế có người che khuất -> chuyển sang OCCLUDED
                    event.occlusion_duration += dt
                    if event.status != EventStatus.OCCLUDED:
                        self.event_manager.transition_event(
                            event_id=event.event_id,
                            new_status=EventStatus.OCCLUDED,
                            frame_number=current_frame,
                            video_time_sec=current_time_sec,
                            reason=assessment.reason,
                            evidence=assessment.evidence,
                            frame=frame,
                        )
                else:
                    # Trường hợp B & D: Ghế không có người che khuất nhưng đồ vật biến mất
                    if event.status == EventStatus.OCCLUDED:
                        # Vừa mới hết che khuất (ví dụ người ngồi che vừa đứng lên) -> chuyển RECHECK
                        self.event_manager.transition_event(
                            event_id=event.event_id,
                            new_status=EventStatus.RECHECK,
                            frame_number=current_frame,
                            video_time_sec=current_time_sec,
                            reason="Ghế đã nhìn thấy lại, đang kiểm tra xem vật thể có còn không",
                            evidence=assessment.evidence,
                            frame=frame,
                        )
                        self._recheck_start_times[event.event_id] = current_time_sec
                    elif event.status in (
                        EventStatus.OBSERVED,
                        EventStatus.ITEM_LEFT,
                        EventStatus.SUSPECTED,
                    ):
                        # Đang quan sát bình thường tự nhiên mất detection
                        frames_since = current_frame - event.last_seen_frame
                        if frames_since > 5:  # Bỏ qua rung lắc 1-5 frame
                            self.event_manager.transition_event(
                                event_id=event.event_id,
                                new_status=EventStatus.RECHECK,
                                frame_number=current_frame,
                                video_time_sec=current_time_sec,
                                reason="Mất detection trên vùng ghế trống, đang xác minh lại",
                                evidence=assessment.evidence,
                                frame=frame,
                            )
                            self._recheck_start_times[event.event_id] = current_time_sec
                    elif event.status == EventStatus.RECHECK:
                        # Đang ở RECHECK: kiểm tra thời gian
                        start_recheck = self._recheck_start_times.get(
                            event.event_id, current_time_sec
                        )
                        recheck_elapsed = current_time_sec - start_recheck

                        if recheck_elapsed >= self.recheck_duration:
                            # Hết thời gian xác minh lại mà vẫn không thấy vật thể:
                            # Chuyển sang REVIEW_REQUIRED (không tự kết luận là đã lấy đi)
                            self.event_manager.transition_event(
                                event_id=event.event_id,
                                new_status=EventStatus.REVIEW_REQUIRED,
                                frame_number=current_frame,
                                video_time_sec=current_time_sec,
                                reason=(
                                    f"Ghế {event.seat_id or 'trống'} đã nhìn thấy lại trong {recheck_elapsed:.1f}s "
                                    f"nhưng không còn phát hiện vật thể. Cần nhân viên kiểm tra thực tế."
                                ),
                                frame=frame,
                            )
                            if event.event_id in self._recheck_start_times:
                                del self._recheck_start_times[event.event_id]

        return events_to_return

    def analyze(
        self,
        item_observations: Dict[int, TrackObservation],
        current_frame: int,
        fps: float,
        seat_manager: Optional[SeatManager] = None,
        occlusion_manager: Optional[OcclusionManager] = None,
        event_manager: Optional[EventManager] = None,
    ) -> List[Dict]:
        """
        Phân tích dựa trên TrackObservation — tương thích với các bài test và pipeline cơ bản.
        """
        events_to_process = []

        for track_id, obs in item_observations.items():
            if obs.class_name not in self.watched_classes:
                continue

            frames_since_last_seen = current_frame - obs.last_seen_frame
            if frames_since_last_seen > 30:
                continue

            is_stationary = self._is_stationary(obs)
            duration = obs.duration_seconds
            current_status = self._track_status.get(track_id, None)

            if is_stationary and duration >= self.min_stationary_time:
                person_left = self._check_person_left(obs, current_frame, fps)

                if person_left and current_status != EventStatus.SUSPECTED:
                    self._track_status[track_id] = EventStatus.SUSPECTED
                    events_to_process.append({
                        "track_id": track_id,
                        "status": EventStatus.SUSPECTED,
                        "observation": obs,
                        "reason": (
                            f"Vật thể '{obs.class_name}' đứng yên {duration:.1f}s. "
                            f"Người từng ở gần đã rời khỏi vùng quan sát."
                        ),
                    })
                elif not person_left and duration >= self.min_stationary_time:
                    if current_status is None:
                        self._track_status[track_id] = EventStatus.MONITORING
                        events_to_process.append({
                            "track_id": track_id,
                            "status": EventStatus.MONITORING,
                            "observation": obs,
                            "reason": (
                                f"Vật thể '{obs.class_name}' đứng yên {duration:.1f}s. "
                                f"Đang theo dõi."
                            ),
                        })
                    elif (
                        current_status == EventStatus.MONITORING
                        and duration >= self.min_stationary_time * 3
                    ):
                        self._track_status[track_id] = EventStatus.SUSPECTED
                        events_to_process.append({
                            "track_id": track_id,
                            "status": EventStatus.SUSPECTED,
                            "observation": obs,
                            "reason": (
                                f"Vật thể '{obs.class_name}' đứng yên {duration:.1f}s "
                                f"(vượt ngưỡng {self.min_stationary_time * 3:.0f}s). "
                                f"Nghi ngờ bỏ quên."
                            ),
                        })

        return events_to_process

    def _is_stationary(self, obs: TrackObservation) -> bool:
        """Kiểm tra vật thể có đứng yên không."""
        displacement = obs.max_displacement(n_recent=20)
        return displacement < self.movement_threshold

    def _check_person_left(
        self, obs: TrackObservation, current_frame: int, fps: float
    ) -> bool:
        """Kiểm tra dấu hiệu người từng ở gần đã rời đi."""
        if not obs.nearby_person_track_ids:
            return False
        if obs.person_last_seen_frame < 0:
            return False

        frames_since = current_frame - obs.person_last_seen_frame
        time_since = frames_since / fps if fps > 0 else 0
        return time_since >= self.alert_delay

    def get_status_for_track(self, track_id: int) -> Optional[str]:
        return self._track_status.get(track_id)

    @staticmethod
    def is_stationary_from_positions(
        positions: List[Tuple[int, int, int, int]],
        threshold: float = STATIONARY_MOVEMENT_THRESHOLD,
    ) -> bool:
        """Kiểm tra vật thể đứng yên từ danh sách bbox (dùng trong unit test)."""
        if len(positions) < 2:
            return True

        first = positions[0]
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)

        for bbox in positions[1:]:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = ((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2) ** 0.5
            if dist >= threshold:
                return False
        return True
