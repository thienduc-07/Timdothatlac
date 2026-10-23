"""
src/events/event_manager.py
Module quản lý vòng đời (Lifecycle) và chống trùng lặp sự kiện nghi ngờ đồ bỏ quên:
- Quản lý State Machine: OBSERVED, ITEM_LEFT, SUSPECTED, OCCLUDED, RECHECK,
  REVIEW_REQUIRED, RESOLVED, DISMISSED.
- Đảm bảo mỗi sự kiện có một event_id ổn định, không phụ thuộc vào ByteTrack ID.
- Tự động lưu lịch sử chuyển đổi trạng thái (state_history) và bằng chứng liên quan.
- Chụp và lưu ảnh snapshot khi có sự kiện nghi ngờ hoặc cần nhân viên kiểm tra.
- Không tự động kết luận hay xóa cảnh báo khi mất detection tạm thời.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.models import Event, TrackObservation, TrackedObject
from src.utils.logger import get_logger
from src.utils.text_renderer import draw_text_unicode
from config.settings import EventStatus, SNAPSHOTS_DIR, CLASS_NAMES_VI, get_current_time

logger = get_logger(__name__)


class EventManager:
    """
    Quản lý tập trung vòng đời các sự kiện đồ bỏ quên.
    """

    def __init__(self):
        # Mapping: event_id -> Event
        self._events: Dict[str, Event] = {}
        # Mapping: track_id -> event_id (cho tra cứu nhanh)
        self._track_to_event: Dict[int, str] = {}
        # Mapping: seat_id -> event_id (cho các sự kiện đang hoạt động)
        self._seat_to_event: Dict[str, str] = {}

    @property
    def events(self) -> List[Event]:
        """Trả về danh sách tất cả các sự kiện."""
        return list(self._events.values())

    def reset(self):
        """Reset toàn bộ sự kiện cho video hoặc phiên phân tích mới."""
        self._events.clear()
        self._track_to_event.clear()
        self._seat_to_event.clear()

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Lấy sự kiện theo event_id."""
        return self._events.get(event_id)

    def get_event_by_track_id(self, track_id: int) -> Optional[Event]:
        """Tìm sự kiện gắn với track_id gần nhất."""
        event_id = self._track_to_event.get(track_id)
        if event_id:
            return self._events.get(event_id)
        return None

    def get_active_event_for_seat(self, seat_id: str) -> Optional[Event]:
        """Lấy sự kiện đang hoạt động (chưa RESOLVED hoặc DISMISSED) trên một ghế."""
        if not seat_id:
            return None
        event_id = self._seat_to_event.get(seat_id)
        if event_id:
            event = self._events.get(event_id)
            if event and event.status not in (EventStatus.RESOLVED, EventStatus.DISMISSED):
                return event
        return None

    def create_event(
        self,
        class_name: str,
        seat_id: str,
        initial_status: str,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_number: int,
        video_time_sec: float,
        video_name: str = "",
        reason: str = "",
        frame: Optional[np.ndarray] = None,
    ) -> Event:
        """
        Tạo mới một sự kiện với ID ổn định.
        """
        event_id = f"EVT_{str(uuid.uuid4())[:8].upper()}"
        now_iso = get_current_time().isoformat()

        event = Event(
            event_id=event_id,
            timestamp=now_iso,
            class_name=class_name,
            seat_id=seat_id,
            track_id=track_id,
            last_track_id=track_id,
            confidence=confidence,
            duration_seconds=0.0,
            status=initial_status,
            video_name=video_name,
            video_time_sec=video_time_sec,
            frame_number=frame_number,
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            notes=reason,
            first_seen_timestamp=now_iso,
            last_seen_timestamp=now_iso,
            last_seen_frame=frame_number,
            last_seen_time_sec=video_time_sec,
            observed_duration=0.0,
            vacant_duration=0.0,
            occlusion_duration=0.0,
            last_known_bbox=bbox,
        )

        event.add_transition(
            from_state="NONE",
            to_state=initial_status,
            frame=frame_number,
            time_sec=video_time_sec,
            reason=reason or f"Phát hiện vật thể {class_name} tại {seat_id or 'khung hình'}",
        )

        if initial_status == EventStatus.SUSPECTED:
            event.suspected_timestamp = now_iso
            event.suspected_time_sec = video_time_sec

        # Lưu ảnh snapshot
        if frame is not None:
            self._save_snapshot(event, bbox, frame)

        self._events[event_id] = event
        if track_id != -1:
            self._track_to_event[track_id] = event_id
        if seat_id:
            self._seat_to_event[seat_id] = event_id

        logger.info(
            f"Tạo sự kiện mới: {event_id} | Ghế: {seat_id or 'N/A'} | "
            f"Lớp: {class_name} | Trạng thái: {initial_status}"
        )
        return event

    def transition_event(
        self,
        event_id: str,
        new_status: str,
        frame_number: int,
        video_time_sec: float,
        reason: str,
        evidence: Optional[Dict] = None,
        frame: Optional[np.ndarray] = None,
        new_bbox: Optional[Tuple[int, int, int, int]] = None,
        new_track_id: Optional[int] = None,
    ) -> bool:
        """
        Chuyển đổi trạng thái của một sự kiện theo State Machine và ghi lại bằng chứng.
        """
        event = self._events.get(event_id)
        if not event:
            logger.warning(f"Không tìm thấy sự kiện để chuyển trạng thái: {event_id}")
            return False

        old_status = event.status
        if old_status == new_status:
            return True

        now_iso = get_current_time().isoformat()

        # Xử lý các logic đặc biệt theo trạng thái đích
        if new_status == EventStatus.SUSPECTED:
            if not event.suspected_timestamp:
                event.suspected_timestamp = now_iso
                event.suspected_time_sec = video_time_sec
        elif new_status == EventStatus.OCCLUDED:
            if old_status != EventStatus.OCCLUDED:
                event.occlusion_start_timestamp = now_iso
                event.occlusion_start_time_sec = video_time_sec
        elif new_status in (EventStatus.ITEM_LEFT, EventStatus.SUSPECTED, EventStatus.OBSERVED):
            if old_status == EventStatus.OCCLUDED:
                # Đồ vật xuất hiện lại từ trạng thái che khuất
                logger.info(f"Sự kiện {event_id} kết thúc che khuất sau {event.occlusion_duration:.1f}s")

        if new_bbox is not None:
            event.last_known_bbox = new_bbox
            event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2 = new_bbox

        if new_track_id is not None and new_track_id != -1:
            event.last_track_id = new_track_id
            self._track_to_event[new_track_id] = event_id

        event.status = new_status
        event.notes = reason

        if evidence:
            event.evidence_metadata.update(evidence)

        event.add_transition(
            from_state=old_status,
            to_state=new_status,
            frame=frame_number,
            time_sec=video_time_sec,
            reason=reason,
        )

        logger.info(
            f"Chuyển trạng thái sự kiện {event_id}: {old_status} → {new_status} | "
            f"Lý do: {reason}"
        )

        # Chụp ảnh snapshot khi chuyển sang trạng thái quan trọng
        if frame is not None and new_status in (
            EventStatus.SUSPECTED,
            EventStatus.REVIEW_REQUIRED,
            EventStatus.OCCLUDED,
        ):
            self._save_snapshot(event, event.last_known_bbox, frame)

        return True

    def reassociate_detection_to_event(
        self,
        event_id: str,
        new_track_id: int,
        new_bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_number: int,
        video_time_sec: float,
        reason: str,
    ) -> bool:
        """
        Nối lại một detection mới vào sự kiện cũ khi bằng chứng đã được xác nhận khớp.
        """
        event = self._events.get(event_id)
        if not event:
            return False

        event.last_track_id = new_track_id
        event.last_known_bbox = new_bbox
        event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2 = new_bbox
        event.last_seen_frame = frame_number
        event.last_seen_time_sec = video_time_sec
        event.confidence = confidence

        self._track_to_event[new_track_id] = event_id

        # Nếu đang ở OCCLUDED hoặc RECHECK, chuyển về trạng thái phù hợp
        if event.status in (EventStatus.OCCLUDED, EventStatus.RECHECK):
            target_status = (
                EventStatus.SUSPECTED
                if event.suspected_timestamp
                else EventStatus.ITEM_LEFT
            )
            self.transition_event(
                event_id=event_id,
                new_status=target_status,
                frame_number=frame_number,
                video_time_sec=video_time_sec,
                reason=f"Đồ vật xuất hiện trở lại: {reason}",
            )
        return True

    def update_event_status(self, event_id: str, new_status: str) -> bool:
        """
        Cập nhật trạng thái sự kiện (dùng cho hành động thủ công từ người dùng).
        """
        event = self._events.get(event_id)
        if not event:
            logger.warning(f"Không tìm thấy sự kiện: {event_id}")
            return False

        old_status = event.status
        event.status = new_status
        event.add_transition(
            from_state=old_status,
            to_state=new_status,
            frame=event.last_seen_frame,
            time_sec=event.last_seen_time_sec,
            reason="Người vận hành cập nhật thủ công trên giao diện",
        )
        logger.info(f"Cập nhật thủ công sự kiện {event_id}: {old_status} → {new_status}")
        return True

    def get_events_by_status(self, status: str) -> List[Event]:
        """Lọc sự kiện theo trạng thái cụ thể."""
        return [e for e in self._events.values() if e.status == status]

    def get_suspected_events(self) -> List[Event]:
        """Lấy tất cả sự kiện đang nghi ngờ bỏ quên."""
        return self.get_events_by_status(EventStatus.SUSPECTED)

    def get_occluded_events(self) -> List[Event]:
        """Lấy tất cả sự kiện đang bị che khuất."""
        return self.get_events_by_status(EventStatus.OCCLUDED)

    def get_review_required_events(self) -> List[Event]:
        """Lấy tất cả sự kiện cần nhân viên kiểm tra."""
        return self.get_events_by_status(EventStatus.REVIEW_REQUIRED)

    def get_summary(self) -> Dict:
        """Trả về thống kê tổng hợp số lượng sự kiện theo từng trạng thái."""
        summary = {
            "total": len(self._events),
            "observed": len(self.get_events_by_status(EventStatus.OBSERVED)),
            "item_left": len(self.get_events_by_status(EventStatus.ITEM_LEFT)),
            "suspected": len(self.get_events_by_status(EventStatus.SUSPECTED)),
            "occluded": len(self.get_events_by_status(EventStatus.OCCLUDED)),
            "recheck": len(self.get_events_by_status(EventStatus.RECHECK)),
            "review_required": len(self.get_events_by_status(EventStatus.REVIEW_REQUIRED)),
            "resolved": len(self.get_events_by_status(EventStatus.RESOLVED)),
            "dismissed": len(self.get_events_by_status(EventStatus.DISMISSED)),
            "monitoring": len(self.get_events_by_status(EventStatus.MONITORING)),
            "confirmed": len(self.get_events_by_status(EventStatus.CONFIRMED)),
        }
        return summary

    def _save_snapshot(
        self, event: Event, bbox: Tuple[int, int, int, int], frame: np.ndarray
    ):
        """Lưu ảnh chụp frame bằng chứng tại thời điểm sự kiện diễn ra."""
        try:
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"event_{event.event_id}_{event.class_name}_{event.status}.jpg"
            snapshot_path = SNAPSHOTS_DIR / filename

            snapshot = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)

            vi_name = CLASS_NAMES_VI.get(event.class_name, event.class_name)
            label = f"{event.event_id} - {vi_name} [{event.status.upper()}]"
            draw_text_unicode(
                frame=snapshot,
                text=label,
                pos=(x1, max(4, y1 - 22)),
                font_size=13,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 200),
                padding=3,
            )

            cv2.imwrite(str(snapshot_path), snapshot)
            event.snapshot_path = str(snapshot_path)
            logger.debug(f"Snapshot lưu tại: {snapshot_path}")
        except Exception as e:
            logger.warning(f"Không thể lưu snapshot: {e}")

    # Phương thức tương thích ngược với ForgottenItemDetector cũ
    def process_detection_result(
        self,
        track_id: int,
        new_status: str,
        observation: TrackObservation,
        reason: str,
        video_name: str,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Event]:
        """
        Cung cấp khả năng tương thích ngược với pipeline cũ.
        """
        existing = self.get_event_by_track_id(track_id)
        bbox = observation.positions[-1] if observation.positions else (0, 0, 0, 0)
        time_sec = observation.last_seen_time
        frame_idx = observation.last_seen_frame

        if existing:
            self.transition_event(
                event_id=existing.event_id,
                new_status=new_status,
                frame_number=frame_idx,
                video_time_sec=time_sec,
                reason=reason,
                frame=frame,
                new_bbox=bbox,
                new_track_id=track_id,
            )
            existing.duration_seconds = observation.duration_seconds
            existing.confidence = observation.avg_confidence
            return None

        # Tạo sự kiện mới
        return self.create_event(
            class_name=observation.class_name,
            seat_id="",
            initial_status=new_status,
            track_id=track_id,
            bbox=bbox,
            confidence=observation.avg_confidence,
            frame_number=frame_idx,
            video_time_sec=time_sec,
            video_name=video_name,
            reason=reason,
            frame=frame,
        )
