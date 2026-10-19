"""
src/tracking/object_tracker.py
Module phụ trách theo dõi vật thể qua nhiều frame bằng ByteTrack
(tích hợp sẵn trong Ultralytics).

- Duy trì trạng thái tracker liên tục giữa các frame.
- Ghi lại lịch sử quan sát (TrackObservation) cho mỗi vật thể.
- Không giả định tracking ID tồn tại vĩnh viễn.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from src.models import Detection, TrackedObject, TrackObservation
from src.utils.logger import get_logger
from config.settings import (
    DEVICE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ITEM_CLASSES,
    PERSON_CLASS,
    PERSON_ITEM_PROXIMITY_THRESHOLD,
)

logger = get_logger(__name__)


class ObjectTracker:
    """
    Quản lý tracking vật thể qua các frame.
    Sử dụng model.track() của Ultralytics (ByteTrack tích hợp sẵn).
    Ghi lại lịch sử quan sát để phục vụ phân tích sự kiện.
    """

    def __init__(self):
        self._observations: Dict[int, TrackObservation] = {}
        # track_id -> TrackObservation
        self._frame_count = 0
        self._current_tracked: List[TrackedObject] = []

    @property
    def observations(self) -> Dict[int, TrackObservation]:
        """Trả về toàn bộ lịch sử quan sát."""
        return self._observations

    @property
    def current_tracked_objects(self) -> List[TrackedObject]:
        """Trả về danh sách vật thể đang được theo dõi ở frame hiện tại."""
        return self._current_tracked

    def reset(self):
        """Reset trạng thái tracker cho video mới."""
        self._observations.clear()
        self._frame_count = 0
        self._current_tracked = []

    def update(
        self,
        model,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        target_class_ids: Optional[List[int]] = None,
    ) -> List[TrackedObject]:
        """
        Chạy tracking trên một frame bằng model.track() của Ultralytics.
        Phương thức này duy trì trạng thái tracker nội bộ (persist=True).

        Args:
            model: Model YOLO đã load
            frame: Frame BGR từ OpenCV
            frame_index: Chỉ số frame hiện tại
            fps: FPS của video (để tính thời gian)
            confidence_threshold: Ngưỡng confidence
            target_class_ids: Danh sách class ID cần theo dõi

        Returns:
            Danh sách TrackedObject trong frame hiện tại
        """
        self._frame_count = frame_index
        current_time = frame_index / fps if fps > 0 else 0.0

        try:
            # Gọi model.track() — persist=True giữ trạng thái ByteTrack
            results = model.track(
                frame,
                conf=confidence_threshold,
                device=DEVICE,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=target_class_ids,
            )

            tracked_objects = []
            current_person_positions = {}  # track_id -> center position (cho người)
            current_item_ids = []

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                for i, box in enumerate(result.boxes):
                    track_id = int(box.id[i]) if box.id is not None else -1
                    if track_id == -1:
                        continue

                    class_id = int(box.cls[i])
                    class_name = model.names.get(class_id, f"class_{class_id}")
                    confidence = float(box.conf[i])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[i].tolist()]
                    bbox = (x1, y1, x2, y2)

                    obj = TrackedObject(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        class_id=class_id,
                    )
                    tracked_objects.append(obj)

                    # Ghi nhận vị trí người
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    if class_name == PERSON_CLASS:
                        current_person_positions[track_id] = center
                    elif class_name in ITEM_CLASSES:
                        current_item_ids.append(track_id)

                    # Cập nhật lịch sử quan sát
                    self._update_observation(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        frame_index=frame_index,
                        current_time=current_time,
                    )

            # Cập nhật thông tin người gần đồ vật
            self._update_person_proximity(
                current_person_positions, current_item_ids, frame_index
            )

            self._current_tracked = tracked_objects
            return tracked_objects

        except Exception as e:
            logger.error(f"Lỗi tracking tại frame {frame_index}: {e}")
            self._current_tracked = []
            return []

    def _update_observation(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        frame_index: int,
        current_time: float,
    ):
        """Cập nhật hoặc tạo mới TrackObservation cho một track."""
        if track_id in self._observations:
            obs = self._observations[track_id]
            obs.last_seen_frame = frame_index
            obs.last_seen_time = current_time
            obs.positions.append(bbox)
            obs.confidence_history.append(confidence)
            # Giới hạn số lượng positions lưu trữ (tiết kiệm bộ nhớ)
            if len(obs.positions) > 300:
                obs.positions = obs.positions[-200:]
            if len(obs.confidence_history) > 300:
                obs.confidence_history = obs.confidence_history[-200:]
        else:
            self._observations[track_id] = TrackObservation(
                track_id=track_id,
                class_name=class_name,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
                first_seen_time=current_time,
                last_seen_time=current_time,
                positions=[bbox],
                confidence_history=[confidence],
            )

    def _update_person_proximity(
        self,
        person_positions: Dict[int, Tuple[float, float]],
        item_track_ids: List[int],
        frame_index: int,
    ):
        """
        Kiểm tra xem có người nào đứng gần đồ vật hay không.
        Cập nhật thông tin nearby_person_track_ids và person_last_seen_frame.
        """
        threshold = PERSON_ITEM_PROXIMITY_THRESHOLD

        for item_tid in item_track_ids:
            if item_tid not in self._observations:
                continue
            item_obs = self._observations[item_tid]
            item_center = item_obs.center

            for person_tid, person_center in person_positions.items():
                dist = (
                    (item_center[0] - person_center[0]) ** 2
                    + (item_center[1] - person_center[1]) ** 2
                ) ** 0.5

                if dist < threshold:
                    if person_tid not in item_obs.nearby_person_track_ids:
                        item_obs.nearby_person_track_ids.append(person_tid)
                    item_obs.person_last_seen_frame = frame_index

    def get_item_observations(self) -> Dict[int, TrackObservation]:
        """Trả về chỉ các observation của đồ vật (không bao gồm người)."""
        return {
            tid: obs
            for tid, obs in self._observations.items()
            if obs.class_name in ITEM_CLASSES
        }

    def get_person_observations(self) -> Dict[int, TrackObservation]:
        """Trả về chỉ các observation của người."""
        return {
            tid: obs
            for tid, obs in self._observations.items()
            if obs.class_name == PERSON_CLASS
        }
