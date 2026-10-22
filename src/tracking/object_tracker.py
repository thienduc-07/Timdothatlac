"""
src/tracking/object_tracker.py
Module phụ trách theo dõi vật thể qua nhiều frame bằng ByteTrack
(tích hợp sẵn trong Ultralytics).

- Duy trì trạng thái tracker liên tục giữa các frame.
- Ghi lại lịch sử quan sát (TrackObservation) cho mỗi vật thể.
- Không giả định tracking ID tồn tại vĩnh viễn.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

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


_CLASS_NAME_TO_ID_MAP = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "seat": 10,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "bottle": 39,
    "cell phone": 67,
    "laptop": 63,
    "book": 73,
    "item": 80,
    "empty_seat": 81,
    "human_occupied": 82,
    "object_occupied": 83,
}


class ObjectTracker:
    """
    Quản lý tracking vật thể qua các frame.
    Hỗ trợ tracking đa chế độ:
    - Supervision ByteTrack cho normalized detections (Local, Roboflow-only, Hybrid).
    - Ultralytics model.track() khi chạy trực tiếp với model instance.
    - Ghi lại lịch sử quan sát (TrackObservation) phục vụ phát hiện đồ bỏ quên.
    """

    def __init__(self):
        self._observations: Dict[int, TrackObservation] = {}
        self._frame_count = 0
        self._current_tracked: List[TrackedObject] = []
        self._sv_tracker = None
        self._init_sv_tracker()

    def _init_sv_tracker(self):
        try:
            import supervision as sv
            self._sv_tracker = sv.ByteTrack()
        except Exception as e:
            logger.warning(f"Supervision ByteTrack không khả dụng: {e}")
            self._sv_tracker = None

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
        self._init_sv_tracker()

    def update(
        self,
        model=None,
        frame: Optional[np.ndarray] = None,
        frame_index: int = 0,
        fps: float = 10.0,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        target_class_ids: Optional[List[int]] = None,
        detections: Optional[List[Any]] = None,
    ) -> List[TrackedObject]:
        """
        Chạy tracking trên một frame.
        Hỗ trợ:
        1. Pre-computed detections (Supervision ByteTrack: Local, Roboflow, Hybrid).
        2. Direct model instance (Ultralytics ByteTrack).

        Returns:
            Danh sách TrackedObject trong frame hiện tại với track_id nhất quán.
        """
        self._frame_count = frame_index
        current_time = frame_index / fps if fps > 0 else 0.0

        # Ưu tiên tracking pre-computed normalized detections bằng Supervision ByteTrack
        if detections is not None:
            tracked_objects: List[TrackedObject] = []
            current_person_positions = {}
            current_item_ids = []

            if self._sv_tracker is not None:
                try:
                    import supervision as sv
                    if detections:
                        boxes = []
                        confs = []
                        class_ids = []
                        class_names = []
                        polygons = []
                        for det in detections:
                            b = det.bbox
                            boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                            confs.append(float(det.confidence))
                            cname = getattr(det, "class_name", "item")
                            class_names.append(cname)
                            cid = getattr(det, "class_id", None)
                            if cid is None:
                                cid = _CLASS_NAME_TO_ID_MAP.get(cname, abs(hash(cname)) % 1000)
                            class_ids.append(cid)
                            polygons.append(getattr(det, "polygon", None))

                        sv_dets = sv.Detections(
                            xyxy=np.array(boxes, dtype=np.float32),
                            confidence=np.array(confs, dtype=np.float32),
                            class_id=np.array(class_ids, dtype=int),
                            data={
                                "class_name": np.array(class_names),
                                "polygon": np.array(polygons, dtype=object),
                            },
                        )
                    else:
                        sv_dets = sv.Detections.empty()

                    tracked_sv = self._sv_tracker.update_with_detections(sv_dets)

                    for i in range(len(tracked_sv)):
                        tid = int(tracked_sv.tracker_id[i]) if tracked_sv.tracker_id is not None else (i + 1)
                        x1, y1, x2, y2 = [int(v) for v in tracked_sv.xyxy[i]]
                        c_name = str(tracked_sv.data["class_name"][i]) if "class_name" in tracked_sv.data else "item"
                        c_conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0
                        cid = int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0
                        poly = tracked_sv.data["polygon"][i] if "polygon" in tracked_sv.data else None
                        if poly is None:
                            # Bảo toàn polygon từ lịch sử quan sát gần nhất (lên đến 15 frames)
                            # để tránh hiện tượng mất viền polygon và thụt lùi về bounding box khi tracking
                            prev_obs = self._observations.get(tid)
                            if prev_obs and hasattr(prev_obs, "polygons") and prev_obs.polygons:
                                for p in reversed(prev_obs.polygons[-15:]):
                                    if p is not None and len(p) >= 3:
                                        poly = p
                                        break

                        obj = TrackedObject(
                            track_id=tid,
                            class_name=c_name,
                            confidence=c_conf,
                            bbox=(x1, y1, x2, y2),
                            class_id=cid,
                            polygon=poly,
                        )
                        tracked_objects.append(obj)

                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        if c_name == PERSON_CLASS:
                            current_person_positions[tid] = center
                        elif c_name in ITEM_CLASSES or c_name == "item":
                            current_item_ids.append(tid)

                        self._update_observation(
                            track_id=tid,
                            class_name=c_name,
                            confidence=c_conf,
                            bbox=(x1, y1, x2, y2),
                            frame_index=frame_index,
                            current_time=current_time,
                            polygon=poly,
                        )

                    self._update_person_proximity(
                        current_person_positions, current_item_ids, frame_index
                    )
                    self._current_tracked = tracked_objects
                    return tracked_objects
                except Exception as e:
                    logger.error(f"Lỗi Supervision tracking tại frame {frame_index}: {e}")

            # Fallback nếu Supervision chưa khởi tạo hoặc lỗi
            for idx, det in enumerate(detections):
                tid = getattr(det, "track_id", -1)
                if tid == -1:
                    tid = idx + 1
                cname = getattr(det, "class_name", "item")
                bbox = getattr(det, "bbox", (0, 0, 0, 0))
                poly = getattr(det, "polygon", None)
                if poly is None:
                    prev_obs = self._observations.get(tid)
                    if prev_obs and hasattr(prev_obs, "polygons") and prev_obs.polygons:
                        for p in reversed(prev_obs.polygons[-15:]):
                            if p is not None and len(p) >= 3:
                                poly = p
                                break

                obj = TrackedObject(
                    track_id=tid,
                    class_name=cname,
                    confidence=getattr(det, "confidence", 1.0),
                    bbox=bbox,
                    class_id=getattr(det, "class_id", 0),
                    polygon=poly,
                )
                tracked_objects.append(obj)
                self._update_observation(
                    track_id=tid,
                    class_name=cname,
                    confidence=getattr(det, "confidence", 1.0),
                    bbox=bbox,
                    frame_index=frame_index,
                    current_time=current_time,
                    polygon=getattr(det, "polygon", None),
                )
            self._current_tracked = tracked_objects
            return tracked_objects

        # Nếu không truyền detections mà truyền model, dùng Ultralytics track()
        elif model is not None and hasattr(model, "track") and frame is not None:
            try:
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
                current_person_positions = {}
                current_item_ids = []

                for result in results:
                    if result.boxes is None or result.boxes.id is None:
                        continue

                    has_masks = (
                        hasattr(result, "masks")
                        and result.masks is not None
                        and hasattr(result.masks, "xy")
                        and result.masks.xy is not None
                    )

                    for i, box in enumerate(result.boxes):
                        try:
                            track_id = int(box.id[0]) if box.id is not None else -1
                        except Exception:
                            track_id = int(result.boxes.id[i]) if result.boxes.id is not None else -1

                        if track_id == -1:
                            continue

                        try:
                            class_id = int(box.cls[0])
                        except Exception:
                            class_id = int(result.boxes.cls[i])

                        class_name = model.names.get(class_id, f"class_{class_id}")

                        try:
                            confidence = float(box.conf[0])
                        except Exception:
                            confidence = float(result.boxes.conf[i])

                        try:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        except Exception:
                            x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i].tolist()]
                        bbox = (x1, y1, x2, y2)

                        polygon: Optional[List[Tuple[float, float]]] = None
                        if has_masks and i < len(result.masks.xy):
                            poly_pts = result.masks.xy[i]
                            if poly_pts is not None and len(poly_pts) >= 3:
                                h, w = frame.shape[:2]
                                valid_pts = []
                                for pt in poly_pts:
                                    px = max(0.0, min(float(w), float(pt[0])))
                                    py = max(0.0, min(float(h), float(pt[1])))
                                    valid_pts.append((px, py))
                                if len(valid_pts) >= 3:
                                    polygon = valid_pts

                        obj = TrackedObject(
                            track_id=track_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            class_id=class_id,
                            polygon=polygon,
                        )
                        tracked_objects.append(obj)

                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        if class_name == PERSON_CLASS:
                            current_person_positions[track_id] = center
                        elif class_name in ITEM_CLASSES or class_name == "item":
                            current_item_ids.append(track_id)

                        self._update_observation(
                            track_id=track_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            frame_index=frame_index,
                            current_time=current_time,
                            polygon=polygon,
                        )

                self._update_person_proximity(
                    current_person_positions, current_item_ids, frame_index
                )

                self._current_tracked = tracked_objects
                return tracked_objects

            except Exception as e:
                logger.error(f"Lỗi tracking tại frame {frame_index}: {e}")
                self._current_tracked = []
                return []
        else:
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
        polygon: Optional[List[Tuple[float, float]]] = None,
    ):
        """Cập nhật hoặc tạo mới TrackObservation cho một track."""
        if track_id in self._observations:
            obs = self._observations[track_id]
            obs.last_seen_frame = frame_index
            obs.last_seen_time = current_time
            obs.positions.append(bbox)
            obs.confidence_history.append(confidence)
            if hasattr(obs, "polygons"):
                obs.polygons.append(polygon)
                if len(obs.polygons) > 300:
                    obs.polygons = obs.polygons[-200:]
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
                polygons=[polygon],
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
