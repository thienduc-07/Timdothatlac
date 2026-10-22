"""
src/video/video_processor.py
Module điều phối pipeline xử lý video:
  Đọc frame → Detection/Tracking → Event detection → Ghi frame chú thích.
"""

import numpy as np
from typing import Callable, Optional

from src.detection.yolo_detector import YOLODetector
from src.tracking.object_tracker import ObjectTracker
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.events.event_manager import EventManager
from src.video.video_reader import VideoReader
from src.video.video_writer import VideoWriter
from src.models import AnalysisResult, VideoMetadata, Detection
from src.utils.logger import get_logger
from src.utils.file_utils import get_unique_output_path
from config.settings import (
    OUTPUTS_DIR,
    MAX_PROCESSING_FPS,
    MAX_FRAME_WIDTH,
    OUTPUT_VIDEO_EXTENSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    INFERENCE_MODE,
)

# Import Roboflow client for remote inference
from src.roboflow.roboflow_client import RoboflowWorkflowClient

from src.detection.class_mapper import map_class

# Simple IoU helper for bounding box overlap
def _iou(box_a, box_b):
    """Calculate Intersection over Union for two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def fuse_hybrid_detections(
    local_detections: list, remote_detections: list, iou_threshold: float = 0.5
) -> list:
    """
    Fuse local and remote detections without discarding valid detections.

    1. If a local and remote detection overlap with IoU > iou_threshold AND share
       the SAME semantic class (via map_class):
       - Keep the detection with higher confidence.
       - Preserve polygon mask if available on either detection.
    2. If they overlap but have DIFFERENT semantic classes (e.g. person and item,
       or seat and item):
       - Keep BOTH detections so personal items or distinct entities are not discarded.
    3. Retain all non-overlapping detections from both sources.
    """
    fused = list(local_detections)
    for rd in remote_detections:
        rd_sem = map_class(rd.class_name)
        matched_idx = None
        for idx, ld in enumerate(local_detections):
            ld_sem = map_class(ld.class_name)
            if rd_sem == ld_sem and _iou(rd.bbox, ld.bbox) > iou_threshold:
                matched_idx = idx
                break

        if matched_idx is not None:
            ld = local_detections[matched_idx]
            poly = getattr(rd, "polygon", None) or getattr(ld, "polygon", None)
            if rd.confidence > ld.confidence:
                fused[matched_idx] = Detection(
                    class_name=rd.class_name,
                    confidence=rd.confidence,
                    bbox=rd.bbox,
                    polygon=poly,
                    class_id=getattr(rd, "class_id", -1),
                )
            else:
                fused[matched_idx] = Detection(
                    class_name=ld.class_name,
                    confidence=ld.confidence,
                    bbox=ld.bbox,
                    polygon=poly,
                    class_id=getattr(ld, "class_id", -1),
                )
        else:
            fused.append(rd)
    return fused


logger = get_logger(__name__)
from src.seats.seat_manager import SeatManager
from src.seats.seat_config import SeatConfigManager

class VideoProcessor:
    """
    Pipeline xử lý video:
    1. Đọc frame-by-frame từ VideoReader.
    2. Chạy YOLO + ByteTrack qua ObjectTracker.
    3. Phân tích sự kiện bằng ForgottenItemDetector + EventManager + SeatManager.
    4. Ghi frame có chú thích vùng ghế và trạng thái bằng VideoWriter.
    """

    def __init__(
        self,
        detector: YOLODetector,
        tracker: ObjectTracker,
        event_detector: ForgottenItemDetector,
        event_manager: EventManager,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        seat_manager: Optional[SeatManager] = None,
    ):
        # Store components
        self.detector = detector
        self.tracker = tracker
        self.event_detector = event_detector
        self.event_manager = event_manager
        self.confidence_threshold = confidence_threshold
        self.seat_manager = seat_manager
        self.inference_mode = INFERENCE_MODE  # 'local', 'roboflow', or 'hybrid'
        # Initialize Roboflow client lazily (only if needed)
        self.roboflow_client = None
        if self.inference_mode in ("roboflow", "hybrid"):
            try:
                self.roboflow_client = RoboflowWorkflowClient()
            except Exception as exc:
                logger.error(f"Failed to initialize Roboflow client: {exc}")
                # Fallback to local mode if remote init fails
                self.inference_mode = "local"
        # Connect seat_manager and event_manager into event_detector
        self.event_detector.event_manager = self.event_manager
        if self.seat_manager:
            self.event_detector.seat_manager = self.seat_manager

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnalysisResult:
        """
        Xử lý toàn bộ video và trả về kết quả phân tích.

        Args:
            video_path: Đường dẫn file video
            progress_callback: Hàm callback (current_frame, total_frames)
                                để cập nhật tiến trình.

        Returns:
            AnalysisResult chứa kết quả tổng hợp.
        """
        result = AnalysisResult()

        # Reset tracker, event detector và event manager cho video mới
        self.tracker.reset()
        self.event_detector.reset()
        self.event_manager.reset()
        if self.seat_manager:
            self.seat_manager.reset()

        # Mở video
        reader = VideoReader(video_path)
        meta = reader.open()
        result.video_name = meta.filename

        if not meta.is_valid:
            result.error_message = meta.error
            reader.close()
            return result

        # Tính skip ratio dựa trên FPS video
        skip_ratio = max(1, int(meta.fps / MAX_PROCESSING_FPS))
        effective_fps = meta.fps / skip_ratio

        # Xác định kích thước frame đầu ra
        preview = reader.get_preview_frame()
        if preview is None:
            result.error_message = "Không thể đọc frame xem trước"
            reader.close()
            return result

        out_h, out_w = preview.shape[:2]
        if out_w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / out_w
            out_w = MAX_FRAME_WIDTH
            out_h = int(out_h * scale)

        # Chuẩn bị output path
        output_path = get_unique_output_path(
            OUTPUTS_DIR, meta.filename, "_analyzed", OUTPUT_VIDEO_EXTENSION
        )
        result.output_video_path = str(output_path)

        # Mở video writer
        writer = VideoWriter(str(output_path), effective_fps, out_w, out_h)
        if not writer.open():
            result.error_message = "Không thể tạo video đầu ra"
            reader.close()
            return result

        logger.info(
            f"Bắt đầu xử lý video: {meta.filename} | "
            f"Skip ratio: {skip_ratio} | "
            f"Effective FPS: {effective_fps:.1f} | "
            f"Ghế đã cấu hình: {len(self.seat_manager.seats) if self.seat_manager else 0}"
        )

        total_objects = 0
        frames_processed = 0

        try:
            # Lấy target class IDs
            target_ids = list(self.detector._target_class_ids) if self.detector._target_class_ids else None

            for frame_idx, frame in reader.read_frames(
                skip_ratio=skip_ratio, max_width=MAX_FRAME_WIDTH
            ):
                cur_h, cur_w = frame.shape[:2]

                # -------- Inference handling based on mode --------
                local_detections = []
                remote_detections = []

                if self.inference_mode in ("local", "hybrid"):
                    # Ensure model is loaded if detector supports it
                    if hasattr(self.detector, "is_loaded") and not self.detector.is_loaded:
                        if hasattr(self.detector, "load_model"):
                            self.detector.load_model()
                    elif not hasattr(self.detector, "is_loaded") and hasattr(self.detector, "load_model"):
                        self.detector.load_model()

                    # Use YOLO detector directly if it has detect()
                    if hasattr(self.detector, "detect"):
                        local_detections = self.detector.detect(frame)
                    else:
                        local_detections = None

                if self.inference_mode in ("roboflow", "hybrid") and self.roboflow_client:
                    # Submit frame to Roboflow workflow (as temporary JPEG file)
                    import cv2, tempfile, pathlib
                    tmp_path = pathlib.Path(tempfile.mkstemp(suffix=".jpg")[1])
                    cv2.imwrite(str(tmp_path), frame)
                    payload = self.roboflow_client.submit_image(tmp_path)
                    tmp_path.unlink(missing_ok=True)
                    if payload:
                        parsed = RoboflowWorkflowClient.parse_workflow_output(payload)
                        # parse_workflow_output normalises bbox to [x1, y1, x2, y2]
                        for pred in parsed.get("predictions", []):
                            cls = pred.get("class", "")
                            conf = float(pred.get("confidence", 0))
                            if conf < self.confidence_threshold:
                                continue
                            bbox = pred.get("bbox", [0, 0, 0, 0])
                            polygon = pred.get("polygon")
                            remote_detections.append(
                                Detection(
                                    class_name=cls,
                                    confidence=conf,
                                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                    polygon=polygon,
                                )
                            )

                # Merge detections for hybrid mode
                if self.inference_mode == "hybrid":
                    detections_to_track = fuse_hybrid_detections(
                        local_detections or [], remote_detections
                    )
                elif self.inference_mode == "roboflow":
                    detections_to_track = remote_detections
                else:  # local
                    detections_to_track = local_detections

                # Tracking (inspect signature so both ObjectTracker and DummyTracker are supported)
                import inspect
                try:
                    sig = inspect.signature(self.tracker.update)
                    supports_det = (
                        detections_to_track is not None
                        and ("detections" in sig.parameters or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                        ))
                    )
                except Exception:
                    supports_det = detections_to_track is not None

                if supports_det:
                    tracked = self.tracker.update(
                        model=getattr(self.detector, "model", None) if self.inference_mode != "roboflow" else None,
                        frame=frame,
                        frame_index=frame_idx,
                        fps=meta.fps,
                        confidence_threshold=self.confidence_threshold,
                        target_class_ids=target_ids,
                        detections=detections_to_track,
                    )
                else:
                    tracked = self.tracker.update(
                        model=getattr(self.detector, "model", None),
                        frame=frame,
                        frame_index=frame_idx,
                        fps=meta.fps,
                        confidence_threshold=self.confidence_threshold,
                        target_class_ids=target_ids,
                    )

                total_objects += len(tracked)

                # Cập nhật phát hiện ghế từ AI predictions trên frame (AI Seat Detection)
                if self.seat_manager:
                    self.seat_manager.update_detected_seats(
                        raw_predictions=detections_to_track if detections_to_track is not None else tracked,
                        frame_idx=frame_idx,
                        frame_width=cur_w,
                        frame_height=cur_h,
                    )

                # Phân tích sự kiện theo phương thức phù hợp
                if self.seat_manager and self.seat_manager.is_configured:
                    self.event_detector.process_frame(
                        tracked_objects=tracked,
                        current_frame=frame_idx,
                        fps=meta.fps,
                        frame_width=cur_w,
                        frame_height=cur_h,
                        video_name=meta.filename,
                        frame=frame,
                    )
                else:
                    item_obs = self.tracker.get_item_observations()
                    events_to_process = self.event_detector.analyze(
                        item_observations=item_obs,
                        current_frame=frame_idx,
                        fps=meta.fps,
                    )
                    for evt_info in events_to_process:
                        self.event_manager.process_detection_result(
                            track_id=evt_info["track_id"],
                            new_status=evt_info["status"],
                            observation=evt_info["observation"],
                            reason=evt_info["reason"],
                            video_name=meta.filename,
                            frame=frame,
                        )

                # Tạo dict trạng thái sự kiện cho video writer
                event_statuses = {}
                for evt in self.event_manager.events:
                    if evt.last_track_id != -1:
                        event_statuses[evt.last_track_id] = evt.status
                    elif evt.track_id != -1:
                        event_statuses[evt.track_id] = evt.status

                # Ghi frame có chú thích đầy đủ
                writer.write_annotated_frame(
                    frame=frame,
                    tracked_objects=tracked,
                    event_statuses=event_statuses,
                    seat_manager=self.seat_manager,
                    active_events=self.event_manager.events,
                )

                frames_processed += 1


                # Cập nhật tiến trình
                if progress_callback:
                    progress_callback(frame_idx, meta.total_frames)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý video: {e}")
            result.error_message = f"Lỗi xử lý: {str(e)}"
        finally:
            writer.close()
            reader.close()

        result.total_frames_processed = frames_processed
        result.total_objects_detected = total_objects
        result.events = list(self.event_manager.events)
        result.total_events_created = len(result.events)
        result.success = frames_processed > 0 and not result.error_message

        logger.info(
            f"Hoàn thành: {frames_processed} frames | "
            f"{total_objects} objects | "
            f"{result.total_events_created} events"
        )

        return result
