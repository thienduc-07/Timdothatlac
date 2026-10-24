"""
src/services/analysis_service.py
Lớp kết nối giao diện (app.py) với pipeline xử lý bên trong.
app.py không cần biết chi tiết về YOLO, tracker, database, v.v.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple
import numpy as np


from src.detection.yolo_detector import YOLODetector
from src.tracking.object_tracker import ObjectTracker
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.events.event_manager import EventManager
from src.video.video_processor import VideoProcessor
from src.video.video_reader import VideoReader
from src.database.db import Database
from src.database.event_repository import EventRepository
from src.models import AnalysisResult, VideoMetadata
from src.utils.logger import get_logger
from src.utils.file_utils import validate_video_file, sanitize_filename
from config.settings import (
    DATA_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_STATIONARY_TIME_SEC,
    STATIONARY_MOVEMENT_THRESHOLD,
    ITEM_CLASSES,
    EventStatus,
)

from src.seats.seat_config import SeatConfigManager
from src.seats.seat_manager import SeatManager
from src.models import SeatConfig, SeatStatus, Event, Alert, Detection
from src.database.alert_repository import AlertRepository
from src.database.dataset_repository import DatasetRepository
from src.database.model_repository import ModelRepository
from src.services.alert_service import AlertService
from src.services.notification_service import NotificationService
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService
from src.video.camera_service import CameraStreamManager
from src.notifications.telegram_notifier import get_telegram_notifier


logger = get_logger(__name__)


class AnalysisService:
    """
    Dịch vụ phân tích video — lớp trung gian giữa giao diện và pipeline.
    Hỗ trợ quản lý vùng ghế (Seat ROI), phát hiện che khuất,
    quản lý Alert đa kênh, Dataset annotation và Model versioning.
    """

    def __init__(self):
        self._detector: Optional[YOLODetector] = None
        self._db: Optional[Database] = None
        self._repo: Optional[EventRepository] = None
        self._alert_repo: Optional[AlertRepository] = None
        self._alert_service: Optional[AlertService] = None
        self._dataset_repo: Optional[DatasetRepository] = None
        self._dataset_service: Optional[DatasetService] = None
        self._model_repo: Optional[ModelRepository] = None
        self._model_service: Optional[ModelService] = None
        self._camera_manager = CameraStreamManager()
        self._seat_config_mgr = SeatConfigManager()
        self._seat_manager = SeatManager()
        self._model_loaded = False
        from config.settings import INFERENCE_MODE
        self.inference_mode = INFERENCE_MODE
        self.roboflow_client = None

    def initialize(self) -> bool:
        """
        Khởi tạo model AI, database và cấu hình ghế.

        Returns:
            True nếu khởi tạo thành công.
        """
        # Khởi tạo database
        self._db = Database()
        if not self._db.connect():
            logger.error("Không thể kết nối database")
            return False

        conn = self._db.get_connection()
        self._repo = EventRepository(conn)
        self._alert_repo = AlertRepository(conn)
        self._alert_service = AlertService(self._alert_repo)
        self._dataset_repo = DatasetRepository(conn)
        self._dataset_service = DatasetService(self._dataset_repo)
        self._model_repo = ModelRepository(conn)
        self._model_service = ModelService(self._model_repo)

        # Khởi tạo Roboflow client nếu có cấu hình
        try:
            from src.roboflow.roboflow_client import RoboflowWorkflowClient
            self.roboflow_client = RoboflowWorkflowClient()
            self._seat_manager.detector.set_roboflow_client(self.roboflow_client)
            logger.info("AnalysisService: Khởi tạo RoboflowWorkflowClient thành công")
        except Exception as e:
            logger.info(f"AnalysisService: Chưa cấu hình Roboflow client: {e}")

        # Khởi tạo YOLO detector
        self._detector = YOLODetector()
        if not self._detector.load_model():
            logger.error("Không thể tải model YOLO")
            return False

        # Nếu model có chứa lớp ghế (như custom model YOLO26: seat, person, item),
        # chia sẻ instance đã nạp cho SeatManager để tránh tốn RAM tải trùng model
        if self._detector.is_loaded and self._detector.model is not None:
            c_names = getattr(self._detector.model, "names", {})
            has_seat = any("seat" in str(c).lower() for c in c_names.values())
            if has_seat:
                self._seat_manager.set_shared_model(self._detector.model)
                logger.info("Đã tự động chia sẻ model instance cho SeatManager")

        # SeatManager sẵn sàng nhận diện ghế động từ camera feed
        logger.info("SeatManager đã sẵn sàng nhận diện ghế động từ camera feed")

        self._model_loaded = True
        logger.info("AnalysisService khởi tạo thành công toàn bộ hệ thống")
        return True


    @property
    def alert_service(self) -> Optional[AlertService]:
        return self._alert_service

    @property
    def dataset_service(self) -> Optional[DatasetService]:
        return self._dataset_service

    @property
    def model_service(self) -> Optional[ModelService]:
        return self._model_service

    @property
    def camera_manager(self) -> CameraStreamManager:
        return self._camera_manager

    @property
    def alert_repo(self) -> Optional[AlertRepository]:
        return self._alert_repo

    @property
    def model_repo(self) -> Optional[ModelRepository]:
        return self._model_repo

    @property
    def is_ready(self) -> bool:
        """Kiểm tra service đã sẵn sàng chưa."""
        return self._model_loaded and self._db is not None

    @property
    def seat_manager(self) -> SeatManager:
        """Trả về instance SeatManager."""
        return self._seat_manager

    @property
    def detector(self) -> Optional[YOLODetector]:
        """Trả về instance YOLODetector."""
        return self._detector

    def is_seat_configured(self) -> bool:
        """Kiểm tra xem vùng ghế đã được thiết lập chưa."""
        return self._seat_manager.is_configured

    def get_seats(self) -> List[SeatConfig]:
        """Lấy danh sách cấu hình ghế hiện tại."""
        return self._seat_manager.seats

    def save_seats(self, seats: List[SeatConfig]) -> bool:
        """Lưu danh sách ghế mới vào file và cập nhật bộ nhớ."""
        success = self._seat_config_mgr.save_seats(seats)
        if success:
            self._seat_manager.set_seats(seats)
        return success

    def load_preset_seats(self) -> List[SeatConfig]:
        """Tải mẫu 4 ghế gợi ý chuẩn hóa."""
        presets = SeatConfigManager.create_preset_seats()
        self.save_seats(presets)
        return presets

    def auto_detect_seats(
        self,
        frame: np.ndarray,
        allow_fallback: bool = False,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        save_to_disk: bool = True,
    ):
        """Tự động phát hiện và phân tách từng ghế trên xe buýt từ frame camera."""
        return self._seat_manager.auto_detect_and_configure_seats(
            frame=frame,
            allow_fallback=allow_fallback,
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
            save_to_disk=save_to_disk,
            roboflow_client=self.roboflow_client,
            inference_mode=self.inference_mode,
        )


    def get_seat_diagnostics(self) -> Dict:
        """Lấy thông tin chẩn đoán về model nhận diện ghế."""
        return self._seat_manager.get_model_diagnostics(
            inference_mode=self.inference_mode,
            roboflow_client=self.roboflow_client,
        )

    def get_model_info(self) -> Dict:
        """Trả về thông tin model để hiển thị."""
        if self._detector:
            return self._detector.get_model_info()
        return {"loaded": False, "model_name": "N/A"}

    def save_uploaded_video(self, uploaded_file) -> Optional[str]:
        """
        Lưu file video được upload từ Streamlit vào thư mục data/.

        Args:
            uploaded_file: UploadedFile từ st.file_uploader

        Returns:
            Đường dẫn file đã lưu, hoặc None nếu lỗi.
        """
        try:
            safe_name = sanitize_filename(uploaded_file.name)
            save_path = DATA_DIR / safe_name

            # Tránh ghi đè
            counter = 1
            stem = save_path.stem
            suffix = save_path.suffix
            while save_path.exists():
                save_path = DATA_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info(f"Lưu video upload: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Lỗi lưu video upload: {e}")
            return None

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Đọc metadata video mà không xử lý."""
        reader = VideoReader(video_path)
        meta = reader.open()
        reader.close()
        return meta

    def analyze_video(
        self,
        video_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_stationary_time: float = MIN_STATIONARY_TIME_SEC,
        movement_threshold: float = STATIONARY_MOVEMENT_THRESHOLD,
        watched_classes: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnalysisResult:
        """
        Chạy toàn bộ pipeline phân tích video.

        Args:
            video_path: Đường dẫn file video
            confidence_threshold: Ngưỡng confidence
            min_stationary_time: Thời gian đứng yên tối thiểu
            movement_threshold: Ngưỡng di chuyển
            watched_classes: Danh sách lớp đồ vật theo dõi
            progress_callback: Callback cập nhật tiến trình

        Returns:
            AnalysisResult
        """
        if not self.is_ready:
            return AnalysisResult(
                error_message="Service chưa được khởi tạo. Gọi initialize() trước."
            )

        # Validate file
        validation = validate_video_file(video_path)
        if not validation["valid"]:
            return AnalysisResult(error_message=validation["error"])

        # Tạo các component
        tracker = ObjectTracker()
        event_manager = EventManager()
        event_detector = ForgottenItemDetector(
            min_stationary_time=min_stationary_time,
            movement_threshold=movement_threshold,
            watched_classes=watched_classes or ITEM_CLASSES,
            seat_manager=self._seat_manager,
            event_manager=event_manager,
        )

        # Cập nhật confidence cho detector
        self._detector.confidence_threshold = confidence_threshold

        # Tạo processor
        processor = VideoProcessor(
            detector=self._detector,
            tracker=tracker,
            event_detector=event_detector,
            event_manager=event_manager,
            confidence_threshold=confidence_threshold,
            seat_manager=self._seat_manager,
        )

        # Chạy pipeline
        result = processor.process_video(
            video_path=video_path,
            progress_callback=progress_callback,
        )

        # Lưu vào database
        if result.success and self._repo:
            self._repo.save_analysis(result)
            self._repo.save_events(result.events, result.analysis_id)

        return result

    def get_dashboard_stats(self) -> Dict:
        """Lấy thống kê cho trang tổng quan."""
        if self._repo:
            return self._repo.get_dashboard_stats()
        return {}

    def get_events_by_status(self, status: str) -> List[Dict]:
        """Lấy sự kiện theo trạng thái cụ thể."""
        if self._repo:
            return self._repo.get_events_by_status(status)
        return []


    def get_all_events(self) -> List[Dict]:
        """Lấy tất cả sự kiện."""
        if self._repo:
            return self._repo.get_all_events()
        return []

    def get_all_analyses(self) -> List[Dict]:
        """Lấy lịch sử phân tích."""
        if self._repo:
            return self._repo.get_all_analyses()
        return []

    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Lấy sự kiện gần đây."""
        if self._repo:
            return self._repo.get_recent_events(limit)
        return []

    def update_event_status(self, event_id: str, new_status: str) -> bool:
        """Cập nhật trạng thái sự kiện trong database."""
        if self._repo:
            return self._repo.update_event_status(event_id, new_status)
        return False

    def process_webcam_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        tracker: ObjectTracker,
        event_detector: ForgottenItemDetector,
        event_manager: EventManager,
        seat_manager: Optional[SeatManager] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        show_seats: bool = True,
        show_persons: bool = True,
        show_objects: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        show_event_markers: bool = True,
        privacy_blur_faces: bool = False,
        show_masks: bool = True,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, List[Event], Dict]:
        """
        Xử lý 1 frame camera thời gian thực:
        - Chạy tracking & detection
        - Cập nhật trạng thái từng ghế
        - Phát hiện đồ bỏ quên & che khuất
        - Kích hoạt Alert Center & thông báo nếu nghi ngờ
        - Đo lường độ trễ suy luận (ms)
        - Chú thích màu sắc và số ghế trực tiếp lên frame
        """
        import time
        from src.video.video_writer import VideoWriter
        t_start = time.time()
        cur_h, cur_w = frame.shape[:2]
        target_ids = list(self._detector._target_class_ids) if (self._detector and getattr(self._detector, "_target_class_ids", None)) else None

        # -------- Multi-mode Inference handling --------
        local_detections = []
        remote_detections = []

        imgsz = kwargs.get("imgsz", None)

        if self.inference_mode in ("local", "hybrid"):
            if self._detector and hasattr(self._detector, "detect"):
                local_detections = self._detector.detect(frame, imgsz=imgsz)
            else:
                local_detections = None

        if self.inference_mode in ("roboflow", "hybrid") and self.roboflow_client:
            import cv2, tempfile, pathlib
            tmp_path = pathlib.Path(tempfile.mkstemp(suffix=".jpg")[1])
            cv2.imwrite(str(tmp_path), frame)
            payload = self.roboflow_client.submit_image(tmp_path)
            tmp_path.unlink(missing_ok=True)
            if payload:
                from src.roboflow.roboflow_client import RoboflowWorkflowClient
                parsed = RoboflowWorkflowClient.parse_workflow_output(payload)
                for pred in parsed.get("predictions", []):
                    cls = pred.get("class", "")
                    conf = float(pred.get("confidence", 0))
                    if conf < confidence_threshold:
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

        if self.inference_mode == "hybrid":
            from src.video.video_processor import fuse_hybrid_detections
            detections_to_track = fuse_hybrid_detections(local_detections or [], remote_detections)
        elif self.inference_mode == "roboflow":
            detections_to_track = remote_detections
        else:
            detections_to_track = local_detections

        if detections_to_track is not None:
            tracked = tracker.update(
                model=getattr(self._detector, "model", None) if self.inference_mode != "roboflow" else None,
                frame=frame,
                frame_index=frame_idx,
                fps=fps,
                confidence_threshold=confidence_threshold,
                target_class_ids=target_ids,
                detections=detections_to_track,
            )
        else:
            tracked = tracker.update(
                model=self._detector.model if self._detector else None,
                frame=frame,
                frame_index=frame_idx,
                fps=fps,
                confidence_threshold=confidence_threshold,
                target_class_ids=target_ids,
            )

        # Cập nhật nhận diện ghế thực tế từ AI model (AI Seat Detection từ camera feed)
        active_seat_mgr = seat_manager if seat_manager is not None else self._seat_manager
        if active_seat_mgr:
            active_seat_mgr.update_detected_seats(
                raw_predictions=(detections_to_track if detections_to_track is not None else tracked),
                frame_idx=frame_idx,
                frame_width=cur_w,
                frame_height=cur_h,
            )


        # Phân tích sự kiện
        current_time_sec = frame_idx / fps if fps > 0 else 0.0
        if active_seat_mgr and active_seat_mgr.is_configured:
            event_detector.process_frame(
                tracked_objects=tracked,
                current_frame=frame_idx,
                fps=fps,
                frame_width=cur_w,
                frame_height=cur_h,
                video_name="Live_Camera",
                frame=frame,
            )
        else:
            item_obs = tracker.get_item_observations()
            events_to_process = event_detector.analyze(
                item_observations=item_obs,
                current_frame=frame_idx,
                fps=fps,
            )
            for evt_info in events_to_process:
                event_manager.process_detection_result(
                    track_id=evt_info["track_id"],
                    new_status=evt_info["status"],
                    observation=evt_info["observation"],
                    reason=evt_info["reason"],
                    video_name="Live_Camera",
                    frame=frame,
                )

        # Trạng thái hiển thị
        event_statuses = {}
        for evt in event_manager.events:
            if evt.last_track_id != -1:
                event_statuses[evt.last_track_id] = evt.status
            elif evt.track_id != -1:
                event_statuses[evt.track_id] = evt.status

        # Lưu sự kiện vào DB và kích hoạt Alert Center
        if event_manager.events:
            for evt in event_manager.events:
                if evt.status in (EventStatus.SUSPECTED, EventStatus.OCCLUDED, EventStatus.REVIEW_REQUIRED):
                    if self._repo:
                        self._repo.save_event(evt, analysis_id="live_stream")
                    if self._alert_service and evt.status in (EventStatus.SUSPECTED, EventStatus.REVIEW_REQUIRED):
                        self._alert_service.process_event_for_alert(
                            event=evt,
                            current_time_sec=current_time_sec,
                            telegram_config=telegram_config,
                            email_config=email_config,
                            webhook_config=webhook_config,
                        )

        # Đo lường độ trễ suy luận
        latency_ms = (time.time() - t_start) * 1000.0
        self._camera_manager.update_metrics(
            latency_ms=latency_ms,
            tracked_count=len(tracked),
            pending_events_count=len(event_manager.events),
        )

        # Vẽ chú thích trực tiếp lên frame với các tùy chọn lớp hiển thị
        annotated = VideoWriter.annotate_frame(
            frame=frame,
            tracked_objects=tracked,
            event_statuses=event_statuses,
            seat_manager=active_seat_mgr,
            active_events=event_manager.events,
            width=cur_w,
            height=cur_h,
            show_seats=show_seats,
            show_persons=show_persons,
            show_objects=show_objects,
            show_track_ids=show_track_ids,
            show_confidence=show_confidence,
            show_event_markers=show_event_markers,
            privacy_blur_faces=privacy_blur_faces,
            show_masks=show_masks,
        )

        summary = event_manager.get_summary()
        return annotated, event_manager.events, summary

    # ──────────────────────────────────────────────
    # Alert & Notification Delegators
    def get_recent_alerts(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        active_only: bool = False,
        *args,
        **kwargs,
    ) -> List[Dict]:
        """Lấy danh sách các Alert gần nhất."""
        if self._alert_repo:
            return self._alert_repo.get_recent_alerts(
                limit=limit,
                status=status,
                active_only=active_only,
                *args,
                **kwargs,
            )
        return []

    def update_alert_status(
        self,
        alert_id: str,
        new_status: str,
        resolution_type: Optional[str] = None,
        resolved_at: Optional[str] = None,
        *args,
        **kwargs,
    ) -> bool:
        """Cập nhật trạng thái alert (PENDING, CONFIRMED, DISMISSED, ITEM_RETRIEVED, FALSE_POSITIVE)."""
        if self._alert_repo:
            return self._alert_repo.update_alert_status(
                alert_id,
                new_status,
                resolution_type=resolution_type,
                resolved_at=resolved_at,
                *args,
                **kwargs,
            )
        return False

    def delete_alert_permanently(self, alert_id: str, *args, **kwargs) -> bool:
        """Xóa vĩnh viễn Alert khỏi database."""
        if self._alert_repo:
            return self._alert_repo.delete_alert_permanently(alert_id, *args, **kwargs)
        return False

    def get_unresolved_alert_count(self) -> int:
        """Đếm số cảnh báo đang chờ xử lý."""
        if self._alert_repo:
            return self._alert_repo.get_unresolved_alert_count()
        return 0

    def get_notification_logs(self, limit: int = 50) -> List[Dict]:
        """Lấy lịch sử gửi thông báo."""
        if self._alert_repo:
            return self._alert_repo.get_notification_logs(limit=limit)
        return []

    def send_test_alert(
        self,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
        seat_id: str = "S01",
    ) -> Optional[Alert]:
        """Bắn thử 1 cảnh báo test qua các kênh đã cấu hình."""
        if self._alert_service:
            return self._alert_service.send_test_alert(
                telegram_config=telegram_config,
                email_config=email_config,
                webhook_config=webhook_config,
                seat_id=seat_id,
            )
        return None

    def get_telegram_delivery_status(self) -> Dict[str, Any]:
        """Lấy trạng thái kết quả phân phối thông báo Telegram gần nhất."""
        try:
            notifier = get_telegram_notifier()
            if hasattr(notifier, "get_last_delivery_status"):
                return notifier.get_last_delivery_status()
        except Exception as e:
            logger.warning(f"Lỗi khi lấy trạng thái phân phối Telegram: {e}")
        return {
            "has_attempt": False,
            "success": False,
            "status_code": 0,
            "message": "",
            "timestamp": "",
            "chat_id": getattr(notifier, "chat_id", "") if "notifier" in locals() and notifier else "",
            "alert_id": "",
        }

    def test_telegram_connection(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Kiểm tra kết nối và gửi tin nhắn test tới kênh Telegram đã cấu hình."""
        return get_telegram_notifier().test_connection(bot_token=bot_token, chat_id=chat_id)

    def process_browser_camera_events(
        self,
        suspected_items: List[Dict[str, Any]],
        snapshot_base64: Optional[str] = None,
        camera_name: str = "Live Camera (Browser)",
        bus_license_plate: Optional[str] = None,
        bus_route: Optional[str] = None,
    ) -> List[Alert]:
        """
        Cầu nối sự kiện: Tiếp nhận danh sách đồ vật nghi bị bỏ quên từ Browser-Native Camera (WASM/WebGPU),
        đồng bộ hóa sang mô hình Event của backend, ghi nhận Database và kích hoạt Alert Service gửi Telegram.
        """
        if not suspected_items:
            return []

        # Tự động khởi tạo nếu chưa sẵn sàng
        if not self._alert_service or not self._repo:
            logger.info("[Browser Camera] Khởi tạo các repository và dịch vụ phân tích...")
            self.initialize()

        if not self._alert_service:
            logger.error("[Browser Camera] Lỗi: Không thể khởi tạo AlertService để tiếp nhận sự kiện")
            return []

        import base64
        import re
        import time
        from config.settings import SNAPSHOTS_DIR, get_current_time, EventStatus
        from src.detection.class_mapper import get_user_facing_label
        from src.models import Event
        from src.utils.bus_info import get_bus_info

        # Đồng bộ thông tin biển số và tuyến xe chuẩn
        def_plate, def_route = get_bus_info(auto_generate_if_missing=True)
        active_plate = (bus_license_plate or def_plate or "").strip()
        active_route = (bus_route or def_route or "Tuyến số 1").strip()

        logger.info(
            f"[Browser Camera] Nhận được {len(suspected_items)} đồ vật nghi bị bỏ quên từ camera '{camera_name}' (Xe: {active_plate} - {active_route})"
        )

        def _safe_decode_and_save_snapshot(b64_str: Optional[str], track_ident: Any) -> Optional[str]:
            """Giải mã an toàn snapshot base64 và lưu vào thư mục SNAPSHOTS_DIR."""
            if not b64_str or not isinstance(b64_str, str):
                return None
            try:
                clean_b64 = b64_str
                if "," in clean_b64:
                    clean_b64 = clean_b64.split(",", 1)[1]
                clean_b64 = clean_b64.strip()
                if not clean_b64:
                    return None
                # Giới hạn kích thước tối đa 10MB
                if len(clean_b64) > 14 * 1024 * 1024:
                    logger.warning(f"[Browser Camera] Snapshot vượt quá giới hạn 10MB cho track {track_ident}")
                    return None
                img_bytes = base64.b64decode(clean_b64, validate=True)
                if len(img_bytes) < 16:
                    logger.warning(f"[Browser Camera] Snapshot quá nhỏ ({len(img_bytes)} bytes) cho track {track_ident}")
                    return None
                # Kiểm tra magic bytes
                is_jpeg = img_bytes.startswith(b"\xff\xd8\xff")
                is_png = img_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                is_webp = img_bytes.startswith(b"RIFF") and b"WEBP" in img_bytes[:16]
                if not (is_jpeg or is_png or is_webp):
                    logger.warning(f"[Browser Camera] Định dạng ảnh không hợp lệ (không phải JPEG/PNG/WebP) cho track {track_ident}")
                    return None

                ext = ".jpg" if is_jpeg else (".png" if is_png else ".webp")
                ts_now = int(time.time() * 1000)
                SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                snap_file = SNAPSHOTS_DIR / f"browser_cam_{ts_now}_{track_ident}{ext}"
                with open(snap_file, "wb") as f:
                    f.write(img_bytes)
                return str(snap_file)
            except Exception as e:
                logger.warning(f"[Browser Camera] Lỗi giải mã snapshot cho track {track_ident}: {e}")
                return None

        # Giải mã ảnh chụp khung hình chung nếu có
        global_saved_snapshot = _safe_decode_and_save_snapshot(snapshot_base64, "global")

        alerts = []
        now_ts = time.time()
        now_iso = get_current_time().isoformat()

        for item in suspected_items:
            track_id = item.get("track_id", "") or item.get("display_id", "")
            seat_id = (item.get("seat_id") or "").strip()
            class_name = item.get("class_name") or "item"
            confidence = float(item.get("confidence", 0.8))
            status_str = str(item.get("status") or "SUSPECTED").upper()
            vacant_dur = float(item.get("vacant_duration", 5.0))

            seat_suffix = seat_id if seat_id and seat_id.lower() not in ("chưa rõ", "none", "unknown_seat") else "NOSEAT"
            evt_id = f"EVT_BROWSER_{track_id}_{seat_suffix}"

            try:
                numeric_track_id = int(re.sub(r"\D", "", str(track_id))) if any(c.isdigit() for c in str(track_id)) else -1
            except Exception:
                numeric_track_id = -1

            seat_display = seat_id if seat_suffix != "NOSEAT" else "Chưa rõ"
            vi_name = get_user_facing_label(class_name)
            reason = f"Phát hiện vật thể {vi_name} bị bỏ lại trên ghế trống {seat_display}"

            # Ưu tiên lấy snapshot gắn trực tiếp với chính item đó
            item_b64 = item.get("snapshot_base64")
            item_snapshot_path = None
            if item_b64:
                item_snapshot_path = _safe_decode_and_save_snapshot(item_b64, track_id)

            chosen_snapshot = item_snapshot_path or global_saved_snapshot or ""

            logger.info(
                f"[Browser Camera] Xử lý item: track_id={track_id}, seat={seat_display}, "
                f"class={class_name}, conf={confidence:.2f}, vacant_duration={vacant_dur:.1f}s, "
                f"has_snapshot={bool(chosen_snapshot)}"
            )

            evt = Event(
                event_id=evt_id,
                track_id=numeric_track_id,
                last_track_id=numeric_track_id,
                seat_id=seat_display,
                class_name=class_name,
                status=EventStatus.SUSPECTED if status_str == "SUSPECTED" else EventStatus.REVIEW_REQUIRED,
                confidence=confidence,
                notes=reason,
                snapshot_path=chosen_snapshot,
                timestamp=now_iso,
                vacant_duration=vacant_dur,
                video_name=camera_name,
            )
            setattr(evt, "reason", reason)

            if self._repo:
                try:
                    self._repo.save_event(evt, analysis_id="browser_live_camera")
                    logger.info(f"[Browser Camera] Đã lưu Event {evt_id} vào database")
                except Exception as ex:
                    logger.warning(f"[Browser Camera] Lỗi khi lưu Event {evt_id}: {ex}")

            tg_cfg = {
                "bus_license_plate": active_plate,
                "bus_route": active_route,
            }

            alert = self._alert_service.process_event_for_alert(
                event=evt,
                current_time_sec=now_ts,
                telegram_config=tg_cfg,
            )
            if alert:
                logger.info(
                    f"[Browser Camera] Đã tạo thành công Alert {alert.alert_id} cho Event {evt_id} (Priority: {alert.priority})"
                )
                alerts.append(alert)
            else:
                logger.info(
                    f"[Browser Camera] Không tạo Alert mới cho item {track_id} (Đã có cảnh báo tồn tại hoặc trong cooldown)"
                )

        return alerts

    # ──────────────────────────────────────────────
    # Dataset Delegators
    # ──────────────────────────────────────────────
    def get_dataset_images(self, split: Optional[str] = None, labeled_only: Optional[bool] = None) -> List[Dict]:
        """Lấy danh sách ảnh trong dataset."""
        if self._dataset_service:
            return self._dataset_service.get_images(split=split, labeled_only=labeled_only)
        return []

    def get_dataset_stats(self) -> Dict:
        """Lấy thống kê dataset."""
        if self._dataset_service:
            return self._dataset_service.get_stats()
        return {"total_images": 0, "labeled_images": 0, "splits": {}, "class_counts": {}}

    def add_dataset_image(self, file_bytes: bytes, filename: str, split: str = "train") -> Optional[str]:
        """Thêm ảnh vào dataset."""
        if self._dataset_service:
            return self._dataset_service.add_image_from_bytes(file_bytes, filename, split=split)
        return None

    def extract_frames_from_video(self, video_path: str, interval_seconds: float = 2.0) -> List[str]:
        """Trích xuất frame từ video để làm dataset."""
        if self._dataset_service:
            return self._dataset_service.extract_frames_from_video(video_path, interval_seconds=interval_seconds)
        return []

    def save_image_annotations(self, image_id: str, annotations: List[Dict]) -> bool:
        """Lưu nhãn cho ảnh."""
        if self._dataset_service:
            return self._dataset_service.save_annotations(image_id, annotations)
        return False

    def get_image_annotations(self, image_id: str) -> List[Dict]:
        """Lấy nhãn của ảnh."""
        if self._dataset_service:
            return self._dataset_service.get_annotations(image_id)
        return []

    def delete_dataset_image(self, image_id: str) -> bool:
        """Xóa ảnh dataset."""
        if self._dataset_service:
            return self._dataset_service.delete_image(image_id)
        return False

    def export_yolo_dataset(self) -> Tuple[bool, str, str]:
        """Xuất dataset ra file zip chuẩn YOLO."""
        if self._dataset_service:
            return self._dataset_service.export_yolo_dataset()
        return False, "", "DatasetService chưa sẵn sàng"

    # ──────────────────────────────────────────────
    # Model Delegators
    # ──────────────────────────────────────────────
    def get_all_models(self) -> List[Dict]:
        """Lấy tất cả model versions."""
        if self._model_service:
            return self._model_service.get_all_models()
        return []

    def get_active_model(self) -> Optional[Dict]:
        """Lấy model active."""
        if self._model_service:
            return self._model_service.get_active_model()
        return None

    def activate_model(self, model_id: str) -> Tuple[bool, str]:
        """Kích hoạt model."""
        if self._model_service:
            return self._model_service.activate_model(model_id)
        return False, "ModelService chưa sẵn sàng"

    @property
    def model_service(self) -> Optional[ModelService]:
        return self._model_service

    def get_training_hardware_info(self) -> Dict:
        """Kiểm tra thông tin phần cứng khả dụng cho huấn luyện."""
        if self._model_service:
            return self._model_service.get_training_hardware_info()
        return {"device": "cpu", "device_name": "CPU", "description": "ModelService chưa sẵn sàng"}

    def validate_training_dataset(self, yaml_path: str) -> Tuple[bool, str, Dict]:
        """Kiểm tra tính hợp lệ của dataset trước khi huấn luyện."""
        if self._model_service:
            return self._model_service.validate_training_dataset(yaml_path)
        return False, "ModelService chưa sẵn sàng", {}

    def generate_training_command(
        self,
        dataset_yaml_path: str,
        epochs: int = 30,
        batch_size: int = 16,
        imgsz: int = 640,
        model: str = "yolov8n.pt",
        task: str = "detect",
    ) -> str:
        """Sinh lệnh CLI Ultralytics cho huấn luyện."""
        if self._model_service:
            return self._model_service.generate_training_command(
                dataset_yaml_path=dataset_yaml_path,
                base_model=model,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                task=task,
            )
        return ""

    def start_real_training_job(
        self,
        dataset_yaml_path: str,
        new_version: str,
        base_model: str = "yolov8n.pt",
        epochs: int = 30,
        batch_size: int = 16,
        imgsz: int = 640,
        task: str = "detect",
    ) -> Tuple[bool, str, Dict]:
        """Khởi chạy huấn luyện thật bằng Ultralytics trong tiến trình riêng."""
        if self._model_service:
            return self._model_service.start_real_training_job(
                dataset_yaml_path=dataset_yaml_path,
                new_version=new_version,
                base_model=base_model,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                task=task,
            )
        return False, "ModelService chưa sẵn sàng", {}

    def get_training_job_status(self, job_id: str) -> Dict:
        """Kiểm tra tiến độ và đọc metrics thật của job huấn luyện."""
        if self._model_service:
            return self._model_service.get_training_job_status(job_id)
        return {"status": "NOT_FOUND", "message": "ModelService chưa sẵn sàng"}

    def approve_and_register_completed_job(
        self, job_id: str, dataset_version: str = "v1.0", notes: str = ""
    ) -> Tuple[bool, str]:
        """Phê duyệt và lưu candidate model với metrics thực tế."""
        if self._model_service:
            return self._model_service.approve_and_register_completed_job(
                job_id=job_id, dataset_version=dataset_version, notes=notes
            )
        return False, "ModelService chưa sẵn sàng"

    def simulate_training_job(self, new_version: str, dataset_version: str, epochs: int = 30) -> Tuple[bool, Dict, str]:
        """(Vô hiệu hóa) Phương thức giả lập đã bị loại bỏ."""
        if self._model_service:
            return self._model_service.simulate_training_job(new_version, dataset_version, epochs=epochs)
        return False, {}, "ModelService chưa sẵn sàng"

    def cleanup(self):
        """Giải phóng tài nguyên."""
        if self._camera_manager:
            self._camera_manager.stop_stream()
        if self._db:
            self._db.close()

