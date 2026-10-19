"""
src/services/analysis_service.py
Lớp kết nối giao diện (app.py) với pipeline xử lý bên trong.
app.py không cần biết chi tiết về YOLO, tracker, database, v.v.
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
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
)

from src.seats.seat_config import SeatConfigManager
from src.seats.seat_manager import SeatManager
from src.models import AnalysisResult, VideoMetadata, SeatConfig, SeatStatus, Event, Alert
from src.database.alert_repository import AlertRepository
from src.database.dataset_repository import DatasetRepository
from src.database.model_repository import ModelRepository
from src.services.alert_service import AlertService
from src.services.notification_service import NotificationService
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService
from src.video.camera_service import CameraStreamManager


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

        # Khởi tạo YOLO detector
        self._detector = YOLODetector()
        if not self._detector.load_model():
            logger.error("Không thể tải model YOLO")
            return False

        # Tải cấu hình vùng ghế nếu có
        saved_seats = self._seat_config_mgr.load_seats()
        if saved_seats:
            self._seat_manager.set_seats(saved_seats)
            logger.info(f"Đã tải {len(saved_seats)} ghế từ cấu hình")
        else:
            logger.warning("Chưa có cấu hình vùng ghế. Người dùng cần thiết lập Seat ROI.")

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
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        save_to_disk: bool = True,
    ):
        """Tự động phát hiện và phân tách từng ghế trên xe buýt từ frame camera."""
        return self._seat_manager.auto_detect_and_configure_seats(
            frame=frame,
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
            save_to_disk=save_to_disk,
        )

    def get_seat_diagnostics(self) -> Dict:
        """Lấy thông tin chẩn đoán về model nhận diện ghế."""
        return self._seat_manager.get_model_diagnostics()

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
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        show_seats: bool = True,
        show_persons: bool = True,
        show_objects: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        show_event_markers: bool = True,
        privacy_blur_faces: bool = False,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
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
        target_ids = list(self._detector._target_class_ids) if (self._detector and self._detector._target_class_ids) else None

        tracked = tracker.update(
            model=self._detector.model if self._detector else None,
            frame=frame,
            frame_index=frame_idx,
            fps=fps,
            confidence_threshold=confidence_threshold,
            target_class_ids=target_ids,
        )

        # Phân tích sự kiện
        current_time_sec = frame_idx / fps if fps > 0 else 0.0
        if self._seat_manager and self._seat_manager.is_configured:
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
            seat_manager=self._seat_manager,
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
        )

        summary = event_manager.get_summary()
        return annotated, event_manager.events, summary

    # ──────────────────────────────────────────────
    # Alert & Notification Delegators
    # ──────────────────────────────────────────────
    def get_recent_alerts(self, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """Lấy danh sách các Alert gần nhất."""
        if self._alert_repo:
            return self._alert_repo.get_recent_alerts(limit=limit, status=status)
        return []

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Cập nhật trạng thái alert (PENDING, CONFIRMED, DISMISSED, RESOLVED)."""
        if self._alert_repo:
            return self._alert_repo.update_alert_status(alert_id, new_status)
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

    def simulate_training_job(self, new_version: str, dataset_version: str, epochs: int = 30) -> Tuple[bool, Dict, str]:
        """Huấn luyện & tạo candidate model."""
        if self._model_service:
            return self._model_service.simulate_training_job(new_version, dataset_version, epochs=epochs)
        return False, {}, "ModelService chưa sẵn sàng"

    def cleanup(self):
        """Giải phóng tài nguyên."""
        if self._camera_manager:
            self._camera_manager.stop_stream()
        if self._db:
            self._db.close()

