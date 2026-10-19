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
from src.models import AnalysisResult, VideoMetadata, SeatConfig, SeatStatus, Event


logger = get_logger(__name__)


class AnalysisService:
    """
    Dịch vụ phân tích video — lớp trung gian giữa giao diện và pipeline.
    Hỗ trợ quản lý vùng ghế (Seat ROI) và phân tích tình huống che khuất.
    """

    def __init__(self):
        self._detector: Optional[YOLODetector] = None
        self._db: Optional[Database] = None
        self._repo: Optional[EventRepository] = None
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
        self._repo = EventRepository(self._db.get_connection())

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
        logger.info("AnalysisService khởi tạo thành công")
        return True

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
    ) -> Tuple[np.ndarray, List[Event], Dict]:
        """
        Xử lý 1 frame webcam thời gian thực:
        - Chạy tracking & detection
        - Cập nhật trạng thái từng ghế
        - Phát hiện đồ bỏ quên & che khuất
        - Chú thích vùng ghế và số ghế trực tiếp lên frame
        """
        from src.video.video_writer import VideoWriter

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
        if self._seat_manager and self._seat_manager.is_configured:
            event_detector.process_frame(
                tracked_objects=tracked,
                current_frame=frame_idx,
                fps=fps,
                frame_width=cur_w,
                frame_height=cur_h,
                video_name="Live_Webcam",
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
                    video_name="Live_Webcam",
                    frame=frame,
                )

        # Trạng thái hiển thị
        event_statuses = {}
        for evt in event_manager.events:
            if evt.last_track_id != -1:
                event_statuses[evt.last_track_id] = evt.status
            elif evt.track_id != -1:
                event_statuses[evt.track_id] = evt.status

        # Lưu sự kiện vào DB định kỳ hoặc khi có sự kiện quan trọng
        if self._repo and event_manager.events:
            for evt in event_manager.events:
                if evt.status in (EventStatus.SUSPECTED, EventStatus.OCCLUDED, EventStatus.REVIEW_REQUIRED):
                    self._repo.save_event(evt, analysis_id="live_webcam")

        # Vẽ chú thích trực tiếp lên frame
        annotated = VideoWriter.annotate_frame(
            frame=frame,
            tracked_objects=tracked,
            event_statuses=event_statuses,
            seat_manager=self._seat_manager,
            active_events=event_manager.events,
            width=cur_w,
            height=cur_h,
        )

        summary = event_manager.get_summary()
        return annotated, event_manager.events, summary

    def cleanup(self):
        """Giải phóng tài nguyên."""
        if self._db:
            self._db.close()

