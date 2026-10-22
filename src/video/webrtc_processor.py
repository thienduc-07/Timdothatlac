"""
src/video/webrtc_processor.py
Bộ xử lý luồng WebRTC thời gian thực cho Streamlit Cloud & Client Browser:
- Truyền trực tiếp webcam từ trình duyệt người dùng tới máy chủ thông qua WebRTC (Zero-Click Client Camera).
- Khắc phục hoàn toàn hạn chế của cv2.VideoCapture() trên Streamlit Cloud (Linux Container không có camera vật lý).
- Tách biệt phiên hoàn toàn (Multi-User Session Isolation): Mỗi kết nối WebRTC có tracker, event detector và seat manager độc lập.
- Tương thích 100% với pipeline YOLO26, ByteTrack, ForgottenItemDetector và VideoWriter.annotate_frame hiện có.
"""

from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.models import Event, TrackedObject, SeatState
from src.tracking.object_tracker import ObjectTracker
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.seats.seat_manager import SeatManager
from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ITEM_CLASSES,
    STATIONARY_MOVEMENT_THRESHOLD,
    DATA_DIR,
)

logger = get_logger(__name__)

# Kiểm tra thư viện streamlit-webrtc và av
try:
    import av
    from streamlit_webrtc import (
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError as err:
    logger.warning(f"streamlit-webrtc hoặc av chưa được cài đặt: {err}")
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # Fallback dummy class
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None

# Cấu hình STUN servers công khai đáng tin cậy của Google để xuyên NAT/Firewall trên Streamlit Cloud
if WEBRTC_AVAILABLE and RTCConfiguration is not None:
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        }
    )
else:
    RTC_CONFIGURATION = None


class BusWebRtcVideoProcessor(VideoProcessorBase):
    """
    Video processor chạy trên worker thread của aiortc/webrtc cho từng phiên client.
    
    Đặc điểm an toàn & hiệu năng:
    1. Độc lập phiên (Per-Session Isolation): Mỗi người dùng mở Live Camera sẽ sở hữu
       tracker, event_detector, event_manager và seat_manager riêng biệt.
    2. Không nghẽn luồng Streamlit: Xử lý frame bất đồng bộ trong recv(), trả về frame có chú thích.
    3. Thread-Safe Metrics: Streamlit UI có thể đọc dict `latest_metrics` bất kỳ lúc nào để hiển thị HUD.
    4. Bền bỉ (Resilient): Nếu frame inference gặp lỗi, vẫn trả về frame gốc an toàn mà không làm gián đoạn WebRTC.
    """

    def __init__(
        self,
        service: Any = None,
        conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_stat_time: float = 3.0,
        show_seats: bool = True,
        show_persons: bool = True,
        show_objects: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        show_event_markers: bool = True,
        privacy_blur: bool = False,
        show_masks: bool = True,
        stride: int = 1,
    ):
        self.service = service
        self.conf_thresh = conf_thresh
        self.min_stat_time = min_stat_time
        self.show_seats = show_seats
        self.show_persons = show_persons
        self.show_objects = show_objects
        self.show_track_ids = show_track_ids
        self.show_confidence = show_confidence
        self.show_event_markers = show_event_markers
        self.privacy_blur = privacy_blur
        self.show_masks = show_masks
        self.stride = max(1, stride)

        # ── Khởi tạo Pipeline AI độc lập cho phiên WebRTC này ──
        self.tracker = ObjectTracker()
        self.event_manager = EventManager()
        
        # Khởi tạo SeatManager riêng cho phiên để tránh xung đột ghế giữa các client
        self.seat_manager = SeatManager()
        if service and getattr(service, "seat_manager", None):
            base_sm = service.seat_manager
            if base_sm.detector:
                self.seat_manager.detector = base_sm.detector
                self.seat_manager.active_detection_method = base_sm.active_detection_method
            if base_sm.seats:
                self.seat_manager.set_seats(list(base_sm.seats))

        self.event_detector = ForgottenItemDetector(
            min_stationary_time=min_stat_time,
            movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
            watched_classes=ITEM_CLASSES,
            seat_manager=self.seat_manager,
            event_manager=self.event_manager,
        )

        self.frame_idx = 0
        self.last_annotated: Optional[np.ndarray] = None
        self.last_events: List[Event] = []
        self.last_summary: Dict[str, Any] = {}
        self.latest_frame_bgr: Optional[np.ndarray] = None
        self.latest_annotated_bgr: Optional[np.ndarray] = None

        self._lock = threading.Lock()
        self._last_time = time.time()
        self.latest_metrics: Dict[str, Any] = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "total_seats": 0,
            "occ_seats": 0,
            "vac_seats": 0,
            "person_count": 0,
            "item_count": 0,
            "sus_count": 0,
            "status_tag": "🟢 KHỞI TẠO",
            "status_color": "#10B981",
            "frame_idx": 0,
            "last_updated": time.time(),
        }

    def update_config(self, **kwargs):
        """Cập nhật tùy chọn hiển thị và ngưỡng tin cậy từ giao diện người dùng."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def get_metrics(self) -> Dict[str, Any]:
        """Lấy bản sao metrics mới nhất cho giao diện Streamlit."""
        with self._lock:
            return dict(self.latest_metrics)

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Lấy frame có chú thích mới nhất để lưu bằng chứng."""
        with self._lock:
            if self.latest_annotated_bgr is not None:
                return self.latest_annotated_bgr.copy()
            if self.latest_frame_bgr is not None:
                return self.latest_frame_bgr.copy()
            return None

    def recv(self, frame: Any) -> Any:
        """
        Nhận frame từ WebRTC client, đưa qua pipeline AI và trả về frame có chú thích.
        Hàm này chạy trên worker thread của aiortc.
        """
        if not WEBRTC_AVAILABLE or frame is None:
            return frame

        t0 = time.time()
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as e:
            logger.error(f"Lỗi giải mã frame WebRTC sang bgr24: {e}")
            return frame

        if img is None:
            return frame

        with self._lock:
            cur_idx = self.frame_idx
            self.frame_idx += 1
            self.latest_frame_bgr = img

            should_infer = (cur_idx % self.stride == 0) or (self.last_annotated is None)

            if should_infer and self.service is not None:
                try:
                    annotated, events, summary = self.service.process_webcam_frame(
                        frame=img,
                        frame_idx=cur_idx,
                        fps=25.0,
                        tracker=self.tracker,
                        event_detector=self.event_detector,
                        event_manager=self.event_manager,
                        seat_manager=self.seat_manager,
                        confidence_threshold=self.conf_thresh,
                        show_seats=self.show_seats,
                        show_persons=self.show_persons,
                        show_objects=self.show_objects,
                        show_track_ids=self.show_track_ids,
                        show_confidence=self.show_confidence,
                        show_event_markers=self.show_event_markers,
                        privacy_blur_faces=self.privacy_blur,
                        show_masks=self.show_masks,
                    )
                    self.last_annotated = annotated
                    self.last_events = events
                    self.last_summary = summary
                except Exception as e:
                    logger.error(f"Lỗi suy luận Live AI tại frame {cur_idx}: {e}")
                    annotated = self.last_annotated if self.last_annotated is not None else img
                    events = self.last_events
                    summary = self.last_summary
            else:
                annotated = self.last_annotated if self.last_annotated is not None else img
                events = self.last_events
                summary = self.last_summary

            self.latest_annotated_bgr = annotated

            # Tính toán các chỉ số thống kê thời gian thực
            latency_ms = (time.time() - t0) * 1000.0
            now = time.time()
            dt = max(0.001, (now - self._last_time))
            fps = 1.0 / dt
            self._last_time = now

            total_seats = len(self.seat_manager.seats)
            statuses = self.seat_manager.get_all_statuses()
            occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
            vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)
            person_count = sum(1 for obj in self.tracker.current_tracked_objects if obj.class_name == "person")
            item_count = sum(1 for obj in self.tracker.current_tracked_objects if obj.class_name in ITEM_CLASSES or obj.class_name == "item")
            sus_count = summary.get("suspected", 0)
            status_tag = "🔴 PHÁT HIỆN ĐỒ BỎ QUÊN!" if sus_count > 0 else "🟢 BÌNH THƯỜNG"
            status_color = "#EF4444" if sus_count > 0 else "#10B981"

            self.latest_metrics = {
                "fps": fps,
                "latency_ms": latency_ms,
                "total_seats": total_seats,
                "occ_seats": occ_seats,
                "vac_seats": vac_seats,
                "person_count": person_count,
                "item_count": item_count,
                "sus_count": sus_count,
                "status_tag": status_tag,
                "status_color": status_color,
                "frame_idx": cur_idx,
                "last_updated": now,
            }

        try:
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as e:
            logger.error(f"Lỗi đóng gói frame bgr24 sang VideoFrame: {e}")
            return frame
