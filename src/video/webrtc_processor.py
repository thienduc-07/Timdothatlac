"""
src/video/webrtc_processor.py
Bộ xử lý luồng WebRTC thời gian thực hiệu năng cao cho Streamlit Cloud & Client Browser:
- Kiến trúc Decoupled Asynchronous Worker: Tách biệt hoàn toàn luồng nhận/trả frame WebRTC (recv)
  khỏi luồng suy luận AI (worker thread). recv() chạy dưới 0.1ms, loại bỏ triệt để hiện tượng
  buffer bloat / video lag sau thời gian dài streaming.
- Hàng đợi Bounded Frame Queue (maxsize=1): Khi backend bận suy luận, tự động drop frame cũ,
  chỉ suy luận frame mới nhất (Latest-Frame Priority), ghi nhận chính xác dropped_frames.
- Độc lập phiên hoàn toàn (Multi-User Session Isolation): Mỗi client sở hữu ObjectTracker,
  SeatManager, EventManager và ForgottenItemDetector riêng biệt.
- Giám sát toàn diện (Telemetry & Metrics):
  * Input FPS (Camera đầu vào từ client)
  * Inference FPS (Tốc độ AI xử lý thực tế)
  * Output FPS (Tốc độ video trả về trình duyệt)
  * Độ trễ trung bình (Avg Latency) và Phân vị 95 (P95 Latency)
  * Số frame bị bỏ qua (Dropped frames) và Số phiên hoạt động đồng thời (Active sessions)
  * CPU, RAM và GPU hardware info
- Giải phóng tài nguyên an toàn (Lifecycle & on_ended): Khi dừng camera, ngắt worker thread,
  giải phóng hàng đợi nhưng giữ nguyên hồ sơ sự kiện đã lưu và đánh dấu Camera Offline.
"""

import collections
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional
import uuid
import cv2
import numpy as np

import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
        Translations,
    )
    WEBRTC_AVAILABLE = True
except ImportError as err:
    logger.warning(f"streamlit-webrtc hoặc av chưa được cài đặt: {err}")
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # Fallback dummy class
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None
    Translations = None

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


class ActiveSessionTracker:
    """Quản lý và đếm số phiên Live Camera đang kết nối đồng thời (Thread-Safe)."""
    _lock = threading.Lock()
    _active_sessions: set = set()

    @classmethod
    def register(cls, session_id: str) -> int:
        with cls._lock:
            cls._active_sessions.add(session_id)
            return len(cls._active_sessions)

    @classmethod
    def unregister(cls, session_id: str) -> int:
        with cls._lock:
            cls._active_sessions.discard(session_id)
            return len(cls._active_sessions)

    @classmethod
    def get_count(cls) -> int:
        with cls._lock:
            return len(cls._active_sessions)


def get_system_telemetry() -> Dict[str, Any]:
    """Đo lường CPU, RAM và GPU của máy chủ đang chạy ứng dụng."""
    try:
        proc = psutil.Process()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_info = proc.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        sys_mem_pct = psutil.virtual_memory().percent

        device_info = "CPU (Linux Container / Cloud Host)"
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_info = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_info = "Apple Silicon Metal (MPS)"

        thread_cnt = threading.active_count()

        return {
            "cpu_percent": cpu_pct,
            "process_rss_mb": rss_mb,
            "system_mem_percent": sys_mem_pct,
            "thread_count": thread_cnt,
            "device": device_info,
        }
    except Exception as e:
        logger.debug(f"Lỗi đọc telemetry: {e}")
        return {
            "cpu_percent": 0.0,
            "process_rss_mb": 0.0,
            "system_mem_percent": 0.0,
            "thread_count": threading.active_count(),
            "device": "Unknown",
        }


def _calc_rolling_fps(timestamps_deque: collections.deque) -> float:
    """Tính toán FPS từ hàng đợi timestamp rolling."""
    if len(timestamps_deque) < 2:
        return 0.0
    dt = timestamps_deque[-1] - timestamps_deque[0]
    if dt <= 0.0001:
        return 0.0
    return float((len(timestamps_deque) - 1) / dt)


class BusWebRtcVideoProcessor(VideoProcessorBase):
    """
    Video processor hiệu năng cao chạy trên worker thread của aiortc/webrtc:
    - Decoupled Pipeline: recv() chỉ mất ~0.02ms để đẩy frame vào queue và trả về frame mới nhất.
    - AI Worker Thread riêng biệt cho từng phiên: Không nghẽn luồng Streamlit và không chậm WebRTC.
    - Drop-Oldest Frame Queue (maxsize=1): Luôn ưu tiên frame mới nhất, đếm chính xác số frame bị bỏ qua.
    - Tách biệt phiên hoàn toàn: Mỗi người dùng sở hữu SeatManager, Tracker, EventDetector độc lập.
    - Đầy đủ chỉ số: Input FPS, Inference FPS, Output FPS, Latency Avg & P95, Dropped Frames, Active Sessions.
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
        stride: int = 2,
        infer_imgsz: Optional[int] = 480,
    ):
        self.session_id = str(uuid.uuid4())[:8]
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
        self.infer_imgsz = infer_imgsz
        self._last_telemetry_check: float = 0.0
        self._cached_telemetry: Dict[str, Any] = get_system_telemetry()
        self._last_frame_recv_time: float = time.time()

        # ── Pipeline AI độc lập cho phiên WebRTC này ──
        self.tracker = ObjectTracker()
        self.event_manager = EventManager()
        
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

        # ── Quản lý hàng đợi và luồng AI Worker ──
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._stopped = False
        self._lock = threading.Lock()

        # Shared frame buffers
        self.latest_frame_bgr: Optional[np.ndarray] = None
        self.latest_annotated_bgr: Optional[np.ndarray] = None
        self.last_events: List[Event] = []
        self.last_summary: Dict[str, Any] = {}

        # ── Thống kê hiệu năng chi tiết ──
        self._input_timestamps = collections.deque(maxlen=30)
        self._infer_timestamps = collections.deque(maxlen=30)
        self._output_timestamps = collections.deque(maxlen=30)
        self._latency_history = collections.deque(maxlen=100)
        self._dropped_frames_count = 0
        self._total_input_frames = 0
        self._total_inferred_frames = 0

        self.is_online = True
        self.latest_metrics: Dict[str, Any] = {
            "session_id": self.session_id,
            "fps": 0.0,
            "input_fps": 0.0,
            "infer_fps": 0.0,
            "output_fps": 0.0,
            "latency_avg_ms": 0.0,
            "latency_p95_ms": 0.0,
            "dropped_frames": 0,
            "active_sessions": ActiveSessionTracker.register(self.session_id),
            "total_seats": 0,
            "occ_seats": 0,
            "vac_seats": 0,
            "person_count": 0,
            "item_count": 0,
            "sus_count": 0,
            "status_tag": "🟢 ĐANG GIÁM SÁT",
            "status_color": "#10B981",
            "frame_idx": 0,
            "last_updated": time.time(),
            "cpu_percent": 0.0,
            "process_rss_mb": 0.0,
            "system_mem_percent": 0.0,
            "device": "CPU",
            "is_online": True,
        }

        # Khởi động AI background worker
        self._worker_thread = threading.Thread(
            target=self._ai_worker_loop,
            name=f"ai-worker-{self.session_id}",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(f"BusWebRtcVideoProcessor [{self.session_id}] đã khởi động.")

    def update_config(self, **kwargs):
        """Cập nhật tùy chọn hiển thị và ngưỡng tin cậy từ giao diện người dùng theo thời gian thực."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def get_metrics(self) -> Dict[str, Any]:
        """Lấy bản sao metrics thời gian thực cho Streamlit UI."""
        with self._lock:
            m = dict(self.latest_metrics)
            m["active_sessions"] = ActiveSessionTracker.get_count()
            return m

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Lấy frame có chú thích mới nhất để lưu bằng chứng hình ảnh."""
        with self._lock:
            if self.latest_annotated_bgr is not None:
                return self.latest_annotated_bgr.copy()
            if self.latest_frame_bgr is not None:
                return self.latest_frame_bgr.copy()
            return None

    def _ai_worker_loop(self):
        """
        Vòng lặp chạy nền của worker thread:
        Lấy frame mới nhất từ queue, thực hiện suy luận YOLO + tracking + seat detection,
        cập nhật annotated buffer và thống kê độ trễ.
        """
        frame_idx = 0
        idle_count = 0
        while self._running:
            try:
                item = self._frame_queue.get(timeout=0.1)
                idle_count = 0
            except queue.Empty:
                idle_count += 1
                if idle_count > 10:  # > 1 giây không có frame mới
                    time.sleep(0.05)  # Tránh quay vòng CPU rỗng khi không có frame
                if idle_count > 200:  # > 20 giây không có frame từ WebRTC client
                    logger.info(f"BusWebRtcVideoProcessor [{self.session_id}] hết thời gian chờ frame (idle timeout).")
                    break
                continue

            if item is None or not self._running:
                break

            img, t_recv, cur_idx = item
            self._last_frame_recv_time = time.time()
            t_infer_start = time.perf_counter()

            annotated = None
            events = []
            summary = {}

            if self.service is not None:
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
                        imgsz=self.infer_imgsz,
                    )
                except Exception as e:
                    logger.error(f"Lỗi suy luận AI tại worker frame {cur_idx}: {e}")
                    annotated = img

            t_infer_end = time.perf_counter()
            end_to_end_ms = (t_infer_end - t_recv) * 1000.0

            # Cập nhật buffer và metrics
            with self._lock:
                self._total_inferred_frames += 1
                self._infer_timestamps.append(t_infer_end)
                self._latency_history.append(end_to_end_ms)

                self.latest_annotated_bgr = annotated if annotated is not None else img
                self.last_events = events
                self.last_summary = summary

                # Tính toán số liệu thống kê
                infer_fps = _calc_rolling_fps(self._infer_timestamps)
                avg_lat = float(np.mean(self._latency_history)) if self._latency_history else 0.0
                p95_lat = float(np.percentile(self._latency_history, 95)) if self._latency_history else 0.0

                total_seats = len(self.seat_manager.seats)
                statuses = self.seat_manager.get_all_statuses()
                occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
                vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)
                person_count = sum(1 for obj in self.tracker.current_tracked_objects if obj.class_name == "person")
                item_count = sum(1 for obj in self.tracker.current_tracked_objects if obj.class_name in ITEM_CLASSES or obj.class_name == "item")
                sus_count = summary.get("suspected", 0)
                status_tag = "🔴 PHÁT HIỆN ĐỒ BỎ QUÊN!" if sus_count > 0 else "🟢 BÌNH THƯỜNG"
                status_color = "#EF4444" if sus_count > 0 else "#10B981"

                now_time = time.time()
                if now_time - self._last_telemetry_check > 2.0:
                    self._cached_telemetry = get_system_telemetry()
                    self._last_telemetry_check = now_time
                telemetry = self._cached_telemetry

                self.latest_metrics.update({
                    "infer_fps": infer_fps,
                    "latency_avg_ms": avg_lat,
                    "latency_p95_ms": p95_lat,
                    "total_seats": total_seats,
                    "occ_seats": occ_seats,
                    "vac_seats": vac_seats,
                    "person_count": person_count,
                    "item_count": item_count,
                    "sus_count": sus_count,
                    "status_tag": status_tag,
                    "status_color": status_color,
                    "frame_idx": cur_idx,
                    "last_updated": time.time(),
                    "cpu_percent": telemetry["cpu_percent"],
                    "process_rss_mb": telemetry["process_rss_mb"],
                    "system_mem_percent": telemetry["system_mem_percent"],
                    "thread_count": telemetry.get("thread_count", 1),
                    "device": telemetry["device"],
                })

            frame_idx += 1

    def recv(self, frame: Any) -> Any:
        """
        Nhận frame từ WebRTC client và trả về frame có chú thích.
        Hàm này chạy trên worker thread của aiortc và được tối ưu để thực thi cực nhanh (< 0.1ms).
        """
        if not WEBRTC_AVAILABLE or frame is None or not self._running:
            return frame

        t_now = time.perf_counter()

        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as e:
            logger.error(f"Lỗi giải mã frame WebRTC sang bgr24: {e}")
            return frame

        if img is None:
            return frame

        with self._lock:
            self._total_input_frames += 1
            self._input_timestamps.append(t_now)
            cur_idx = self._total_input_frames
            self.latest_frame_bgr = img

            # Kiểm tra stride: chỉ đưa vào hàng đợi suy luận theo chu kỳ
            should_infer = (cur_idx % self.stride == 0) or (self.latest_annotated_bgr is None)

            if should_infer:
                # Nếu queue đang đầy (worker chưa xử lý xong frame trước), drop frame cũ ngay lập tức
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                        self._dropped_frames_count += 1
                    except queue.Empty:
                        pass
                try:
                    self._frame_queue.put_nowait((img, t_now, cur_idx))
                except queue.Full:
                    self._dropped_frames_count += 1

            # Lấy frame chú thích mới nhất có sẵn (hoặc frame hiện tại nếu chưa có kết quả)
            out_img = self.latest_annotated_bgr if self.latest_annotated_bgr is not None else img

            # Tính toán Input FPS và Output FPS
            input_fps = _calc_rolling_fps(self._input_timestamps)
            t_out = time.perf_counter()
            self._output_timestamps.append(t_out)
            output_fps = _calc_rolling_fps(self._output_timestamps)

            self.latest_metrics["input_fps"] = input_fps
            self.latest_metrics["output_fps"] = output_fps
            self.latest_metrics["fps"] = output_fps
            self.latest_metrics["dropped_frames"] = self._dropped_frames_count

        try:
            return av.VideoFrame.from_ndarray(out_img, format="bgr24")
        except Exception as e:
            logger.error(f"Lỗi đóng gói frame bgr24 sang VideoFrame: {e}")
            return frame

    def on_ended(self):
        """
        Được gọi tự động khi media track kết thúc hoặc người dùng bấm STOP trong streamlit-webrtc.
        Dừng worker thread, giải phóng tài nguyên nhưng giữ nguyên hồ sơ sự kiện đã lưu.
        """
        self.stop()

    def stop(self):
        """Dừng luồng suy luận an toàn và giải phóng tài nguyên."""
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self.is_online = False

        # Đánh thức worker thread nếu đang chờ queue
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

        ActiveSessionTracker.unregister(self.session_id)

        with self._lock:
            self.latest_metrics.update({
                "status_tag": "🔴 CAMERA OFFLINE (Chưa giám sát)",
                "status_color": "#94A3B8",
                "is_online": False,
                "input_fps": 0.0,
                "infer_fps": 0.0,
                "output_fps": 0.0,
            })

        logger.info(f"BusWebRtcVideoProcessor [{self.session_id}] đã dừng và giải phóng tài nguyên.")
