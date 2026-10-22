"""
src/detection/inference_backend.py
Kiến trúc Trừu tượng hóa Backend Suy luận (Hybrid AI Inference Backend Architecture):
- Định nghĩa hợp đồng chung (contracts) cho cả Client-Side (Trình duyệt WebGPU/WASM) và Server-Side (Python YOLO/Roboflow).
- Quản lý 3 chế độ: AUTO (Tự động thích ứng), ON_DEVICE (Ưu tiên client), SERVER (Chạy trên máy chủ).
- Hỗ trợ cơ chế chuyển đổi ổn định với thời gian chờ (cooldown & hysteresis) để tránh dao động liên tục giữa các backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceBackendType(str, Enum):
    CLIENT_WEBGPU = "client_webgpu"
    CLIENT_WASM = "client_wasm"
    SERVER_LOCAL_YOLO = "server_local_yolo"
    SERVER_ROBOFLOW = "server_roboflow"
    SERVER_HYBRID = "server_hybrid"


class HybridMode(str, Enum):
    AUTO = "auto"           # Tự động chọn Client nếu hỗ trợ và mượt, fallback sang Server khi cần
    ON_DEVICE = "on_device" # Cố định chạy trực tiếp trên thiết bị (Client WebGPU/WASM)
    SERVER = "server"       # Cố định chạy trên máy chủ (WebRTC/OpenCV Python)


@dataclass
class InferenceResult:
    """Kết quả suy luận chuẩn hóa từ bất kỳ backend nào (Client hoặc Server)."""
    detections: List[Detection] = field(default_factory=list)
    backend: str = InferenceBackendType.SERVER_LOCAL_YOLO.value
    inference_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    frame_width: int = 640
    frame_height: int = 480
    is_client: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.class_name == "person")

    @property
    def item_count(self) -> int:
        return sum(1 for d in self.detections if d.class_name not in ("person", "seat", "bus_seat"))


class InferenceBackend(ABC):
    """Giao diện trừu tượng cho backend suy luận."""

    def __init__(self, backend_type: str, is_client_side: bool = False):
        self.backend_type = backend_type
        self.is_client_side = is_client_side
        self.is_ready = False

    @abstractmethod
    def initialize(self) -> bool:
        """Khởi tạo tài nguyên model hoặc cấu hình backend."""
        pass

    @abstractmethod
    def infer(self, frame: Optional[np.ndarray], **kwargs) -> InferenceResult:
        """Thực thi suy luận và trả về kết quả chuẩn hóa."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Kiểm tra backend có đang sẵn sàng hoạt động hay không."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Giải phóng tài nguyên."""
        pass


class ServerInferenceBackend(InferenceBackend):
    """Backend suy luận phía máy chủ sử dụng YOLODetector hoặc Roboflow."""

    def __init__(self, yolo_detector=None, backend_type: str = InferenceBackendType.SERVER_LOCAL_YOLO.value):
        super().__init__(backend_type=backend_type, is_client_side=False)
        self.detector = yolo_detector
        if self.detector is not None:
            self.is_ready = True

    def initialize(self) -> bool:
        if self.detector is None:
            try:
                from src.detection.yolo_detector import YOLODetector
                self.detector = YOLODetector()
                self.is_ready = True
                return True
            except Exception as e:
                logger.error(f"Khởi tạo ServerInferenceBackend thất bại: {e}")
                self.is_ready = False
                return False
        self.is_ready = True
        return True

    def infer(self, frame: Optional[np.ndarray], **kwargs) -> InferenceResult:
        t0 = time.perf_counter()
        if frame is None or not self.is_ready:
            return InferenceResult(
                detections=[],
                backend=self.backend_type,
                inference_time_ms=0.0,
                is_client=False,
            )

        conf = kwargs.get("confidence_threshold", 0.35)
        imgsz = kwargs.get("imgsz", 480)
        h, w = frame.shape[:2]

        try:
            detections = self.detector.detect(frame, confidence_threshold=conf, imgsz=imgsz)
        except Exception as e:
            logger.error(f"Lỗi suy luận Server: {e}")
            detections = []

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            detections=detections,
            backend=self.backend_type,
            inference_time_ms=elapsed_ms,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            is_client=False,
            metadata={"imgsz": imgsz, "conf": conf},
        )

    def is_healthy(self) -> bool:
        return self.is_ready and self.detector is not None

    def release(self) -> None:
        self.detector = None
        self.is_ready = False


class ClientInferenceBackend(InferenceBackend):
    """
    Mô tả cấu hình và trạng thái của Backend suy luận trực tiếp trên thiết bị (Client WebGPU/WASM).
    Nhận telemetry và kết quả suy luận từ trình duyệt chuyển về để đồng bộ hóa nếu cần.
    """

    def __init__(
        self,
        model_url: str = "/app/static/models/yolo26n-seg.onnx",
        preferred_execution_provider: str = "webgpu",
    ):
        super().__init__(backend_type=InferenceBackendType.CLIENT_WEBGPU.value, is_client_side=True)
        self.model_url = model_url
        self.preferred_execution_provider = preferred_execution_provider
        self.client_fps: float = 0.0
        self.client_latency_ms: float = 0.0
        self.client_provider: str = preferred_execution_provider
        self.last_telemetry_time: float = 0.0
        self.is_ready = True

    def initialize(self) -> bool:
        self.is_ready = True
        return True

    def update_telemetry(self, fps: float, latency_ms: float, provider: str = "webgpu"):
        self.client_fps = float(fps)
        self.client_latency_ms = float(latency_ms)
        self.client_provider = provider
        self.last_telemetry_time = time.time()
        if "wasm" in provider.lower():
            self.backend_type = InferenceBackendType.CLIENT_WASM.value
        else:
            self.backend_type = InferenceBackendType.CLIENT_WEBGPU.value

    def infer(self, frame: Optional[np.ndarray], **kwargs) -> InferenceResult:
        # Client backend suy luận trực tiếp trên trình duyệt, Python không xử lý raw frame
        return InferenceResult(
            detections=kwargs.get("client_detections", []),
            backend=self.backend_type,
            inference_time_ms=self.client_latency_ms,
            timestamp=time.time(),
            frame_width=kwargs.get("frame_width", 640),
            frame_height=kwargs.get("frame_height", 480),
            is_client=True,
            metadata={"provider": self.client_provider, "fps": self.client_fps},
        )

    def is_healthy(self) -> bool:
        # Nếu đang trong phiên client, kiểm tra xem telemetry có được cập nhật gần đây (trong vòng 10s)
        if self.last_telemetry_time > 0 and (time.time() - self.last_telemetry_time) > 10.0:
            return False
        return self.is_ready

    def release(self) -> None:
        self.client_fps = 0.0
        self.client_latency_ms = 0.0
        self.is_ready = False


class InferenceBackendManager:
    """
    Quản lý lựa chọn backend và chính sách chuyển đổi tự động (Fallback Policy):
    - Đảm bảo thời gian cooldown để không đổi qua lại liên tục giữa các frame.
    - Duy trì tính cô lập phiên (Session isolation).
    """

    def __init__(
        self,
        mode: HybridMode = HybridMode.AUTO,
        server_backend: Optional[ServerInferenceBackend] = None,
        client_backend: Optional[ClientInferenceBackend] = None,
        min_client_fps: float = 8.0,
        switch_cooldown_sec: float = 5.0,
    ):
        self.mode = mode
        self.server_backend = server_backend or ServerInferenceBackend()
        self.client_backend = client_backend or ClientInferenceBackend()
        self.min_client_fps = min_client_fps
        self.switch_cooldown_sec = switch_cooldown_sec

        self._active_backend: InferenceBackend = self.server_backend
        self._last_switch_time: float = 0.0
        self._consecutive_low_fps_count: int = 0
        self._client_init_failed: bool = False
        self._fallback_reason: Optional[str] = None

        self._evaluate_active_backend(force=True)
        self._last_switch_time = 0.0

    @property
    def active_backend(self) -> InferenceBackend:
        return self._active_backend

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    def set_mode(self, new_mode: HybridMode):
        if self.mode != new_mode:
            self.mode = new_mode
            self._consecutive_low_fps_count = 0
            self._client_init_failed = False
            self._fallback_reason = None
            self._evaluate_active_backend(force=True)

    def report_client_status(self, success: bool, error_msg: Optional[str] = None, fps: float = 0.0, latency_ms: float = 0.0, provider: str = "webgpu"):
        """Ghi nhận phản hồi trạng thái từ trình duyệt."""
        if not success:
            self._client_init_failed = True
            self._fallback_reason = error_msg or "Trình duyệt không hỗ trợ WebGPU/WASM hoặc tải model thất bại"
            if self.mode == HybridMode.AUTO:
                self._switch_to(self.server_backend, reason=self._fallback_reason)
            return

        self._client_init_failed = False
        self.client_backend.update_telemetry(fps=fps, latency_ms=latency_ms, provider=provider)

        # Trong chế độ AUTO, kiểm tra hiệu năng
        if self.mode == HybridMode.AUTO:
            now = time.time()
            if fps > 0 and fps < self.min_client_fps:
                self._consecutive_low_fps_count += 1
                if self._consecutive_low_fps_count >= 5 and (now - self._last_switch_time) >= self.switch_cooldown_sec:
                    self._switch_to(
                        self.server_backend,
                        reason=f"Hiệu năng thiết bị thấp ({fps:.1f} FPS < {self.min_client_fps} FPS), tự động chuyển sang máy chủ."
                    )
            else:
                self._consecutive_low_fps_count = 0
                if self._active_backend != self.client_backend and (now - self._last_switch_time) >= self.switch_cooldown_sec:
                    self._switch_to(self.client_backend, reason="Thiết bị client sẵn sàng và hoạt động mượt mà.")

    def _switch_to(self, target_backend: InferenceBackend, reason: str):
        now = time.time()
        if self._active_backend != target_backend:
            logger.info(f"Chuyển backend từ {self._active_backend.backend_type} sang {target_backend.backend_type}: {reason}")
            self._active_backend = target_backend
            self._last_switch_time = now
            self._fallback_reason = reason

    def _evaluate_active_backend(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_switch_time) < self.switch_cooldown_sec:
            return

        if self.mode == HybridMode.SERVER:
            self._switch_to(self.server_backend, reason="Người dùng chọn chế độ Máy chủ (Server)")
        elif self.mode == HybridMode.ON_DEVICE:
            if self._client_init_failed:
                # Vẫn giữ client hoặc thông báo rõ ràng
                self._fallback_reason = "Thiết bị client không khởi tạo được runtime"
            self._switch_to(self.client_backend, reason="Người dùng chọn chế độ Trên thiết bị (On-Device)")
        elif self.mode == HybridMode.AUTO:
            if self._client_init_failed:
                self._switch_to(self.server_backend, reason=self._fallback_reason or "Fallback tự động sang Máy chủ do lỗi Client")
            else:
                self._switch_to(self.client_backend, reason="Ưu tiên tự động chạy trên thiết bị client")
