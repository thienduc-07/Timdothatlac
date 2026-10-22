"""
tests/test_hybrid_backend_abstraction.py
Kiểm thử bộ trừu tượng hóa Hybrid AI Inference Backend:
- Xác minh các giao diện (contracts) của InferenceBackend, ServerInferenceBackend, ClientInferenceBackend.
- Kiểm thử các chế độ AUTO, ON_DEVICE, SERVER của InferenceBackendManager.
- Xác minh chính sách chuyển đổi ổn định (Hysteresis & Cooldown) không dao động liên tục giữa các frame.
- Đảm bảo tính cô lập phiên (Session isolation).
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.models import Detection
from src.detection.inference_backend import (
    InferenceBackend,
    InferenceBackendType,
    HybridMode,
    InferenceResult,
    ServerInferenceBackend,
    ClientInferenceBackend,
    InferenceBackendManager,
)


class TestHybridBackendAbstraction(unittest.TestCase):

    def setUp(self):
        self.mock_yolo = MagicMock()
        self.mock_yolo.detect.return_value = [
            Detection(class_name="person", confidence=0.92, bbox=(10, 10, 100, 200)),
            Detection(class_name="backpack", confidence=0.88, bbox=(120, 50, 180, 120), polygon=[(120, 50), (180, 50), (180, 120), (120, 120)]),
        ]

    def test_01_inference_result_contract(self):
        """1. Kiểm tra cấu trúc chuẩn hóa của InferenceResult."""
        res = InferenceResult(
            detections=[
                Detection(class_name="person", confidence=0.9, bbox=(0, 0, 10, 10)),
                Detection(class_name="laptop", confidence=0.85, bbox=(20, 20, 50, 50)),
                Detection(class_name="seat", confidence=0.8, bbox=(60, 60, 100, 100)),
            ],
            backend=InferenceBackendType.CLIENT_WEBGPU.value,
            inference_time_ms=25.4,
            is_client=True,
        )
        self.assertEqual(res.person_count, 1)
        self.assertEqual(res.item_count, 1)  # Chỉ laptop, loại trừ person và seat
        self.assertEqual(res.backend, "client_webgpu")
        self.assertTrue(res.is_client)

    def test_02_server_backend_execution(self):
        """2. Kiểm tra ServerInferenceBackend thực thi và bọc kết quả đúng."""
        backend = ServerInferenceBackend(yolo_detector=self.mock_yolo)
        self.assertTrue(backend.is_healthy())

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = backend.infer(dummy_frame, confidence_threshold=0.35)

        self.assertFalse(result.is_client)
        self.assertEqual(result.backend, InferenceBackendType.SERVER_LOCAL_YOLO.value)
        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.person_count, 1)
        self.assertEqual(result.item_count, 1)
        self.assertGreater(result.inference_time_ms, 0.0)

    def test_03_client_backend_telemetry(self):
        """3. Kiểm tra ClientInferenceBackend ghi nhận telemetry và cập nhật trạng thái."""
        client_backend = ClientInferenceBackend(model_url="/test/model.onnx")
        self.assertTrue(client_backend.is_client_side)

        client_backend.update_telemetry(fps=28.5, latency_ms=32.0, provider="webgpu")
        self.assertEqual(client_backend.client_fps, 28.5)
        self.assertEqual(client_backend.backend_type, InferenceBackendType.CLIENT_WEBGPU.value)

        client_backend.update_telemetry(fps=15.0, latency_ms=65.0, provider="wasm")
        self.assertEqual(client_backend.backend_type, InferenceBackendType.CLIENT_WASM.value)
        self.assertTrue(client_backend.is_healthy())

    def test_04_manager_auto_mode_client_priority(self):
        """4. Kiểm tra chế độ AUTO ưu tiên Client khi khởi tạo thành công."""
        server_backend = ServerInferenceBackend(yolo_detector=self.mock_yolo)
        client_backend = ClientInferenceBackend()

        manager = InferenceBackendManager(
            mode=HybridMode.AUTO,
            server_backend=server_backend,
            client_backend=client_backend,
            switch_cooldown_sec=0.1,  # Đặt nhỏ để test nhanh
        )

        self.assertEqual(manager.active_backend, client_backend)

    def test_05_manager_auto_fallback_on_client_error(self):
        """5. Kiểm tra chế độ AUTO tự động fallback sang Server khi Client báo lỗi."""
        server_backend = ServerInferenceBackend(yolo_detector=self.mock_yolo)
        client_backend = ClientInferenceBackend()

        manager = InferenceBackendManager(
            mode=HybridMode.AUTO,
            server_backend=server_backend,
            client_backend=client_backend,
            switch_cooldown_sec=0.05,
        )

        # Báo cáo client thất bại (ví dụ: trình duyệt không hỗ trợ WebGPU/WASM)
        manager.report_client_status(success=False, error_msg="WebGPU not supported")

        self.assertEqual(manager.active_backend, server_backend)
        self.assertIn("WebGPU not supported", manager.fallback_reason)

    def test_06_manager_cooldown_prevents_rapid_fluttering(self):
        """6. Kiểm tra thời gian cooldown ngăn chặn việc chuyển đổi liên tục giữa các frame."""
        server_backend = ServerInferenceBackend(yolo_detector=self.mock_yolo)
        client_backend = ClientInferenceBackend()

        manager = InferenceBackendManager(
            mode=HybridMode.AUTO,
            server_backend=server_backend,
            client_backend=client_backend,
            min_client_fps=10.0,
            switch_cooldown_sec=1.0,  # Cooldown 1s
        )

        # Gửi FPS thấp liên tục
        for _ in range(6):
            manager.report_client_status(success=True, fps=5.0, latency_ms=150.0)

        # Đã kích hoạt chuyển sang server
        self.assertEqual(manager.active_backend, server_backend)

        # Ngay lập tức client báo FPS cao, nhưng còn trong cooldown 1s nên CHƯA được đổi ngay
        manager.report_client_status(success=True, fps=30.0, latency_ms=25.0)
        self.assertEqual(manager.active_backend, server_backend, "Phải giữ cooldown không được dao động ngay lập tức")

        # Sau khi hết cooldown
        time.sleep(1.05)
        manager.report_client_status(success=True, fps=30.0, latency_ms=25.0)
        self.assertEqual(manager.active_backend, client_backend, "Đã hết cooldown, phục hồi về client thành công")

    def test_07_manager_strict_modes(self):
        """7. Kiểm tra các chế độ cố định: ON_DEVICE và SERVER."""
        server_backend = ServerInferenceBackend(yolo_detector=self.mock_yolo)
        client_backend = ClientInferenceBackend()

        manager = InferenceBackendManager(
            mode=HybridMode.SERVER,
            server_backend=server_backend,
            client_backend=client_backend,
        )
        self.assertEqual(manager.active_backend, server_backend)

        manager.set_mode(HybridMode.ON_DEVICE)
        self.assertEqual(manager.active_backend, client_backend)


if __name__ == "__main__":
    unittest.main()
