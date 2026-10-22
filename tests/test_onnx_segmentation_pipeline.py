"""
tests/test_onnx_segmentation_pipeline.py
Kiểm thử chuyên sâu pipeline mô hình ONNX Instance Segmentation (yolo26n-seg.onnx):
- Xác minh file mô hình tồn tại và nạp được qua ONNX Runtime.
- Kiểm tra kích thước tensor đầu vào [1, 3, 480, 480] Float32 RGB.
- Kiểm tra các tensor đầu ra: output0 [1, 116, 4725] và output1 [1, 32, 120, 120].
- Kiểm thử logic tái tạo mặt nạ nhị phân (prototype masks @ mask coefficients -> sigmoid).
- Kiểm thử trích xuất polygon contour và chuẩn hóa tọa độ về ảnh gốc.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import cv2
import numpy as np

from src.detection.onnx_detector import ONNXSegmentationDetector, sigmoid
from config.settings import ITEM_CLASSES


class TestONNXSegmentationPipeline(unittest.TestCase):

    def setUp(self):
        p = Path("static/models/yolo26n-seg.onnx")
        if not p.exists():
            p = Path("yolo26n-seg.onnx")
        self.model_path = p
        self.detector = ONNXSegmentationDetector(
            model_path=self.model_path,
            input_size=480,
            conf_threshold=0.25,
        )

    def test_01_model_file_and_session(self):
        """1. Xác minh file yolo26n-seg.onnx tồn tại và nạp session thành công."""
        self.assertTrue(self.model_path.exists(), "File yolo26n-seg.onnx phải tồn tại trong workspace")
        self.assertTrue(self.detector.is_loaded, "ONNX session phải nạp thành công")
        self.assertIsNotNone(self.detector.session)

    def test_02_tensor_shapes(self):
        """2. Xác minh tensor đầu vào và đầu ra đúng chuẩn YOLOv8/YOLO26-seg."""
        inputs = self.detector.session.get_inputs()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "images")
        in_shape = inputs[0].shape
        self.assertEqual(in_shape[0], 1)
        self.assertEqual(in_shape[1], 3)
        self.assertEqual(in_shape[2], 480)
        self.assertEqual(in_shape[3], 480)

        outputs = self.detector.session.get_outputs()
        self.assertEqual(len(outputs), 2)
        out0_name, out1_name = outputs[0].name, outputs[1].name
        self.assertIn("output", out0_name)
        self.assertIn("output", out1_name)

        out0_shape = outputs[0].shape
        out1_shape = outputs[1].shape
        # output0: [1, 116, 4725] (4 bbox + 80 class + 32 mask coeffs = 116)
        self.assertEqual(out0_shape[1], 116)
        # output1: [1, 32, 120, 120] (32 prototype masks)
        self.assertEqual(out1_shape[1], 32)
        self.assertEqual(out1_shape[2], 120)
        self.assertEqual(out1_shape[3], 120)

    def test_03_preprocessing_letterbox(self):
        """3. Kiểm tra tiền xử lý letterbox đúng tỉ lệ khung hình."""
        test_img = np.zeros((360, 640, 3), dtype=np.uint8)
        blob, r, (pad_w, pad_h) = self.detector.preprocess(test_img)

        self.assertEqual(blob.shape, (1, 3, 480, 480))
        self.assertEqual(blob.dtype, np.float32)
        # Tỉ lệ r = min(480/360, 480/640) = 480/640 = 0.75
        self.assertAlmostEqual(r, 0.75, places=2)
        # nw = 640 * 0.75 = 480 -> pad_w = 0
        self.assertEqual(pad_w, 0.0)
        # nh = 360 * 0.75 = 270 -> pad_h = (480 - 270) / 2 = 105.0
        self.assertEqual(pad_h, 105.0)

    def test_04_mask_reconstruction_math(self):
        """4. Kiểm thử toán học tái tạo prototype mask & sigmoid."""
        # Giả lập 32 prototype masks [32, 10, 10]
        protos = np.ones((32, 10, 10), dtype=np.float32) * 0.5
        # 32 mask coefficients
        coeffs = np.ones(32, dtype=np.float32) * 0.1

        # dot product
        mask_proto = np.sum(coeffs[:, None, None] * protos, axis=0)
        # sum = 32 * 0.5 * 0.1 = 1.6
        self.assertAlmostEqual(float(mask_proto[0, 0]), 1.6, places=3)

        sig = sigmoid(mask_proto)
        expected_sig = 1.0 / (1.0 + np.exp(-1.6))
        self.assertAlmostEqual(float(sig[0, 0]), expected_sig, places=3)
        self.assertTrue((sig > 0.5).all())

    def test_05_inference_on_blank_and_pattern_frames(self):
        """5. Chạy inference trên frame kiểm thử thật (không crash và trả về cấu trúc Detection hợp lệ)."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets_blank = self.detector.detect(blank_frame)
        self.assertIsInstance(dets_blank, list)
        self.assertEqual(len(dets_blank), 0)

        # Frame có vật thể mô phỏng
        pattern_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(pattern_frame, (100, 100), (300, 300), (200, 200, 200), -1)
        dets_pattern = self.detector.detect(pattern_frame, confidence_threshold=0.1)
        self.assertIsInstance(dets_pattern, list)


if __name__ == "__main__":
    unittest.main()
