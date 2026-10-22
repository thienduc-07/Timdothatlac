"""
src/detection/onnx_detector.py
Bộ dò tìm và phân đoạn đối tượng sử dụng mô hình ONNX Runtime (yolo26n-seg.onnx):
- Nạp model ONNX Instance Segmentation (1, 3, imgsz, imgsz).
- Giải mã tensor output0 [1, 116, anchors] và output1 [1, 32, proto_h, proto_w].
- Tái tạo mặt nạ nhị phân (prototype masks @ mask coefficients -> sigmoid).
- Trích xuất polygon contour thật sự cho mọi đồ vật được phát hiện.
- Đối chiếu chính xác tọa độ về frame gốc (un-pad và rescale).
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.models import Detection
from src.utils.logger import get_logger
from config.settings import (
    ITEM_CLASSES,
    PERSON_CLASS,
    SEAT_CLASS,
    DEFAULT_CONFIDENCE_THRESHOLD,
)

logger = get_logger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ONNXSegmentationDetector:
    """Detector chuyên biệt chạy model ONNX Instance Segmentation."""

    # 80 lớp COCO chuẩn
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo26n-seg.onnx",
        target_classes: Optional[List[str]] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = 0.45,
        input_size: int = 480,
    ):
        p = Path(model_path)
        if not p.exists():
            for alt in [Path("static/models/yolo26n-seg.onnx"), Path("models/yolo26n-seg.onnx"), Path("yolo26n-seg.onnx")]:
                if alt.exists():
                    p = alt
                    break
        self.model_path = p
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.session: Optional[Any] = None
        self.is_loaded = False
        self.task = "segment"

        # Định nghĩa các lớp mục tiêu
        default_targets = list(set([PERSON_CLASS, SEAT_CLASS, "chair"] + ITEM_CLASSES))
        self.target_classes = target_classes or default_targets

        if ORT_AVAILABLE and self.model_path.exists():
            self._load_model()

    def _load_model(self):
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self.is_loaded = True
            logger.info(f"ONNXSegmentationDetector đã nạp thành công mô hình: {self.model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi nạp mô hình ONNX: {e}")
            self.session = None
            self.is_loaded = False

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Letterbox padding chuẩn giữ tỉ lệ khung hình."""
        h0, w0 = img.shape[:2]
        r = min(self.input_size / h0, self.input_size / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pad_w = (self.input_size - nw) / 2.0
        pad_h = (self.input_size - nh) / 2.0

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_h)), int(round(pad_w))
        canvas[top:top + nh, left:left + nw] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB & Chuẩn hóa 0-1
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob, r, (pad_w, pad_h)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        if not self.is_loaded or self.session is None or frame is None:
            return []

        conf_thresh = confidence_threshold if confidence_threshold is not None else self.conf_threshold
        h0, w0 = frame.shape[:2]
        blob, r, (pad_w, pad_h) = self.preprocess(frame)

        # Chạy suy luận ONNX
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        out0 = outputs[0]  # [1, 116, 4725]
        out1 = outputs[1]  # [1, 32, 120, 120]

        preds = np.transpose(out0[0], (1, 0))  # [4725, 116]
        protos = out1[0]  # [32, 120, 120]

        boxes, scores, class_ids, mask_coeffs = [], [], [], []
        num_classes = 80
        for row in preds:
            cls_scores = row[4:4 + num_classes]
            max_id = int(np.argmax(cls_scores))
            score = float(cls_scores[max_id])
            if score >= conf_thresh:
                cname = self.COCO_CLASSES[max_id] if max_id < len(self.COCO_CLASSES) else "unknown"
                # Map chair thành seat
                if cname == "chair":
                    cname = "seat"
                if self.target_classes and cname not in self.target_classes and cname not in ITEM_CLASSES:
                    continue

                cx, cy, w, h = row[0:4]
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(max_id)
                mask_coeffs.append(row[4 + num_classes:4 + num_classes + 32])

        if not boxes:
            return []

        # NMS
        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores, conf_thresh, self.iou_threshold)
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        proto_scale = 120.0 / self.input_size
        detections: List[Detection] = []

        for idx in indices:
            b = boxes[idx]
            mc = mask_coeffs[idx]
            cid = class_ids[idx]
            cname = self.COCO_CLASSES[cid] if cid < len(self.COCO_CLASSES) else "item"
            if cname == "chair":
                cname = "seat"

            # Tái tạo mask nhị phân từ 32 prototypes
            mask_proto = np.sum(mc[:, None, None] * protos, axis=0)
            mask_sig = sigmoid(mask_proto)

            # Cắt theo bounding box trong không gian proto
            px1 = max(0, int(np.floor(b[0] * proto_scale)))
            py1 = max(0, int(np.floor(b[1] * proto_scale)))
            px2 = min(120, int(np.ceil(b[2] * proto_scale)))
            py2 = min(120, int(np.ceil(b[3] * proto_scale)))

            cropped_mask = np.zeros_like(mask_sig)
            cropped_mask[py1:py2, px1:px2] = mask_sig[py1:py2, px1:px2]
            bin_mask = (cropped_mask > 0.5).astype(np.uint8)

            # Khôi phục mask về không gian ảnh gốc
            mask_input = cv2.resize(bin_mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            top = int(round(pad_h))
            left = int(round(pad_w))
            nh = int(round(h0 * r))
            nw = int(round(w0 * r))
            unpad_mask = mask_input[top:top + nh, left:left + nw]
            orig_mask = cv2.resize(unpad_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

            # Trích xuất polygon contour
            contours, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 3:
                    pts = largest.squeeze()
                    if pts.ndim == 2:
                        # Simplify nhẹ để tối ưu render
                        epsilon = 0.005 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True).squeeze()
                        if approx.ndim == 2 and len(approx) >= 3:
                            polygon = [(float(pt[0]), float(pt[1])) for pt in approx]
                        else:
                            polygon = [(float(pt[0]), float(pt[1])) for pt in pts]

            # Khôi phục bbox về tọa độ frame gốc
            ox1 = int(np.clip(round((b[0] - pad_w) / r), 0, w0))
            oy1 = int(np.clip(round((b[1] - pad_h) / r), 0, h0))
            ox2 = int(np.clip(round((b[2] - pad_w) / r), 0, w0))
            oy2 = int(np.clip(round((b[3] - pad_h) / r), 0, h0))

            detections.append(Detection(
                class_name=cname,
                confidence=scores[idx],
                bbox=(ox1, oy1, ox2, oy2),
                class_id=cid,
                polygon=polygon,
            ))

        return detections
