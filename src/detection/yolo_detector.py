"""
src/detection/yolo_detector.py
Module phụ trách khởi tạo model YOLOv8 và thực hiện nhận diện vật thể.

- Tải model pretrained (lần đầu cần Internet, sau đó dùng cache).
- Kiểm tra model đã có trong máy hay chưa.
- Thực hiện inference trên từng frame.
- Trả về danh sách Detection với tên lớp, confidence và bounding box.

Giấy phép: YOLOv8 (Ultralytics) sử dụng AGPL-3.0.
Nếu thương mại hóa, cần mua giấy phép Enterprise từ Ultralytics.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Set

from src.models import Detection
from src.utils.logger import get_logger
from config.settings import (
    YOLO_MODEL_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEVICE,
    ITEM_CLASSES,
    PERSON_CLASS,
    MODELS_DIR,
)

logger = get_logger(__name__)


class YOLODetector:
    """
    Bao bọc (wrapper) cho model YOLOv8 của Ultralytics.
    Tách riêng logic nhận diện để có thể thay model dễ dàng.
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str = DEVICE,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name: Tên file model (ví dụ: "yolov8n.pt")
            confidence_threshold: Ngưỡng confidence tối thiểu
            device: "cpu" hoặc "cuda"
            target_classes: Danh sách lớp cần phát hiện (None = tất cả)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._class_names = {}  # Mapping: class_id -> class_name
        self._target_class_ids: Optional[Set[int]] = None

        # Danh sách lớp mà người dùng muốn theo dõi
        self._target_class_names = target_classes or (ITEM_CLASSES + [PERSON_CLASS])

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra model đã được tải chưa."""
        return self._loaded

    def load_model(self) -> bool:
        """
        Tải model YOLOv8. Lần đầu cần Internet để tải weights.

        Returns:
            True nếu tải thành công, False nếu thất bại.
        """
        try:
            from ultralytics import YOLO

            # Kiểm tra xem model đã được cache bởi Ultralytics chưa
            logger.info(f"Đang tải model {self.model_name} trên thiết bị '{self.device}'...")
            self.model = YOLO(self.model_name)

            # Lấy danh sách tên lớp từ model
            self._class_names = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}
            logger.info(f"Model có {len(self._class_names)} lớp")

            # Xác định các class ID tương ứng với target classes
            self._target_class_ids = set()
            found_classes = []
            for class_id, class_name in self._class_names.items():
                if class_name in self._target_class_names:
                    self._target_class_ids.add(class_id)
                    found_classes.append(class_name)

            logger.info(f"Các lớp theo dõi tìm thấy trong model: {found_classes}")

            not_found = set(self._target_class_names) - set(found_classes)
            if not_found:
                logger.warning(f"Các lớp không có trong model: {not_found}")

            self._loaded = True
            return True

        except ImportError:
            logger.error("Không thể import ultralytics. Hãy cài: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi tải model YOLO: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Nhận diện vật thể trên một frame.

        Args:
            frame: Ảnh BGR (numpy array) từ OpenCV

        Returns:
            Danh sách Detection cho các vật thể được phát hiện.
            Trả về danh sách rỗng nếu không có kết quả.
        """
        if not self._loaded or self.model is None:
            logger.error("Model chưa được tải. Gọi load_model() trước.")
            return []

        try:
            # Chạy inference — verbose=False để không in ra console
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                classes=list(self._target_class_ids) if self._target_class_ids else None,
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self._class_names.get(class_id, f"class_{class_id}")
                    confidence = float(box.conf[0])
                    # Tọa độ xyxy (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=class_id,
                    ))

            return detections

        except Exception as e:
            logger.error(f"Lỗi khi chạy detection: {e}")
            return []

    def get_model_info(self) -> dict:
        """Trả về thông tin model để hiển thị trên giao diện."""
        info = {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "total_classes": len(self._class_names),
            "tracked_classes": list(self._target_class_ids) if self._target_class_ids else [],
            "tracked_class_names": [
                self._class_names.get(cid, "?")
                for cid in (self._target_class_ids or [])
            ],
        }
        return info
