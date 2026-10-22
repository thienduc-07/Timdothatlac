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
from typing import List, Optional, Set, Tuple, Union, Dict

from src.models import Detection
from src.utils.logger import get_logger
from config.settings import (
    YOLO_MODEL_NAME,
    CUSTOM_SEGMENTATION_WEIGHTS_PATH,
    SUPPORTED_CUSTOM_WEIGHTS_PATHS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEVICE,
    ITEM_CLASSES,
    PERSON_CLASS,
    SEAT_CLASS,
    MODELS_DIR,
    PROJECT_ROOT,
)

logger = get_logger(__name__)


class YOLODetector:
    """
    Bao bọc (wrapper) cho model YOLO (Detection & Instance Segmentation) của Ultralytics.
    Hỗ trợ cả model Object Detection thông thường và YOLO26 / YOLO Instance Segmentation:
    - Tự động ưu tiên weights custom models/best.pt hoặc yolo26n-seg.pt nếu có.
    - Nhận diện tác vụ model (task='segment' hoặc task='detect').
    - Trích xuất polygon từ segmentation masks (result.masks.xy).
    - Hỗ trợ chia sẻ model instance với SeatDetector để tránh load trùng bộ nhớ.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str = DEVICE,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name: Đường dẫn hoặc tên file model. Nếu None, tự động kiểm tra weights segmentation ưu tiên
            confidence_threshold: Ngưỡng confidence tối thiểu
            device: "cpu" hoặc "cuda"
            target_classes: Danh sách lớp cần phát hiện (None = tất cả các lớp quan tâm)
        """
        # Xác định đường dẫn model ưu tiên (Ưu tiên Instance Segmentation weights)
        if model_name is None:
            if Path(CUSTOM_SEGMENTATION_WEIGHTS_PATH).exists():
                self.model_name = str(CUSTOM_SEGMENTATION_WEIGHTS_PATH)
            elif (PROJECT_ROOT / "yolo26n-seg.pt").exists():
                self.model_name = str(PROJECT_ROOT / "yolo26n-seg.pt")
            elif (MODELS_DIR / "yolo26n-seg.pt").exists():
                self.model_name = str(MODELS_DIR / "yolo26n-seg.pt")
            else:
                self.model_name = YOLO_MODEL_NAME
        else:
            self.model_name = model_name

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._class_names: Dict[int, str] = {}  # Mapping: class_id -> class_name
        self._target_class_ids: Optional[Set[int]] = None

        # Danh sách lớp mà người dùng muốn theo dõi (bao gồm person, seat và các item)
        default_targets = list(set([PERSON_CLASS, SEAT_CLASS] + ITEM_CLASSES))
        self._target_class_names = target_classes or default_targets

        self._loaded = False
        self.task: str = "detect"
        self.is_segmentation: bool = False
        self.is_custom_3class: bool = False
        self._weights_source: str = "default"

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra model đã được tải chưa."""
        return self._loaded

    def set_shared_model(self, model_instance) -> bool:
        """
        Tái sử dụng instance model đã nạp từ trước (tránh nạp duplicate vào RAM).
        """
        if model_instance is None:
            return False
        try:
            self.model = model_instance
            self._class_names = getattr(model_instance, "names", {})
            self.task = getattr(model_instance, "task", "detect")
            self.is_segmentation = (self.task == "segment")
            self._setup_classes()
            self._loaded = True
            self._weights_source = "shared"
            logger.info(f"Đã gán shared model: task='{self.task}', {len(self._class_names)} lớp")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi gán shared model: {e}")
            return False

    def _setup_classes(self):
        """Xác định class IDs theo dõi và kiểm tra xem có phải custom 3-class (seat, person, item) không."""
        self._target_class_ids = set()
        found_classes = []
        for class_id, class_name in self._class_names.items():
            if class_name in self._target_class_names:
                self._target_class_ids.add(class_id)
                found_classes.append(class_name)

        c_names_lower = [str(c).lower() for c in self._class_names.values()]
        has_seat = any("seat" in c for c in c_names_lower)
        has_person = any("person" in c for c in c_names_lower)
        has_item = any("item" in c for c in c_names_lower)
        self.is_custom_3class = (has_seat and has_person and has_item)

        logger.info(
            f"Model setup classes: task='{self.task}', is_seg={self.is_segmentation}, "
            f"custom_3class={self.is_custom_3class}, target_ids={self._target_class_ids}"
        )

    def load_model(self) -> bool:
        """
        Tải model YOLO. Tự động kiểm tra file custom weights (models/best.pt) nếu có.

        Returns:
            True nếu tải thành công, False nếu thất bại.
        """
        try:
            from ultralytics import YOLO

            # Kiểm tra nếu đường dẫn hiện tại không tồn tại, tự động tìm weights thay thế
            target_path = Path(self.model_name)
            if not target_path.exists():
                if Path(CUSTOM_SEGMENTATION_WEIGHTS_PATH).exists():
                    self.model_name = str(CUSTOM_SEGMENTATION_WEIGHTS_PATH)
                    logger.info(f"Phát hiện file weights custom tại {CUSTOM_SEGMENTATION_WEIGHTS_PATH}. Tự động ưu tiên sử dụng.")
                elif (PROJECT_ROOT / "yolo26n-seg.pt").exists():
                    self.model_name = str(PROJECT_ROOT / "yolo26n-seg.pt")
                    logger.info("Phát hiện file weights yolo26n-seg.pt tại PROJECT_ROOT. Tự động ưu tiên sử dụng.")
                elif (MODELS_DIR / "yolo26n-seg.pt").exists():
                    self.model_name = str(MODELS_DIR / "yolo26n-seg.pt")
                    logger.info("Phát hiện file weights yolo26n-seg.pt tại MODELS_DIR. Tự động ưu tiên sử dụng.")

            logger.info(f"Đang tải model {self.model_name} trên thiết bị '{self.device}'...")
            self.model = YOLO(self.model_name)

            # Lấy thông tin task và danh sách lớp
            self.task = getattr(self.model, "task", "detect")
            self.is_segmentation = (self.task == "segment")
            self._class_names = self.model.names if hasattr(self.model, "names") else {}
            logger.info(f"Model đã tải: {len(self._class_names)} lớp, task='{self.task}', is_segmentation={self.is_segmentation}")

            self._setup_classes()
            self._loaded = True
            self._weights_source = "loaded"
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
        Hỗ trợ trích xuất cả bounding box và segmentation polygon (nếu là instance segmentation).

        Args:
            frame: Ảnh BGR (numpy array) từ OpenCV

        Returns:
            Danh sách Detection cho các vật thể được phát hiện.
            Trả về danh sách rỗng nếu không có kết quả hoặc lỗi.
        """
        if not self._loaded or self.model is None:
            logger.error("Model chưa được tải. Gọi load_model() trước.")
            return []

        try:
            h, w = frame.shape[:2]
            # Chạy inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                classes=list(self._target_class_ids) if self._target_class_ids else None,
            )

            detections = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Kiểm tra xem có masks (Instance Segmentation) hay không
                has_masks = (
                    hasattr(result, "masks")
                    and result.masks is not None
                    and hasattr(result.masks, "xy")
                    and result.masks.xy is not None
                )

                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = self._class_names.get(class_id, f"class_{class_id}")
                    confidence = float(box.conf[0])

                    # Tọa độ xyxy (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Trích xuất polygon nếu model trả về mask
                    polygon: Optional[List[Tuple[float, float]]] = None
                    if has_masks and i < len(result.masks.xy):
                        poly_pts = result.masks.xy[i]
                        if poly_pts is not None and len(poly_pts) >= 3:
                            # Chuẩn hóa giới hạn nằm trong khung ảnh [0, w], [0, h]
                            valid_pts = []
                            for pt in poly_pts:
                                px = max(0.0, min(float(w), float(pt[0])))
                                py = max(0.0, min(float(h), float(pt[1])))
                                valid_pts.append((px, py))
                            if len(valid_pts) >= 3:
                                polygon = valid_pts

                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        class_id=class_id,
                        polygon=polygon,
                    ))

            return detections

        except Exception as e:
            logger.error(f"Lỗi khi chạy detection: {e}")
            return []

    def get_model_info(self) -> dict:
        """Trả về thông tin model để hiển thị trên giao diện."""
        return {
            "model_name": Path(self.model_name).name,
            "weights_path": str(self.model_name),
            "loaded": self._loaded,
            "device": self.device,
            "task": self.task,
            "is_segmentation": self.is_segmentation,
            "is_custom_3class": self.is_custom_3class,
            "confidence_threshold": self.confidence_threshold,
            "total_classes": len(self._class_names),
            "tracked_classes": list(self._target_class_ids) if self._target_class_ids else [],
            "tracked_class_names": [
                self._class_names.get(cid, "?")
                for cid in (self._target_class_ids or [])
            ],
            "all_classes": list(self._class_names.values()),
        }

    @property
    def class_names(self) -> dict:
        """Từ điển id -> tên lớp."""
        return dict(self._class_names)

    @property
    def target_class_ids(self) -> set:
        """Tập hợp ID các lớp mục tiêu được theo dõi."""
        return set(self._target_class_ids)
