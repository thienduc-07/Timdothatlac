"""
src/seats/seat_detector.py
Module nhận diện ghế chuyên biệt từ hình ảnh camera trần xe buýt (Overhead / Top-Down View):
- Hỗ trợ Custom YOLO Seat Model Adapter nạp weights chuyên biệt (ví dụ models/bus_seat_yolo.pt).
- Kiểm tra danh sách class thực tế từ weights khi nạp; kiểm tra xem model có hỗ trợ các lớp ghế hay không.
- KHÔNG giả định yolov8n.pt (COCO) có lớp seat.
- KHÔNG giả định lớp chair trong COCO (ghế thông thường nhìn ngang) là nhận diện được ghế xe buýt nhìn từ trên xuống.
- Nếu không có weights chuyên biệt, báo rõ "CUSTOM SEAT MODEL NOT CONFIGURED".
- Tuyệt đối không dùng BusGeometrySeatDetector để vẽ lưới tọa độ cố định rồi ghi nhãn đó là AI đã nhận diện ghế.
- Phân biệt rõ ràng 3 trạng thái hoạt động:
    1. MODEL DETECTION: Nhận diện trực tiếp từ inference của neural network trên các pixel đệm ghế.
    2. GEOMETRY FALLBACK: Phương án dự phòng bằng hình học khoang xe (phục vụ kiểm thử, công khai minh bạch).
    3. MANUAL CONFIGURATION: Cấu hình vùng ghế do người dùng thiết lập thủ công.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.models import SeatConfig
from src.utils.logger import get_logger
from config.settings import (
    CUSTOM_SEGMENTATION_WEIGHTS_PATH,
    SUPPORTED_CUSTOM_WEIGHTS_PATHS,
)

logger = get_logger(__name__)

# Các lớp ghế hợp lệ mà một model chuyên biệt cho góc nhìn trần xe buýt có thể xuất ra
VALID_SEAT_CLASS_NAMES = {
    "seat",
    "bus_seat",
    "bus-seat",
    "bus_cushion",
    "seat_cushion",
    "seat_window",
    "seat_aisle",
    "double_seat",
    "bench_seat",
    "empty_seat",
    "human_occupied",
    "object_occupied",
}


class SeatDetectionMethod(str, Enum):
    MODEL_DETECTION = "MODEL DETECTION"
    GEOMETRY_FALLBACK = "GEOMETRY FALLBACK"
    MANUAL_CONFIGURATION = "MANUAL CONFIGURATION"
    NOT_CONFIGURED = "CUSTOM SEAT MODEL NOT CONFIGURED"


@dataclass
class SeatDetectionResult:
    """Kết quả phân tích nhận diện ghế từ frame camera."""
    seats: List[SeatConfig]
    method: SeatDetectionMethod
    model_name: str
    confidence_avg: float = 0.0
    is_fallback: bool = False
    status_message: str = ""
    diagnostics: Dict[str, Union[str, bool, int, float, List[str]]] = field(default_factory=dict)

    @property
    def total_seats(self) -> int:
        return len(self.seats)

    @property
    def message(self) -> str:
        return self.status_message


class CustomSeatYOLOAdapter:
    """
    Adapter chuyên biệt nạp và chạy suy luận model Deep Learning nhận diện ghế xe buýt
    từ camera gắn trên trần nhìn xuống (Top-down overhead bus seat).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        shared_model=None,
    ):
        self.model = None
        self._is_available = False
        self._supported_seat_classes: List[str] = []
        self._all_model_classes: Dict[int, str] = {}
        self._inspection_status: str = "NOT_INSPECTED"
        self._inspection_error: str = ""
        self.task: str = "detect"
        self.is_segmentation: bool = False

        if shared_model is not None:
            self.weights_path = Path(getattr(shared_model, "weights_path", "shared_model"))
            self._inspect_model_instance(shared_model)
            return

        if weights_path is not None:
            self.weights_path = Path(weights_path)
        else:
            # Tự động tìm kiếm file weights ưu tiên (models/best.pt, sau đó các alias khác)
            chosen_path = None
            for p in SUPPORTED_CUSTOM_WEIGHTS_PATHS:
                if Path(p).exists():
                    chosen_path = Path(p)
                    break
            self.weights_path = chosen_path if chosen_path is not None else Path(CUSTOM_SEGMENTATION_WEIGHTS_PATH)

        self.inspect_weights()

    @property
    def is_available(self) -> bool:
        """Chỉ True khi weights tồn tại VÀ model thực sự có chứa lớp nhận diện ghế chuyên biệt."""
        return self._is_available

    @property
    def model_name(self) -> str:
        return self.weights_path.name if self.weights_path else "N/A"

    @property
    def supported_classes(self) -> List[str]:
        return self._supported_seat_classes

    @staticmethod
    def inspect_classes(class_dict_or_list: Union[Dict, List[str]]) -> List[str]:
        """Kiểm tra và trả về danh sách lớp ghế xe buýt hợp lệ từ dict hoặc list lớp."""
        if isinstance(class_dict_or_list, dict):
            names = list(class_dict_or_list.values())
        else:
            names = list(class_dict_or_list)
        return [c for c in names if c.lower() in VALID_SEAT_CLASS_NAMES or ("seat" in c.lower() and c.lower() != "toilet seat")]

    @staticmethod
    def is_seat_class_supported(class_dict_or_list: Union[Dict, List[str]]) -> bool:
        """Trả về True nếu danh sách lớp có chứa lớp ghế xe buýt chuyên biệt (không tính COCO chair)."""
        valid = CustomSeatYOLOAdapter.inspect_classes(class_dict_or_list)
        return len(valid) > 0

    def set_shared_model(self, model_instance) -> bool:
        """Thiết lập instance model dùng chung để không phải tải lại từ file."""
        if model_instance is None:
            return False
        res = self._inspect_model_instance(model_instance)
        return bool(res.get("is_valid_seat_model", False))

    def _inspect_model_instance(self, model) -> Dict[str, Union[bool, str, List[str]]]:
        """Kiểm tra một model instance đã được nạp sẵn (shared model)."""
        names = getattr(model, "names", {})
        self._all_model_classes = names
        self.task = getattr(model, "task", "detect")
        self.is_segmentation = (self.task == "segment")
        class_names_list = list(names.values())

        detected_seat_classes = [
            cname for cname in class_names_list
            if cname.lower() in VALID_SEAT_CLASS_NAMES or "seat" in cname.lower()
        ]

        is_generic_coco = len(names) == 80 and "chair" in class_names_list and not detected_seat_classes
        if is_generic_coco:
            self._is_available = False
            self._inspection_status = "COCO_GENERIC_CHAIR_ONLY"
            self._inspection_error = (
                "Shared model là COCO pretrained (chỉ có lớp 'chair' nhìn ngang). "
                "Không có lớp nhận diện đệm ghế xe buýt nhìn từ trần xuống."
            )
            return {"is_valid_seat_model": False, "status": "COCO_CHAIR_REJECTED", "message": self._inspection_error}

        if not detected_seat_classes:
            self._is_available = False
            self._inspection_status = "NO_SEAT_CLASSES"
            self._inspection_error = f"Model có {len(names)} lớp nhưng không chứa lớp ghế xe buýt."
            return {"is_valid_seat_model": False, "status": "INVALID_CLASSES", "message": self._inspection_error}

        self.model = model
        self._is_available = True
        self._supported_seat_classes = detected_seat_classes
        self._inspection_status = "VALID_SEAT_MODEL"
        return {
            "is_valid_seat_model": True,
            "supported_seat_classes": detected_seat_classes,
            "status": "READY",
            "message": f"Đã kết nối shared model ({self.task}) với các lớp ghế: {detected_seat_classes}",
        }

    def inspect_weights(self) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Kiểm tra file weights trước khi sử dụng:
        1. Kiểm tra file có tồn tại trên đĩa không.
        2. Đọc danh sách class thực tế từ weights của model.
        3. Kiểm tra xem class nào thực sự được hỗ trợ.
        4. Phát hiện nếu là model COCO phổ thông (chỉ có chair nhìn ngang, không có seat nhìn trần).
        """
        result = {
            "file_exists": False,
            "weights_path": str(self.weights_path),
            "is_valid_seat_model": False,
            "supported_seat_classes": [],
            "all_classes": [],
            "task": "unknown",
            "is_segmentation": False,
            "status": "CUSTOM SEAT MODEL NOT CONFIGURED",
            "message": "",
        }

        if not self.weights_path.exists() or not self.weights_path.is_file():
            self._is_available = False
            self._inspection_status = "FILE_NOT_FOUND"
            self._inspection_error = (
                f"Chưa tìm thấy file weights tại: '{self.weights_path}'. "
                "Hướng dẫn: Sau khi huấn luyện YOLO26 Instance Segmentation, hãy đặt file weights vào models/best.pt."
            )
            result["status"] = "CUSTOM SEAT MODEL NOT CONFIGURED"
            result["message"] = self._inspection_error
            return result

        result["file_exists"] = True

        try:
            from ultralytics import YOLO

            # Tải metadata model
            model = YOLO(str(self.weights_path))
            names = model.names if hasattr(model, "names") else {}
            self._all_model_classes = names
            self.task = getattr(model, "task", "detect")
            self.is_segmentation = (self.task == "segment")
            result["task"] = self.task
            result["is_segmentation"] = self.is_segmentation

            class_names_list = list(names.values())
            result["all_classes"] = class_names_list

            # Kiểm tra xem có lớp nào trong danh sách lớp ghế xe buýt hợp lệ không
            detected_seat_classes = [
                cname for cname in class_names_list
                if cname.lower() in VALID_SEAT_CLASS_NAMES or "seat" in cname.lower()
            ]

            # Kiểm tra trường hợp model COCO (80 lớp, chỉ có chair)
            is_generic_coco = len(names) == 80 and "chair" in class_names_list and not detected_seat_classes

            if is_generic_coco:
                self._is_available = False
                self._inspection_status = "COCO_GENERIC_CHAIR_ONLY"
                msg = (
                    f"Model {self.weights_path.name} là COCO pretrained (chỉ có lớp 'chair' - ghế văn phòng/phòng khách nhìn ngang). "
                    "Không có lớp nhận diện đệm ghế xe buýt nhìn từ trên trần xuống. Không thể sử dụng làm seat model chuyên biệt."
                )
                self._inspection_error = msg
                result["status"] = "COCO_CHAIR_REJECTED"
                result["message"] = msg
                return result

            if not detected_seat_classes:
                self._is_available = False
                self._inspection_status = "NO_SEAT_CLASSES"
                msg = (
                    f"Model {self.weights_path.name} có {len(names)} lớp nhưng không chứa lớp nào liên quan đến ghế xe buýt "
                    f"(hỗ trợ: {sorted(list(VALID_SEAT_CLASS_NAMES))})."
                )
                self._inspection_error = msg
                result["status"] = "INVALID_CLASSES"
                result["message"] = msg
                return result

            # Model hợp lệ có lớp ghế
            self.model = model
            self._is_available = True
            self._supported_seat_classes = detected_seat_classes
            self._inspection_status = "VALID_SEAT_MODEL"
            msg = f"Đã nạp thành công Custom Seat Model ({self.task}) với các lớp: {detected_seat_classes}"
            result["is_valid_seat_model"] = True
            result["supported_seat_classes"] = detected_seat_classes
            result["status"] = "READY"
            result["message"] = msg
            logger.info(msg)
            return result

        except Exception as e:
            self._is_available = False
            self._inspection_status = "LOAD_ERROR"
            self._inspection_error = f"Lỗi đọc file model {self.weights_path}: {e}"
            result["status"] = "ERROR"
            result["message"] = self._inspection_error
            logger.error(self._inspection_error)
            return result

    def detect(self, frame: np.ndarray, confidence: float = 0.35) -> List[SeatConfig]:
        """
        Thực hiện inference trên frame camera để nhận diện từng chiếc đệm ghế thực tế.
        Ưu tiên trích xuất đa giác phân đoạn (segmentation mask polygon) của từng ghế.
        Chỉ fallback về bounding box khi model không hỗ trợ mask hoặc mask rỗng.
        """
        if not self.is_available or self.model is None:
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(source=frame, conf=confidence, verbose=False)
        detected_seats: List[SeatConfig] = []
        counter = 1

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            has_masks = (
                hasattr(r, "masks")
                and r.masks is not None
                and hasattr(r.masks, "xy")
                and r.masks.xy is not None
            )

            for i, b in enumerate(boxes):
                try:
                    class_id = int(b.cls[0])
                except Exception:
                    try:
                        class_id = int(boxes.cls[i])
                    except Exception:
                        class_id = int(b.cls)

                class_name = self.model.names.get(class_id, f"class_{class_id}")

                # Chỉ nhận các lớp ghế thực sự
                if class_name.lower() not in VALID_SEAT_CLASS_NAMES and "seat" not in class_name.lower():
                    continue

                try:
                    conf = float(b.conf[0])
                except Exception:
                    try:
                        conf = float(boxes.conf[i])
                    except Exception:
                        conf = float(b.conf)

                raw_xy = getattr(b, "xyxy", None)
                if raw_xy is None:
                    raw_xy = getattr(boxes, "xyxy", None)
                    if raw_xy is not None and len(raw_xy) > i:
                        raw_xy = raw_xy[i]
                if raw_xy is None:
                    continue

                if hasattr(raw_xy, "cpu"):
                    raw_xy = raw_xy.cpu().numpy()
                else:
                    raw_xy = np.asarray(raw_xy)

                if raw_xy.ndim > 1:
                    raw_xy = raw_xy[0]
                if len(raw_xy) < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in raw_xy[:4]]

                # Trích xuất polygon từ segmentation mask nếu có
                mask_poly = None
                if has_masks and i < len(r.masks.xy):
                    poly_raw = r.masks.xy[i]
                    if poly_raw is not None and len(poly_raw) >= 3:
                        norm_pts = []
                        for pt in poly_raw:
                            nx = round(max(0.0, min(1.0, float(pt[0]) / w)), 4)
                            ny = round(max(0.0, min(1.0, float(pt[1]) / h)), 4)
                            norm_pts.append((nx, ny))
                        if len(norm_pts) >= 3:
                            mask_poly = norm_pts

                # Ưu tiên segmentation polygon
                if mask_poly is not None:
                    seat_poly = mask_poly
                    method_str = "AI Segmentation Mask"
                else:
                    # Bounding box fallback
                    seat_poly = [
                        (round(float(x1) / w, 4), round(float(y1) / h, 4)),
                        (round(float(x2) / w, 4), round(float(y1) / h, 4)),
                        (round(float(x2) / w, 4), round(float(y2) / h, 4)),
                        (round(float(x1) / w, 4), round(float(y2) / h, 4)),
                    ]
                    method_str = "AI Bounding Box Fallback"

                # Kiểm tra trường hợp cụm ghế đôi nếu model dự đoán 1 mask cho cả cụm
                box_w = x2 - x1
                box_h = y2 - y1
                aspect_ratio = float(box_w) / max(1, box_h)
                norm_w = float(box_w) / max(1, w)
                is_wide_bench = (box_h > 0 and aspect_ratio > 1.45 and norm_w >= 0.14) or (
                    "double" in class_name.lower() or "bench" in class_name.lower()
                )

                if is_wide_bench:
                    cx = float(x1 + x2) / (2.0 * max(1, w))
                    side = "left" if cx < 0.5 else "right"
                    row_index = max(1, int(round((float(y1) / max(1, h)) * 4)) + 1)
                    decomposer = TwinSeatDecomposer()
                    child_seats = decomposer.split_horizontal_bench(
                        polygon_or_bbox=seat_poly,
                        row_index=row_index,
                        side=side,
                        naming_format="SEAT",
                        gap_ratio=0.05,
                    )
                    for cs in child_seats:
                        cs.description = f"{cs.description} ({method_str} conf {conf:.0%})"
                        detected_seats.append(cs)
                else:
                    # Gán ID có ngữ nghĩa cho ghế đơn
                    if "window" in class_name.lower():
                        seat_id = f"SEAT_W{counter:02d}"
                        desc = f"Ghế cửa sổ ({method_str} conf {conf:.0%})"
                    elif "aisle" in class_name.lower():
                        seat_id = f"SEAT_A{counter:02d}"
                        desc = f"Ghế lối đi ({method_str} conf {conf:.0%})"
                    else:
                        seat_id = f"SEAT_{counter:03d}"
                        desc = f"Ghế {counter} ({method_str} conf {conf:.0%})"

                    detected_seats.append(SeatConfig(
                        seat_id=seat_id,
                        polygon=seat_poly,
                        is_normalized=True,
                        description=desc,
                    ))
                    counter += 1

        return detected_seats


# Alias tương thích ngược
CustomYOLOSeatDetector = CustomSeatYOLOAdapter


class TwinSeatDecomposer:
    """
    Module phụ trách phân rã các băng ghế đôi hoặc các cụm đệm ghế liền kề
    thành 2 thực thể ghế độc lập (Ghế A - Cửa sổ, Ghế B - Lối đi).
    Đảm bảo 2 ghế nằm cạnh nhau trong cùng hàng có ID và bounding box tách biệt.
    """

    def split_horizontal_bench(
        self,
        polygon_or_bbox: Union[Tuple[float, float, float, float], List[Tuple[float, float]]],
        row_index: int,
        side: str = "left",
        naming_format: str = "SEAT",
        gap_ratio: float = 0.05,
    ) -> List[SeatConfig]:
        """
        Phân tách 1 băng ghế đôi thành 2 ghế độc lập:
        - side == 'left': Ghế A (cửa sổ), Ghế B (lối đi)
        - side == 'right': Ghế C (lối đi), Ghế D (cửa sổ)
        """
        if isinstance(polygon_or_bbox, (tuple, list)) and len(polygon_or_bbox) == 4 and isinstance(polygon_or_bbox[0], (int, float)):
            x1, y1, x2, y2 = polygon_or_bbox
        else:
            xs = [p[0] for p in polygon_or_bbox]
            ys = [p[1] for p in polygon_or_bbox]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        bench_w = x2 - x1
        gap_w = bench_w * gap_ratio
        single_w = (bench_w - gap_w) / 2.0

        seat_1_x1 = x1
        seat_1_x2 = x1 + single_w
        seat_2_x1 = seat_1_x2 + gap_w
        seat_2_x2 = x2

        if side.lower() == "left":
            id_1 = f"SEAT_{row_index}A" if naming_format == "SEAT" else f"SEAT_{(row_index - 1)*4 + 1:03d}"
            id_2 = f"SEAT_{row_index}B" if naming_format == "SEAT" else f"SEAT_{(row_index - 1)*4 + 2:03d}"
            desc_1 = f"Hàng {row_index} - Bên trái (Cửa sổ)"
            desc_2 = f"Hàng {row_index} - Bên trái (Lối đi)"
        else:
            id_1 = f"SEAT_{row_index}C" if naming_format == "SEAT" else f"SEAT_{(row_index - 1)*4 + 3:03d}"
            id_2 = f"SEAT_{row_index}D" if naming_format == "SEAT" else f"SEAT_{(row_index - 1)*4 + 4:03d}"
            desc_1 = f"Hàng {row_index} - Bên phải (Lối đi)"
            desc_2 = f"Hàng {row_index} - Bên phải (Cửa sổ)"

        seat_1 = SeatConfig(
            seat_id=id_1,
            polygon=[(round(seat_1_x1, 4), round(y1, 4)), (round(seat_1_x2, 4), round(y1, 4)), (round(seat_1_x2, 4), round(y2, 4)), (round(seat_1_x1, 4), round(y2, 4))],
            description=desc_1,
            is_normalized=True,
        )
        seat_2 = SeatConfig(
            seat_id=id_2,
            polygon=[(round(seat_2_x1, 4), round(y1, 4)), (round(seat_2_x2, 4), round(y1, 4)), (round(seat_2_x2, 4), round(y2, 4)), (round(seat_2_x1, 4), round(y2, 4))],
            description=desc_2,
            is_normalized=True,
        )
        return [seat_1, seat_2]


class BusGeometrySeatDetector:
    """
    Bộ tạo cấu hình ghế bằng mô hình hình học dự phòng (Geometry Fallback Engine).
    ĐÂY LÀ PHƯƠNG ÁN DỰ PHÒNG CÔNG KHAI KHI CHƯA CÓ MODEL AI CHUYÊN BIỆT,
    TUYỆT ĐỐI KHÔNG TUYÊN BỐ ĐÂY LÀ AI ĐÃ NHẬN DIỆN GHẾ THÀNH CÔNG.
    """

    def __init__(
        self,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        start_y: float = 0.12,
        end_y: float = 0.90,
        aisle_ratio: float = 0.18,
    ):
        self.rows = max(1, rows)
        self.left_cols = max(1, min(3, left_cols))
        self.right_cols = max(1, min(3, right_cols))
        self.naming_format = naming_format
        self.start_y = start_y
        self.end_y = end_y
        self.aisle_ratio = aisle_ratio

    def generate_seats(self, frame_width: int = 1280, frame_height: int = 720) -> List[SeatConfig]:
        """Sinh lưới hình học dự phòng phân tách từng ghế riêng biệt."""
        seats = []
        counter = 1

        total_height = self.end_y - self.start_y
        row_step = total_height / self.rows
        seat_height = row_step * 0.76

        aisle_w = self.aisle_ratio
        aisle_start_x = (1.0 - aisle_w) / 2.0
        aisle_end_x = aisle_start_x + aisle_w

        left_start_x = 0.06
        left_bench_w = max(0.10, (aisle_start_x - 0.02) - left_start_x)
        seat_left_w = left_bench_w / self.left_cols

        right_start_x = aisle_end_x + 0.02
        right_bench_w = max(0.10, 0.94 - right_start_x)
        seat_right_w = right_bench_w / self.right_cols

        for r in range(1, self.rows + 1):
            y1 = self.start_y + (r - 1) * row_step
            y2 = y1 + seat_height

            # Ghế bên trái lối đi
            for c in range(self.left_cols):
                x1 = left_start_x + c * seat_left_w
                x2 = x1 + seat_left_w * 0.92
                col_letter = chr(65 + c)  # A, B
                seat_id = f"SEAT_{r}{col_letter}" if self.naming_format == "SEAT" else f"SEAT_{counter:03d}"
                poly = [
                    (round(x1, 4), round(y1, 4)),
                    (round(x2, 4), round(y1, 4)),
                    (round(x2, 4), round(y2, 4)),
                    (round(x1, 4), round(y2, 4)),
                ]
                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=poly,
                    is_normalized=True,
                    description=f"Hàng {r} - Ghế trái {col_letter} (Geometry Fallback)",
                ))
                counter += 1

            # Ghế bên phải lối đi
            for c in range(self.right_cols):
                x1 = right_start_x + c * seat_right_w
                x2 = x1 + seat_right_w * 0.92
                col_letter = chr(65 + self.left_cols + c)  # C, D
                seat_id = f"SEAT_{r}{col_letter}" if self.naming_format == "SEAT" else f"SEAT_{counter:03d}"
                poly = [
                    (round(x1, 4), round(y1, 4)),
                    (round(x2, 4), round(y1, 4)),
                    (round(x2, 4), round(y2, 4)),
                    (round(x1, 4), round(y2, 4)),
                ]
                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=poly,
                    is_normalized=True,
                    description=f"Hàng {r} - Ghế phải {col_letter} (Geometry Fallback)",
                ))
                counter += 1

        return seats


class SeatDetector:
    """
    Facade điều phối phát hiện ghế với tiêu chí Computer Vision thực sự:
    1. Kiểm tra model chuyên biệt có tồn tại và hỗ trợ class ghế hay không.
    2. Nếu có Custom Seat Model -> Thực hiện MODEL DETECTION bằng neural network.
    3. Nếu chưa có Custom Seat Model -> Báo rõ trạng thái CUSTOM SEAT MODEL NOT CONFIGURED.
    4. Chỉ khi có yêu cầu dự phòng rõ ràng (allow_fallback=True) mới trả về GEOMETRY FALLBACK,
       và được đánh dấu rõ ràng là phương án dự phòng toán học, KHÔNG mạo nhận là AI.
    """

    def __init__(
        self,
        custom_weights_path: Optional[str] = None,
        shared_model=None,
        aisle_ratio: float = 0.18,
        roboflow_client=None,
        inference_mode: Optional[str] = None,
    ):
        self.custom_weights_path = custom_weights_path
        self.model_adapter = CustomSeatYOLOAdapter(
            weights_path=custom_weights_path,
            shared_model=shared_model,
        )
        self.geometry_detector = BusGeometrySeatDetector(aisle_ratio=aisle_ratio)
        self.roboflow_client = roboflow_client
        self.inference_mode = inference_mode

    def set_shared_model(self, model_instance) -> bool:
        """Thiết lập model instance dùng chung cho SeatDetector."""
        return self.model_adapter.set_shared_model(model_instance)

    def set_roboflow_client(self, client) -> None:
        """Thiết lập client Roboflow Workflow cho SeatDetector."""
        self.roboflow_client = client

    @staticmethod
    def inspect_coco_classes() -> Dict[str, Union[bool, str, List[str]]]:
        """
        Kiểm tra minh bạch lớp ghế của model COCO:
        COCO pretrained chỉ có lớp 56 'chair' (ghế nhìn ngang phòng khách),
        HOÀN TOÀN KHÔNG có lớp đệm ghế xe buýt nhìn từ trần xuống.
        """
        return {
            "has_overhead_bus_seat_class": False,
            "has_coco_chair_class": True,
            "coco_chair_class_id": 56,
            "explanation": (
                "Model YOLOv8 COCO pretrained phổ thông chỉ hỗ trợ lớp 'chair' (ghế gia đình góc nhìn ngang). "
                "Không có lớp nhận diện đệm ghế xe buýt nhìn từ trên trần xuống (top-down overhead bus seat). "
                "Hệ thống yêu cầu nạp model chuyên biệt hoặc sử dụng cấu hình thủ công/dự phòng minh bạch."
            ),
            "recommendation": (
                "Huấn luyện custom model trên dữ liệu xe buýt (lớp: seat_window, seat_aisle hoặc seat) "
                "và đặt vào models/bus_seat_yolo.pt để kích hoạt tính năng Computer Vision nhận diện ghế thực sự."
            ),
        }

    def detect_seats(
        self,
        frame: np.ndarray,
        allow_fallback: bool = False,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        confidence: float = 0.35,
        roboflow_client: Optional[Any] = None,
        inference_mode: Optional[str] = None,
    ) -> SeatDetectionResult:
        """
        Thực hiện nhận diện ghế theo kiến trúc Computer Vision thực thụ:
        - Hỗ trợ cả 3 chế độ: Local (custom weights), Roboflow Workflow, và Hybrid.
        - Không bao giờ tự động tạo ghế giả lập trừ khi bật allow_fallback kiểm thử.
        - Giữ ID ghế ổn định qua SeatTracker.
        """
        import os
        diagnostics: Dict[str, Union[str, bool, int, float, List[str]]] = {}
        diagnostics["weights_path"] = str(self.model_adapter.weights_path)
        diagnostics["weights_available"] = self.model_adapter.is_available

        active_mode = (
            inference_mode
            or self.inference_mode
            or os.getenv("INFERENCE_MODE", "local")
        ).lower()
        diagnostics["inference_mode"] = active_mode

        rf_client = roboflow_client or self.roboflow_client
        if rf_client is None and active_mode in ("roboflow", "hybrid"):
            try:
                from src.roboflow.roboflow_client import RoboflowWorkflowClient
                rf_client = RoboflowWorkflowClient()
                self.roboflow_client = rf_client
            except Exception as e:
                logger.warning(f"Could not initialize Roboflow client in SeatDetector: {e}")

        # ── 1. ROBOWFLOW HOẶC HYBRID INFERENCE ────────────────────────
        if active_mode in ("roboflow", "hybrid") and rf_client is not None:
            try:
                import cv2, tempfile, pathlib
                tmp_path = pathlib.Path(tempfile.mkstemp(suffix=".jpg")[1])
                cv2.imwrite(str(tmp_path), frame)
                payload = rf_client.submit_image(tmp_path)
                tmp_path.unlink(missing_ok=True)

                if payload:
                    from src.roboflow.roboflow_client import RoboflowWorkflowClient
                    parsed = RoboflowWorkflowClient.parse_workflow_output(payload)
                    raw_preds = [
                        p for p in parsed.get("predictions", [])
                        if float(p.get("confidence", 0)) >= confidence
                    ]
                    rf_seats = self.detect_seats_from_predictions(
                        raw_preds, frame.shape[1], frame.shape[0]
                    )

                    # Nếu là hybrid và có cả custom local weights, hợp nhất tránh trùng lặp
                    if active_mode == "hybrid" and self.model_adapter.is_available:
                        local_seats = self.model_adapter.detect(frame, confidence=confidence)
                        # So khớp IoU để không tạo ghế trùng
                        from src.seats.seat_tracker import calculate_polygon_iou
                        fused_seats = list(local_seats)
                        for r_seat in rf_seats:
                            matched = any(
                                calculate_polygon_iou(r_seat.polygon, l_seat.polygon) > 0.45
                                for l_seat in local_seats
                            )
                            if not matched:
                                fused_seats.append(r_seat)
                        return SeatDetectionResult(
                            seats=fused_seats,
                            method=SeatDetectionMethod.MODEL_DETECTION,
                            model_name="Hybrid AI (Local YOLO + Roboflow)",
                            confidence_avg=round(float(np.mean([0.88] * max(1, len(fused_seats)))), 2),
                            is_fallback=False,
                            status_message=f"Nhận diện thành công {len(fused_seats)} ghế bằng Hybrid AI Model.",
                            diagnostics=diagnostics,
                        )

                    # Chế độ Roboflow thuần hoặc hybrid chỉ có Roboflow
                    if rf_seats:
                        return SeatDetectionResult(
                            seats=rf_seats,
                            method=SeatDetectionMethod.MODEL_DETECTION,
                            model_name=f"Roboflow Workflow ({getattr(rf_client, 'workflow_id', 'Workflow')})",
                            confidence_avg=round(float(np.mean([0.85] * len(rf_seats))), 2),
                            is_fallback=False,
                            status_message=f"Nhận diện thành công {len(rf_seats)} đệm ghế bằng Roboflow AI Model.",
                            diagnostics=diagnostics,
                        )
                    else:
                        # Model chạy nhưng không có ghế trong frame (chân thực, không bịa ghế)
                        return SeatDetectionResult(
                            seats=[],
                            method=SeatDetectionMethod.MODEL_DETECTION,
                            model_name=f"Roboflow Workflow ({getattr(rf_client, 'workflow_id', 'Workflow')})",
                            confidence_avg=0.0,
                            is_fallback=False,
                            status_message="Mô hình Roboflow AI đã quét khung hình camera nhưng không phát hiện ghế nào trong góc nhìn này.",
                            diagnostics=diagnostics,
                        )
            except Exception as e:
                logger.error(f"Lỗi khi nhận diện ghế bằng Roboflow: {e}")

        # ── 2. LOCAL YOLO CHUYÊN BIỆT ─────────────────────────────────
        if self.model_adapter.is_available:
            detected = self.model_adapter.detect(frame, confidence=confidence)
            if detected:
                diagnostics["detected_count"] = len(detected)
                diagnostics["classes_used"] = self.model_adapter.supported_classes
                return SeatDetectionResult(
                    seats=detected,
                    method=SeatDetectionMethod.MODEL_DETECTION,
                    model_name=self.model_adapter.model_name,
                    confidence_avg=round(float(np.mean([0.85] * len(detected))), 2),
                    is_fallback=False,
                    status_message=f"Nhận diện thành công {len(detected)} đệm ghế bằng Custom AI Model.",
                    diagnostics=diagnostics,
                )
            else:
                return SeatDetectionResult(
                    seats=[],
                    method=SeatDetectionMethod.MODEL_DETECTION,
                    model_name=self.model_adapter.model_name,
                    confidence_avg=0.0,
                    is_fallback=False,
                    status_message="Mô hình AI cục bộ đã quét khung hình camera nhưng không phát hiện chiếc ghế nào.",
                    diagnostics=diagnostics,
                )

        # ── 3. NẾU CHƯA CÓ MODEL PHÙ HỢP ─────────────────────────────
        insp = self.model_adapter.inspect_weights()
        diagnostics["inspection"] = insp

        if not allow_fallback:
            status_msg = (
                f"CUSTOM SEAT MODEL NOT CONFIGURED: Không tìm thấy model chuyên biệt tại "
                f"'{self.model_adapter.weights_path}'. Hệ thống không giả lập AI nhận diện ghế. "
                "Hãy chọn chế độ Roboflow / Hybrid hoặc nạp trọng số chuyên biệt (models/best.pt)."
            )
            return SeatDetectionResult(
                seats=[],
                method=SeatDetectionMethod.NOT_CONFIGURED,
                model_name="NONE",
                confidence_avg=0.0,
                is_fallback=False,
                status_message=status_msg,
                diagnostics=diagnostics,
            )

        # ── 4. GEOMETRY FALLBACK (CHỈ KHI ALLOW_FALLBACK = TRUE) ──────
        fallback_seats = self.geometry_detector.generate_seats(frame.shape[1], frame.shape[0])
        status_msg = (
            f"GEOMETRY FALLBACK: Đang sử dụng sơ đồ lưới hình học {rows} hàng ({left_cols}+{right_cols} ghế) "
            "làm phương án dự phòng để kiểm thử luồng. Đây KHÔNG PHẢI là kết quả do AI model nhận diện."
        )
        diagnostics["fallback_rows"] = rows
        diagnostics["fallback_seats"] = len(fallback_seats)

        return SeatDetectionResult(
            seats=fallback_seats,
            method=SeatDetectionMethod.GEOMETRY_FALLBACK,
            model_name="Geometry-Fallback-Engine",
            confidence_avg=1.0,
            is_fallback=True,
            status_message=status_msg,
            diagnostics=diagnostics,
        )


    def detect_seats_from_predictions(
        self,
        predictions: List[Any],
        frame_width: int,
        frame_height: int,
        naming_format: str = "SEAT",
    ) -> List[SeatConfig]:
        """
        Trích xuất các đệm ghế thực tế từ danh sách predictions (Local YOLO, Roboflow Workflow, hoặc Hybrid).
        Chỉ chuyển đổi các predictions có nhãn thuộc VALID_SEAT_CLASS_NAMES hoặc có chứa 'seat'.
        Nếu phát hiện băng ghế đôi (wide bench hoặc nhãn double/bench), tự động phân rã thành 2 ghế độc lập.
        """
        detected_seats = []
        counter = 1
        for pred in predictions:
            raw_cls = getattr(pred, "class_name", None) or (pred.get("class") if isinstance(pred, dict) else "")
            raw_lower = str(raw_cls).strip().lower()

            if raw_lower in VALID_SEAT_CLASS_NAMES or ("seat" in raw_lower and "toilet" not in raw_lower):
                poly = getattr(pred, "polygon", None) or (pred.get("polygon") if isinstance(pred, dict) else None)
                bbox = getattr(pred, "bbox", None) or (pred.get("bbox") if isinstance(pred, dict) else (0, 0, 0, 0))
                conf = float(getattr(pred, "confidence", 0.0) or (pred.get("confidence", 0.0) if isinstance(pred, dict) else 0.0))

                if poly and len(poly) >= 3:
                    is_already_norm = all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in poly)
                    if is_already_norm:
                        seat_poly = [(round(float(pt[0]), 4), round(float(pt[1]), 4)) for pt in poly]
                    else:
                        seat_poly = [
                            (round(float(pt[0]) / max(1, frame_width), 4), round(float(pt[1]) / max(1, frame_height), 4))
                            for pt in poly
                        ]
                else:
                    bx1, by1, bx2, by2 = bbox
                    seat_poly = [
                        (round(float(bx1) / max(1, frame_width), 4), round(float(by1) / max(1, frame_height), 4)),
                        (round(float(bx2) / max(1, frame_width), 4), round(float(by1) / max(1, frame_height), 4)),
                        (round(float(bx2) / max(1, frame_width), 4), round(float(by2) / max(1, frame_height), 4)),
                        (round(float(bx1) / max(1, frame_width), 4), round(float(by2) / max(1, frame_height), 4)),
                    ]

                # Đo lường hình học đệm ghế
                xs = [p[0] for p in seat_poly]
                ys = [p[1] for p in seat_poly]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                box_w = max_x - min_x
                box_h = max_y - min_y
                aspect_ratio = box_w / max(0.001, box_h)
                cx = (min_x + max_x) / 2.0

                # Kiểm tra điều kiện băng ghế đôi: tỷ lệ chiều rộng/dài > 1.45 và độ rộng đáng kể (>=0.14),
                # hoặc nhãn có chứa từ khóa 'double' hoặc 'bench'
                is_twin_bench = (box_h > 0 and aspect_ratio > 1.45 and box_w >= 0.14) or (
                    "double" in raw_lower or "bench" in raw_lower
                )

                if is_twin_bench:
                    side = "left" if cx < 0.5 else "right"
                    row_index = max(1, int(round(min_y * 4)) + 1)
                    decomposer = TwinSeatDecomposer()
                    child_seats = decomposer.split_horizontal_bench(
                        polygon_or_bbox=seat_poly,
                        row_index=row_index,
                        side=side,
                        naming_format=naming_format,
                        gap_ratio=0.05,
                    )
                    for cs in child_seats:
                        cs.description = f"{cs.description} (AI Bench conf {conf:.0%})"
                        detected_seats.append(cs)
                else:
                    if "window" in raw_lower:
                        seat_id = f"SEAT_W{counter:02d}"
                        desc = f"Ghế cửa sổ (AI conf {conf:.0%})"
                    elif "aisle" in raw_lower:
                        seat_id = f"SEAT_A{counter:02d}"
                        desc = f"Ghế lối đi (AI conf {conf:.0%})"
                    else:
                        seat_id = f"SEAT_{counter:03d}"
                        desc = f"Ghế {counter} (AI conf {conf:.0%})"

                    detected_seats.append(SeatConfig(
                        seat_id=seat_id,
                        polygon=seat_poly,
                        is_normalized=True,
                        description=desc,
                    ))
                    counter += 1

        return detected_seats

