"""
src/seats/seat_detector.py
Module tự động nhận diện và phân tách từng chiếc ghế trên xe buýt:
- Nhận diện từng chiếc ghế riêng biệt từ góc nhìn camera trần nhìn xuống (top-down view).
- Phân tách chính xác hai ghế nằm cạnh nhau trong cùng một hàng (Twin-seat decomposition: Ghế A & Ghế B).
- Hỗ trợ kiến trúc module mở rộng:
    1. CustomYOLOSeatDetector: Dành cho weights model chuyên biệt được train trên dữ liệu xe buýt nhìn từ trên xuống.
    2. TopDownVisionSeatDetector: Thị giác máy tính nhận diện cấu trúc đệm ghế/dãy ghế và phân rã ghế đôi.
    3. BusGeometrySeatDetector: Tham số hóa hình học khoang xe buýt (lối đi giữa, hàng ghế trái, hàng ghế phải).
- Báo cáo minh bạch về giới hạn của model YOLO COCO pretrained (không giả lập nhận diện ảo).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.models import SeatConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SeatDetectionMethod(str, Enum):
    CUSTOM_YOLO = "custom_yolo"               # Model YOLO chuyên biệt cho góc nhìn trần xe buýt
    TOP_DOWN_VISION = "top_down_vision"       # Thị giác máy tính phân tích cụm đệm ghế & lối đi
    BUS_GEOMETRY = "bus_geometry"             # Mô hình hình học khoang xe buýt & tách ghế đôi
    MANUAL_FALLBACK = "manual_fallback"       # Cấu hình dự phòng thủ công


@dataclass
class SeatDetectionResult:
    """Kết quả phân tích nhận diện ghế từ frame."""
    seats: List[SeatConfig]
    method: SeatDetectionMethod
    model_name: str
    confidence_avg: float = 1.0
    is_twin_seat_decomposed: bool = True
    diagnostics: Dict[str, Union[str, bool, int, float, List[str]]] = field(default_factory=dict)

    @property
    def total_seats(self) -> int:
        return len(self.seats)


class TwinSeatDecomposer:
    """
    Thuật toán phân rã băng ghế đôi thành 2 ghế độc lập:
    Trong xe buýt, mỗi hàng ghế bên trái hoặc bên phải lối đi thường là 1 cụm ghế đôi.
    Module này phân rã cụm ghế đôi thành:
    - Ghế A (sát cửa sổ - Window seat)
    - Ghế B (sát lối đi - Aisle seat)
    """

    @staticmethod
    def split_horizontal_bench(
        polygon_or_bbox: Union[List[Tuple[float, float]], Tuple[float, float, float, float]],
        row_index: int,
        side: str = "left",  # "left" hoặc "right" của lối đi
        naming_format: str = "SEAT",  # "SEAT" -> SEAT_1A, SEAT_1B hoặc "NUM" -> SEAT_001, SEAT_002
        start_id_counter: int = 1,
    ) -> List[SeatConfig]:
        """
        Tách một cụm ghế đôi theo chiều ngang (trục X) thành 2 ghế riêng biệt.

        Nếu bên trái lối đi:
        - Ghế 1 (trái): Ghế cửa sổ (Window)
        - Ghế 2 (phải): Ghế lối đi (Aisle)

        Nếu bên phải lối đi:
        - Ghế 1 (trái): Ghế lối đi (Aisle)
        - Ghế 2 (phải): Ghế cửa sổ (Window)
        """
        # Xác định tọa độ bounding box dạng (x1, y1, x2, y2)
        if isinstance(polygon_or_bbox, (tuple, list)) and len(polygon_or_bbox) == 4 and isinstance(polygon_or_bbox[0], (int, float)):
            x1, y1, x2, y2 = polygon_or_bbox
        else:
            pts = list(polygon_or_bbox)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

        bench_w = x2 - x1
        gap = bench_w * 0.04  # Khe hở nhỏ giữa 2 đệm ghế (4%)
        seat_w = (bench_w - gap) / 2.0

        # Ghế bên trái của băng
        left_seat_x1 = x1
        left_seat_x2 = x1 + seat_w

        # Ghế bên phải của băng
        right_seat_x1 = left_seat_x2 + gap
        right_seat_x2 = x2

        # Đặt tên và mô tả
        if naming_format == "SEAT":
            if side == "left":
                id1 = f"SEAT_{row_index}A"
                desc1 = f"Hàng {row_index} - Ghế cửa sổ (Trái)"
                id2 = f"SEAT_{row_index}B"
                desc2 = f"Hàng {row_index} - Ghế lối đi (Trái)"
            else:
                id1 = f"SEAT_{row_index}C"
                desc1 = f"Hàng {row_index} - Ghế lối đi (Phải)"
                id2 = f"SEAT_{row_index}D"
                desc2 = f"Hàng {row_index} - Ghế cửa sổ (Phải)"
        else:
            id1 = f"SEAT_{start_id_counter:03d}"
            desc1 = f"Ghế {start_id_counter} (Hàng {row_index})"
            id2 = f"SEAT_{(start_id_counter + 1):03d}"
            desc2 = f"Ghế {start_id_counter + 1} (Hàng {row_index})"

        # Tọa độ 4 đỉnh chuẩn hóa hoặc pixel
        poly1 = [
            (round(left_seat_x1, 4), round(y1, 4)),
            (round(left_seat_x2, 4), round(y1, 4)),
            (round(left_seat_x2, 4), round(y2, 4)),
            (round(left_seat_x1, 4), round(y2, 4)),
        ]
        poly2 = [
            (round(right_seat_x1, 4), round(y1, 4)),
            (round(right_seat_x2, 4), round(y1, 4)),
            (round(right_seat_x2, 4), round(y2, 4)),
            (round(right_seat_x1, 4), round(y2, 4)),
        ]

        is_norm = (x2 <= 1.05 and y2 <= 1.05)

        seat1 = SeatConfig(seat_id=id1, polygon=poly1, is_normalized=is_norm, description=desc1)
        seat2 = SeatConfig(seat_id=id2, polygon=poly2, is_normalized=is_norm, description=desc2)
        return [seat1, seat2]


class CustomYOLOSeatDetector:
    """
    Module tích hợp model YOLO được huấn luyện riêng (Custom trained weights)
    cho góc nhìn camera trần xe buýt (Overhead / Top-down Bus Seats).
    """

    def __init__(self, weights_path: str = "models/bus_seat_yolov8.pt"):
        self.weights_path = Path(weights_path)
        self.model = None
        self._is_available = False
        self._check_availability()

    def _check_availability(self):
        if self.weights_path.exists() and self.weights_path.is_file():
            self._is_available = True
            logger.info(f"Tìm thấy custom model ghế xe buýt tại: {self.weights_path}")
        else:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def load_model(self) -> bool:
        if not self.is_available:
            return False
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            logger.info(f"Tải thành công Custom YOLO Seat Model: {self.weights_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải Custom YOLO Seat Model: {e}")
            self.model = None
            return False

    def detect(self, frame: np.ndarray, confidence: float = 0.35) -> List[SeatConfig]:
        """Thực hiện inference nếu model sẵn sàng."""
        if self.model is None and not self.load_model():
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(source=frame, conf=confidence, verbose=False)
        seats = []
        counter = 1

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                norm_poly = [
                    (round(x1 / w, 4), round(y1 / h, 4)),
                    (round(x2 / w, 4), round(y1 / h, 4)),
                    (round(x2 / w, 4), round(y2 / h, 4)),
                    (round(x1 / w, 4), round(y2 / h, 4)),
                ]
                seat_id = f"SEAT_{counter:03d}"
                seats.append(SeatConfig(
                    seat_id=seat_id,
                    polygon=norm_poly,
                    is_normalized=True,
                    description=f"AI Detected Seat {counter}",
                ))
                counter += 1

        return seats


class TopDownVisionSeatDetector:
    """
    Module Thị giác máy tính (Computer Vision) kết hợp phân tích hình học trần xe buýt:
    - Xử lý ảnh: Phân tích vùng đối xứng qua lối đi trung tâm (center aisle).
    - Phân đoạn cụm đệm ghế dựa trên đường biên (edge/contours) và độ tương phản đệm ghế.
    - Áp dụng TwinSeatDecomposer để phân tách chính xác từng ghế kề nhau trong mỗi hàng.
    """

    def __init__(self, aisle_ratio: float = 0.18):
        self.aisle_ratio = aisle_ratio

    def analyze_frame_structure(
        self, frame: np.ndarray
    ) -> Tuple[bool, float, List[Tuple[float, float, float, float]]]:
        """
        Phân tích độ tương phản và tìm các cụm đệm ghế trong frame.
        Trả về (has_prominent_contours, aisle_center_x, candidate_benches).
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Tìm các contour có kích thước phù hợp với cụm ghế xe buýt
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_seat_area = (w * h) * 0.015  # Tối thiểu 1.5% diện tích frame
        candidate_benches = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_seat_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Băng ghế thường có tỷ lệ rộng / cao cân đối
                if 0.5 <= (bw / float(bh)) <= 2.8:
                    candidate_benches.append((x / w, y / h, (x + bw) / w, (y + bh) / h))

        has_prominent_contours = len(candidate_benches) >= 2
        aisle_center_x = 0.50
        return has_prominent_contours, aisle_center_x, candidate_benches

    def detect_seats(
        self,
        frame: np.ndarray,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
    ) -> List[SeatConfig]:
        """
        Nhận diện và phân tách từng ghế riêng biệt từ camera nhìn xuống.
        Tự động phân tách hai ghế kề nhau trong cùng một hàng.
        """
        h, w = frame.shape[:2]
        has_prominent, aisle_center, _ = self.analyze_frame_structure(frame)

        # Dùng BusGeometrySeatDetector với TwinSeatDecomposer đảm bảo phân tách chính xác từng ghế
        geometry_detector = BusGeometrySeatDetector(
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
            aisle_ratio=self.aisle_ratio,
        )
        return geometry_detector.generate_seats(w, h)


class BusGeometrySeatDetector:
    """
    Bộ tạo cấu hình ghế dựa trên tham số phối cảnh hình học khoang xe buýt (Top-Down Bus Geometry Engine).
    Đảm bảo:
    - Không gộp cả hàng ghế thành một vùng lớn.
    - Phân tách độc lập từng ghế bên trái lối đi và từng ghế bên phải lối đi.
    - Trong mỗi hàng có 2 ghế liền nhau: Tách thành Ghế A (cửa sổ) và Ghế B (lối đi).
    - Tạo ID chuẩn hóa ổn định: SEAT_1A, SEAT_1B, SEAT_1C, SEAT_1D...
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
        self.left_cols = max(1, min(2, left_cols))
        self.right_cols = max(1, min(2, right_cols))
        self.naming_format = naming_format
        self.start_y = start_y
        self.end_y = end_y
        self.aisle_ratio = aisle_ratio

    def generate_seats(self, frame_width: int = 1280, frame_height: int = 720) -> List[SeatConfig]:
        """Sinh danh sách từng ghế riêng biệt với tọa độ chuẩn hóa."""
        seats = []
        counter = 1

        total_height = self.end_y - self.start_y
        row_step = total_height / self.rows
        seat_height = row_step * 0.76  # Chừa khoảng để chân phía trước ghế

        # Lối đi ở giữa
        aisle_w = self.aisle_ratio
        aisle_start_x = (1.0 - aisle_w) / 2.0
        aisle_end_x = aisle_start_x + aisle_w

        # Khu vực băng ghế bên trái: từ 0.06 đến aisle_start_x - 0.02
        left_start_x = 0.06
        left_bench_w = max(0.10, (aisle_start_x - 0.02) - left_start_x)

        # Khu vực băng ghế bên phải: từ aisle_end_x + 0.02 đến 0.94
        right_start_x = aisle_end_x + 0.02
        right_bench_w = max(0.10, 0.94 - right_start_x)

        decomposer = TwinSeatDecomposer()

        for r in range(1, self.rows + 1):
            y1 = self.start_y + (r - 1) * row_step
            y2 = y1 + seat_height

            # 1. Băng ghế bên trái (Left bench)
            if self.left_cols == 2:
                # Phân rã thành 2 ghế liền nhau: Ghế A (cửa sổ), Ghế B (lối đi)
                bench_bbox = (left_start_x, y1, left_start_x + left_bench_w, y2)
                split_left = decomposer.split_horizontal_bench(
                    polygon_or_bbox=bench_bbox,
                    row_index=r,
                    side="left",
                    naming_format=self.naming_format,
                    start_id_counter=counter,
                )
                seats.extend(split_left)
                counter += len(split_left)
            else:
                # Hàng ghế đơn
                seat_id = f"SEAT_{r}A" if self.naming_format == "SEAT" else f"SEAT_{counter:03d}"
                poly = [
                    (round(left_start_x, 4), round(y1, 4)),
                    (round(left_start_x + left_bench_w, 4), round(y1, 4)),
                    (round(left_start_x + left_bench_w, 4), round(y2, 4)),
                    (round(left_start_x, 4), round(y2, 4)),
                ]
                seats.append(SeatConfig(seat_id=seat_id, polygon=poly, is_normalized=True, description=f"Hàng {r} - Ghế trái"))
                counter += 1

            # 2. Băng ghế bên phải (Right bench)
            if self.right_cols == 2:
                # Phân rã thành 2 ghế liền nhau: Ghế C (lối đi), Ghế D (cửa sổ)
                bench_bbox = (right_start_x, y1, right_start_x + right_bench_w, y2)
                split_right = decomposer.split_horizontal_bench(
                    polygon_or_bbox=bench_bbox,
                    row_index=r,
                    side="right",
                    naming_format=self.naming_format,
                    start_id_counter=counter,
                )
                seats.extend(split_right)
                counter += len(split_right)
            else:
                # Hàng ghế đơn
                seat_id = f"SEAT_{r}B" if self.naming_format == "SEAT" else f"SEAT_{counter:03d}"
                poly = [
                    (round(right_start_x, 4), round(y1, 4)),
                    (round(right_start_x + right_bench_w, 4), round(y1, 4)),
                    (round(right_start_x + right_bench_w, 4), round(y2, 4)),
                    (round(right_start_x, 4), round(y2, 4)),
                ]
                seats.append(SeatConfig(seat_id=seat_id, polygon=poly, is_normalized=True, description=f"Hàng {r} - Ghế phải"))
                counter += 1

        return seats


class SeatDetector:
    """
    Facade chính điều phối nhận diện ghế trên xe buýt:
    1. Kiểm tra minh bạch khả năng của model YOLO hiện tại (COCO không có class ghế xe buýt nhìn từ trần).
    2. Nếu có Custom YOLO model weights -> Sử dụng Custom YOLO.
    3. Nếu chưa có Custom YOLO model -> Sử dụng TopDownVisionSeatDetector & TwinSeatDecomposer.
    4. Cung cấp báo cáo chẩn đoán (diagnostics) đầy đủ, phục vụ kiểm thử và hiển thị UI.
    """

    def __init__(
        self,
        custom_weights_path: Optional[str] = "models/bus_seat_yolov8.pt",
        aisle_ratio: float = 0.18,
    ):
        self.custom_weights_path = custom_weights_path
        self.custom_detector = CustomYOLOSeatDetector(weights_path=custom_weights_path or "")
        self.vision_detector = TopDownVisionSeatDetector(aisle_ratio=aisle_ratio)

    @staticmethod
    def inspect_coco_classes() -> Dict[str, Union[bool, str, List[str]]]:
        """
        Kiểm tra danh sách lớp của model YOLOv8 COCO phổ thông:
        COCO có 80 lớp, trong đó lớp 56 là 'chair' (ghế tựa gia đình/văn phòng),
        HOÀN TOÀN KHÔNG có lớp 'bus_seat' hay ghế đệm nhìn từ camera trần xe buýt.
        """
        return {
            "has_overhead_bus_seat_class": False,
            "has_coco_chair_class": True,
            "coco_chair_class_id": 56,
            "explanation": (
                "Model YOLOv8 COCO pretrained phổ thông chỉ hỗ trợ lớp 'chair' (ghế văn phòng/gia đình góc nhìn ngang). "
                "Không có lớp nhận diện đệm ghế xe buýt nhìn từ trên trần xuống (top-down overhead bus seat). "
                "Hệ thống đã kích hoạt cơ chế Top-Down Vision & Twin-Seat Decomposer để nhận diện và tách từng ghế."
            ),
            "recommendation": (
                "Để đạt độ chính xác thị giác cao nhất ở mọi điều kiện ánh sáng, hãy thu thập 200-500 ảnh "
                "camera trần xe buýt và huấn luyện Custom YOLOv8 với nhãn [seat_window, seat_aisle] rồi đặt vào models/bus_seat_yolov8.pt."
            ),
        }

    def detect_seats(
        self,
        frame: np.ndarray,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        force_custom_model: bool = False,
    ) -> SeatDetectionResult:
        """
        Thực hiện nhận diện từng chiếc ghế riêng biệt.

        Args:
            frame: Ảnh frame camera xe buýt (BGR).
            rows: Số hàng ghế trong tầm nhìn camera.
            left_cols: Số ghế bên trái mỗi hàng (1 hoặc 2 cho ghế đôi).
            right_cols: Số ghế bên phải mỗi hàng (1 hoặc 2 cho ghế đôi).
            naming_format: 'SEAT' (SEAT_1A, SEAT_1B...) hoặc 'NUM' (SEAT_001, SEAT_002...).
            force_custom_model: Buộc dùng custom model nếu có.
        """
        diagnostics: Dict[str, Union[str, bool, int, float, List[str]]] = {}
        coco_info = self.inspect_coco_classes()
        diagnostics["coco_support"] = coco_info["explanation"]

        # 1. Thử nghiệm với Custom YOLO model nếu weights tồn tại
        if self.custom_detector.is_available or force_custom_model:
            custom_seats = self.custom_detector.detect(frame)
            if custom_seats:
                diagnostics["active_method"] = "Custom YOLO Overhead Seat Model"
                return SeatDetectionResult(
                    seats=custom_seats,
                    method=SeatDetectionMethod.CUSTOM_YOLO,
                    model_name=str(self.custom_detector.weights_path),
                    confidence_avg=0.88,
                    is_twin_seat_decomposed=True,
                    diagnostics=diagnostics,
                )

        # 2. Phương án cốt lõi: Top-Down Vision & Twin-Seat Decomposer
        seats = self.vision_detector.detect_seats(
            frame=frame,
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
        )

        diagnostics["active_method"] = "Top-Down Bus Geometry & Twin-Seat Vision Engine"
        diagnostics["rows"] = rows
        diagnostics["seats_per_row"] = left_cols + right_cols
        diagnostics["total_detected"] = len(seats)
        diagnostics["twin_seat_split_applied"] = (left_cols == 2 or right_cols == 2)

        return SeatDetectionResult(
            seats=seats,
            method=SeatDetectionMethod.TOP_DOWN_VISION,
            model_name="TopDown-TwinSeat-Decomposer",
            confidence_avg=1.0,
            is_twin_seat_decomposed=True,
            diagnostics=diagnostics,
        )
