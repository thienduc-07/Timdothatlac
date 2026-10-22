"""
src/seats/seat_manager.py
Quản lý trạng thái các vùng ghế (Seat ROI) trên xe buýt:
- Tự động nhận diện và phân biệt từng chiếc ghế riêng biệt từ góc nhìn trần xe buýt.
- Phân rã và quản lý từng ghế kề nhau trong cùng một hàng (Ghế A & Ghế B).
- Xác định ghế đang có người ngồi (OCCUPIED) hay trống (VACANT) bằng vùng tiếp xúc cơ thể.
- Gán đồ vật cho ghế phù hợp nhất hoặc đánh dấu UNKNOWN_SEAT nếu nằm ở ranh giới giữa hai ghế.
- Áp dụng làm mượt theo thời gian (temporal hysteresis) để tránh rung lắc trạng thái do nhiễu detection.
- Không kết luận ghế trống chỉ vì 1 frame không phát hiện thấy người.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.models import SeatConfig, SeatState, SeatStatus, TrackedObject
from src.seats.seat_config import (
    SeatConfigManager,
    calculate_overlap_ratio,
    calculate_polygon_overlap_ratio,
    is_point_in_seat,
)
from src.seats.seat_detector import (
    SeatDetector,
    SeatDetectionResult,
    SeatDetectionMethod,
)
from src.seats.seat_tracker import SeatTracker
from src.seats.seat_occupancy import SeatOccupancyEstimator
from src.utils.logger import get_logger
from config.settings import (
    PERSON_SEAT_OVERLAP_THRESHOLD,
    SEAT_OCCUPIED_CONFIRM_SECONDS,
    SEAT_VACANT_CONFIRM_SECONDS,
    SEAT_HOLD_SECONDS,
    CUSTOM_SEGMENTATION_WEIGHTS_PATH,
)

logger = get_logger(__name__)


class SeatManager:
    """
    Quản lý danh sách ghế, tự động nhận diện và cập nhật trạng thái có người/trống qua từng frame.
    """

    def __init__(
        self,
        seats: Optional[List[SeatConfig]] = None,
        overlap_threshold: float = PERSON_SEAT_OVERLAP_THRESHOLD,
        occupied_confirm_sec: float = SEAT_OCCUPIED_CONFIRM_SECONDS,
        vacant_confirm_sec: float = SEAT_VACANT_CONFIRM_SECONDS,
        hold_sec: float = SEAT_HOLD_SECONDS,
        custom_weights_path: Optional[str] = None,
        shared_model=None,
    ):
        self.overlap_threshold = overlap_threshold
        self.occupied_confirm_sec = occupied_confirm_sec
        self.vacant_confirm_sec = vacant_confirm_sec
        self.hold_sec = hold_sec

        self._seats: Dict[str, SeatConfig] = {}
        self._statuses: Dict[str, SeatStatus] = {}

        # Khởi tạo các module con chuyên biệt
        self.detector = SeatDetector(
            custom_weights_path=custom_weights_path,
            shared_model=shared_model,
        )
        self.tracker = SeatTracker()
        self.occupancy_estimator = SeatOccupancyEstimator(
            overlap_threshold=overlap_threshold,
            occupied_confirm_sec=occupied_confirm_sec,
            vacant_confirm_sec=vacant_confirm_sec,
            hold_sec=hold_sec,
        )

        self.active_detection_method: SeatDetectionMethod = (
            SeatDetectionMethod.MANUAL_CONFIGURATION if seats else SeatDetectionMethod.NOT_CONFIGURED
        )

        if seats is not None:
            self.set_seats(seats)
        # KHÔNG tự động nạp cấu hình ghế cố định từ seats_config.json.
        # Hệ thống chỉ ghi nhận ghế khi AI model phát hiện trực tiếp từ camera.

    def set_shared_model(self, model_instance) -> bool:
        """Chia sẻ model instance cho detector của SeatManager."""
        ok = self.detector.set_shared_model(model_instance)
        if ok:
            self.active_detection_method = SeatDetectionMethod.MODEL_DETECTION
        return ok

    @property
    def is_configured(self) -> bool:
        """Kiểm tra xem hệ thống đã có cấu hình ghế hay chưa."""
        return len(self._seats) > 0

    @property
    def seats(self) -> List[SeatConfig]:
        """Trả về danh sách cấu hình ghế."""
        return list(self._seats.values())

    def set_seats(self, seats: List[SeatConfig]):
        """Cập nhật danh sách ghế và khởi tạo trạng thái."""
        self._seats.clear()
        self._statuses.clear()
        for s in seats:
            self._seats[s.seat_id] = s
            self._statuses[s.seat_id] = SeatStatus(
                seat_id=s.seat_id,
                state=SeatState.UNKNOWN,
                notes="Chưa bắt đầu phân tích",
            )
        self.tracker.register_seats(seats)
        logger.info(f"SeatManager đã cấu hình {len(self._seats)} ghế: {list(self._seats.keys())}")

    def reset(self):
        """Reset trạng thái tất cả các ghế về UNKNOWN cho video mới."""
        for seat_id in self._statuses:
            self._statuses[seat_id] = SeatStatus(
                seat_id=seat_id,
                state=SeatState.UNKNOWN,
                notes="Reset trạng thái",
            )
        self.tracker.reset()
        if self._seats:
            self.tracker.register_seats(list(self._seats.values()))

    def update_detected_seats(
        self,
        raw_predictions: List[Any],
        frame_idx: int,
        frame_width: int,
        frame_height: int,
    ) -> List[SeatConfig]:
        """
        Cập nhật danh sách ghế phát hiện trực tiếp từ camera frame:
        - Trích xuất candidate đệm ghế từ raw predictions (Local YOLO, Roboflow Workflow, hoặc Hybrid).
        - Gửi candidate vào SeatTracker để so khớp IoU, làm mượt EMA và gán ID ổn định (SEAT_001, SEAT_002...).
        - Cập nhật danh sách ghế hoạt động trong _seats và khởi tạo trạng thái nếu là ghế mới.
        """
        candidate_seats = self.detector.detect_seats_from_predictions(
            raw_predictions, frame_width, frame_height
        )

        if not candidate_seats and not self.tracker.get_active_seats():
            # Không phát hiện thấy ghế nào từ model AI
            return []

        active_seats = self.tracker.update(candidate_seats, frame_idx)

        # Cập nhật danh sách _seats và bảo toàn trạng thái đã có
        new_seat_map = {}
        for s in active_seats:
            new_seat_map[s.seat_id] = s
            if s.seat_id not in self._statuses:
                self._statuses[s.seat_id] = SeatStatus(
                    seat_id=s.seat_id,
                    state=SeatState.UNKNOWN,
                    notes="AI phát hiện trực tiếp từ camera",
                )

        self._seats = new_seat_map
        if self._seats:
            self.active_detection_method = SeatDetectionMethod.MODEL_DETECTION

        return self.seats

    @staticmethod
    def generate_bus_grid(
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_style: str = "SEAT",
        start_y: float = 0.12,
        end_y: float = 0.90,
        aisle_ratio: float = 0.18,
    ) -> List[SeatConfig]:
        """Tự động tạo sơ đồ lưới ghế xe buýt với từng ghế riêng biệt."""
        return SeatConfigManager.generate_bus_grid(
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_style="GHE" if naming_style in ("GHE", "alphanumeric") else "SEAT",
            start_y=start_y,
            end_y=end_y,
            aisle_ratio=aisle_ratio,
        )

    def auto_detect_and_configure_seats(
        self,
        frame: np.ndarray,
        allow_fallback: bool = False,
        rows: int = 4,
        left_cols: int = 2,
        right_cols: int = 2,
        naming_format: str = "SEAT",
        save_to_disk: bool = True,
        roboflow_client: Optional[Any] = None,
        inference_mode: Optional[str] = None,
    ) -> SeatDetectionResult:
        """
        Tự động nhận diện từng ghế từ frame camera bằng Computer Vision Model:
        - Hỗ trợ Roboflow Workflow, Hybrid hoặc Local Custom Model.
        - Nếu có Custom Model hoặc Roboflow: Trả về MODEL DETECTION.
        - Nếu không có Custom Model và allow_fallback=False: Báo CUSTOM SEAT MODEL NOT CONFIGURED.
        - Nếu allow_fallback=True: Sinh GEOMETRY FALLBACK được công khai rõ ràng.
        """
        result = self.detector.detect_seats(
            frame=frame,
            allow_fallback=allow_fallback,
            rows=rows,
            left_cols=left_cols,
            right_cols=right_cols,
            naming_format=naming_format,
            roboflow_client=roboflow_client,
            inference_mode=inference_mode,
        )

        if result.seats:
            self.set_seats(result.seats)
            self.active_detection_method = result.method
            if save_to_disk:
                SeatConfigManager().save_seats(
                    result.seats,
                    description=f"Seats configured via {result.method.value}",
                )
        elif result.method == SeatDetectionMethod.NOT_CONFIGURED:
            self.active_detection_method = SeatDetectionMethod.NOT_CONFIGURED
        elif result.method == SeatDetectionMethod.MODEL_DETECTION and not result.seats:
            # AI quét nhưng không có ghế trong frame
            self.active_detection_method = SeatDetectionMethod.MODEL_DETECTION

        return result


    def get_pixel_polygon(
        self, seat_id: str, frame_width: int, frame_height: int
    ) -> List[Tuple[int, int]]:
        """Lấy danh sách đỉnh polygon theo pixel thực tế trên frame."""
        cfg = self._seats.get(seat_id)
        if not cfg:
            return []
        return cfg.get_pixel_polygon(frame_width, frame_height)

    def update(
        self,
        persons: List[Union[TrackedObject, Tuple[int, int, int, int]]],
        frame_index: int,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, SeatStatus]:
        """
        Cập nhật trạng thái người ngồi trên từng ghế ở frame hiện tại:
        - Sử dụng SeatOccupancyEstimator phân tích vùng tiếp xúc cơ thể.
        - Phân xử tình huống ghế đôi kề nhau trong một hàng.
        - Áp dụng làm mượt theo thời gian.
        """
        if not self._seats:
            return self._statuses

        return self.occupancy_estimator.update_occupancy(
            seats=list(self._seats.values()),
            current_statuses=self._statuses,
            persons=persons,
            frame_index=frame_index,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    # Alias để tương thích với các module và test gọi update_occupancy
    update_occupancy = update

    def find_seat_for_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        min_overlap: float = 0.10,
        ambiguity_diff_threshold: float = 0.08,
    ) -> Optional[str]:
        """
        Xác định vùng ghế mà một vật thể (bbox) đang thuộc về:
        - Nếu vật thể nằm gần ranh giới giữa 2 ghế hoặc không đủ bằng chứng: Trả về 'UNKNOWN_SEAT'.
        - Nếu vật thể nằm trọn trong 1 ghế hoặc có tỷ lệ giao thoa vượt trội: Trả về seat_id đó.
        - Tránh gán nhầm lẫn giữa 2 ghế nằm cạnh nhau trong cùng hàng.
        - Trả về None nếu nằm ngoài tất cả các ghế.
        """
        if not self._seats:
            return None

        bx1, by1, bx2, by2 = bbox
        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

        seat_scores: List[Tuple[str, float, bool]] = []  # (seat_id, score, is_center_inside)

        for seat_id, config in self._seats.items():
            poly_pts = config.get_pixel_polygon(frame_width, frame_height)
            is_inside = is_point_in_seat(center, poly_pts)
            overlap_seat, overlap_bbox = calculate_overlap_ratio(bbox, poly_pts)
            score = max(overlap_seat, overlap_bbox)

            if is_inside or score >= min_overlap:
                effective_score = score + (0.35 if is_inside else 0.0)
                seat_scores.append((seat_id, effective_score, is_inside))

        if not seat_scores:
            return None

        # Sắp xếp theo điểm số giảm dần
        seat_scores.sort(key=lambda x: x[1], reverse=True)

        if len(seat_scores) == 1:
            return seat_scores[0][0]

        top1_id, top1_score, top1_center = seat_scores[0]
        top2_id, top2_score, top2_center = seat_scores[1]

        # Kiểm tra tính mơ hồ ở ranh giới (Boundary Ambiguity):
        # Nếu cả 2 ghế đều có điểm cao xấp xỉ nhau (chênh lệch nhỏ)
        diff = abs(top1_score - top2_score)
        if diff < ambiguity_diff_threshold and top2_score >= min_overlap:
            logger.debug(
                f"Đồ vật nằm ở ranh giới giữa {top1_id} và {top2_id} (diff {diff:.2f}) -> UNKNOWN_SEAT"
            )
            return "UNKNOWN_SEAT"

        # Nếu tâm nằm trọn trong ghế 1 và ghế 2 chỉ chạm nhẹ viền
        if top1_center and not top2_center:
            return top1_id

        if top1_score >= min_overlap:
            return top1_id

        return None

    def find_seat_for_polygon(
        self,
        polygon: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
        min_overlap: float = 0.08,
        ambiguity_diff_threshold: float = 0.08,
    ) -> Optional[str]:
        """
        Xác định vùng ghế mà một vật thể có segmentation mask polygon đang thuộc về:
        - Sử dụng calculate_polygon_overlap_ratio trực tiếp giữa đa giác vật thể và đa giác ghế.
        - Trả về seat_id, 'UNKNOWN_SEAT' nếu nằm ở ranh giới mơ hồ, hoặc None nếu nằm ngoài.
        """
        if not self._seats or len(polygon) < 3:
            return None

        # Chuyển đổi polygon sang pixel
        is_norm = all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in polygon)
        if is_norm:
            obj_pts = [(int(pt[0] * frame_width), int(pt[1] * frame_height)) for pt in polygon]
        else:
            obj_pts = [(int(pt[0]), int(pt[1])) for pt in polygon]

        cx = sum(p[0] for p in obj_pts) / len(obj_pts)
        cy = sum(p[1] for p in obj_pts) / len(obj_pts)
        center = (cx, cy)

        seat_scores: List[Tuple[str, float, bool]] = []
        for seat_id, config in self._seats.items():
            poly_pts = config.get_pixel_polygon(frame_width, frame_height)
            is_inside = is_point_in_seat(center, poly_pts)
            overlap_seat, overlap_obj = calculate_polygon_overlap_ratio(obj_pts, poly_pts)
            score = max(overlap_seat, overlap_obj)

            if is_inside or score >= min_overlap:
                effective_score = score + (0.35 if is_inside else 0.0)
                seat_scores.append((seat_id, effective_score, is_inside))

        if not seat_scores:
            return None

        seat_scores.sort(key=lambda x: x[1], reverse=True)
        if len(seat_scores) == 1:
            return seat_scores[0][0]

        top1_id, top1_score, top1_center = seat_scores[0]
        top2_id, top2_score, top2_center = seat_scores[1]

        diff = abs(top1_score - top2_score)
        if diff < ambiguity_diff_threshold and top2_score >= min_overlap:
            logger.debug(
                f"Đồ vật (polygon) nằm ở ranh giới giữa {top1_id} và {top2_id} (diff {diff:.2f}) -> UNKNOWN_SEAT"
            )
            return "UNKNOWN_SEAT"

        if top1_center and not top2_center:
            return top1_id

        if top1_score >= min_overlap:
            return top1_id

        return None

    def find_seat_for_object(
        self,
        tracked_obj: TrackedObject,
        frame_width: int,
        frame_height: int,
        min_overlap: float = 0.10,
    ) -> Optional[str]:
        """
        Tìm ghế cho một đối tượng TrackedObject.
        Ưu tiên sử dụng segmentation polygon nếu có, fallback về bounding box.
        """
        if getattr(tracked_obj, "polygon", None) is not None and len(tracked_obj.polygon) >= 3:
            seat_by_poly = self.find_seat_for_polygon(
                tracked_obj.polygon, frame_width, frame_height, min_overlap=min_overlap
            )
            if seat_by_poly is not None:
                return seat_by_poly

        return self.find_seat_for_bbox(
            tracked_obj.bbox, frame_width, frame_height, min_overlap=min_overlap
        )

    def get_last_departed_person(self, seat_id: str) -> Optional[Tuple[int, float]]:
        """Lấy thông tin người vừa rời khỏi ghế (person_track_id, departure_time_sec)."""
        return self.occupancy_estimator.get_last_departed_person(seat_id)

    def is_seat_occupied(self, seat_id: str) -> bool:
        """Kiểm tra ghế có đang ở trạng thái OCCUPIED hay không."""
        status = self._statuses.get(seat_id)
        return status is not None and status.state == SeatState.OCCUPIED

    def is_seat_vacant(self, seat_id: str) -> bool:
        """Kiểm tra ghế có đang ở trạng thái VACANT hay không."""
        status = self._statuses.get(seat_id)
        return status is not None and status.state == SeatState.VACANT

    def get_seat_status(self, seat_id: str) -> Optional[SeatStatus]:
        """Lấy trạng thái chi tiết của ghế."""
        return self._statuses.get(seat_id)

    def get_all_statuses(self) -> Dict[str, SeatStatus]:
        """Lấy trạng thái của toàn bộ ghế."""
        return self._statuses

    def get_model_diagnostics(
        self,
        inference_mode: Optional[str] = None,
        roboflow_client: Optional[Any] = None,
    ) -> Dict[str, Union[str, bool, int, float, List[str]]]:
        """Lấy thông tin chẩn đoán toàn diện về khả năng nhận diện ghế theo chế độ inference hiện hành."""
        import os
        from config.settings import INFERENCE_MODE, ROBOFLOW_API_KEY, ROBOFLOW_WORKFLOW_ID

        rf_client = roboflow_client or self.detector.roboflow_client
        custom_ready = self.detector.model_adapter.is_available
        roboflow_configured = bool(
            getattr(rf_client, "configured", False)
            or (ROBOFLOW_API_KEY and ROBOFLOW_WORKFLOW_ID)
        )
        mode = (
            inference_mode
            or self.detector.inference_mode
            or os.getenv("INFERENCE_MODE", INFERENCE_MODE)
        ).lower()

        # Nhận diện ghế sẵn sàng nếu có local custom weights hoặc nếu đang ở roboflow/hybrid mode và có roboflow
        if mode in ("roboflow", "hybrid"):
            seat_ready = roboflow_configured or custom_ready
        else:
            seat_ready = custom_ready

        coco_info = self.detector.inspect_coco_classes()
        supported_classes = list(self.detector.model_adapter.supported_classes)
        if (mode in ("roboflow", "hybrid") or roboflow_configured) and not supported_classes:
            supported_classes = ["empty_seat", "human_occupied", "object_occupied"]

        if not seat_ready:
            diagnostic_msg = (
                f"Chế độ hiện tại '{mode.upper()}' sử dụng mô hình YOLO COCO cục bộ không có lớp đệm ghế xe buýt nhìn từ trần xuống "
                f"(chỉ có lớp 56 'chair' - ghế phòng khách nhìn ngang). "
                f"{'Mô hình Roboflow Serverless Workflow đã được cấu hình với các lớp đệm ghế (empty_seat, human_occupied, object_occupied). Hãy chuyển sang chế độ ROBOFLOW hoặc HYBRID để nhận diện ghế bằng AI thực thụ.' if roboflow_configured else 'Chưa nạp mô hình chuyên biệt models/best.pt.'}"
            )
        else:
            diagnostic_msg = f"Sẵn sàng nhận diện ghế bằng AI ({'Roboflow Workflow' if (mode in ('roboflow', 'hybrid') and roboflow_configured) else 'Custom YOLO'})."

        return {
            "custom_adapter_ready": custom_ready,
            "roboflow_ready": roboflow_configured,
            "seat_detection_ready": seat_ready,
            "inference_mode": mode,
            "model_weights": str(self.detector.model_adapter.weights_path),
            "supported_classes": supported_classes,
            "active_detection_method": self.active_detection_method.value,
            "has_overhead_bus_seat_class": seat_ready,
            "has_coco_chair_class": True,
            "coco_chair_class_id": 56,
            "explanation": coco_info.get("explanation", ""),
            "recommendation": coco_info.get("recommendation", ""),
            "diagnostic_message": diagnostic_msg,
        }

