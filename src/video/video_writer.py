"""
src/video/video_writer.py
Module phụ trách ghi video đầu ra có chú thích:
- Vẽ vùng ghế (Seat ROI polygon) với màu sắc phản ánh trạng thái OCCUPIED / VACANT.
- Vẽ bounding box, tracking ID và trạng thái chi tiết theo State Machine.
- Vẽ vị trí cuối cùng của vật thể đang bị che khuất (OCCLUDED/RECHECK).
- Đảm bảo video có thể phát trong trình duyệt.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.models import Event, TrackedObject, SeatState
from src.seats.seat_manager import SeatManager
from src.utils.logger import get_logger
from config.settings import (
    OUTPUT_VIDEO_CODEC,
    OUTPUT_VIDEO_EXTENSION,
    CLASS_NAMES_VI,
    EventStatus,
)

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Chuẩn màu sắc cố định (Part 4) - BGR cho OpenCV
# ──────────────────────────────────────────────
# 1. GHẾ (SEATS) - Cyan / Xanh dương
COLOR_SEAT_VACANT = (255, 229, 0)      # Cyan: Ghế trống
COLOR_SEAT_OCCUPIED = (255, 120, 0)    # Xanh dương đậm: Ghế có người
COLOR_SEAT_UNKNOWN = (160, 160, 160)   # Xám: Chưa rõ

# 2. NGƯỜI (PERSONS) - Xanh lá cây (Green)
COLOR_PERSON = (76, 217, 100)          # Xanh lá cây

# 3. ĐỒ VẬT (OBJECTS) - Hổ phách / Vàng cam (Amber)
COLOR_OBJECT = (11, 158, 245)          # Hổ phách / Vàng cam
COLOR_OBJECT_OCCLUDED = (180, 100, 240)# Tím nhạt cho đồ bị che khuất (dự đoán)
COLOR_ALERT_CRITICAL = (68, 68, 239)   # Đỏ cho cảnh báo khẩn cấp


class VideoWriter:
    """Ghi video đầu ra với bounding box, label, trạng thái sự kiện và vùng ghế."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None
        self._opened = False
        self._frame_count = 0

    def open(self) -> bool:
        """Mở video writer."""
        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
            self._writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )

            if not self._writer.isOpened():
                logger.error(f"Không thể mở video writer: {self.output_path}")
                return False

            self._opened = True
            logger.info(f"Video writer mở tại: {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi tạo video writer: {e}")
            return False

    def write_annotated_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        event_statuses: Optional[dict] = None,
        seat_manager: Optional[SeatManager] = None,
        active_events: Optional[List[Event]] = None,
        show_seats: bool = True,
        show_persons: bool = True,
        show_objects: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        show_event_markers: bool = True,
        privacy_blur_faces: bool = False,
    ):
        """Vẽ chú thích toàn diện lên frame và ghi vào video đầu ra."""
        if not self._opened or self._writer is None:
            return

        annotated = self.annotate_frame(
            frame=frame,
            tracked_objects=tracked_objects,
            event_statuses=event_statuses,
            seat_manager=seat_manager,
            active_events=active_events,
            width=self.width,
            height=self.height,
            show_seats=show_seats,
            show_persons=show_persons,
            show_objects=show_objects,
            show_track_ids=show_track_ids,
            show_confidence=show_confidence,
            show_event_markers=show_event_markers,
            privacy_blur_faces=privacy_blur_faces,
        )

        self._writer.write(annotated)
        self._frame_count += 1

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        event_statuses: Optional[dict] = None,
        seat_manager: Optional[SeatManager] = None,
        active_events: Optional[List[Event]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        show_seats: bool = True,
        show_persons: bool = True,
        show_objects: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        show_event_markers: bool = True,
        privacy_blur_faces: bool = False,
    ) -> np.ndarray:
        """
        Vẽ chú thích toàn diện lên 1 frame tuân thủ nghiêm ngặt chuẩn Part 4:
        - GHẾ (Cyan/Blue): S01, S02... kèm trạng thái
        - NGƯỜI (Xanh lá): P01, P12...
        - ĐỒ VẬT (Hổ phách/Amber): O01, O31...
        - Đồ bị che khuất: vẽ viền mờ với nhãn [Dự đoán/Che khuất]
        - Hỗ trợ làm mờ khuôn mặt (Privacy Blur)
        - Hỗ trợ lọc hiển thị từng lớp riêng biệt
        """
        annotated = frame.copy()
        cur_h, cur_w = frame.shape[:2]
        w = width or cur_w
        h = height or cur_h
        event_statuses = event_statuses or {}

        # 1. Vẽ các vùng ghế (Seat ROI) - Màu Cyan / Blue
        if show_seats and seat_manager and seat_manager.is_configured:
            VideoWriter._draw_seat_rois_static(annotated, seat_manager, w, h)

        # 2. Vẽ vị trí cuối cùng của các vật thể đang bị che khuất (OCCLUDED / RECHECK)
        if show_objects and active_events:
            VideoWriter._draw_occluded_and_missing_events_static(
                annotated, active_events, tracked_objects, show_track_ids, show_event_markers
            )

        # 3. Vẽ các đối tượng đang được track trong frame
        for obj in tracked_objects:
            is_person = (obj.class_name == "person")

            # Bộ lọc lớp
            if is_person and not show_persons:
                continue
            if not is_person and not show_objects:
                continue

            x1, y1, x2, y2 = obj.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Làm mờ khuôn mặt nếu bật Privacy Blur
            if is_person and privacy_blur_faces:
                face_h = max(5, int((y2 - y1) * 0.35))
                face_roi = annotated[y1:y1 + face_h, x1:x2]
                if face_roi.size > 0:
                    blurred = cv2.GaussianBlur(face_roi, (25, 25), 30)
                    annotated[y1:y1 + face_h, x1:x2] = blurred

            status = event_statuses.get(obj.track_id, None)

            # Gán ghế liên quan
            associated_seat = ""
            if seat_manager and seat_manager.is_configured:
                associated_seat = seat_manager.find_seat_for_bbox(obj.bbox, w, h) or ""

            # Chọn màu và format ID chuẩn theo Part 4
            if is_person:
                color = COLOR_PERSON
                id_prefix = f"P{obj.track_id}" if show_track_ids else "Người"
                seat_info = f" | {associated_seat}" if associated_seat else ""
                conf_info = f" | {obj.confidence:.0%}" if show_confidence else ""
                label = f"{id_prefix}{seat_info}{conf_info}"
                thickness = 2
            else:
                vi_name = CLASS_NAMES_VI.get(obj.class_name, obj.class_name)
                id_prefix = f"O{obj.track_id}" if show_track_ids else vi_name
                seat_info = f" | {associated_seat}" if associated_seat else ""
                conf_info = f" | {obj.confidence:.0%}" if show_confidence else ""

                if status == EventStatus.SUSPECTED:
                    color = COLOR_ALERT_CRITICAL
                    marker = " | ⚠ NGHI NGỜ" if show_event_markers else ""
                    thickness = 3
                elif status == EventStatus.ITEM_LEFT:
                    color = COLOR_OBJECT
                    marker = " | 🪑 GHẾ TRỐNG" if show_event_markers else ""
                    thickness = 2
                elif status == EventStatus.OCCLUDED:
                    color = COLOR_OBJECT_OCCLUDED
                    marker = " | 🛡 CHE KHUẤT" if show_event_markers else ""
                    thickness = 2
                elif status == EventStatus.REVIEW_REQUIRED:
                    color = COLOR_ALERT_CRITICAL
                    marker = " | 📋 CẦN DUYỆT" if show_event_markers else ""
                    thickness = 2
                else:
                    color = COLOR_OBJECT
                    marker = ""
                    thickness = 2

                label = f"{id_prefix} ({vi_name}){seat_info}{conf_info}{marker}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
            label_y = max(y1 - 6, text_h + 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 4),
                (x1 + text_w + 6, label_y + 4),
                color,
                -1,
            )
            # Text màu đen nếu nền sáng (Cyan/Amber), màu trắng nếu nền tối
            text_color = (0, 0, 0) if (color == COLOR_SEAT_VACANT or color == COLOR_OBJECT) else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, label_y - 2),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated


    @staticmethod
    def _draw_seat_rois_static(frame: np.ndarray, seat_manager: SeatManager, width: int, height: int):
        """Vẽ polygon vùng ghế và trạng thái có người/trống."""
        overlay = frame.copy()
        alpha = 0.20  # Độ trong suốt của màu nền ghế

        for seat in seat_manager.seats:
            poly_pts = seat_manager.get_pixel_polygon(seat.seat_id, width, height)
            if len(poly_pts) < 3:
                continue

            status = seat_manager.get_seat_status(seat.seat_id)
            seat_state = status.state if status else SeatState.UNKNOWN

            if seat_state == SeatState.OCCUPIED:
                color = COLOR_SEAT_OCCUPIED
                state_text = "CÓ NGƯỜI"
            elif seat_state == SeatState.VACANT:
                color = COLOR_SEAT_VACANT
                state_text = "TRỐNG"
            else:
                color = COLOR_SEAT_UNKNOWN
                state_text = "CHƯA RÕ"

            pts_arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_arr], color)
            cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=2)

            # Tính tâm ghế để vẽ nhãn
            cx = int(np.mean([pt[0] for pt in poly_pts]))
            cy = int(np.mean([pt[1] for pt in poly_pts]))

            label = f"{seat.seat_id} [{state_text}]"
            cv2.putText(
                frame,
                label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _draw_occluded_and_missing_events_static(
        frame: np.ndarray,
        active_events: List[Event],
        current_tracked: List[TrackedObject],
        show_track_ids: bool = True,
        show_event_markers: bool = True,
    ):
        """
        Vẽ vị trí lưu giữ cuối cùng của các đồ vật đang bị che khuất hoặc đang xác minh.
        """
        currently_detected_tracks = {obj.track_id for obj in current_tracked}

        for event in active_events:
            # Nếu vật thể đang có bbox phát hiện thực tế trong frame, bỏ qua để vẽ ở bước 3
            if event.last_track_id in currently_detected_tracks:
                continue

            if event.status in (EventStatus.OCCLUDED, EventStatus.RECHECK, EventStatus.REVIEW_REQUIRED):
                bx1, by1, bx2, by2 = event.last_known_bbox
                if bx1 == 0 and by1 == 0 and bx2 == 0 and by2 == 0:
                    continue

                if event.status == EventStatus.OCCLUDED:
                    color = COLOR_OBJECT_OCCLUDED
                    tag = f"O{event.last_track_id} [Dự đoán/Che khuất {event.occlusion_duration:.0f}s]"
                elif event.status == EventStatus.RECHECK:
                    color = COLOR_OBJECT
                    tag = f"O{event.last_track_id} [Đang xác minh]"
                else:
                    color = COLOR_ALERT_CRITICAL
                    tag = f"O{event.last_track_id} [Cần kiểm tra]"

                # Vẽ viền nét đứt hoặc nét kép biểu thị vị trí được ghi nhớ
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(
                    frame,
                    tag,
                    (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def close(self):
        """Đóng video writer và giải phóng tài nguyên."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._opened = False
        logger.info(f"Đã ghi {self._frame_count} frame vào video đầu ra")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
