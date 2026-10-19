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

# Bảng màu BGR
COLOR_PERSON = (255, 200, 50)          # Xanh dương nhạt cho người
COLOR_ITEM_NORMAL = (50, 255, 50)      # Xanh lá cho đồ vật bình thường
COLOR_OBSERVED = (0, 255, 255)         # Vàng chanh cho OBSERVED / MONITORING
COLOR_ITEM_LEFT = (0, 165, 255)        # Cam cho ITEM_LEFT (ghế trống còn đồ)
COLOR_SUSPECTED = (0, 0, 255)          # Đỏ cho SUSPECTED (nghi ngờ bỏ quên)
COLOR_OCCLUDED = (180, 50, 180)        # Tím cho OCCLUDED (bị che khuất)
COLOR_RECHECK = (0, 215, 255)          # Vàng đậm cho RECHECK (đang xác minh)
COLOR_REVIEW_REQUIRED = (147, 20, 255) # Hồng tím cho REVIEW_REQUIRED (cần kiểm tra)
COLOR_RESOLVED = (0, 200, 0)           # Xanh lá đậm cho RESOLVED / CONFIRMED
COLOR_DISMISSED = (128, 128, 128)      # Xám cho DISMISSED

# Màu vùng ghế (Seat ROI)
COLOR_SEAT_VACANT = (50, 200, 50)      # Xanh lá: Ghế trống
COLOR_SEAT_OCCUPIED = (50, 50, 220)    # Đỏ cam: Ghế có người
COLOR_SEAT_UNKNOWN = (150, 150, 150)   # Xám: Chưa xác định


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
    ):
        """
        Vẽ chú thích toàn diện lên frame và ghi vào video đầu ra.
        """
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
    ) -> np.ndarray:
        """
        Vẽ chú thích toàn diện lên 1 frame (dùng chung cho cả video file và live webcam):
        - Vùng ghế (Seat ROI polygon) với màu sắc trạng thái.
        - Bounding box vật thể với tên số ghế: ví dụ '[Ghế 1A] Ba lô #2 85% ⚠ NGHI NGỜ'.
        - Vị trí vật thể đang bị che khuất.
        """
        annotated = frame.copy()
        cur_h, cur_w = frame.shape[:2]
        w = width or cur_w
        h = height or cur_h
        event_statuses = event_statuses or {}

        # 1. Vẽ các vùng ghế (Seat ROI)
        if seat_manager and seat_manager.is_configured:
            VideoWriter._draw_seat_rois_static(annotated, seat_manager, w, h)

        # 2. Vẽ vị trí cuối cùng của các vật thể đang bị che khuất (OCCLUDED / RECHECK / REVIEW_REQUIRED)
        if active_events:
            VideoWriter._draw_occluded_and_missing_events_static(annotated, active_events, tracked_objects)

        # 3. Vẽ các vật thể đang được phát hiện / tracking trong frame
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            status = event_statuses.get(obj.track_id, None)

            # Phân loại màu sắc theo trạng thái
            if status == EventStatus.SUSPECTED:
                color = COLOR_SUSPECTED
                tag = " ⚠ NGHI NGỜ"
                thickness = 3
            elif status == EventStatus.OCCLUDED:
                color = COLOR_OCCLUDED
                tag = " 🛡 BỊ CHE KHUẤT"
                thickness = 2
            elif status == EventStatus.RECHECK:
                color = COLOR_RECHECK
                tag = " 🔄 XÁC MINH"
                thickness = 2
            elif status == EventStatus.REVIEW_REQUIRED:
                color = COLOR_REVIEW_REQUIRED
                tag = " 📋 CẦN KIỂM TRA"
                thickness = 3
            elif status == EventStatus.ITEM_LEFT:
                color = COLOR_ITEM_LEFT
                tag = " 🪑 GHẾ TRỐNG"
                thickness = 2
            elif status in (EventStatus.OBSERVED, EventStatus.MONITORING):
                color = COLOR_OBSERVED
                tag = " 👁 QUAN SÁT"
                thickness = 2
            elif status in (EventStatus.RESOLVED, EventStatus.CONFIRMED):
                color = COLOR_RESOLVED
                tag = " ✅ ĐÃ XÁC NHẬN"
                thickness = 2
            elif status == EventStatus.DISMISSED:
                color = COLOR_DISMISSED
                tag = " ❌ LOẠI BỎ"
                thickness = 1
            elif obj.class_name == "person":
                color = COLOR_PERSON
                tag = ""
                thickness = 1
            else:
                color = COLOR_ITEM_NORMAL
                tag = ""
                thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Nhận diện số ghế mà vật thể hoặc người đang nằm trên đó
            seat_tag = ""
            if seat_manager and seat_manager.is_configured:
                found_seat = seat_manager.find_seat_for_bbox(obj.bbox, w, h)
                if found_seat:
                    seat_tag = f"[{found_seat}] "

            vi_name = CLASS_NAMES_VI.get(obj.class_name, obj.class_name)
            label = f"{seat_tag}{vi_name} #{obj.track_id} {obj.confidence:.0%}{tag}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
            label_y = max(y1 - 10, text_h + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 5),
                (x1 + text_w + 5, label_y + 5),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),
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
                    color = COLOR_OCCLUDED
                    tag = f"🛡 {event.event_id} BỊ CHE ({event.occlusion_duration:.0f}s)"
                elif event.status == EventStatus.RECHECK:
                    color = COLOR_RECHECK
                    tag = f"🔄 {event.event_id} ĐANG XÁC MINH"
                else:
                    color = COLOR_REVIEW_REQUIRED
                    tag = f"📋 {event.event_id} CẦN KIỂM TRA"

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
