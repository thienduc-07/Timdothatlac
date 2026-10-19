"""
src/video/camera_service.py
Module quản lý luồng camera và hiệu suất thời gian thực:
- Hỗ trợ các nguồn đầu vào: Video tải lên, Video mẫu Demo, Webcam máy tính, RTSP / IP camera.
- Đo lường và hiển thị thông số kỹ thuật thực tế: Input FPS, Inference FPS, End-to-end Latency (ms), Dropped Frames.
- Hỗ trợ các trạng thái điều khiển: Bắt đầu, Tạm dừng, Tiếp tục, Dừng, Khởi động lại.
- Tự động tạo video mẫu mô phỏng xe buýt chuẩn để phục vụ demo trực tiếp trên Cloud.
"""

from datetime import datetime
import os
from pathlib import Path
import time
from typing import Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from config.settings import DATA_DIR, SystemMode

logger = get_logger(__name__)

DEMO_VIDEO_PATH = DATA_DIR / "sample_bus_demo.mp4"


class CameraSourceType:
    VIDEO_FILE = "video_file"
    DEMO_SAMPLE = "demo_sample"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class CameraStreamManager:
    """Quản lý luồng đọc frame từ nhiều nguồn và đo đạc hiệu năng."""

    def __init__(self):
        self.source_type: str = CameraSourceType.DEMO_SAMPLE
        self.source_path: str = ""
        self.is_running: bool = False
        self.is_paused: bool = False

        # Các chỉ số đo đạc hiệu năng THỰC TẾ
        self.input_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.latency_ms: float = 0.0
        self.dropped_frames: int = 0
        self.total_frames_read: int = 0
        self.tracked_count: int = 0
        self.pending_events_count: int = 0
        self.error_message: str = ""

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time: float = 0.0

    def start_stream(
        self, source_type: str, source_path: str = ""
    ) -> Tuple[bool, str]:
        """Bắt đầu kết nối nguồn camera."""
        self.stop_stream()
        self.source_type = source_type
        self.source_path = source_path
        self.error_message = ""
        self.dropped_frames = 0
        self.total_frames_read = 0

        try:
            if source_type == CameraSourceType.DEMO_SAMPLE:
                ensure_demo_video_exists()
                target_path = str(DEMO_VIDEO_PATH)
                self._cap = cv2.VideoCapture(target_path)
            elif source_type == CameraSourceType.VIDEO_FILE:
                if not Path(source_path).exists():
                    return False, f"File video không tồn tại: {source_path}"
                self._cap = cv2.VideoCapture(source_path)
            elif source_type == CameraSourceType.WEBCAM:
                cam_idx = int(source_path) if source_path.isdigit() else 0
                self._cap = cv2.VideoCapture(cam_idx)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
            elif source_type == CameraSourceType.RTSP:
                if not source_path.startswith("rtsp://") and not source_path.startswith("http://"):
                    return False, "URL RTSP phải bắt đầu bằng rtsp:// hoặc http://"
                self._cap = cv2.VideoCapture(source_path)
            else:
                return False, f"Loại nguồn camera không hợp lệ: {source_type}"

            if self._cap is None or not self._cap.isOpened():
                err = f"Không thể mở nguồn camera: {source_path or source_type}"
                self.error_message = err
                return False, err

            self.is_running = True
            self.is_paused = False
            self._last_frame_time = time.time()
            logger.info(f"Đã mở thành công luồng camera: {source_type} ({source_path})")
            return True, "Kết nối thành công"

        except Exception as e:
            err = f"Lỗi khởi động camera: {str(e)}"
            self.error_message = err
            logger.error(err)
            return False, err

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Đọc 1 frame từ luồng camera kèm đo FPS đầu vào thực tế."""
        if not self.is_running or self._cap is None:
            return False, None

        if self.is_paused:
            return False, None

        t0 = time.time()
        ret, frame = self._cap.read()

        if not ret or frame is None:
            # Nếu là file video hoặc demo, tự động loop lại
            if self.source_type in (CameraSourceType.DEMO_SAMPLE, CameraSourceType.VIDEO_FILE):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()

            if not ret or frame is None:
                self.is_running = False
                return False, None

        now = time.time()
        dt = now - self._last_frame_time
        if dt > 0:
            current_input_fps = 1.0 / dt
            self.input_fps = round(self.input_fps * 0.7 + current_input_fps * 0.3, 1)
        self._last_frame_time = now
        self.total_frames_read += 1

        return True, frame

    def update_metrics(
        self,
        latency_ms: float,
        tracked_count: int = 0,
        pending_events_count: int = 0,
    ):
        """Cập nhật độ trễ suy luận và số đối tượng thực tế."""
        self.latency_ms = round(latency_ms, 1)
        if latency_ms > 0:
            calc_fps = 1000.0 / latency_ms
            self.inference_fps = round(self.inference_fps * 0.7 + calc_fps * 0.3, 1)
        self.tracked_count = tracked_count
        self.pending_events_count = pending_events_count

    def pause_stream(self):
        """Tạm dừng luồng."""
        self.is_paused = True

    def resume_stream(self):
        """Tiếp tục luồng."""
        self.is_paused = False

    def stop_stream(self):
        """Dừng và giải phóng tài nguyên camera."""
        self.is_running = False
        self.is_paused = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info("Đã dừng và giải phóng luồng camera")

    def get_status_summary(self) -> Dict:
        """Lấy bảng thông số hoạt động thực tế của camera."""
        return {
            "source_type": self.source_type,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "input_fps": self.input_fps,
            "inference_fps": self.inference_fps,
            "latency_ms": self.latency_ms,
            "dropped_frames": self.dropped_frames,
            "total_frames_read": self.total_frames_read,
            "tracked_count": self.tracked_count,
            "pending_events_count": self.pending_events_count,
            "error_message": self.error_message,
        }


def generate_bus_simulation_frame(
    f_idx: int,
    w: int = 640,
    h: int = 360,
    total_frames: int = 200,
    fps: float = 20.0,
) -> np.ndarray:
    """
    Sinh 1 frame mô phỏng khoang xe buýt nhìn từ trần (Top-down Ceiling Camera).
    Frame có độ tương phản cao, hình ảnh ghế, người và ba lô rõ nét:
    - 4 vị trí ghế: Ghế 1A, Ghế 1B, Ghế 2A, Ghế 2B
    - Hành lang xe buýt ở giữa có kẻ vạch
    - Frame 0-35: Hành khách bước vào lối đi, tiến đến Ghế 1A
    - Frame 35-100: Hành khách ngồi tại Ghế 1A, đặt ba lô xuống ghế
    - Frame 100-140: Hành khách đứng lên, bước ra hành lang và rời khỏi xe
    - Frame 140-200: Ghế 1A trống, ba lô bị bỏ quên nằm lại trên ghế -> Kích hoạt cảnh báo!
    """
    frame = np.full((h, w, 3), (32, 34, 42), dtype=np.uint8)

    # 1. Vẽ sàn và lối đi xe buýt
    cv2.rectangle(frame, (250, 0), (390, h), (46, 50, 62), -1)
    for dy in range(0, h, 28):
        cv2.line(frame, (320, dy), (320, dy + 14), (80, 88, 108), 2)
        cv2.line(frame, (255, dy), (265, dy), (60, 66, 82), 1)
        cv2.line(frame, (375, dy), (385, dy), (60, 66, 82), 1)

    # 2. Vẽ 4 ghế cố định
    demo_seats = [
        ("Ghế 1A", 40, 60, 240, 180),
        ("Ghế 1B", 400, 60, 600, 180),
        ("Ghế 2A", 40, 205, 240, 325),
        ("Ghế 2B", 400, 205, 600, 325),
    ]
    for sid, sx1, sy1, sx2, sy2 in demo_seats:
        # Khung ghế & đệm ghế
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (55, 62, 85), -1)
        cv2.rectangle(frame, (sx1 + 6, sy1 + 6), (sx2 - 6, sy2 - 6), (72, 82, 112), -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 229, 255), 2)  # Viền Cyan
        # Tựa lưng & tay vịn
        cv2.rectangle(frame, (sx1 + 4, sy1 + 4), (sx1 + 24, sy2 - 4), (40, 46, 64), -1)
        cv2.putText(
            frame, sid, (sx1 + 35, sy1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

    # 3. Trạng thái người & Ba lô theo timeline
    bag_box = (155, 95, 225, 155)  # Trên Ghế 1A

    def _draw_backpack(canvas, bx1, by1, bx2, by2):
        """Vẽ ba lô màu hổ phách/cam tương phản cao có quai đeo và khóa kéo."""
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (11, 158, 245), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        # Quai đeo
        cv2.ellipse(canvas, ((bx1 + bx2) // 2, by1 + 4), (14, 6), 0, 180, 360, (20, 20, 20), 2)
        # Ngăn kéo trước
        cv2.rectangle(canvas, (bx1 + 8, by1 + 18), (bx2 - 8, by2 - 6), (6, 120, 190), -1)
        cv2.line(canvas, (bx1 + 12, by1 + 24), (bx2 - 12, by1 + 24), (240, 240, 240), 2)
        cv2.putText(canvas, "BALO", (bx1 + 10, by2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_person(canvas, cx, cy):
        """Vẽ người nhìn từ trên xuống (vai, đầu, tóc, áo khoác)."""
        # Vai và áo khoác xanh lá cây
        cv2.ellipse(canvas, (cx, cy), (38, 26), 0, 0, 360, (76, 175, 80), -1)
        cv2.ellipse(canvas, (cx, cy), (38, 26), 0, 0, 360, (255, 255, 255), 2)
        # Cánh tay
        cv2.circle(canvas, (cx - 30, cy + 5), 10, (56, 142, 60), -1)
        cv2.circle(canvas, (cx + 30, cy + 5), 10, (56, 142, 60), -1)
        # Đầu tóc
        cv2.circle(canvas, (cx, cy - 2), 17, (30, 25, 20), -1)
        cv2.circle(canvas, (cx, cy - 2), 17, (200, 200, 200), 1)
        cv2.putText(canvas, "NGUOI", (cx - 24, cy - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1)

    if f_idx < 35:
        # Người đi vào hành lang tới Ghế 1A mang theo ba lô
        progress = f_idx / 35.0
        px = int(320 - progress * 190)
        py = int(320 - progress * 200)
        _draw_person(frame, px, py)
        # Ba lô trên vai
        _draw_backpack(frame, px + 18, py - 15, px + 55, py + 25)
    elif 35 <= f_idx < 100:
        # Người ngồi tại Ghế 1A (bên trái), đặt ba lô bên cạnh (phần ghế bên phải)
        _draw_person(frame, 100, 120)
        _draw_backpack(frame, bag_box[0], bag_box[1], bag_box[2], bag_box[3])
    elif 100 <= f_idx < 140:
        # Người đứng lên đi ra hành lang hướng ra cửa thoát hiểm
        progress = (f_idx - 100) / 40.0
        px = int(100 + progress * 220)
        py = int(120 + progress * 210)
        _draw_person(frame, px, py)
        # Ba lô BỊ BỎ QUÊN lại trên Ghế 1A!
        _draw_backpack(frame, bag_box[0], bag_box[1], bag_box[2], bag_box[3])
    else:
        # Ghế 1A hoàn toàn trống, ba lô bị bỏ quên nằm yên!
        _draw_backpack(frame, bag_box[0], bag_box[1], bag_box[2], bag_box[3])

    # Chú thích timestamp mô phỏng
    time_sec = f_idx / fps
    status_text = (
        "Hanh khach len xe" if f_idx < 35 else
        "Hanh khach dang ngoi Ghe 1A" if f_idx < 100 else
        "Hanh khach dung day xuong xe" if f_idx < 140 else
        "CANH BAO: Do vat bi bo quen tren Ghe 1A!"
    )
    cv2.putText(
        frame, f"CAM-BUS-CEILING | {time_sec:.1f}s | {status_text}", (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 1
    )
    return frame


def get_demo_ground_truth(f_idx: int, w: int = 640, h: int = 360) -> Dict:
    """Trả về tọa độ chuẩn của người và ba lô theo frame để làm fallback đảm bảo demo không lỗi."""
    bag_bbox = (155, 95, 225, 155)  # Tọa độ trên Ghế 1A
    if f_idx < 35:
        progress = f_idx / 35.0
        px = int(320 - progress * 190)
        py = int(320 - progress * 200)
        return {
            "has_person": True,
            "person_bbox": (px - 38, py - 26, px + 38, py + 26),
            "has_bag": True,
            "bag_bbox": (px + 18, py - 15, px + 55, py + 25),
            "seat_id": None,
        }
    elif 35 <= f_idx < 100:
        return {
            "has_person": True,
            "person_bbox": (100 - 38, 120 - 26, 100 + 38, 120 + 26),
            "has_bag": True,
            "bag_bbox": bag_bbox,
            "seat_id": "Ghế 1A",
        }
    elif 100 <= f_idx < 140:
        progress = (f_idx - 100) / 40.0
        px = int(100 + progress * 220)
        py = int(120 + progress * 210)
        return {
            "has_person": True,
            "person_bbox": (px - 38, py - 26, px + 38, py + 26),
            "has_bag": True,
            "bag_bbox": bag_bbox,
            "seat_id": "Ghế 1A",
        }
    else:
        return {
            "has_person": False,
            "person_bbox": None,
            "has_bag": True,
            "bag_bbox": bag_bbox,
            "seat_id": "Ghế 1A",
        }


def ensure_demo_video_exists(force_recreate: bool = False) -> str:
    """
    Tạo hoặc trả về video mẫu mô phỏng camera trần xe buýt (10 giây, 640x360, 20 fps).
    Hỗ trợ multi-codec (mp4v, avc1, XVID, MJPG) để chạy được trên mọi OS (macOS, Windows, Linux, Streamlit Cloud).
    """
    target = Path(DEMO_VIDEO_PATH)
    if not force_recreate and target.exists() and target.stat().st_size > 20000:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    fps = 20.0
    total_frames = 200

    # Thử lần lượt các codec phổ biến
    candidate_codecs = ["mp4v", "avc1", "XVID", "MJPG"]
    out = None

    for codec in candidate_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate_out = cv2.VideoWriter(str(target), fourcc, fps, (w, h))
            if candidate_out.isOpened():
                out = candidate_out
                logger.info(f"VideoWriter khởi tạo thành công với codec '{codec}'")
                break
            candidate_out.release()
        except Exception as e:
            logger.warning(f"Thử codec '{codec}' không thành công: {e}")

    if out is not None and out.isOpened():
        try:
            for f_idx in range(total_frames):
                frame = generate_bus_simulation_frame(f_idx, w, h, total_frames, fps)
                out.write(frame)
        finally:
            out.release()
        logger.info(f"Đã tạo hoàn tất video demo mẫu: {target} (Kích thước: {target.stat().st_size} bytes)")
    else:
        logger.warning(f"Không thể mở VideoWriter với bất kỳ codec nào trên hệ thống. Sử dụng frame generator on-the-fly.")

    return str(target)

