"""
src/video/video_reader.py
Module phụ trách mở video, đọc metadata và đọc frame-by-frame.
Xử lý lỗi cho các trường hợp: file không tồn tại, codec không hỗ trợ,
video không có frame, FPS không hợp lệ.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple

from src.models import VideoMetadata
from src.utils.logger import get_logger
from config.settings import MAX_FRAME_WIDTH

logger = get_logger(__name__)


class VideoReader:
    """
    Đọc video từ file, trích xuất metadata và cung cấp frame-by-frame.
    Không tải toàn bộ video vào RAM.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._opened = False

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        return self._metadata

    @property
    def is_opened(self) -> bool:
        return self._opened

    def open(self) -> VideoMetadata:
        """
        Mở video và đọc metadata.

        Returns:
            VideoMetadata chứa thông tin video.
        """
        path = Path(self.file_path)
        meta = VideoMetadata(filename=path.name)

        if not path.exists():
            meta.error = f"File không tồn tại: {path.name}"
            self._metadata = meta
            return meta

        meta.file_size_bytes = path.stat().st_size

        try:
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                meta.error = "Không thể mở video. File có thể bị hỏng hoặc codec không được hỗ trợ."
                self._metadata = meta
                return meta

            meta.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta.fps = self._cap.get(cv2.CAP_PROP_FPS)
            meta.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Lấy codec fourcc
            fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            meta.codec = "".join(
                [chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]
            )

            # Validate
            if meta.fps <= 0 or meta.fps > 240:
                meta.error = f"FPS không hợp lệ: {meta.fps}"
                self._metadata = meta
                return meta

            if meta.total_frames <= 0:
                meta.error = "Video không có khung hình nào"
                self._metadata = meta
                return meta

            if meta.width <= 0 or meta.height <= 0:
                meta.error = f"Độ phân giải không hợp lệ: {meta.width}x{meta.height}"
                self._metadata = meta
                return meta

            meta.duration_seconds = meta.total_frames / meta.fps

            # Đọc thử frame đầu tiên
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                meta.error = "Không thể đọc frame đầu tiên"
                self._metadata = meta
                return meta

            # Reset về frame đầu
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            meta.is_valid = True
            self._opened = True
            self._metadata = meta

            logger.info(
                f"Mở video: {path.name} | "
                f"{meta.width}x{meta.height} | "
                f"{meta.fps:.1f} FPS | "
                f"{meta.total_frames} frames | "
                f"{meta.duration_seconds:.1f}s"
            )

        except Exception as e:
            meta.error = f"Lỗi khi mở video: {str(e)}"
            logger.error(meta.error)

        self._metadata = meta
        return meta

    def read_frames(
        self, skip_ratio: int = 1, max_width: int = MAX_FRAME_WIDTH
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Đọc frame-by-frame từ video (generator — không tải toàn bộ vào RAM).

        Args:
            skip_ratio: Đọc 1 frame trong mỗi skip_ratio frame (để tăng tốc).
                        Ví dụ: skip_ratio=2 → đọc mỗi frame thứ 2.
            max_width: Resize frame nếu chiều rộng lớn hơn giá trị này.

        Yields:
            Tuple (frame_index, frame) — frame là numpy array BGR.
        """
        if not self._opened or self._cap is None:
            logger.error("Video chưa được mở")
            return

        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if frame_index % skip_ratio == 0:
                # Resize nếu cần
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    new_w = max_width
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                yield frame_index, frame

            frame_index += 1

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Lấy frame đầu tiên làm ảnh xem trước."""
        if not self._opened or self._cap is None:
            return None

        current_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if ret and frame is not None:
            return frame
        return None

    def close(self):
        """Đóng video và giải phóng tài nguyên."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.debug("Đã đóng video reader")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
