"""
src/utils/text_renderer.py

Bộ kết xuất văn bản hỗ trợ Unicode (tiếng Việt có dấu) chuyên dụng cho OpenCV frames.
Khắc phục triệt để hạn chế của cv2.putText() vốn dùng Hershey fonts ASCII và biến các ký tự
có dấu tiếng Việt thành dấu chấm hỏi (?).

Tính năng:
- Tự động phát hiện và nạp font TrueType hỗ trợ tiếng Việt trên macOS, Linux và Windows.
- Sử dụng functools.lru_cache để lưu trữ font đã nạp, tránh đọc file disk trên mỗi frame.
- Hỗ trợ vẽ nhãn kèm nền (background badge/pill) với độ tương phản cao.
- Sử dụng kỹ thuật crop-and-paste cục bộ hoặc vẽ toàn khung đảm bảo hiệu năng cao (>100 FPS).
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Danh sách đường dẫn font ưu tiên hỗ trợ Unicode tiếng Việt đầy đủ
CANDIDATE_FONT_PATHS: List[str] = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFCompact.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    # Linux / Ubuntu / Debian
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


@lru_cache(maxsize=1)
def find_system_unicode_font() -> Optional[str]:
    """Tìm đường dẫn font TrueType đầu tiên trên hệ thống hỗ trợ tiếng Việt."""
    for path_str in CANDIDATE_FONT_PATHS:
        p = Path(path_str)
        if p.exists() and p.is_file():
            try:
                # Kiểm tra font có nạp được không
                ImageFont.truetype(str(p), 14)
                return str(p)
            except Exception:
                continue

    # Thử tìm font trong matplotlib nếu có cài đặt
    try:
        import matplotlib
        mpl_font = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if mpl_font.exists():
            return str(mpl_font)
    except Exception:
        pass

    return None


@lru_cache(maxsize=64)
def get_cached_font(font_size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Lấy font từ bộ nhớ cache theo kích cỡ. Không đọc lại file từ ổ cứng."""
    resolved_path = font_path or find_system_unicode_font()
    if resolved_path:
        try:
            return ImageFont.truetype(resolved_path, font_size)
        except Exception as e:
            logger.warning(f"Không thể nạp font {resolved_path}: {e}")

    # Fallback cuối cùng
    return ImageFont.load_default()


def measure_text_unicode(
    text: str,
    font_size: int = 14,
    font_path: Optional[str] = None,
) -> Tuple[int, int]:
    """Đo kích thước (width, height) tính bằng pixel của chuỗi văn bản."""
    font = get_cached_font(font_size, font_path)
    try:
        bbox = font.getbbox(text)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        return (w, h)
    except Exception:
        # Fallback cho ImageFont cũ
        return (len(text) * int(font_size * 0.6), font_size)


def draw_text_unicode(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 14,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 3,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """
    Vẽ văn bản Unicode (tiếng Việt có dấu) trực tiếp lên frame BGR của OpenCV.

    Args:
        frame: Ảnh OpenCV BGR (sửa đổi in-place hoặc trả về bản sao)
        text: Chuỗi văn bản tiếng Việt
        pos: (x, y) - Tọa độ góc trên bên trái của văn bản hoặc badge
        font_size: Cỡ chữ (pixel)
        text_color: Màu chữ BGR (mặc định trắng (255, 255, 255))
        bg_color: Màu nền BGR tùy chọn. Nếu có, vẽ hình chữ nhật nền
        padding: Khoảng đệm xung quanh chữ khi có nền
        font_path: Đường dẫn font tùy chọn

    Returns:
        Frame có chữ đã vẽ.
    """
    if frame is None or not text:
        return frame

    h_frame, w_frame = frame.shape[:2]
    x, y = int(pos[0]), int(pos[1])

    font = get_cached_font(font_size, font_path)
    text_w, text_h = measure_text_unicode(text, font_size, font_path)

    # Tính toán bounding box của vùng vẽ
    if bg_color is not None:
        rx1 = max(0, x - padding)
        ry1 = max(0, y - padding)
        rx2 = min(w_frame, x + text_w + padding * 2)
        ry2 = min(h_frame, y + text_h + padding * 2)
        text_offset_x = x - rx1
        text_offset_y = y - ry1
    else:
        rx1 = max(0, x)
        ry1 = max(0, y)
        rx2 = min(w_frame, x + text_w + 4)
        ry2 = min(h_frame, y + text_h + 4)
        text_offset_x = 0
        text_offset_y = 0

    patch_w = rx2 - rx1
    patch_h = ry2 - ry1

    if patch_w <= 0 or patch_h <= 0:
        return frame

    # Crop vùng nhỏ cần vẽ
    patch = frame[ry1:ry2, rx1:rx2].copy()

    # Chuyển đổi BGR -> RGB cho PIL
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Chuyển màu BGR sang RGB cho PIL
    rgb_text_color = (int(text_color[2]), int(text_color[1]), int(text_color[0]))

    if bg_color is not None:
        rgb_bg_color = (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))
        draw.rectangle(
            [(0, 0), (patch_w - 1, patch_h - 1)],
            fill=rgb_bg_color,
        )

    # Vẽ chữ tiếng Việt chuẩn xác
    draw.text(
        (text_offset_x, text_offset_y),
        text,
        font=font,
        fill=rgb_text_color,
    )

    # Chuyển ngược RGB -> BGR và gán lại vào frame
    res_patch = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[ry1:ry2, rx1:rx2] = res_patch

    return frame
