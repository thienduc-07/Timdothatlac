"""
src/utils/file_utils.py
Tiện ích kiểm tra file, tạo tên file an toàn và quản lý đường dẫn.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict

from config.settings import SUPPORTED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE_MB
from src.utils.logger import get_logger

logger = get_logger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Tạo tên file an toàn bằng cách loại bỏ ký tự đặc biệt.

    Args:
        filename: Tên file gốc

    Returns:
        Tên file đã được làm sạch
    """
    # Chuẩn hóa unicode
    filename = unicodedata.normalize("NFKD", filename)
    # Giữ lại chữ cái, số, dấu gạch ngang, dấu gạch dưới và dấu chấm
    filename = re.sub(r"[^\w\-.]", "_", filename)
    # Loại bỏ nhiều dấu gạch dưới liên tiếp
    filename = re.sub(r"_+", "_", filename)
    # Giới hạn độ dài
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    return name + ext


def validate_video_file(file_path: str) -> Dict[str, object]:
    """
    Kiểm tra tính hợp lệ của file video.

    Args:
        file_path: Đường dẫn tới file video

    Returns:
        Dict với keys: 'valid' (bool), 'error' (str hoặc None),
        'size_mb' (float), 'extension' (str)
    """
    result = {"valid": False, "error": None, "size_mb": 0.0, "extension": ""}

    path = Path(file_path)

    # Kiểm tra file tồn tại
    if not path.exists():
        result["error"] = f"File không tồn tại: {path.name}"
        return result

    # Kiểm tra phần mở rộng
    ext = path.suffix.lower()
    result["extension"] = ext
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        result["error"] = (
            f"Định dạng '{ext}' không được hỗ trợ. "
            f"Các định dạng cho phép: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
        )
        return result

    # Kiểm tra dung lượng
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    result["size_mb"] = round(size_mb, 2)

    if size_mb == 0:
        result["error"] = "File video rỗng (0 bytes)"
        return result

    if size_mb > MAX_VIDEO_SIZE_MB:
        result["error"] = (
            f"File quá lớn ({size_mb:.1f} MB). "
            f"Giới hạn: {MAX_VIDEO_SIZE_MB} MB"
        )
        return result

    result["valid"] = True
    logger.info(f"File hợp lệ: {path.name} ({size_mb:.1f} MB)")
    return result


def ensure_directory(dir_path: Path) -> bool:
    """
    Tạo thư mục nếu chưa tồn tại.

    Returns:
        True nếu thư mục tồn tại hoặc được tạo thành công
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Không thể tạo thư mục {dir_path}: {e}")
        return False


def get_unique_output_path(output_dir: Path, original_name: str, suffix: str = "_analyzed", extension: str = ".mp4") -> Path:
    """
    Tạo đường dẫn output duy nhất, tránh ghi đè file có sẵn.

    Args:
        output_dir: Thư mục đầu ra
        original_name: Tên file gốc (không có extension)
        suffix: Hậu tố thêm vào
        extension: Phần mở rộng file

    Returns:
        Path duy nhất cho file đầu ra
    """
    ensure_directory(output_dir)
    safe_name = sanitize_filename(original_name)
    base_name = Path(safe_name).stem

    output_path = output_dir / f"{base_name}{suffix}{extension}"
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{base_name}{suffix}_{counter}{extension}"
        counter += 1

    return output_path


def format_file_size(size_bytes: int) -> str:
    """Chuyển đổi kích thước bytes sang dạng đọc được."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """Chuyển đổi thời lượng (giây) sang dạng mm:ss hoặc hh:mm:ss."""
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
