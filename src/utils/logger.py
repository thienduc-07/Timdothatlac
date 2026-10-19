"""
src/utils/logger.py
Thiết lập logging thống nhất cho toàn bộ dự án.
Ghi log ra file và console với mức độ có thể cấu hình.
"""

import logging
import sys
from config.settings import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Tạo và trả về một logger với tên cho trước.

    Args:
        name: Tên module (thường dùng __name__)

    Returns:
        Logger đã được cấu hình
    """
    logger = logging.getLogger(name)

    # Tránh thêm handler trùng lặp nếu logger đã được cấu hình
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Format log
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — không fail nếu không tạo được file log
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"Không thể tạo file log: {e}")

    return logger
