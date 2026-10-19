"""
config/settings.py
Quản lý tập trung các thông số cấu hình cho toàn bộ dự án Lost & Found AI.
Mọi giá trị cấu hình nên được import từ file này thay vì hard-code trong các module.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Đường dẫn gốc của dự án
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Các thư mục chính
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"
SNAPSHOTS_DIR = OUTPUTS_DIR / "snapshots"  # Ảnh chụp frame sự kiện

# Đảm bảo các thư mục tồn tại
for _dir in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, ASSETS_DIR, SNAPSHOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / "lost_and_found.db"

# ──────────────────────────────────────────────
# YOLO Model
# ──────────────────────────────────────────────
# Model nhỏ nhất, phù hợp máy cá nhân. Lần đầu chạy sẽ tự tải từ Ultralytics.
YOLO_MODEL_NAME = "yolov8n.pt"

# Confidence threshold mặc định cho object detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

# Thiết bị: "cpu" hoặc "cuda" (tự động fallback về cpu nếu không có GPU)
DEVICE = os.environ.get("DEVICE", "cpu")

# ──────────────────────────────────────────────
# Các lớp vật thể COCO cần theo dõi
# ──────────────────────────────────────────────
# Đây là các lớp từ COCO dataset mà YOLOv8 đã được huấn luyện.
# Chỉ liệt kê các lớp thực sự tồn tại trong model.

# Lớp đồ vật có khả năng bị bỏ quên
ITEM_CLASSES = [
    "backpack",     # Ba lô
    "handbag",      # Túi xách
    "suitcase",     # Vali
    "umbrella",     # Ô / Dù
    "bottle",       # Chai
    "cell phone",   # Điện thoại
    "laptop",       # Máy tính xách tay
    "book",         # Sách
]

# Lớp con người (dùng để theo dõi sự hiện diện của người)
PERSON_CLASS = "person"

# Tên tiếng Việt cho các lớp (dùng hiển thị trên giao diện)
CLASS_NAMES_VI = {
    "person": "Người",
    "backpack": "Ba lô",
    "handbag": "Túi xách",
    "suitcase": "Vali",
    "umbrella": "Ô/Dù",
    "bottle": "Chai",
    "cell phone": "Điện thoại",
    "laptop": "Laptop",
    "book": "Sách",
}

# ──────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────
# Tracker mặc định trong Ultralytics (ByteTrack)
TRACKER_TYPE = "bytetrack"

# ──────────────────────────────────────────────
# Forgotten-item detection (heuristic)
# ──────────────────────────────────────────────
# Thời gian tối thiểu (giây) vật thể đứng yên để bắt đầu nghi ngờ
MIN_STATIONARY_TIME_SEC = 10.0

# Ngưỡng di chuyển (pixel) — nếu vật thể di chuyển ít hơn ngưỡng này
# giữa các lần quan sát, coi như đứng yên
STATIONARY_MOVEMENT_THRESHOLD = 30.0

# Khoảng cách tối đa (pixel) giữa người và đồ vật để coi là "gần nhau"
PERSON_ITEM_PROXIMITY_THRESHOLD = 150.0

# Thời gian (giây) sau khi người rời đi mà đồ vật vẫn còn → tạo cảnh báo
ALERT_DELAY_AFTER_PERSON_LEFT_SEC = 5.0

# ──────────────────────────────────────────────
# Video
# ──────────────────────────────────────────────
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi"]
MAX_VIDEO_SIZE_MB = 500  # Cảnh báo nếu video vượt quá

# FPS tối đa để xử lý (skip frame nếu video FPS cao hơn)
MAX_PROCESSING_FPS = 15

# Resize frame về chiều rộng này nếu lớn hơn (để tăng tốc)
MAX_FRAME_WIDTH = 1280

# Codec đầu ra — mp4v tương thích tốt trên cả macOS và Windows
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_VIDEO_EXTENSION = ".mp4"

# ──────────────────────────────────────────────
# Cấu hình Seat ROI & Occlusion
# ──────────────────────────────────────────────
DEFAULT_SEATS_CONFIG_PATH = PROJECT_ROOT / "config" / "seats_config.json"

# Ngưỡng giao nhau (overlap ratio) giữa bounding box người và ROI của ghế để ước lượng có người
PERSON_SEAT_OVERLAP_THRESHOLD = 0.20

# Thời gian (giây) xác nhận trạng thái ghế có người (temporal smoothing)
SEAT_OCCUPIED_CONFIRM_SECONDS = 0.5

# Thời gian (giây) xác nhận trạng thái ghế đã trống sau khi không còn người
SEAT_VACANT_CONFIRM_SECONDS = 1.0

# Thời gian duy trì trạng thái (giây) khi detection người bị mất tạm thời (chống nhiễu)
SEAT_HOLD_SECONDS = 1.0

# Thời gian kiểm tra lại (giây) trong trạng thái RECHECK khi vùng quan sát nhìn thấy lại
RECHECK_DURATION_SECONDS = 3.0

# Khoảng cách tối đa (pixel) giữa vị trí detection mới và vị trí cũ để xem xét nối lại sự kiện
MAX_REASSOCIATION_DISTANCE_PX = 100.0

# Thời gian che khuất tối đa (giây) mà hệ thống vẫn có thể xem xét nối lại sự kiện cũ
MAX_OCCLUSION_REASSOCIATION_TIME_SEC = 180.0

# ──────────────────────────────────────────────
# Trạng thái sự kiện (State Machine)
# ──────────────────────────────────────────────
class EventStatus:
    """
    Các trạng thái của một sự kiện theo State Machine chuẩn:
    - OBSERVED: Vật thể đã được phát hiện trong khung hình.
    - ITEM_LEFT: Vật thể còn lại sau khi ghế được xác nhận trống.
    - SUSPECTED: Tình huống đã đủ điều kiện tạo cảnh báo nghi ngờ bỏ quên.
    - OCCLUDED: Vật thể không nhìn thấy và có bằng chứng vùng quan sát bị che khuất.
    - RECHECK: Vùng quan sát đã nhìn thấy lại và đang được xác minh.
    - REVIEW_REQUIRED: Không đủ bằng chứng để tự động kết luận, cần nhân viên kiểm tra.
    - RESOLVED: Sự kiện đã được kết thúc theo quy tắc rõ ràng (đã xử lý/xác nhận).
    - DISMISSED: Nhân viên đã loại bỏ cảnh báo.
    """
    OBSERVED = "observed"              # Đã phát hiện vật thể
    ITEM_LEFT = "item_left"            # Ghế trống nhưng đồ vật vẫn còn
    SUSPECTED = "suspected"            # Đủ điều kiện nghi ngờ bỏ quên
    OCCLUDED = "occluded"              # Bị che khuất (người ngồi lên / che khuất)
    RECHECK = "recheck"                # Đang kiểm tra lại khi ghế mở lại
    REVIEW_REQUIRED = "review_required"  # Cần nhân viên kiểm tra
    RESOLVED = "resolved"              # Đã kết thúc sự kiện
    DISMISSED = "dismissed"            # Nhân viên loại bỏ cảnh báo

    # Giữ tương thích ngược với code cũ
    MONITORING = "monitoring"          # Alias tương đương OBSERVED / ITEM_LEFT
    CONFIRMED = "confirmed"            # Alias tương đương RESOLVED (đã xác nhận)

# Tên hiển thị tiếng Việt và màu sắc đại diện cho từng trạng thái
EVENT_STATUS_META = {
    EventStatus.OBSERVED: {"label": "Phát hiện", "color": "#17a2b8", "badge": "👁"},
    EventStatus.ITEM_LEFT: {"label": "Đồ trên ghế trống", "color": "#fd7e14", "badge": "🪑"},
    EventStatus.SUSPECTED: {"label": "Nghi ngờ bỏ quên", "color": "#dc3545", "badge": "⚠️"},
    EventStatus.OCCLUDED: {"label": "Bị che khuất", "color": "#6f42c1", "badge": "🛡️"},
    EventStatus.RECHECK: {"label": "Đang xác minh lại", "color": "#ffc107", "badge": "🔄"},
    EventStatus.REVIEW_REQUIRED: {"label": "Cần nhân viên kiểm tra", "color": "#e83e8c", "badge": "📋"},
    EventStatus.RESOLVED: {"label": "Đã xử lý / Xác nhận", "color": "#28a745", "badge": "✅"},
    EventStatus.DISMISSED: {"label": "Đã loại bỏ", "color": "#6c757d", "badge": "❌"},
    EventStatus.MONITORING: {"label": "Đang theo dõi", "color": "#ffc107", "badge": "👁"},
    EventStatus.CONFIRMED: {"label": "Đã xác nhận", "color": "#28a745", "badge": "✅"},
}

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"

