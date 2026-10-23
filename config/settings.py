"""
config/settings.py
Quản lý tập trung các thông số cấu hình cho toàn bộ dự án Lost & Found AI.
Mọi giá trị cấu hình nên được import từ file này thay vì hard-code trong các module.
"""

import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Load environment variables from .env if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or .env missing – ignore
    pass

def get_secret_or_env(key: str, default: str = "") -> str:
    """
    Lấy giá trị cấu hình an toàn từ biến môi trường (os.getenv),
    Streamlit Session State, hoặc Streamlit Secrets (st.secrets) trên Streamlit Cloud.
    Tự động đồng bộ vào os.environ để các module phụ thuộc cũng nhận diện được.
    """
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "session_state") and st.session_state.get(key):
                val = str(st.session_state.get(key)).strip()
            elif hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key]).strip()
            if val:
                os.environ[key] = val
        except Exception:
            pass
    return val.strip() if val else default

# ──────────────────────────────────────────────
# Múi giờ hệ thống (Timezone Settings)
# Mặc định: Giờ Việt Nam (Asia/Ho_Chi_Minh - GMT+7)
# ──────────────────────────────────────────────
DEFAULT_TIMEZONE_NAME = os.environ.get("APP_TIMEZONE", "Asia/Ho_Chi_Minh")
os.environ["TZ"] = DEFAULT_TIMEZONE_NAME
if hasattr(time, "tzset"):
    try:
        time.tzset()
    except Exception:
        pass

try:
    from zoneinfo import ZoneInfo
    APP_TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = timezone(timedelta(hours=7))


def get_current_time() -> datetime:
    """Trả về datetime hiện tại chuẩn theo múi giờ hệ thống (mặc định GMT+7 Việt Nam)."""
    return datetime.now(APP_TIMEZONE)

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
# YOLO Model & Instance Segmentation (YOLO26 / YOLO-seg)
# ──────────────────────────────────────────────
# Model Instance Segmentation mặc định (YOLO26-seg)
YOLO_SEGMENTATION_MODEL_NAME = "yolo26n-seg.pt"

# Đường dẫn weights custom ưu tiên cho model Instance Segmentation (3 lớp: seat, person, item)
CUSTOM_SEGMENTATION_WEIGHTS_PATH = MODELS_DIR / "best.pt"

# Danh sách các file weights segmentation được ưu tiên tìm kiếm
SUPPORTED_CUSTOM_WEIGHTS_PATHS = [
    MODELS_DIR / "best.pt",
    PROJECT_ROOT / "yolo26n-seg.pt",
    MODELS_DIR / "yolo26n-seg.pt",
    PROJECT_ROOT / "static/models/yolo26n-seg.onnx",
    MODELS_DIR / "yolo26n-seg.onnx",
    MODELS_DIR / "bus_seat_yolo.pt",
    MODELS_DIR / "bus_seat_yolov8.pt",
]

# Model fallback: Tự động ưu tiên yolo26n-seg.pt hoặc yolo26n-seg.onnx (Instance Segmentation) nếu có trong project
YOLO_MODEL_NAME = os.getenv(
    "YOLO_MODEL_NAME",
    "yolo26n-seg.pt" if (PROJECT_ROOT / "yolo26n-seg.pt").exists() else (
        str(PROJECT_ROOT / "static/models/yolo26n-seg.onnx")
        if (PROJECT_ROOT / "static/models/yolo26n-seg.onnx").exists()
        else "yolov8n-seg.pt"
    ),
)

# Cấu hình hiển thị Segmentation Mask
DEFAULT_SHOW_SEGMENTATION_MASKS = True
SEGMENTATION_MASK_ALPHA = 0.35

# Confidence threshold mặc định cho object detection & segmentation
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

# Thiết bị: "cpu" hoặc "cuda" (tự động fallback về cpu nếu không có GPU)
DEVICE = os.environ.get("DEVICE", "cpu")
# Enable Supervision visual utilities (set to 'false' to disable)
USE_SUPERVISION = os.getenv("USE_SUPERVISION", "true").lower() == "true"

# Inference mode – local YOLO, Roboflow serverless, or hybrid (local+Roboflow)
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # "local", "roboflow", "hybrid"

# ──────────────────────────────────────────────
# Hybrid AI Inference (Client WebGPU/WASM + Server Fallback)
# ──────────────────────────────────────────────
HYBRID_INFERENCE_MODE = os.getenv("HYBRID_INFERENCE_MODE", "auto").lower()  # "auto", "on_device", "server"
ONNX_SEGMENTATION_MODEL_NAME = "yolo26n-seg.onnx"
ONNX_MODEL_PATH = MODELS_DIR / ONNX_SEGMENTATION_MODEL_NAME
CLIENT_MODEL_URL = os.getenv("CLIENT_MODEL_URL", "/app/static/models/yolo26n-seg.onnx")
CLIENT_PERF_THRESHOLD_FPS = float(os.getenv("CLIENT_PERF_THRESHOLD_FPS", "8.0"))
BACKEND_SWITCH_COOLDOWN_SEC = float(os.getenv("BACKEND_SWITCH_COOLDOWN_SEC", "5.0"))

# Path to YOLO weights – can be overridden via env var; defaults to available segmentation weights.
DEFAULT_MODEL_PATH = (
    str(CUSTOM_SEGMENTATION_WEIGHTS_PATH)
    if CUSTOM_SEGMENTATION_WEIGHTS_PATH.exists()
    else (
        str(PROJECT_ROOT / "yolo26n-seg.pt")
        if (PROJECT_ROOT / "yolo26n-seg.pt").exists()
        else (
            str(PROJECT_ROOT / "static/models/yolo26n-seg.onnx")
            if (PROJECT_ROOT / "static/models/yolo26n-seg.onnx").exists()
            else (
                str(PROJECT_ROOT / "yolo26n.pt")
                if (PROJECT_ROOT / "yolo26n.pt").exists()
                else "yolov8n-seg.pt"
            )
        )
    )
)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# Roboflow Serverless Workflow & Hosted Model Configuration
ROBOFLOW_API_KEY = get_secret_or_env("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKFLOW_URL = get_secret_or_env("ROBOFLOW_WORKFLOW_URL", "https://serverless.roboflow.com")
ROBOFLOW_WORKSPACE = get_secret_or_env("ROBOFLOW_WORKSPACE", "le-uc-s-workspace")
ROBOFLOW_PROJECT_ID = get_secret_or_env("ROBOFLOW_PROJECT_ID", "bus-seat-and-belongings-monitori")
ROBOFLOW_WORKFLOW_ID = get_secret_or_env("ROBOFLOW_WORKFLOW_ID", "bus-seat-and-belongings-monitori")
ROBOFLOW_MODEL_VERSION = get_secret_or_env("ROBOFLOW_MODEL_VERSION", "1")

if ROBOFLOW_API_KEY:
    os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY


# ──────────────────────────────────────────────
# Chế độ hoạt động của hệ thống
# ──────────────────────────────────────────────
class SystemMode:
    DEMO = "DEMO"            # Chế độ chạy video mẫu minh họa
    PROCESSING = "PROCESSING"# Chế độ đang phân tích video file
    LIVE = "LIVE"            # Chế độ giám sát trực tiếp từ Camera / RTSP

# ──────────────────────────────────────────────
# Chuẩn màu sắc cố định cho Computer Vision (BGR cho OpenCV, HEX cho Web)
# ──────────────────────────────────────────────
# 1. GHẾ (SEATS) - Cyan / Xanh dương
COLOR_SEAT_BGR = (255, 229, 0)         # Cyan sáng trong BGR
COLOR_SEAT_HEX = "#00E5FF"
COLOR_SEAT_OCCUPIED_BGR = (255, 120, 0)# Xanh dương đậm khi có người
COLOR_SEAT_OCCUPIED_HEX = "#0284C7"
COLOR_SEAT_VACANT_BGR = (255, 229, 0)
COLOR_SEAT_VACANT_HEX = "#00E5FF"
COLOR_SEAT_UNKNOWN_BGR = (160, 160, 160)
COLOR_SEAT_UNKNOWN_HEX = "#94A3B8"

# 2. NGƯỜI (PERSONS) - Xanh lá cây (Green)
COLOR_PERSON_BGR = (76, 217, 100)      # Green trong BGR
COLOR_PERSON_HEX = "#10B981"

# 3. ĐỒ VẬT (OBJECTS) - Hổ phách / Vàng cam (Amber)
COLOR_OBJECT_BGR = (11, 158, 245)      # Amber trong BGR
COLOR_OBJECT_HEX = "#F59E0B"
COLOR_OBJECT_OCCLUDED_BGR = (180, 100, 240) # Tím nhạt cho đồ bị che khuất
COLOR_OBJECT_OCCLUDED_HEX = "#A855F7"

# 4. CẢNH BÁO / LỖI KHẨN CẤP - Đỏ
COLOR_ALERT_BGR = (68, 68, 239)        # Đỏ trong BGR
COLOR_ALERT_HEX = "#EF4444"

# ──────────────────────────────────────────────
# Các lớp vật thể COCO & Mở rộng (Flexible Object Detection)
# ──────────────────────────────────────────────
ITEM_CLASSES = [
    "item",         # Đồ vật tổng quát (từ custom model YOLO26 Instance Segmentation: seat, person, item)
    "backpack",     # Ba lô
    "handbag",      # Túi xách
    "suitcase",     # Vali
    "umbrella",     # Ô / Dù
    "bottle",       # Chai / Bình nước
    "cell phone",   # Điện thoại
    "laptop",       # Máy tính xách tay
    "book",         # Sách / Vở
]

# Các lớp mở rộng hỗ trợ nhận diện bổ sung hoặc human review
EXTENDED_ITEM_CLASSES = [
    "paper_bag",    # Túi giấy / túi nilon
    "box",          # Hộp / Thùng nhỏ
    "jacket",       # Áo khoác / quần áo
    "hat",          # Mũ / Nón
    "lunch_box",    # Hộp cơm / Hộp đồ ăn
    "wallet",       # Ví tiền
    "unknown_object"# Đồ vật chưa xác định rõ lớp
]

PERSON_CLASS = "person"
SEAT_CLASS = "seat"

# Tên tiếng Việt hiển thị
CLASS_NAMES_VI = {
    "person": "Người",
    "item": "Đồ vật / Hành lý",
    "seat": "Ghế xe buýt",
    "bus_seat": "Ghế xe buýt",
    "empty_seat": "Ghế trống",
    "human_occupied": "Ghế có người",
    "object_occupied": "Ghế có đồ",
    "backpack": "Ba lô",
    "handbag": "Túi xách",
    "suitcase": "Vali",
    "umbrella": "Ô/Dù",
    "bottle": "Bình nước/Chai",
    "cell phone": "Điện thoại",
    "laptop": "Laptop",
    "book": "Sách",
    "paper_bag": "Túi giấy/Nilon",
    "box": "Hộp/Thùng",
    "jacket": "Áo khoác",
    "hat": "Mũ/Nón",
    "lunch_box": "Hộp cơm",
    "wallet": "Ví tiền",
    "unknown_object": "Vật thể chưa rõ",
}

# ──────────────────────────────────────────────
# Cấu hình Thông báo & Cảnh báo (Alerts & Notifications)
# ──────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 60.0          # Tránh gửi cảnh báo trùng lặp cho cùng 1 ghế/vật trong 60s
DEFAULT_NOTIFICATION_RETRY_COUNT = 3  # Số lần thử lại tối đa khi gửi thông báo thất bại

# Cấu hình Telegram Bot
TELEGRAM_BOT_TOKEN = get_secret_or_env("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret_or_env("TELEGRAM_CHAT_ID", "")
TELEGRAM_ALERTS_ENABLED = get_secret_or_env("TELEGRAM_ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
TELEGRAM_ALERT_COOLDOWN_SECONDS = float(get_secret_or_env("TELEGRAM_ALERT_COOLDOWN_SECONDS", "60.0"))


# Mức độ ưu tiên
class AlertPriority:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AlertStatus:
    PENDING = "PENDING"
    ACTIVE = "PENDING"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    ITEM_RETRIEVED = "ITEM_RETRIEVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    RESOLVED = "RESOLVED"

# Kênh thông báo
class NotificationChannel:
    IN_APP = "in_app"
    WEB_PUSH = "web_push"
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"

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

