"""
src/database/db.py
Module phụ trách kết nối SQLite, khởi tạo schema và quản lý vòng đời kết nối.
Database được tạo tự động nếu chưa tồn tại.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from config.settings import DATABASE_PATH

logger = get_logger(__name__)

# Schema cho bảng analyses (lịch sử phân tích)
CREATE_ANALYSES_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT UNIQUE NOT NULL,
    video_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    total_frames INTEGER DEFAULT 0,
    total_objects INTEGER DEFAULT 0,
    total_events INTEGER DEFAULT 0,
    output_video_path TEXT DEFAULT '',
    status TEXT DEFAULT 'completed'
);
"""

# Schema cho bảng events (sự kiện nghi ngờ bỏ quên)
CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    analysis_id TEXT,
    timestamp TEXT NOT NULL,
    class_name TEXT NOT NULL,
    track_id INTEGER DEFAULT -1,
    last_track_id INTEGER DEFAULT -1,
    seat_id TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    duration_seconds REAL DEFAULT 0.0,
    status TEXT DEFAULT 'monitoring',
    video_name TEXT DEFAULT '',
    video_time_sec REAL DEFAULT 0.0,
    frame_number INTEGER DEFAULT 0,
    snapshot_path TEXT DEFAULT '',
    bbox_x1 INTEGER DEFAULT 0,
    bbox_y1 INTEGER DEFAULT 0,
    bbox_x2 INTEGER DEFAULT 0,
    bbox_y2 INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    first_seen_timestamp TEXT DEFAULT '',
    last_seen_timestamp TEXT DEFAULT '',
    suspected_timestamp TEXT DEFAULT '',
    occlusion_start_timestamp TEXT DEFAULT '',
    occlusion_duration REAL DEFAULT 0.0,
    observed_duration REAL DEFAULT 0.0,
    vacant_duration REAL DEFAULT 0.0,
    last_known_location TEXT DEFAULT '',
    evidence_metadata TEXT DEFAULT '{}',
    event_history TEXT DEFAULT '[]',
    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
);
"""

# Schema cho bảng cấu hình vùng ghế (Seat ROI)
CREATE_SEATS_TABLE = """
CREATE TABLE IF NOT EXISTS seat_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seat_id TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    is_normalized INTEGER DEFAULT 0,
    polygon_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# Schema cho bảng alerts (Cảnh báo sinh ra cho nhân viên)
CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    event_id TEXT NOT NULL,
    seat_id TEXT DEFAULT '',
    object_id TEXT DEFAULT '',
    person_id TEXT DEFAULT '',
    priority TEXT DEFAULT 'MEDIUM',
    status TEXT DEFAULT 'PENDING',
    reason TEXT DEFAULT '',
    snapshot_path TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    sent_channels TEXT DEFAULT '[]'
);
"""

# Schema cho bảng notification_logs (Lịch sử gửi thông báo đa kênh)
CREATE_NOTIFICATION_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS notification_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT,
    channel TEXT NOT NULL,
    recipient TEXT DEFAULT '',
    status TEXT NOT NULL,
    status_code INTEGER DEFAULT 0,
    error_message TEXT DEFAULT '',
    payload TEXT DEFAULT '{}',
    sent_at TEXT NOT NULL
);
"""

# Schema cho bảng dataset_images (Ảnh huấn luyện AI)
CREATE_DATASET_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS dataset_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    split TEXT DEFAULT 'train',
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    source_type TEXT DEFAULT 'upload',
    is_labeled INTEGER DEFAULT 0,
    quality_tag TEXT DEFAULT 'good',
    created_at TEXT NOT NULL
);
"""

# Schema cho bảng annotations (Nhãn đối tượng bounding box)
CREATE_ANNOTATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id TEXT UNIQUE NOT NULL,
    image_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    bbox_x1 REAL DEFAULT 0.0,
    bbox_y1 REAL DEFAULT 0.0,
    bbox_x2 REAL DEFAULT 0.0,
    bbox_y2 REAL DEFAULT 0.0,
    is_occluded INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES dataset_images(image_id)
);
"""

# Schema cho bảng model_versions (Quản lý các phiên bản Model AI)
CREATE_MODEL_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,
    version TEXT NOT NULL,
    dataset_version TEXT DEFAULT 'v1.0',
    classes TEXT DEFAULT '[]',
    metrics_json TEXT DEFAULT '{}',
    confidence_threshold REAL DEFAULT 0.35,
    status TEXT DEFAULT 'archived',
    artifact_path TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""

# Schema cho bảng system_config (Lưu cấu hình hệ thống dạng key-value)
CREATE_SYSTEM_CONFIG_TABLE = """
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Database:
    """Quản lý kết nối và khởi tạo SQLite database."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATABASE_PATH)
        self._connection: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        """
        Tạo kết nối tới database. Tạo file và schema nếu chưa tồn tại.

        Returns:
            True nếu kết nối thành công.
        """
        try:
            # Đảm bảo thư mục cha tồn tại
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Cho phép dùng từ nhiều thread (Streamlit)
            )
            self._connection.row_factory = sqlite3.Row  # Trả kết quả dạng dict-like
            self._connection.execute("PRAGMA journal_mode=WAL")  # Performance tốt hơn

            # Tạo bảng nếu chưa có và tự động chạy migration an toàn
            self._init_schema()

            logger.info(f"Kết nối database: {self.db_path}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Lỗi kết nối database: {e}")
            return False

    def _init_schema(self):
        """Tạo các bảng cần thiết và thực hiện migration an toàn nếu bảng cũ thiếu cột."""
        cursor = self._connection.cursor()
        cursor.execute(CREATE_ANALYSES_TABLE)
        cursor.execute(CREATE_EVENTS_TABLE)
        cursor.execute(CREATE_SEATS_TABLE)
        cursor.execute(CREATE_ALERTS_TABLE)
        cursor.execute(CREATE_NOTIFICATION_LOGS_TABLE)
        cursor.execute(CREATE_DATASET_IMAGES_TABLE)
        cursor.execute(CREATE_ANNOTATIONS_TABLE)
        cursor.execute(CREATE_MODEL_VERSIONS_TABLE)
        cursor.execute(CREATE_SYSTEM_CONFIG_TABLE)
        self._connection.commit()

        # Thực hiện migration kiểm tra các cột mới cho bảng events
        self._migrate_events_table(cursor)
        self._init_default_data(cursor)
        self._connection.commit()
        logger.debug("Schema database đã được khởi tạo và kiểm tra migration")

    def _init_default_data(self, cursor: sqlite3.Cursor):
        """Khởi tạo phiên bản model mặc định và cấu hình ban đầu nếu chưa có."""
        cursor.execute("SELECT COUNT(*) AS cnt FROM model_versions")
        if cursor.fetchone()["cnt"] == 0:
            import json
            from datetime import datetime
            now_iso = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO model_versions (
                    model_id, version, dataset_version, classes, metrics_json,
                    confidence_threshold, status, artifact_path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "yolov8n-coco-v1.0",
                    "v1.0",
                    "COCO-Pretrained",
                    json.dumps(["person", "backpack", "handbag", "suitcase", "umbrella", "bottle", "cell phone", "laptop", "book"]),
                    json.dumps({"map50": 0.82, "precision": 0.79, "recall": 0.78}),
                    0.35,
                    "active",
                    "yolov8n.pt",
                    "Mô hình YOLOv8n chuẩn COCO tối ưu tốc độ cho xe buýt",
                    now_iso,
                ),
            )

    def _migrate_events_table(self, cursor: sqlite3.Cursor):
        """Kiểm tra và thêm các cột mới vào bảng events nếu đang dùng database cũ."""
        cursor.execute("PRAGMA table_info(events)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        new_columns = [
            ("last_track_id", "INTEGER DEFAULT -1"),
            ("seat_id", "TEXT DEFAULT ''"),
            ("first_seen_timestamp", "TEXT DEFAULT ''"),
            ("last_seen_timestamp", "TEXT DEFAULT ''"),
            ("suspected_timestamp", "TEXT DEFAULT ''"),
            ("occlusion_start_timestamp", "TEXT DEFAULT ''"),
            ("occlusion_duration", "REAL DEFAULT 0.0"),
            ("observed_duration", "REAL DEFAULT 0.0"),
            ("vacant_duration", "REAL DEFAULT 0.0"),
            ("last_known_location", "TEXT DEFAULT ''"),
            ("evidence_metadata", "TEXT DEFAULT '{}'"),
            ("event_history", "TEXT DEFAULT '[]'"),
        ]

        for col_name, col_def in new_columns:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE events ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Migration: Đã thêm cột '{col_name}' vào bảng events")
                except sqlite3.Error as e:
                    logger.warning(f"Không thể thêm cột {col_name}: {e}")


    def get_connection(self) -> Optional[sqlite3.Connection]:
        """Trả về kết nối hiện tại."""
        return self._connection

    def close(self):
        """Đóng kết nối database."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Đã đóng kết nối database")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
