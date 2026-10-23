"""
src/database/model_repository.py
Repository quản lý phiên bản model AI và cấu hình hệ thống trong SQLite.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3

from src.utils.logger import get_logger
from config.settings import get_current_time

logger = get_logger(__name__)


class ModelRepository:
    """Quản lý các phiên bản Model AI và System Config trong SQLite."""

    def __init__(self, db_connection: sqlite3.Connection):
        self._conn = db_connection

    def get_all_models(self) -> List[Dict]:
        """Lấy tất cả phiên bản model đã đăng ký."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                try:
                    item["classes"] = json.loads(item.get("classes", "[]"))
                    item["metrics"] = json.loads(item.get("metrics_json", "{}"))
                except Exception:
                    item["classes"] = []
                    item["metrics"] = {}
                result.append(item)
            return result
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy danh sách models: {e}")
            return []

    def get_active_model(self) -> Optional[Dict]:
        """Lấy model đang được kích hoạt sử dụng."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE status = 'active' LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            try:
                item["classes"] = json.loads(item.get("classes", "[]"))
                item["metrics"] = json.loads(item.get("metrics_json", "{}"))
            except Exception:
                item["classes"] = []
                item["metrics"] = {}
            return item
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy active model: {e}")
            return None

    def set_active_model(self, model_id: str) -> bool:
        """Kích hoạt một model và đưa các model khác về archived."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("UPDATE model_versions SET status = 'archived' WHERE status = 'active'")
            cursor.execute("UPDATE model_versions SET status = 'active' WHERE model_id = ?", (model_id,))
            self._conn.commit()
            logger.info(f"Đã kích hoạt model: {model_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi kích hoạt model: {e}")
            return False

    def add_model_version(
        self,
        model_id: str,
        version: str,
        dataset_version: str,
        classes: List[str],
        metrics: Dict,
        confidence_threshold: float = 0.35,
        status: str = "candidate",
        artifact_path: str = "",
        notes: str = "",
    ) -> bool:
        """Đăng ký một phiên bản model mới."""
        try:
            cursor = self._conn.cursor()
            now_iso = get_current_time().isoformat()
            cursor.execute(
                """
                INSERT INTO model_versions (
                    model_id, version, dataset_version, classes, metrics_json,
                    confidence_threshold, status, artifact_path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    version,
                    dataset_version,
                    json.dumps(classes),
                    json.dumps(metrics),
                    confidence_threshold,
                    status,
                    artifact_path,
                    notes,
                    now_iso,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi thêm model version: {e}")
            return False

    # ── System Config ──
    def get_config(self, key: str, default: str = "") -> str:
        """Lấy giá trị cấu hình theo key."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default
        except sqlite3.Error as e:
            logger.error(f"Lỗi đọc config {key}: {e}")
            return default

    def set_config(self, key: str, value: str) -> bool:
        """Lưu hoặc cập nhật giá trị cấu hình theo key."""
        try:
            cursor = self._conn.cursor()
            now_iso = get_current_time().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO system_config (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now_iso),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi lưu config {key}: {e}")
            return False
