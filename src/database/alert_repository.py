"""
src/database/alert_repository.py
Repository thao tác với bảng alerts và notification_logs.
Hỗ trợ lưu trữ, cập nhật trạng thái cảnh báo và nhật ký gửi tin nhắn thực tế.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3

from src.models import Alert
from src.utils.logger import get_logger
from config.settings import get_current_time

logger = get_logger(__name__)


class AlertRepository:
    """Quản lý CRUD cho alerts và notification logs trong SQLite."""

    def __init__(self, db_connection: sqlite3.Connection):
        self._conn = db_connection

    def save_alert(self, alert: Alert) -> bool:
        """Lưu hoặc cập nhật thông tin một Alert."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    alert_id, event_id, seat_id, object_id, person_id,
                    priority, status, reason, snapshot_path, created_at, sent_channels
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.event_id,
                    alert.seat_id,
                    alert.object_id,
                    alert.person_id,
                    alert.priority,
                    alert.status,
                    alert.reason,
                    alert.snapshot_path,
                    alert.created_at or get_current_time().isoformat(),
                    json.dumps(alert.sent_channels),
                ),
            )
            self._conn.commit()
            logger.info(f"Đã lưu Alert: {alert.alert_id} (Seat: {alert.seat_id}, Priority: {alert.priority})")
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lưu Alert: {e}")
            return False

    def get_recent_alerts(self, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """Lấy danh sách các alert gần nhất."""
        try:
            cursor = self._conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM alerts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                try:
                    item["sent_channels"] = json.loads(item.get("sent_channels", "[]"))
                except Exception:
                    item["sent_channels"] = []
                result.append(item)
            return result
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy danh sách Alert: {e}")
            return []

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Lấy chi tiết một alert theo ID."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            try:
                item["sent_channels"] = json.loads(item.get("sent_channels", "[]"))
            except Exception:
                item["sent_channels"] = []
            return item
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy Alert theo ID: {e}")
            return None

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Cập nhật trạng thái alert (PENDING, CONFIRMED, DISMISSED, RESOLVED)."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE alerts SET status = ? WHERE alert_id = ?",
                (new_status, alert_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi cập nhật trạng thái Alert: {e}")
            return False

    def get_unresolved_alert_count(self) -> int:
        """Đếm số lượng cảnh báo chưa xử lý (PENDING)."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE status = 'PENDING'")
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi đếm cảnh báo chưa xử lý: {e}")
            return 0

    def log_notification(
        self,
        alert_id: str,
        channel: str,
        recipient: str,
        status: str,
        status_code: int = 0,
        error_message: str = "",
        payload: Optional[Dict] = None,
    ) -> bool:
        """Ghi nhận nhật ký một lần gửi thông báo ra hệ thống bên ngoài."""
        try:
            cursor = self._conn.cursor()
            now_iso = get_current_time().isoformat()
            cursor.execute(
                """
                INSERT INTO notification_logs (
                    alert_id, channel, recipient, status,
                    status_code, error_message, payload, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    channel,
                    recipient,
                    status,
                    status_code,
                    error_message,
                    json.dumps(payload or {}),
                    now_iso,
                ),
            )
            # Nếu gửi thành công, cập nhật channel vào sent_channels của alert
            if status == "SUCCESS" and alert_id:
                cursor.execute("SELECT sent_channels FROM alerts WHERE alert_id = ?", (alert_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        channels = json.loads(row["sent_channels"] or "[]")
                    except Exception:
                        channels = []
                    if channel not in channels:
                        channels.append(channel)
                        cursor.execute(
                            "UPDATE alerts SET sent_channels = ? WHERE alert_id = ?",
                            (json.dumps(channels), alert_id),
                        )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lưu notification log: {e}")
            return False

    def get_notification_logs(self, limit: int = 50) -> List[Dict]:
        """Lấy lịch sử gửi thông báo."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM notification_logs ORDER BY sent_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy notification logs: {e}")
            return []
