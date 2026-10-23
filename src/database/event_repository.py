"""
src/database/event_repository.py
Module phụ trách các thao tác CRUD cho sự kiện và lịch sử phân tích.
"""

import sqlite3
from typing import List, Optional, Dict
from datetime import datetime

from src.models import Event, AnalysisResult
from src.utils.logger import get_logger
from config.settings import get_current_time

logger = get_logger(__name__)


class EventRepository:
    """Thao tác đọc, ghi, cập nhật dữ liệu sự kiện trong SQLite."""

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection

    def save_analysis(self, result: AnalysisResult) -> bool:
        """
        Lưu lịch sử phân tích video.

        Returns:
            True nếu lưu thành công.
        """
        try:
            import uuid
            analysis_id = result.analysis_id or str(uuid.uuid4())[:8]
            result.analysis_id = analysis_id

            self._conn.execute(
                """
                INSERT OR IGNORE INTO analyses
                (analysis_id, video_name, created_at, total_frames,
                 total_objects, total_events, output_video_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    result.video_name,
                    get_current_time().isoformat(),
                    result.total_frames_processed,
                    result.total_objects_detected,
                    result.total_events_created,
                    result.output_video_path,
                    "completed" if result.success else "failed",
                ),
            )
            self._conn.commit()
            logger.info(f"Lưu phân tích {analysis_id}: {result.video_name}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi lưu phân tích: {e}")
            return False

    def save_event(self, event: Event, analysis_id: str = None) -> bool:
        """
        Lưu một sự kiện vào database với đầy đủ thông tin vòng đời và che khuất.

        Returns:
            True nếu lưu thành công.
        """
        try:
            import json

            # Chuẩn hóa vị trí bbox cuối cùng
            last_loc = f"{event.bbox_x1},{event.bbox_y1},{event.bbox_x2},{event.bbox_y2}"
            if event.last_known_bbox != (0, 0, 0, 0):
                last_loc = f"{event.last_known_bbox[0]},{event.last_known_bbox[1]},{event.last_known_bbox[2]},{event.last_known_bbox[3]}"

            evidence_json = json.dumps(event.evidence_metadata, ensure_ascii=False)
            history_json = json.dumps(event.state_history, ensure_ascii=False)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO events
                (event_id, analysis_id, timestamp, class_name, track_id,
                 last_track_id, seat_id, confidence, duration_seconds, status,
                 video_name, video_time_sec, frame_number, snapshot_path,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, notes,
                 first_seen_timestamp, last_seen_timestamp, suspected_timestamp,
                 occlusion_start_timestamp, occlusion_duration, observed_duration,
                 vacant_duration, last_known_location, evidence_metadata, event_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    analysis_id,
                    event.timestamp,
                    event.class_name,
                    event.track_id,
                    event.last_track_id if event.last_track_id != -1 else event.track_id,
                    event.seat_id,
                    event.confidence,
                    event.duration_seconds,
                    event.status,
                    event.video_name,
                    event.video_time_sec,
                    event.frame_number,
                    event.snapshot_path,
                    event.bbox_x1,
                    event.bbox_y1,
                    event.bbox_x2,
                    event.bbox_y2,
                    event.notes,
                    event.first_seen_timestamp or event.timestamp,
                    event.last_seen_timestamp or event.timestamp,
                    event.suspected_timestamp,
                    event.occlusion_start_timestamp,
                    event.occlusion_duration,
                    event.observed_duration,
                    event.vacant_duration,
                    last_loc,
                    evidence_json,
                    history_json,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi lưu sự kiện {event.event_id}: {e}")
            return False

    def save_events(self, events: List[Event], analysis_id: str = None) -> int:
        """
        Lưu nhiều sự kiện cùng lúc.

        Returns:
            Số lượng sự kiện đã lưu thành công.
        """
        saved = 0
        for event in events:
            if self.save_event(event, analysis_id):
                saved += 1
        return saved

    def update_event_status(self, event_id: str, new_status: str) -> bool:
        """Cập nhật trạng thái sự kiện."""
        try:
            cursor = self._conn.execute(
                "UPDATE events SET status = ? WHERE event_id = ?",
                (new_status, event_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning(f"Không tìm thấy sự kiện: {event_id}")
                return False
            logger.info(f"Cập nhật trạng thái sự kiện {event_id} → {new_status}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi cập nhật sự kiện: {e}")
            return False

    def get_all_events(self) -> List[Dict]:
        """Trả về tất cả sự kiện, sắp xếp theo thời gian mới nhất."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn sự kiện: {e}")
            return []

    def get_events_by_video(self, video_name: str) -> List[Dict]:
        """Trả về sự kiện theo tên video."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE video_name = ? ORDER BY timestamp DESC",
                (video_name,),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn sự kiện theo video: {e}")
            return []

    def get_events_by_status(self, status: str) -> List[Dict]:
        """Trả về sự kiện theo trạng thái."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE status = ? ORDER BY timestamp DESC",
                (status,),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn sự kiện theo trạng thái: {e}")
            return []

    def get_all_analyses(self) -> List[Dict]:
        """Trả về lịch sử phân tích."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn lịch sử phân tích: {e}")
            return []

    def get_dashboard_stats(self) -> Dict:
        """Trả về thống kê tổng quan cho dashboard bao gồm che khuất và kiểm tra."""
        try:
            stats = {}
            cursor = self._conn.execute("SELECT COUNT(*) FROM analyses")
            stats["total_analyses"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COALESCE(SUM(total_objects), 0) FROM analyses"
            )
            stats["total_objects"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM events")
            stats["total_events"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'suspected'"
            )
            stats["suspected_events"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'occluded'"
            )
            stats["occluded_events"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'review_required'"
            )
            stats["review_required_events"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status IN ('confirmed', 'resolved')"
            )
            stats["confirmed_events"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'dismissed'"
            )
            stats["dismissed_events"] = cursor.fetchone()[0]

            return stats
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn thống kê: {e}")
            return {
                "total_analyses": 0,
                "total_objects": 0,
                "total_events": 0,
                "suspected_events": 0,
                "occluded_events": 0,
                "review_required_events": 0,
                "confirmed_events": 0,
                "dismissed_events": 0,
            }


    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Trả về N sự kiện gần nhất."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn sự kiện gần đây: {e}")
            return []
