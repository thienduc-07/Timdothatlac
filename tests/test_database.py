"""
tests/test_database.py
Kiểm thử khởi tạo database, lưu/truy vấn sự kiện, cập nhật trạng thái.
Sử dụng database tạm trong bộ nhớ.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db import Database
from src.database.event_repository import EventRepository
from src.models import Event, AnalysisResult


@pytest.fixture
def db_and_repo(tmp_path):
    """Tạo database tạm và repository."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path=db_path)
    assert db.connect() is True
    repo = EventRepository(db.get_connection())
    yield db, repo
    db.close()


class TestDatabaseInit:
    """Test khởi tạo database."""

    def test_create_database_successfully(self, tmp_path):
        """Database tạo thành công với schema."""
        db_path = str(tmp_path / "new_test.db")
        db = Database(db_path=db_path)
        assert db.connect() is True
        assert Path(db_path).exists()
        db.close()

    def test_tables_exist(self, db_and_repo):
        """Kiểm tra bảng analyses và events đã được tạo."""
        db, repo = db_and_repo
        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "analyses" in tables
        assert "events" in tables

    def test_reopen_database_no_error(self, tmp_path):
        """Mở lại database đã có → không lỗi, không mất dữ liệu."""
        db_path = str(tmp_path / "reopen_test.db")

        # Lần 1
        db1 = Database(db_path=db_path)
        db1.connect()
        db1.close()

        # Lần 2
        db2 = Database(db_path=db_path)
        assert db2.connect() is True
        db2.close()


class TestEventCRUD:
    """Test các thao tác CRUD cho sự kiện."""

    def _make_event(self, event_id="evt001", class_name="backpack"):
        return Event(
            event_id=event_id,
            timestamp="2024-01-01T12:00:00",
            class_name=class_name,
            track_id=1,
            confidence=0.85,
            duration_seconds=15.0,
            status="suspected",
            video_name="test_video.mp4",
            video_time_sec=30.0,
            frame_number=900,
        )

    def test_save_event(self, db_and_repo):
        """Lưu sự kiện thành công."""
        _, repo = db_and_repo
        event = self._make_event()
        result = repo.save_event(event, analysis_id="analysis001")
        assert result is True

    def test_get_all_events(self, db_and_repo):
        """Lấy tất cả sự kiện."""
        _, repo = db_and_repo
        repo.save_event(self._make_event("evt001"), "a1")
        repo.save_event(self._make_event("evt002", "handbag"), "a1")

        events = repo.get_all_events()
        assert len(events) == 2

    def test_get_events_by_status(self, db_and_repo):
        """Lọc sự kiện theo trạng thái."""
        _, repo = db_and_repo
        e1 = self._make_event("evt001")
        e1.status = "suspected"
        e2 = self._make_event("evt002")
        e2.status = "confirmed"

        repo.save_event(e1, "a1")
        repo.save_event(e2, "a1")

        suspected = repo.get_events_by_status("suspected")
        assert len(suspected) == 1
        assert suspected[0]["event_id"] == "evt001"

    def test_update_event_status(self, db_and_repo):
        """Cập nhật trạng thái sự kiện."""
        _, repo = db_and_repo
        repo.save_event(self._make_event("evt001"), "a1")

        result = repo.update_event_status("evt001", "confirmed")
        assert result is True

        events = repo.get_events_by_status("confirmed")
        assert len(events) == 1

    def test_update_nonexistent_event(self, db_and_repo):
        """Cập nhật sự kiện không tồn tại → False."""
        _, repo = db_and_repo
        result = repo.update_event_status("nonexistent", "confirmed")
        assert result is False

    def test_save_duplicate_event_replaces(self, db_and_repo):
        """Lưu sự kiện trùng event_id → thay thế (INSERT OR REPLACE)."""
        _, repo = db_and_repo
        e1 = self._make_event("evt001")
        e1.status = "monitoring"
        repo.save_event(e1, "a1")

        e2 = self._make_event("evt001")
        e2.status = "suspected"
        repo.save_event(e2, "a1")

        events = repo.get_all_events()
        assert len(events) == 1
        assert events[0]["status"] == "suspected"


class TestAnalysisCRUD:
    """Test lưu và truy vấn lịch sử phân tích."""

    def test_save_analysis(self, db_and_repo):
        _, repo = db_and_repo
        result = AnalysisResult(
            video_name="test.mp4",
            total_frames_processed=1000,
            total_objects_detected=50,
            total_events_created=3,
            output_video_path="outputs/test_analyzed.mp4",
            success=True,
        )
        assert repo.save_analysis(result) is True

    def test_get_all_analyses(self, db_and_repo):
        _, repo = db_and_repo
        r1 = AnalysisResult(video_name="v1.mp4", success=True)
        r2 = AnalysisResult(video_name="v2.mp4", success=True)
        repo.save_analysis(r1)
        repo.save_analysis(r2)

        analyses = repo.get_all_analyses()
        assert len(analyses) == 2


class TestDashboardStats:
    """Test thống kê tổng quan."""

    def test_empty_stats(self, db_and_repo):
        """Khi chưa có dữ liệu → thống kê trả về 0."""
        _, repo = db_and_repo
        stats = repo.get_dashboard_stats()
        assert stats["total_analyses"] == 0
        assert stats["total_events"] == 0
        assert stats["suspected_events"] == 0

    def test_stats_after_data(self, db_and_repo):
        """Thống kê chính xác sau khi có dữ liệu."""
        _, repo = db_and_repo

        r = AnalysisResult(
            video_name="test.mp4",
            total_objects_detected=100,
            success=True,
        )
        repo.save_analysis(r)

        e1 = Event(event_id="e1", timestamp="2024-01-01", class_name="backpack",
                    status="suspected", video_name="test.mp4")
        e2 = Event(event_id="e2", timestamp="2024-01-01", class_name="handbag",
                    status="confirmed", video_name="test.mp4")
        repo.save_event(e1, r.analysis_id)
        repo.save_event(e2, r.analysis_id)

        stats = repo.get_dashboard_stats()
        assert stats["total_analyses"] == 1
        assert stats["total_objects"] == 100
        assert stats["total_events"] == 2
        assert stats["suspected_events"] == 1
        assert stats["confirmed_events"] == 1
