"""
tests/test_three_issues_fixes.py
Bộ kiểm thử toàn diện cho 3 vấn đề:
1. Dismiss & Quản lý vòng đời Alert Center (loại bỏ khỏi Active, giữ lịch sử, ITEM_RETRIEVED, FALSE_POSITIVE).
2. Camera Browser -> Backend Event -> Alert -> Telegram pipeline (điều kiện tạo, chống spam lặp, logging).
3. Gửi Telegram kèm ảnh đồ vật bị bỏ quên (sendPhoto, caption chuẩn, fallback văn bản khi thiếu ảnh, an toàn base64).
"""

import base64
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import AlertPriority, AlertStatus, NotificationChannel
from src.database.alert_repository import AlertRepository
from src.database.db import Database
from src.database.event_repository import EventRepository
from src.models import Alert, Event
from src.notifications.telegram_notifier import TelegramNotifier, get_telegram_notifier
from src.services.alert_service import AlertService
from src.services.analysis_service import AnalysisService


@pytest.fixture
def temp_db():
    """Tạo database SQLite tạm thời trong bộ nhớ file để test kiểm thử CRUD thực tế."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)
    assert db.connect() is True
    yield db
    db.close()
    if os.path.exists(path):
        os.remove(path)


# ─────────────────────────────────────────────────────────────
# NHÓM 1: DISMISS & ALERT CENTER LIFECYCLE (Yêu cầu 1, 2, 3, 6)
# ─────────────────────────────────────────────────────────────

def test_01_dismiss_updates_database_and_disappears_from_active_list(temp_db):
    """
    Yêu cầu 1 & 4 (Vấn đề 1):
    Khi dismiss, cập nhật trạng thái bản ghi trong DB thành DISMISSED,
    lưu resolved_at, resolution_type và cảnh báo biến mất khỏi danh sách ACTIVE.
    """
    conn = temp_db.get_connection()
    repo = AlertRepository(conn)

    alert = Alert(
        alert_id="ALT_TEST_01",
        event_id="EVT_01",
        seat_id="S01",
        object_id="O01",
        priority="HIGH",
        status=AlertStatus.PENDING,
        reason="Đồ vật bỏ quên tại ghế S01",
        created_at="2026-10-23T12:00:00",
    )
    assert repo.save_alert(alert) is True

    # Ban đầu alert nằm trong danh sách ACTIVE
    active_alerts = repo.get_recent_alerts(active_only=True)
    assert len(active_alerts) == 1
    assert active_alerts[0]["alert_id"] == "ALT_TEST_01"

    # Người dùng nhấn Dismiss
    ok = repo.update_alert_status("ALT_TEST_01", "DISMISSED", resolution_type="DISMISSED")
    assert ok is True

    # Cảnh báo phải biến mất hoàn toàn khỏi danh sách ACTIVE
    active_after = repo.get_recent_alerts(active_only=True)
    assert len(active_after) == 0

    # Nhưng bản ghi vẫn tồn tại trong database với trạng thái DISMISSED
    item = repo.get_alert_by_id("ALT_TEST_01")
    assert item is not None
    assert item["status"] == "DISMISSED"
    assert item["resolution_type"] == "DISMISSED"
    assert item["resolved_at"] != ""


def test_02_dismissed_retained_in_history_after_restart_or_rerun(temp_db):
    """
    Yêu cầu 2 & 6 (Vấn đề 1):
    Bản ghi DISMISSED vẫn được giữ nguyên trong lịch sử sau khi khởi động lại (kết nối lại DB)
    và không bao giờ quay lại ACTIVE.
    """
    conn = temp_db.get_connection()
    repo = AlertRepository(conn)

    alert = Alert(
        alert_id="ALT_TEST_02",
        event_id="EVT_02",
        seat_id="S02",
        object_id="O02",
        status=AlertStatus.PENDING,
        reason="Đồ vật tại ghế S02",
    )
    repo.save_alert(alert)
    repo.update_alert_status("ALT_TEST_02", "DISMISSED")

    # Giả lập restart / rerun: mở kết nối mới tới cùng database
    db_path = temp_db.db_path
    new_db = Database(db_path=db_path)
    new_db.connect()
    new_repo = AlertRepository(new_db.get_connection())

    # Kiểm tra ACTIVE: trống
    assert len(new_repo.get_recent_alerts(active_only=True)) == 0

    # Kiểm tra Lịch sử: vẫn đầy đủ
    history = new_repo.get_recent_alerts(status=None)
    assert len(history) == 1
    assert history[0]["alert_id"] == "ALT_TEST_02"
    assert history[0]["status"] == "DISMISSED"
    new_db.close()


def test_03_item_retrieved_and_false_positive_saved_properly(temp_db):
    """
    Yêu cầu 2 & 3 (Vấn đề 1):
    Phân biệt rõ ràng ITEM_RETRIEVED (đã lấy đồ) và FALSE_POSITIVE (báo sai).
    """
    conn = temp_db.get_connection()
    repo = AlertRepository(conn)

    a1 = Alert(alert_id="ALT_RET", event_id="EVT_R", seat_id="S03", object_id="O03", status="PENDING")
    a2 = Alert(alert_id="ALT_FP", event_id="EVT_F", seat_id="S04", object_id="O04", status="PENDING")
    repo.save_alert(a1)
    repo.save_alert(a2)

    # Đánh dấu ITEM_RETRIEVED
    assert repo.update_alert_status("ALT_RET", "ITEM_RETRIEVED", resolution_type="ITEM_RETRIEVED") is True
    # Đánh dấu FALSE_POSITIVE
    assert repo.update_alert_status("ALT_FP", "FALSE_POSITIVE", resolution_type="FALSE_POSITIVE") is True

    # Kiểm tra lọc theo từng trạng thái
    ret_list = repo.get_recent_alerts(status="ITEM_RETRIEVED")
    assert len(ret_list) == 1
    assert ret_list[0]["alert_id"] == "ALT_RET"
    assert ret_list[0]["resolution_type"] == "ITEM_RETRIEVED"

    fp_list = repo.get_recent_alerts(status="FALSE_POSITIVE")
    assert len(fp_list) == 1
    assert fp_list[0]["alert_id"] == "ALT_FP"
    assert fp_list[0]["resolution_type"] == "FALSE_POSITIVE"

    # Cả hai đều không nằm trong ACTIVE
    assert len(repo.get_recent_alerts(active_only=True)) == 0


def test_06_alert_center_query_active_and_delete_permanently(temp_db):
    """
    Yêu cầu 7 (Vấn đề 1):
    Xóa vĩnh viễn tách biệt với dismiss và loại bỏ vật lý bản ghi khi người dùng xác nhận.
    """
    conn = temp_db.get_connection()
    repo = AlertRepository(conn)

    a1 = Alert(alert_id="ALT_DEL", event_id="EVT_D", seat_id="S05", object_id="O05", status="PENDING")
    repo.save_alert(a1)
    assert repo.get_alert_by_id("ALT_DEL") is not None

    # Thực hiện xóa vĩnh viễn
    deleted = repo.delete_alert_permanently("ALT_DEL")
    assert deleted is True

    # Bản ghi không còn trong database
    assert repo.get_alert_by_id("ALT_DEL") is None


# ─────────────────────────────────────────────────────────────
# NHÓM 2: CAMERA EVENT ROUTING & DEDUPLICATION (Yêu cầu 4, 5, 7)
# ─────────────────────────────────────────────────────────────

def test_04_browser_camera_creates_event_and_alert_when_eligible(temp_db):
    """
    Yêu cầu 4 & 5 (Vấn đề 2):
    Event từ browser camera (suspected, vacant_duration >= 3.5s) tạo thành công Event và Alert.
    """
    conn = temp_db.get_connection()
    service = AnalysisService()
    service._db = temp_db
    service._repo = EventRepository(conn)
    service._alert_repo = AlertRepository(conn)
    service._alert_service = AlertService(service._alert_repo)

    suspected_items = [
        {
            "track_id": 101,
            "display_id": "ITEM_101",
            "class_name": "backpack",
            "seat_id": "SEAT_02",
            "status": "SUSPECTED",
            "confidence": 0.88,
            "vacant_duration": 4.5,
        }
    ]

    with patch.object(service._alert_service, "_dispatch_notifications"):
        alerts = service.process_browser_camera_events(
            suspected_items=suspected_items,
            camera_name="Bus Roof Cam",
        )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.seat_id == "SEAT_02"
    assert "101" in alert.object_id
    assert alert.priority == "HIGH"
    assert alert.status == "PENDING"

    # Kiểm tra Event đã được lưu trong DB
    events = service._repo.get_recent_events(limit=10)
    assert len(events) >= 1
    assert any("101" in e["event_id"] for e in events)

    # Kiểm tra Alert đã được lưu trong DB
    db_alerts = service.get_recent_alerts(active_only=True)
    assert len(db_alerts) == 1
    assert db_alerts[0]["alert_id"] == alert.alert_id


def test_05_ineligible_event_does_not_create_alert(temp_db):
    """
    Yêu cầu 5 (Vấn đề 2):
    Sự kiện không đủ điều kiện (ví dụ: status=monitoring, vacant_duration < 3.5s) không tạo cảnh báo.
    """
    conn = temp_db.get_connection()
    alert_repo = AlertRepository(conn)
    alert_service = AlertService(alert_repo)

    # Event ở trạng thái monitoring (chưa nghi ngờ)
    evt_monitoring = Event(
        event_id="EVT_MONITOR_01",
        class_name="bottle",
        seat_id="S01",
        status="monitoring",
        vacant_duration=1.0,
    )

    alert = alert_service.process_event_for_alert(evt_monitoring, current_time_sec=100.0)
    assert alert is None
    assert alert_repo.get_unresolved_alert_count() == 0


def test_07_repeated_detections_do_not_create_duplicate_alerts(temp_db):
    """
    Yêu cầu 7 (Vấn đề 1 & 2):
    Cùng một vật thể được gửi lặp lại giữa các frame không tạo nhiều Alert ngoài ý muốn.
    Sau khi dismiss, vật thể vẫn trong frame không được sinh lại Alert mới.
    """
    conn = temp_db.get_connection()
    service = AnalysisService()
    service._db = temp_db
    service._repo = EventRepository(conn)
    service._alert_repo = AlertRepository(conn)
    service._alert_service = AlertService(service._alert_repo)

    suspected_item = {
        "track_id": 202,
        "class_name": "handbag",
        "seat_id": "SEAT_05",
        "status": "SUSPECTED",
        "confidence": 0.85,
        "vacant_duration": 5.0,
    }

    with patch.object(service._alert_service, "_dispatch_notifications"):
        # Lần 1: Tạo alert đầu tiên
        a1 = service.process_browser_camera_events([suspected_item])
        assert len(a1) == 1
        first_alert_id = a1[0].alert_id

        # Lần 2: Frame kế tiếp gửi cùng đối tượng -> Không tạo thêm alert mới
        a2 = service.process_browser_camera_events([suspected_item])
        assert len(a2) == 0

        # Người dùng Dismiss cảnh báo đầu tiên
        service.update_alert_status(first_alert_id, "DISMISSED")

        # Lần 3: Camera tiếp tục stream detection đó -> Vẫn bị chặn, không sinh lại alert trùng lặp!
        a3 = service.process_browser_camera_events([suspected_item])
        assert len(a3) == 0

    # Tổng số alert trong DB chỉ là 1 bản ghi duy nhất đã dismiss
    all_alerts = service.get_recent_alerts(status=None)
    assert len(all_alerts) == 1
    assert all_alerts[0]["status"] == "DISMISSED"


# ─────────────────────────────────────────────────────────────
# NHÓM 3: TELEGRAM GỬI ẢNH BẰNG CHỨNG (Yêu cầu 8, 9, 10, 11, 12, 13)
# ─────────────────────────────────────────────────────────────

def test_08_telegram_sends_photo_and_caption_when_snapshot_valid(tmp_path):
    """
    Yêu cầu 8 (Vấn đề 3):
    Telegram sử dụng sendPhoto kèm caption chi tiết khi có file ảnh hợp lệ.
    """
    # Tạo một file ảnh JPEG giả lập hợp lệ
    dummy_jpg = tmp_path / "valid_item.jpg"
    # Magic bytes của JPEG: FF D8 FF
    dummy_jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 100)

    notifier = TelegramNotifier(bot_token="test_token", chat_id="123456", enabled=True)
    notifier.reset_deduplication()

    alert = Alert(
        alert_id="ALT_PHOTO_TEST",
        event_id="EVT_P",
        seat_id="S08",
        object_id="O88",
        priority="HIGH",
        status="PENDING",
        snapshot_path=str(dummy_jpg),
        created_at="2026-10-23T14:30:00",
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 999}}
        mock_post.return_value = mock_resp

        res = notifier.send_alert(alert=alert, is_async=False, force=True)

        assert res.success is True
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        # Phải gọi sendPhoto chứ không phải sendMessage thuần
        assert "sendPhoto" in call_url
        data_payload = mock_post.call_args[1].get("data", {})
        assert data_payload.get("chat_id") == "123456"
        caption = data_payload.get("caption", "")
        assert "CẢNH BÁO ĐỒ VẬT THẤT LẠC" in caption
        assert "S08" in caption
        assert "O88" in caption
        assert len(caption) <= 1024


def test_09_telegram_sends_text_when_snapshot_unavailable():
    """
    Yêu cầu 9 (Vấn đề 3):
    Khi không có snapshot, gửi thông báo văn bản (sendMessage) và nêu rõ ảnh không khả dụng.
    """
    notifier = TelegramNotifier(bot_token="test_token", chat_id="123456", enabled=True)
    notifier.reset_deduplication()

    alert = Alert(
        alert_id="ALT_NO_PHOTO",
        event_id="EVT_NP",
        seat_id="S09",
        object_id="O99",
        priority="HIGH",
        status="PENDING",
        snapshot_path="",  # Không có snapshot
        created_at="2026-10-23T14:35:00",
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 1000}}
        mock_post.return_value = mock_resp

        res = notifier.send_alert(alert=alert, is_async=False, force=True)

        assert res.success is True
        call_url = mock_post.call_args[0][0]
        # Phải dùng sendMessage khi không có ảnh
        assert "sendMessage" in call_url
        sent_json = mock_post.call_args[1].get("json", {})
        msg_text = sent_json.get("text", "")
        assert "CẢNH BÁO ĐỒ VẬT THẤT LẠC" in msg_text
        assert "S09" in msg_text
        assert "O99" in msg_text


def test_10_corrupt_snapshot_does_not_crash_camera_or_lose_alert(temp_db):
    """
    Yêu cầu 10 (Vấn đề 3):
    Base64 snapshot sai định dạng không làm crash camera hoặc làm mất Event/Alert.
    """
    conn = temp_db.get_connection()
    service = AnalysisService()
    service._db = temp_db
    service._repo = EventRepository(conn)
    service._alert_repo = AlertRepository(conn)
    service._alert_service = AlertService(service._alert_repo)

    suspected_items = [
        {
            "track_id": 303,
            "class_name": "cell phone",
            "seat_id": "SEAT_10",
            "status": "SUSPECTED",
            "confidence": 0.92,
            "vacant_duration": 4.0,
            "snapshot_base64": "invalid_base64_%%%_corrupt_data",
        }
    ]

    with patch.object(service._alert_service, "_dispatch_notifications"):
        alerts = service.process_browser_camera_events(
            suspected_items=suspected_items,
            snapshot_base64="corrupt_global_b64",
        )

    # Vẫn tạo thành công Alert bất chấp ảnh hỏng
    assert len(alerts) == 1
    assert alerts[0].seat_id == "SEAT_10"
    # Event và Alert vẫn được lưu an toàn
    assert len(service.get_recent_alerts(active_only=True)) == 1


def test_11_telegram_failure_preserves_event_and_alert(temp_db):
    """
    Yêu cầu 11 (Vấn đề 2 & 3):
    Máy chủ Telegram lỗi (500 hoặc mất mạng) không làm mất Event hoặc Alert đã tạo.
    """
    conn = temp_db.get_connection()
    service = AnalysisService()
    service._db = temp_db
    service._repo = EventRepository(conn)
    service._alert_repo = AlertRepository(conn)
    service._alert_service = AlertService(service._alert_repo)

    evt = Event(
        event_id="EVT_TG_FAIL",
        class_name="backpack",
        seat_id="S11",
        last_track_id=404,
        status="suspected",
        vacant_duration=6.0,
    )

    with patch("requests.post") as mock_post:
        # Giả lập máy chủ Telegram lỗi 502 Bad Gateway
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.return_value = {"ok": False, "description": "Bad Gateway"}
        mock_post.return_value = mock_resp

        alert = service._alert_service.process_event_for_alert(
            event=evt,
            current_time_sec=500.0,
            telegram_config={"enabled": True, "bot_token": "valid_token", "chat_id": "123456"},
            is_test=True,
        )

    # Alert vẫn được tạo
    assert alert is not None
    assert alert.seat_id == "S11"
    # Alert vẫn được lưu an toàn trong Database
    db_alert = service._alert_repo.get_alert_by_id(alert.alert_id)
    assert db_alert is not None
    assert db_alert["status"] == "PENDING"


def test_12_bounded_retry_stops_on_client_error():
    """
    Yêu cầu 12 (Vấn đề 3):
    Retry dừng ngay lập tức khi gặp lỗi client 401/403/404 và không gửi lặp vô hạn.
    """
    notifier = TelegramNotifier(bot_token="invalid_token", chat_id="123456", enabled=True)
    notifier.reset_deduplication()

    alert = Alert(
        alert_id="ALT_RETRY_TEST",
        event_id="EVT_RETRY",
        seat_id="S12",
        object_id="O12",
        status="PENDING",
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"ok": False, "description": "Unauthorized"}
        mock_post.return_value = mock_resp

        res = notifier.send_alert(alert=alert, is_async=False, force=True)

        assert res.success is False
        assert res.status_code == 401
        # Gặp 401 Unauthorized phải dừng ngay ở attempt đầu tiên (call_count == 1)
        assert mock_post.call_count == 1


def test_13_diagnostics_does_not_crash_when_unconfigured():
    """
    Yêu cầu 13 (Vấn đề 2):
    Diagnostics không crash nếu Telegram chưa được cấu hình token hoặc chat ID.
    """
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=True)
    status = notifier.get_last_delivery_status()
    assert isinstance(status, dict)
    assert status["success"] is False
    assert status["status_code"] == 0

    ok, msg = notifier.test_connection(bot_token="", chat_id="")
    assert ok is False
    assert "Chưa cấu hình" in msg or "Thiếu" in msg
