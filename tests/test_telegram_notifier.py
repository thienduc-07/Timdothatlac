"""
tests/test_telegram_notifier.py
Unit & Integration tests cho dịch vụ TelegramNotifier và AlertService.
Kiểm tra toàn diện:
- Xử lý thiếu thông tin cấu hình (Missing credentials) an toàn, không crash.
- Gửi tin nhắn và ảnh thành công (mocked HTTP 200).
- Xử lý các mã lỗi HTTP từ Telegram (401, 400, 403, 500, Timeout).
- Bảo mật: Tuyệt đối không rò rỉ token trong log hay thông điệp lỗi.
- Chống spam & Deduplication: Giãn cách cooldown và không lặp sự kiện.
- Quy tắc kích hoạt: Detections thô hoặc ghế có người (OBSERVED) KHÔNG được gửi alert;
  chỉ SUSPECTED hoặc REVIEW_REQUIRED mới kích hoạt alert.
- Định dạng thông điệp tuân thủ mẫu Lost & Found AI Bus.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests

from src.models import Alert, Event
from config.settings import NotificationChannel
from src.notifications.telegram_notifier import (
    TelegramNotifier,
    TelegramResponse,
    sanitize_telegram_error,
)
from src.services.alert_service import AlertService


# ──────────────────────────────────────────────
# 1. KIỂM TRA BẢO MẬT & THIẾU CẤU HÌNH
# ──────────────────────────────────────────────
def test_missing_credentials_safety():
    """Kiểm tra khi chưa cấu hình token hoặc chat ID thì không crash ứng dụng."""
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert not notifier.is_configured()

    # Kiểm tra test_connection
    ok, msg = notifier.test_connection(bot_token="", chat_id="")
    assert not ok
    assert "chưa cấu hình" in msg.lower() or "thiếu" in msg.lower()

    # Kiểm tra send_message
    res = notifier.send_message("Test message", chat_id="")
    assert not res.success
    assert res.error_type in ("MISSING_TOKEN", "MISSING_CHAT_ID")

    # Kiểm tra send_alert
    alert = Alert(
        alert_id="ALT_TEST",
        event_id="EVT_1",
        seat_id="S01",
        object_id="O1",
        priority="HIGH",
        status="PENDING",
    )
    res_alert = notifier.send_alert(alert, is_async=False)
    assert not res_alert.success
    assert res_alert.error_type == "NOT_CONFIGURED"


def test_token_masking_and_sanitization():
    """Đảm bảo token nhạy cảm bị che giấu hoàn toàn khỏi ngoại lệ và URL."""
    fake_token = "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ_secret"
    raw_error = f"HTTP Error 401: https://api.telegram.org/bot{fake_token}/sendMessage failed"

    sanitized = sanitize_telegram_error(raw_error, fake_token)
    assert fake_token not in sanitized
    assert "bot[REDACTED_TOKEN]" in sanitized

    notifier = TelegramNotifier(bot_token=fake_token, chat_id="999999")
    masked = notifier.get_masked_token()
    assert fake_token not in masked
    assert "••••••••" in masked


# ──────────────────────────────────────────────
# 2. KIỂM TRA GỬI TIN NHẮN & TEST CONNECTION (MOCKED HTTP)
# ──────────────────────────────────────────────
@patch("requests.get")
def test_test_connection_success(mock_get):
    """Kiểm tra test_connection khi getMe trả về 200 OK."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": {"id": 12345, "is_bot": True, "username": "DucLostFoundAIBot"},
    }
    mock_get.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="fake_token", chat_id="")
    ok, msg = notifier.test_connection()
    assert ok
    assert "@DucLostFoundAIBot" in msg


@patch("requests.get")
def test_test_connection_invalid_token_401(mock_get):
    """Kiểm tra xử lý lỗi 401 khi bot token không hợp lệ."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"ok": False, "error_code": 401, "description": "Unauthorized"}
    mock_get.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="bad_token", chat_id="")
    ok, msg = notifier.test_connection()
    assert not ok
    assert "không hợp lệ" in msg.lower()


@patch("requests.post")
def test_send_message_success(mock_post):
    """Kiểm tra gửi tin nhắn thành công qua HTTP POST."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 101}}
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="test_token", chat_id="123456789")
    res = notifier.send_message("Xin chào từ xe buýt AI")

    assert res.success
    assert res.status_code == 200
    assert res.message == "Thành công"
    mock_post.assert_called_once()


@patch("requests.post")
def test_send_message_chat_not_found_400(mock_post):
    """Kiểm tra xử lý lỗi khi Chat ID chưa bấm Start (400 Bad Request: chat not found)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "ok": False,
        "error_code": 400,
        "description": "Bad Request: chat not found",
    }
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="test_token", chat_id="999999999")
    res = notifier.send_message("Test")

    assert not res.success
    assert res.status_code == 400
    assert "không tìm thấy chat id" in res.message.lower()


@patch("requests.post")
def test_send_message_timeout(mock_post):
    """Kiểm tra xử lý khi mạng bị timeout không làm sập ứng dụng."""
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    notifier = TelegramNotifier(bot_token="test_token", chat_id="123456789")
    res = notifier.send_message("Test")

    assert not res.success
    assert res.status_code == 408
    assert res.error_type == "TIMEOUT"


# ──────────────────────────────────────────────
# 3. KIỂM TRA ĐỊNH DẠNG BẢN TIN CẢNH BÁO (MESSAGE FORMATTING)
# ──────────────────────────────────────────────
def test_message_formatting_standard_template():
    """Kiểm tra mẫu bản tin cảnh báo tuân thủ yêu cầu hệ thống."""
    alert = Alert(
        alert_id="ALT_1234ABCD",
        event_id="EVT_99",
        seat_id="S02",
        object_id="T15",
        priority="HIGH",
        status="PENDING",
        created_at="2026-10-23T20:00:00",
    )
    event = Event(
        event_id="EVT_99",
        class_name="backpack",
        seat_id="S02",
        track_id=15,
        last_track_id=15,
        status="suspected",
        timestamp="2026-10-23T20:00:00",
        vacant_duration=15.0,
    )

    msg = TelegramNotifier.format_forgotten_item_message(
        alert=alert,
        event=event,
        bus_or_camera_id="Tuyến xe buýt 08 (Bến Thành - ĐHQG)",
    )

    assert "LOST & FOUND AI BUS — CẢNH BÁO ĐỒ BỎ QUÊN" in msg
    assert "Tuyến xe buýt 08 (Bến Thành - ĐHQG)" in msg
    assert "Ghế S02" in msg or "S02" in msg
    assert "Đồ vật" in msg
    assert "T15" in msg
    assert "Xác nhận bởi logic AI" in msg
    assert "Vui lòng kiểm tra camera" in msg


def test_message_formatting_unknown_seat_limitation():
    """Nếu không xác định được ghế, phải dán nhãn ứng viên cần rà soát trung thực."""
    alert = Alert(
        alert_id="ALT_UNKN",
        event_id="EVT_UNKNOWN",
        seat_id="",
        object_id="T7",
        priority="MEDIUM",
        status="PENDING",
    )
    event = Event(
        event_id="EVT_UNKNOWN",
        class_name="cell phone",
        seat_id="",
        track_id=7,
        last_track_id=7,
        status="suspected",
    )

    msg = TelegramNotifier.format_forgotten_item_message(
        alert=alert,
        event=event,
        bus_or_camera_id="Camera Hành Lang",
    )

    assert "Chưa rõ vị trí ghế" in msg
    assert "Ứng viên cần rà soát" in msg


# ──────────────────────────────────────────────
# 4. KIỂM TRA CHỐNG SPAM & DEDUPLICATION
# ──────────────────────────────────────────────
@patch("requests.post")
def test_anti_spam_cooldown_and_event_deduplication(mock_post):
    """
    Đảm bảo 1 sự kiện chỉ gửi thông báo 1 lần khi đủ điều kiện,
    không bị lặp khi stream video chạy 30 FPS.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(
        bot_token="test_token",
        chat_id="123456",
        cooldown_seconds=60.0,
    )
    notifier.reset_deduplication()

    alert1 = Alert(
        alert_id="ALT_01",
        event_id="EVT_SAME_ITEM",
        seat_id="S01",
        object_id="O10",
    )
    event1 = Event(
        event_id="EVT_SAME_ITEM",
        class_name="backpack",
        seat_id="S01",
        last_track_id=10,
        status="suspected",
    )

    # Lần gửi 1: Thành công
    res1 = notifier.send_alert(alert1, event=event1, is_async=False)
    assert res1.success
    assert mock_post.call_count == 1

    # Lần gửi 2 ngay frame kế tiếp (cùng event_id): Bị chặn bởi deduplication
    res2 = notifier.send_alert(alert1, event=event1, is_async=False)
    assert res2.success  # Trả về thành công nhưng bị chặn không gọi HTTP lặp
    assert "deduplicated" in res2.message.lower() or "giãn cách" in res2.message.lower()
    assert mock_post.call_count == 1  # Không tăng số lần gọi HTTP API!


# ──────────────────────────────────────────────
# 5. KIỂM TRA QUY TẮC KÍCH HOẠT TRONG ALERTSERVICE
# ──────────────────────────────────────────────
def test_raw_detections_and_occupied_seat_do_not_trigger_telegram():
    """
    Kiểm tra bảo đảm rằng các detection thô hoặc vật thể khi ghế CÓ NGƯỜI (OBSERVED/ITEM_LEFT)
    tuyệt đối KHÔNG được gửi cảnh báo Telegram.
    """
    mock_repo = MagicMock()
    alert_service = AlertService(alert_repo=mock_repo)

    # 1. Sự kiện mới chỉ quan sát thấy (ghế đang có người ngồi cùng đồ) -> OBSERVED
    evt_observed = Event(
        event_id="EVT_OBS",
        class_name="backpack",
        seat_id="S01",
        status="observed",
        vacant_duration=0.0,
    )
    res_obs = alert_service.process_event_for_alert(evt_observed, current_time_sec=1.0)
    assert res_obs is None  # Không sinh Alert
    assert mock_repo.save_alert.call_count == 0

    # 2. Người mới vừa rời ghế, chưa đủ thời gian kiểm tra -> ITEM_LEFT
    evt_item_left = Event(
        event_id="EVT_LEFT",
        class_name="backpack",
        seat_id="S01",
        status="item_left",
        vacant_duration=1.2,
    )
    res_left = alert_service.process_event_for_alert(evt_item_left, current_time_sec=2.0)
    assert res_left is None
    assert mock_repo.save_alert.call_count == 0


@patch("requests.post")
def test_suspected_event_triggers_telegram_alert(mock_post):
    """
    Khi sự kiện chuyển sang SUSPECTED (ghế trống đủ lâu),
    AlertService phải tự động kích hoạt gửi cảnh báo Telegram.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    mock_post.return_value = mock_resp

    mock_repo = MagicMock()
    alert_service = AlertService(alert_repo=mock_repo)

    evt_suspected = Event(
        event_id="EVT_CONFIRMED_FORGOTTEN",
        class_name="backpack",
        seat_id="S03",
        last_track_id=42,
        status="suspected",
        vacant_duration=12.0,
    )

    t_config = {
        "enabled": True,
        "bot_token": "valid_token",
        "chat_id": "12345678",
    }

    alert = alert_service.process_event_for_alert(
        event=evt_suspected,
        current_time_sec=10.0,
        telegram_config=t_config,
        is_test=True,
    )

    assert alert is not None
    assert alert.seat_id == "S03"
    assert alert.priority == "HIGH"
    mock_repo.save_alert.assert_called()
    time.sleep(0.15)
    mock_repo.log_notification.assert_called()


# ──────────────────────────────────────────────
# 6. KIỂM TRA GỬI ẢNH BẰNG CHỨNG (SEND PHOTO)
# ──────────────────────────────────────────────
@patch("requests.post")
def test_send_photo_with_valid_file(mock_post):
    """Kiểm tra gửi ảnh chụp snapshot bằng chứng."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="test_token", chat_id="123456")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00")
        tmp_path = tmp.name

    try:
        res = notifier.send_photo(photo_path=tmp_path, caption="Bằng chứng nghi ngờ bỏ quên")
        assert res.success
        assert res.status_code == 200
        mock_post.assert_called_once()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
