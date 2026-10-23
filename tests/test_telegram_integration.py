"""
tests/test_telegram_integration.py
Kiểm thử toàn diện 10 yêu cầu tích hợp thông báo Telegram cho hệ thống phát hiện đồ bỏ quên:
1. Sự kiện đồ bỏ quên kích hoạt dịch vụ thông báo Telegram.
2. Âm thanh cảnh báo trên Web UI vẫn hoạt động nguyên vẹn.
3. Thông báo backend hoạt động độc lập với âm thanh frontend.
4. Các khung hình video lặp lại không gửi tin nhắn trùng lặp (Deduplication).
5. Thiếu thông tin cấu hình Telegram tạo ra thông điệp chẩn đoán rõ ràng.
6. Lỗi API Telegram được báo cáo chính xác, an toàn.
7. Phản hồi thành công từ Telegram (ok: true) được xử lý chính xác.
8. Lần thử thất bại có thể được thử lại mà không gây lặp tin nhắn mất kiểm soát.
9. Tin nhắn gửi tới TELEGRAM_CHAT_ID (kênh cấu hình), không phải personal chat ID.
10. Quá trình xử lý camera tiếp tục hoạt động bình thường khi Telegram mất kết nối/lỗi mạng.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from config.settings import (
    NotificationChannel,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    get_current_time,
)
from src.models import Alert, Event
from src.notifications.telegram_notifier import (
    TelegramNotifier,
    TelegramResponse,
    get_telegram_notifier,
)
from src.services.alert_service import AlertService
from src.services.analysis_service import AnalysisService


# ─────────────────────────────────────────────────────────────
# 1. Forgotten-item event triggers Telegram notification service
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_1_forgotten_item_triggers_telegram_service(mock_post):
    """Yêu cầu 1: Sự kiện đồ bỏ quên từ browser camera kích hoạt gửi thông báo Telegram."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 101}}
    mock_post.return_value = mock_resp

    mock_repo = MagicMock()
    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "123456:TEST_TOKEN"
    notifier.chat_id = "-1004483384365"

    alert_service = AlertService(alert_repo=mock_repo)
    analysis_service = AnalysisService()
    analysis_service._alert_service = alert_service
    analysis_service._repo = mock_repo

    suspected_items = [
        {
            "track_id": "T12",
            "display_id": "T12",
            "class_name": "backpack",
            "seat_id": "S02",
            "status": "SUSPECTED",
            "confidence": 0.88,
        }
    ]

    alerts = analysis_service.process_browser_camera_events(
        suspected_items=suspected_items,
        snapshot_base64=None,
        camera_name="Live Camera Test",
    )

    assert len(alerts) == 1
    assert alerts[0].seat_id == "S02"
    # Đợi worker hoàn tất gửi tin
    time.sleep(0.2)
    assert mock_post.called
    call_args, call_kwargs = mock_post.call_args
    sent_payload = call_kwargs.get("json", {})
    assert sent_payload.get("chat_id") == "-1004483384365"
    assert "S02" in sent_payload.get("text", "")
    assert "T12" in sent_payload.get("text", "")


# ─────────────────────────────────────────────────────────────
# 2. Existing web alert sound remains functional
# ─────────────────────────────────────────────────────────────
def test_2_web_alert_sound_preserved_in_frontend():
    """Yêu cầu 2: Âm thanh cảnh báo trong frontend JavaScript vẫn tồn tại và được gọi khi chuyển SUSPECTED."""
    frontend_path = Path("src/components/hybrid_camera/frontend/index.html")
    assert frontend_path.exists(), "File frontend/index.html phải tồn tại"
    content = frontend_path.read_text(encoding="utf-8")

    # Kiểm tra bộ tổng hợp âm thanh Web Audio API
    assert "AudioAlertSynth" in content
    assert "playAlertChime()" in content
    assert "audioSynth" in content

    # Kiểm tra lời gọi âm thanh khi chuyển trạng thái SUSPECTED
    assert "audioSynth.playAlertChime();" in content
    # Đảm bảo có ít nhất 2 vị trí kích hoạt âm thanh (trên ghế trống và ngoài ghế)
    assert content.count("audioSynth.playAlertChime();") >= 2


# ─────────────────────────────────────────────────────────────
# 3. Backend notification works independently of frontend sound
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_3_backend_notification_independent_of_frontend_sound(mock_post):
    """Yêu cầu 3: Thông báo backend gửi thành công ngay cả khi frontend không phát âm thanh."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 102}}
    mock_post.return_value = mock_resp

    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "123456:TEST_TOKEN"
    notifier.chat_id = "-1004483384365"

    evt = Event(
        event_id="EVT_HEADLESS_ITEM",
        track_id=99,
        last_track_id=99,
        seat_id="S05",
        class_name="handbag",
        status="suspected",
    )
    alert = Alert(
        alert_id="ALT_HEADLESS_1",
        event_id=evt.event_id,
        seat_id="S05",
        object_id="O99",
        priority="HIGH",
        status="PENDING",
    )

    res = notifier.send_alert(alert=alert, event=evt, is_async=False)
    assert res.success is True
    assert res.status_code == 200
    assert mock_post.called


# ─────────────────────────────────────────────────────────────
# 4. Repeated frames do not send duplicate messages
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_4_repeated_frames_do_not_send_duplicate_messages(mock_post):
    """Yêu cầu 4: Các frame video liên tiếp lặp lại cùng 1 vật thể không gửi tin nhắn trùng lặp."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 103}}
    mock_post.return_value = mock_resp

    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "123456:TEST_TOKEN"
    notifier.chat_id = "-1004483384365"

    mock_repo = MagicMock()
    alert_service = AlertService(alert_repo=mock_repo)

    evt = Event(
        event_id="EVT_STABLE_REPEAT",
        track_id=15,
        last_track_id=15,
        seat_id="S04",
        class_name="backpack",
        status="suspected",
    )

    # Frame 1: Gửi cảnh báo lần đầu
    alert_1 = alert_service.process_event_for_alert(event=evt, current_time_sec=100.0)
    time.sleep(0.15)
    assert mock_post.call_count == 1

    # Frame 2-10: Cùng sự kiện ở các frame tiếp theo (100.1s, 100.2s, ...)
    for offset in range(1, 10):
        alert_service.process_event_for_alert(event=evt, current_time_sec=100.0 + offset * 0.1)

    time.sleep(0.15)
    # Tuyệt đối không được gửi thêm yêu cầu HTTP nào
    assert mock_post.call_count == 1


# ─────────────────────────────────────────────────────────────
# 5. Missing Telegram credentials produce a clear diagnostic
# ─────────────────────────────────────────────────────────────
def test_5_missing_credentials_clear_diagnostic():
    """Yêu cầu 5: Thiếu Bot Token hoặc Chat ID thông báo chẩn đoán rõ ràng, không crash."""
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert notifier.is_configured() is False

    ok, msg = notifier.test_connection(bot_token="", chat_id="")
    assert ok is False
    assert "chưa cấu hình" in msg.lower() or "thiếu" in msg.lower()

    status = notifier.get_last_delivery_status()
    assert status.get("has_attempt") is True
    assert status.get("success") is False
    assert status.get("status_code") == 400


# ─────────────────────────────────────────────────────────────
# 6. Telegram API failures are reported correctly
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_6_telegram_api_failures_reported_correctly(mock_post):
    """Yêu cầu 6: Báo cáo lỗi API (401 Unauthorized, 403 Forbidden, 400 Bad Request) chính xác."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {
        "ok": False,
        "error_code": 403,
        "description": "Forbidden: bot was kicked from the channel",
    }
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="invalid_token", chat_id="-1009999")
    notifier.reset_deduplication()

    res = notifier.send_message("Test message")
    assert res.success is False
    assert res.status_code == 403
    assert "403" in res.message or "chặn" in res.message.lower() or "forbidden" in res.message.lower()

    # Xác nhận token không bị rò rỉ
    assert "invalid_token" not in res.message


# ─────────────────────────────────────────────────────────────
# 7. Successful API responses are handled correctly
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_7_successful_api_response_verified(mock_post):
    """Yêu cầu 7: Phản hồi ok: true từ Telegram được xác thực trước khi ghi nhận thành công."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": {
            "message_id": 999,
            "chat": {"id": -1004483384365, "title": "Lost & Found AI"},
        },
    }
    mock_post.return_value = mock_resp

    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "valid_token"
    notifier.chat_id = "-1004483384365"

    alert = Alert(
        alert_id="ALT_TEST_OK",
        event_id="EVT_TEST_OK",
        seat_id="S01",
        object_id="O10",
        priority="HIGH",
        status="PENDING",
    )

    res = notifier.send_alert(alert=alert, is_async=False)
    assert res.success is True
    assert res.status_code == 200

    last_status = notifier.get_last_delivery_status()
    assert last_status.get("success") is True
    assert last_status.get("alert_id") == "ALT_TEST_OK"


# ─────────────────────────────────────────────────────────────
# 8. Failed attempt retries without uncontrolled duplicates
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_8_failed_attempt_retried_controlled(mock_post):
    """Yêu cầu 8: Khi gặp lỗi mạng tạm thời, thử lại có kiểm soát (bounded retry) và không gửi lặp khi đã thành công."""
    # Lần 1: Timeout lỗi tạm thời; Lần 2: Thành công 200 OK
    resp_success = MagicMock()
    resp_success.status_code = 200
    resp_success.json.return_value = {"ok": True, "result": {"message_id": 104}}

    mock_post.side_effect = [
        requests.exceptions.Timeout("Connection timed out"),
        resp_success,
    ]

    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "valid_token"
    notifier.chat_id = "-1004483384365"

    alert = Alert(
        alert_id="ALT_RETRY_1",
        event_id="EVT_RETRY_1",
        seat_id="S02",
        object_id="O22",
        priority="HIGH",
        status="PENDING",
    )

    with patch("time.sleep", return_value=None):
        res = notifier.send_alert(alert=alert, is_async=False)

    assert res.success is True
    assert mock_post.call_count == 2  # Đã thử lại lần 2 và thành công

    # Thử gửi lại ngay: Phải bị deduplicate chặn, không được gọi requests.post lần thứ 3
    res_dup = notifier.send_alert(alert=alert, is_async=False)
    assert res_dup.success is True
    assert "Deduplicated" in res_dup.message or "trước đó" in res_dup.message
    assert mock_post.call_count == 2


# ─────────────────────────────────────────────────────────────
# 9. Channel delivery uses TELEGRAM_CHAT_ID, not personal chat ID
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_9_delivery_uses_configured_telegram_chat_id(mock_post):
    """Yêu cầu 9: Điểm đến gửi tin sử dụng TELEGRAM_CHAT_ID (kênh), không dùng hardcoded personal chat ID."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 105}}
    mock_post.return_value = mock_resp

    expected_chat_id = TELEGRAM_CHAT_ID or "-1004483384365"
    notifier = TelegramNotifier()
    assert notifier.chat_id == expected_chat_id

    notifier.send_message("Test destination")
    assert mock_post.called
    _, kwargs = mock_post.call_args
    assert kwargs.get("json", {}).get("chat_id") == expected_chat_id


# ─────────────────────────────────────────────────────────────
# 10. Camera processing continues if Telegram is unavailable
# ─────────────────────────────────────────────────────────────
@patch("requests.post")
def test_10_camera_processing_continues_on_telegram_failure(mock_post):
    """Yêu cầu 10: Luồng xử lý camera không bị crash hoặc ngưng trệ khi Telegram mất mạng hoàn toàn."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Network is unreachable")

    notifier = get_telegram_notifier()
    notifier.reset_deduplication()
    notifier.bot_token = "valid_token"
    notifier.chat_id = "-1004483384365"

    mock_repo = MagicMock()
    alert_service = AlertService(alert_repo=mock_repo)
    analysis_service = AnalysisService()
    analysis_service._alert_service = alert_service
    analysis_service._repo = mock_repo

    suspected_items = [
        {
            "track_id": "T77",
            "display_id": "T77",
            "class_name": "phone",
            "seat_id": "S08",
            "status": "SUSPECTED",
            "confidence": 0.95,
        }
    ]

    # Quá trình tiếp nhận sự kiện từ camera không được ném Exception ra ngoài
    try:
        alerts = analysis_service.process_browser_camera_events(
            suspected_items=suspected_items,
            snapshot_base64=None,
            camera_name="Live Camera Stream",
        )
        assert len(alerts) == 1
    except Exception as e:
        pytest.fail(f"Camera event processing bị crash do lỗi mạng Telegram: {e}")
