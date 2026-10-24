"""
tests/test_bus_info.py
Kiểm thử bộ tiện ích quản lý thông tin xe buýt (Biển số xe & Tuyến xe):
1. Định dạng biển số demo: nnX-nnnnn.
2. Tự sinh biển số 1 lần và lưu lại nhất quán, không sinh lại khi gọi nhiều lần.
3. Mặc định tuyến xe là "Tuyến số 1", không để trống.
4. Ưu tiên biển số và tuyến xe do người dùng nhập thủ công.
5. Khi người dùng chủ động xóa biển số và lưu, tự sinh một biển số mới.
"""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.bus_info import (
    CONFIG_FILE_PATH,
    DEFAULT_BUS_ROUTE,
    generate_demo_license_plate,
    get_bus_info,
    save_bus_info,
)


@pytest.fixture(autouse=True)
def clean_bus_config(tmp_path):
    """Fixture dọn dẹp biến môi trường và file cấu hình tạm thời cho mỗi test."""
    test_config_path = tmp_path / "bus_config.json"
    with patch("src.utils.bus_info.CONFIG_FILE_PATH", test_config_path):
        with patch.dict(os.environ, {"BUS_LICENSE_PLATE": "", "BUS_ROUTE": ""}):
            yield test_config_path


def test_generate_demo_license_plate_format():
    """Kiểm tra biển số tự sinh tuân thủ chính xác định dạng nnX-nnnnn."""
    pattern = re.compile(r"^\d{2}[A-Z]-\d{5}$")
    for _ in range(50):
        plate = generate_demo_license_plate()
        assert pattern.match(plate), f"Biển số '{plate}' không khớp định dạng nnX-nnnnn"


def test_bus_info_auto_generated_once_and_persisted():
    """Kiểm tra tự sinh biển số một lần và lưu lại, các lần gọi tiếp theo giữ nguyên."""
    plate1, route1 = get_bus_info(auto_generate_if_missing=True)
    assert plate1 != ""
    assert re.match(r"^\d{2}[A-Z]-\d{5}$", plate1)
    assert route1 == "Tuyến số 1"

    # Gọi lần 2: phải giữ nguyên biển số cũ
    plate2, route2 = get_bus_info(auto_generate_if_missing=True)
    assert plate2 == plate1
    assert route2 == route1


def test_bus_info_manual_override_priority():
    """Người dùng nhập biển số và tuyến thủ công thì luôn ưu tiên giá trị đó."""
    save_bus_info(plate="51B-99999", route="Tuyến 08 (Bến Thành - ĐHQG)")

    plate, route = get_bus_info(auto_generate_if_missing=True)
    assert plate == "51B-99999"
    assert route == "Tuyến 08 (Bến Thành - ĐHQG)"


def test_bus_info_clearing_plate_regenerates_new_demo_plate():
    """Khi người dùng chủ động xóa biển số và lưu, hệ thống tự sinh một biển số mới."""
    save_bus_info(plate="51B-11111", route="Tuyến số 1")
    p1, _ = get_bus_info()
    assert p1 == "51B-11111"

    # Người dùng xóa biển số và lưu lại
    p2, r2 = save_bus_info(plate="", route="Tuyến số 1", auto_generate_if_empty=True)
    assert p2 != ""
    assert p2 != "51B-11111"
    assert re.match(r"^\d{2}[A-Z]-\d{5}$", p2)
    assert r2 == "Tuyến số 1"


def test_build_lost_item_telegram_caption_format():
    """Kiểm tra hàm build_lost_item_telegram_caption tạo đúng cấu trúc yêu cầu và không có chuỗi Khu vực."""
    from src.notifications.telegram_notifier import build_lost_item_telegram_caption
    from src.models import Alert, Event

    alert = Alert(
        alert_id="ALT_TEST_1",
        event_id="EVT_01",
        seat_id="023",
        object_id="item_01",
        priority="HIGH",
        status="PENDING",
        created_at="2026-10-24T10:55:23",
    )
    event = Event(
        event_id="EVT_01",
        track_id=27,
        last_track_id=27,
        seat_id="023",
        class_name="backpack",
        status="suspected",
        timestamp="2026-10-24T10:55:23",
    )

    caption = build_lost_item_telegram_caption(
        alert=alert,
        event=event,
        has_photo=True,
        bus_license_plate="51B-12345",
        bus_route="Tuyến số 1",
    )

    assert "🚨 <b>LOST & FOUND AI BUS — CẢNH BÁO ĐỒ BỎ QUÊN</b>" in caption
    assert "🚌 <b>Biển số xe buýt:</b> 51B-12345" in caption
    assert "🛣️ <b>Tuyến xe buýt:</b> Tuyến số 1" in caption
    assert "💺 <b>Ghế:</b> Ghế 023" in caption
    assert "📦 <b>Đồ vật:</b> Đồ vật" in caption
    assert "🆔 <b>Mã theo dõi:</b> <code>T27</code>" in caption
    assert "🕒 <b>Thời điểm:</b> 10:55:23 24/10/2026" in caption
    assert "📋 <b>Trạng thái:</b> Xác nhận bởi logic AI (Ghế trống sau khi khách rời đi)" in caption
    assert "📷 <b>Ảnh chụp bằng chứng:</b> <i>Đính kèm bên dưới</i>" in caption
    assert "<i>Vui lòng kiểm tra camera hoặc khoang hành khách để xác nhận đồ vật.</i>" in caption
    assert "Khu vực:" not in caption
    assert "Live Camera (Browser)" not in caption


def test_live_camera_browser_events_use_unified_caption():
    """Kiểm tra luồng Live Camera tiếp nhận sự kiện và gửi Telegram bằng đúng template thống nhất."""
    from unittest.mock import MagicMock
    from src.services.analysis_service import AnalysisService
    from src.services.alert_service import AlertService

    save_bus_info(plate="51B-12345", route="Tuyến số 1")

    service = AnalysisService()
    mock_notifier = MagicMock()
    mock_notifier.bus_license_plate = "51B-12345"
    mock_notifier.bus_route = "Tuyến số 1"
    mock_notifier.is_configured.return_value = True
    mock_notifier._enabled = True
    mock_notifier.send_alert.return_value = MagicMock(success=True, status_code=200)

    alert_service = AlertService()
    service._alert_service = alert_service
    service._repo = MagicMock()

    with patch("src.services.alert_service.get_telegram_notifier", return_value=mock_notifier):
        suspected_items = [
            {
                "track_id": "T2",
                "display_id": "T2",
                "class_name": "backpack",
                "seat_id": "004",
                "status": "SUSPECTED",
                "confidence": 0.92,
            }
        ]
        alerts = service.process_browser_camera_events(
            suspected_items=suspected_items,
            snapshot_base64=None,
            camera_name="Live Camera (Browser)",
        )
        assert len(alerts) == 1
        assert mock_notifier.send_alert.called

        _, kwargs = mock_notifier.send_alert.call_args
        assert kwargs.get("bus_license_plate") == "51B-12345"
        assert kwargs.get("bus_route") == "Tuyến số 1"

