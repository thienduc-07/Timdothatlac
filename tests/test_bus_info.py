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
