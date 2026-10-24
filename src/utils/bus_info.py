"""
src/utils/bus_info.py
Quản lý thông tin Xe buýt (Biển số xe và Tuyến xe) dùng chung và nhất quán trên toàn hệ thống:
1. Lưu trữ đồng bộ giữa: file JSON cấu hình cố định (data/bus_config.json), .env, os.environ, session_state và TelegramNotifier.
2. Tự động sinh biển số xe theo định dạng nnX-nnnnn (ví dụ: 51B-12345) đúng một lần và lưu cố định nếu chưa có biển số.
3. Không sinh lại biển số khi Streamlit rerun, đổi trang, mở lại camera hay gửi nhiều ảnh cảnh báo.
4. Ưu tiên biển số người dùng nhập thủ công. Khi người dùng chủ động xóa và lưu, tự sinh một biển số demo mới.
5. Tuyến xe buýt mặc định là "Tuyến số 1" nếu chưa nhập. Tuyệt đối không để trống và không sinh tuyến ngẫu nhiên.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BUS_ROUTE = "Tuyến số 1"
CONFIG_FILE_PATH = Path("data/bus_config.json")


def generate_demo_license_plate() -> str:
    """
    Sinh biển số xe buýt demo theo định dạng mô phỏng: nnX-nnnnn
    - Hai ký tự n đầu tiên: chữ số từ 10-99 (mã tỉnh/thành phố, ví dụ 51).
    - X: chữ cái in hoa từ A-Z (thường là B cho xe buýt/xe khách).
    - Năm ký tự cuối: chữ số từ 10000-99999.
    Ví dụ: 51B-12345
    """
    nn = f"{random.randint(10, 99):02d}"
    x = random.choice("BCDEFGHIKLMNPSTUVXYZ")
    nnnnn = f"{random.randint(10000, 99999):05d}"
    return f"{nn}{x}-{nnnnn}"


def _read_persistent_file() -> Dict[str, str]:
    """Đọc thông tin xe buýt từ file JSON lưu trữ cục bộ."""
    if CONFIG_FILE_PATH.exists():
        try:
            data = json.loads(CONFIG_FILE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "bus_license_plate": str(data.get("bus_license_plate", "")).strip(),
                    "bus_route": str(data.get("bus_route", "")).strip(),
                }
        except Exception as e:
            logger.warning(f"Lỗi đọc file cấu hình {CONFIG_FILE_PATH}: {e}")
    return {}


def _write_persistent_file(plate: str, route: str) -> bool:
    """Ghi thông tin xe buýt vào file JSON lưu trữ cục bộ."""
    try:
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bus_license_plate": plate,
            "bus_route": route,
        }
        CONFIG_FILE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Lỗi ghi file cấu hình {CONFIG_FILE_PATH}: {e}")
        return False


def _update_env_file(plate: str, route: str) -> bool:
    """Cập nhật giá trị vào file .env nếu file này tồn tại."""
    env_p = Path(".env")
    if not env_p.exists():
        return False
    try:
        lines = env_p.read_text(encoding="utf-8").splitlines()
        upd = {
            "BUS_LICENSE_PLATE": plate,
            "BUS_ROUTE": route,
        }
        new_lines = []
        matched = set()
        for line in lines:
            line_str = line.strip()
            if "=" in line_str and not line_str.startswith("#"):
                key, _ = line_str.split("=", 1)
                key = key.strip()
                if key in upd:
                    new_lines.append(f"{key}={upd[key]}")
                    matched.add(key)
                    continue
            new_lines.append(line)
        for k, v in upd.items():
            if k not in matched:
                new_lines.append(f"{k}={v}")
        env_p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Không thể cập nhật .env cho bus info: {e}")
        return False


def get_bus_info(auto_generate_if_missing: bool = True) -> Tuple[str, str]:
    """
    Lấy thông tin (Biển số xe, Tuyến xe) theo thứ tự ưu tiên:
    1. st.session_state (nếu ứng dụng Streamlit đang hoạt động)
    2. os.environ
    3. File lưu trữ data/bus_config.json
    4. Cấu hình mặc định trong config/settings.py
    5. Nếu chưa có biển số và auto_generate_if_missing=True: Tự động sinh 1 biển số demo (nnX-nnnnn)
       và LƯU LẠI NGAY để các lần sau (và mọi cảnh báo) dùng cố định giá trị này.
    6. Nếu chưa có tuyến xe: Mặc định là 'Tuyến số 1' (không tự sinh ngẫu nhiên).
    """
    plate = ""
    route = ""

    # 1. Thử lấy từ Streamlit session_state
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            s_plate = st.session_state.get("bus_license_plate")
            s_route = st.session_state.get("bus_route")
            if s_plate and str(s_plate).strip():
                plate = str(s_plate).strip()
            if s_route and str(s_route).strip():
                route = str(s_route).strip()
    except Exception:
        pass

    # 2. Thử lấy từ os.environ
    if not plate:
        plate = os.environ.get("BUS_LICENSE_PLATE", "").strip()
    if not route:
        route = os.environ.get("BUS_ROUTE", "").strip()

    # 3. Thử lấy từ file data/bus_config.json
    if not plate or not route:
        file_info = _read_persistent_file()
        if not plate and file_info.get("bus_license_plate"):
            plate = file_info["bus_license_plate"]
        if not route and file_info.get("bus_route"):
            route = file_info["bus_route"]

    # 4. Thử lấy từ config/settings.py
    if not plate:
        try:
            from config.settings import BUS_LICENSE_PLATE
            if BUS_LICENSE_PLATE and str(BUS_LICENSE_PLATE).strip():
                plate = str(BUS_LICENSE_PLATE).strip()
        except Exception:
            pass

    if not route:
        try:
            from config.settings import BUS_ROUTE
            if BUS_ROUTE and str(BUS_ROUTE).strip():
                route = str(BUS_ROUTE).strip()
        except Exception:
            pass

    # 5. Xử lý quy tắc tự động sinh hoặc gán giá trị mặc định
    need_save = False
    if not plate and auto_generate_if_missing:
        plate = generate_demo_license_plate()
        need_save = True

    if not route:
        route = DEFAULT_BUS_ROUTE
        need_save = True

    # Đồng bộ sang môi trường và file nếu vừa sinh mới
    if need_save:
        save_bus_info(plate=plate, route=route, auto_generate_if_empty=False)
    else:
        # Đảm bảo os.environ và session_state đồng bộ
        os.environ["BUS_LICENSE_PLATE"] = plate
        os.environ["BUS_ROUTE"] = route
        try:
            import streamlit as st
            if hasattr(st, "session_state"):
                st.session_state["bus_license_plate"] = plate
                st.session_state["bus_route"] = route
        except Exception:
            pass

    return plate, route


def save_bus_info(
    plate: Optional[str] = None,
    route: Optional[str] = None,
    auto_generate_if_empty: bool = True,
) -> Tuple[str, str]:
    """
    Lưu thông tin xe buýt và áp dụng thống nhất cho toàn bộ hệ thống:
    - Nếu plate trống:
        + Nếu auto_generate_if_empty=True: Tự động sinh một biển số demo mới (phục vụ khi người dùng xóa để đổi biển số demo)
        + Nếu False: giữ nguyên hoặc lấy từ get_bus_info()
    - Nếu route trống: gán về mặc định 'Tuyến số 1'.
    - Lưu vào: data/bus_config.json, .env, os.environ, session_state và TelegramNotifier.
    """
    cleaned_plate = (plate or "").strip()
    cleaned_route = (route or "").strip()

    if not cleaned_plate:
        if auto_generate_if_empty:
            cleaned_plate = generate_demo_license_plate()
        else:
            existing_plate, _ = get_bus_info(auto_generate_if_missing=True)
            cleaned_plate = existing_plate

    if not cleaned_route:
        cleaned_route = DEFAULT_BUS_ROUTE

    # Lưu file JSON cục bộ
    _write_persistent_file(cleaned_plate, cleaned_route)

    # Cập nhật file .env
    _update_env_file(cleaned_plate, cleaned_route)

    # Cập nhật os.environ
    os.environ["BUS_LICENSE_PLATE"] = cleaned_plate
    os.environ["BUS_ROUTE"] = cleaned_route

    # Cập nhật Streamlit session_state nếu có
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            st.session_state["bus_license_plate"] = cleaned_plate
            st.session_state["bus_route"] = cleaned_route
    except Exception:
        pass

    # Cập nhật TelegramNotifier singleton
    try:
        from src.notifications.telegram_notifier import get_telegram_notifier
        notifier = get_telegram_notifier()
        if notifier:
            notifier.bus_license_plate = cleaned_plate
            notifier.bus_route = cleaned_route
    except Exception as e:
        logger.debug(f"Không thể cập nhật TelegramNotifier với bus info mới: {e}")

    logger.info(f"[BusInfo] Đã lưu thông tin xe: Biển số='{cleaned_plate}', Tuyến='{cleaned_route}'")
    return cleaned_plate, cleaned_route
