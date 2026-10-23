"""
src/notifications/telegram_notifier.py
Dịch vụ gửi thông báo cảnh báo đồ bỏ quên qua Telegram Bot chính thức.
- Bảo mật: Không hard-code token, không in token trong log/lỗi.
- Bất đồng bộ / Non-blocking: Sử dụng ThreadPoolExecutor để không làm nghẽn luồng xử lý video/camera.
- Chống spam & Deduplication: Thread-safe cooldown và lọc trùng lặp sự kiện.
- Tuân thủ định dạng thông điệp Lost & Found AI Bus.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import html
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import requests

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ALERTS_ENABLED,
    TELEGRAM_ALERT_COOLDOWN_SECONDS,
    get_current_time,
    APP_TIMEZONE,
)
from src.detection.class_mapper import format_seat_label, get_user_facing_label
from src.models import Alert, Event
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Biểu thức regex để che giấu token Telegram nếu vô tình xuất hiện trong url hoặc error string
TOKEN_REDACT_REGEX = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


def sanitize_telegram_error(text: Union[str, Exception], token: Optional[str] = None) -> str:
    """Loại bỏ token Telegram khỏi chuỗi lỗi hoặc ngoại lệ để tránh rò rỉ bảo mật."""
    msg = str(text)
    if token and len(token) > 5:
        msg = msg.replace(token, "[REDACTED_TOKEN]")
    msg = TOKEN_REDACT_REGEX.sub("bot[REDACTED_TOKEN]", msg)
    return msg


@dataclass
class TelegramResponse:
    """Kết quả trả về sau khi gửi yêu cầu tới Telegram Bot API."""
    success: bool
    status_code: int
    message: str
    error_type: Optional[str] = None
    raw_response: Optional[dict] = None


class TelegramNotifier:
    """
    Dịch vụ gửi thông báo và cảnh báo tới Telegram Bot API.
    Thiết kế hướng an toàn, non-blocking, và chống spam.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        cooldown_seconds: Optional[float] = None,
    ):
        self._bot_token = (bot_token if bot_token is not None else (TELEGRAM_BOT_TOKEN or "")).strip()
        self._chat_id = (chat_id if chat_id is not None else (TELEGRAM_CHAT_ID or "")).strip()
        self._enabled = enabled if enabled is not None else TELEGRAM_ALERTS_ENABLED
        self._cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None else TELEGRAM_ALERT_COOLDOWN_SECONDS
        )

        # Thread pool chạy ngầm để gửi HTTP request không block inference loop
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="telegram_notifier")

        # Quản lý deduplication thread-safe
        self._lock = threading.Lock()
        self._last_alert_timestamps: Dict[str, float] = {}
        self._notified_event_ids: Set[str] = set()

        # Quản lý trạng thái phân phối gần nhất cho giao diện chẩn đoán UI
        self._last_delivery_status: Dict[str, Any] = {
            "has_attempt": False,
            "success": False,
            "status_code": 0,
            "message": "",
            "timestamp": "",
            "chat_id": self._chat_id,
            "alert_id": "",
        }

    @property
    def bot_token(self) -> str:
        return self._bot_token

    @bot_token.setter
    def bot_token(self, val: str):
        self._bot_token = (val or "").strip()

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @chat_id.setter
    def chat_id(self, val: str):
        self._chat_id = (val or "").strip()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = bool(val)

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    @cooldown_seconds.setter
    def cooldown_seconds(self, val: float):
        self._cooldown_seconds = max(1.0, float(val))

    def is_configured(self) -> bool:
        """Kiểm tra dịch vụ đã có đầy đủ Bot Token và Chat ID chưa."""
        return bool(self._bot_token and self._chat_id)

    def get_masked_token(self) -> str:
        """Trả về token dạng che mờ để hiển thị an toàn trên giao diện UI."""
        if not self._bot_token:
            return ""
        if len(self._bot_token) <= 10:
            return "••••••••"
        return f"{self._bot_token[:4]}••••••••{self._bot_token[-4:]}"

    def get_last_delivery_status(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái phân phối gần nhất phục vụ UI Diagnostics."""
        with self._lock:
            if not hasattr(self, "_last_delivery_status"):
                self._last_delivery_status = {
                    "has_attempt": False,
                    "success": False,
                    "status_code": 0,
                    "message": "",
                    "timestamp": "",
                    "chat_id": getattr(self, "_chat_id", ""),
                    "alert_id": "",
                }
            return dict(self._last_delivery_status)

    def reset_deduplication(self):
        """Xóa bộ đệm chống spam (thường dùng trong unit tests hoặc restart session)."""
        with self._lock:
            self._last_alert_timestamps.clear()
            self._notified_event_ids.clear()

    # ──────────────────────────────────────────────
    # KIỂM TRA KẾT NỐI (TEST CONNECTION)
    # ──────────────────────────────────────────────
    def test_connection(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Kiểm tra thông tin token và kết nối mạng bằng API getMe của Telegram.
        Nếu có chat_id, gửi 1 tin nhắn kiểm tra tới kênh.
        """
        token = (bot_token if bot_token is not None else self._bot_token).strip()
        target_chat = (chat_id if chat_id is not None else self._chat_id).strip()

        if not token:
            err_msg = "Chưa cấu hình Telegram Bot Token (vui lòng kiểm tra file .env)"
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": 400,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg

        url = f"https://api.telegram.org/bot{token}/getMe"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "LostAndFoundAIBus/1.0"},
                timeout=(5.0, 8.0),
            )
            data = resp.json()
        except requests.exceptions.Timeout:
            err_msg = "Lỗi kết nối: Hết thời gian chờ phản hồi từ máy chủ Telegram (Timeout)."
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": 408,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg
        except requests.exceptions.RequestException as e:
            err_msg = f"Lỗi kết nối mạng: {sanitize_telegram_error(e, token)}"
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": 500,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg
        except Exception as e:
            err_msg = f"Lỗi không xác định: {sanitize_telegram_error(e, token)}"
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": 500,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg

        if resp.status_code == 401:
            err_msg = "Bot Token không hợp lệ hoặc đã bị vô hiệu hóa bởi Telegram."
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": 401,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg

        if not data.get("ok"):
            err_desc = data.get("description", "Không xác định")
            err_msg = f"Lỗi Telegram API: {err_desc}"
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": False,
                    "status_code": resp.status_code,
                    "message": err_msg,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            return False, err_msg

        bot_username = data.get("result", {}).get("username", "Bot")
        if target_chat:
            test_text = (
                f"🔔 <b>[Lost & Found AI Bus]</b> Kiểm tra kết nối thành công!\n"
                f"Hệ thống giám sát trần xe buýt đã liên kết với Bot <code>@{bot_username}</code>."
            )
            sub_res = self.send_message(text=test_text, chat_id=target_chat, bot_token=token)
            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": sub_res.success,
                    "status_code": sub_res.status_code,
                    "message": "Thành công" if sub_res.success else sub_res.message,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": target_chat,
                    "alert_id": "TEST_CONN",
                }
            if sub_res.success:
                return True, f"Kết nối thành công! Đã gửi tin nhắn test tới Chat ID: {target_chat} (Bot @{bot_username})"
            return False, f"Bot Token đúng (@{bot_username}) nhưng không gửi được tới Chat ID {target_chat}: {sub_res.message}"

        with self._lock:
            self._last_delivery_status = {
                "has_attempt": True,
                "success": True,
                "status_code": 200,
                "message": f"Bot Token hợp lệ (@{bot_username})",
                "timestamp": get_current_time().isoformat(),
                "chat_id": "",
                "alert_id": "TEST_CONN",
            }
        return True, f"Bot Token hợp lệ! Đã xác thực thành công Bot @{bot_username}."

    # ──────────────────────────────────────────────
    # GỬI TIN NHẮN (SEND MESSAGE)
    # ──────────────────────────────────────────────
    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
        parse_mode: str = "HTML",
        timeout: Tuple[float, float] = (5.0, 10.0),
    ) -> TelegramResponse:
        """Gửi tin nhắn văn bản (HTML) qua Telegram API."""
        token = (bot_token if bot_token is not None else self._bot_token).strip()
        target_chat = (chat_id if chat_id is not None else self._chat_id).strip()

        if not token:
            return TelegramResponse(
                success=False,
                status_code=400,
                message="Thiếu Telegram Bot Token",
                error_type="MISSING_TOKEN",
            )
        if not target_chat:
            return TelegramResponse(
                success=False,
                status_code=400,
                message="Thiếu Telegram Chat ID",
                error_type="MISSING_CHAT_ID",
            )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"User-Agent": "LostAndFoundAIBus/1.0"},
                timeout=timeout,
            )
            status_code = resp.status_code
            try:
                res_data = resp.json()
            except Exception:
                res_data = {}

            if status_code == 200 and res_data.get("ok"):
                return TelegramResponse(
                    success=True,
                    status_code=200,
                    message="Thành công",
                    raw_response=res_data,
                )

            desc = res_data.get("description", resp.reason or f"HTTP {status_code}")
            if status_code == 401:
                friendly_err = "Bot Token không hợp lệ"
            elif status_code == 400 and "chat not found" in desc.lower():
                friendly_err = "Không tìm thấy Chat ID (Người dùng chưa mở bot và bấm Start)"
            elif status_code == 403:
                friendly_err = "Bot bị chặn bởi người dùng (Forbidden)"
            else:
                friendly_err = f"Lỗi Telegram ({status_code}): {desc}"

            return TelegramResponse(
                success=False,
                status_code=status_code,
                message=friendly_err,
                error_type="TELEGRAM_API_ERROR",
                raw_response=res_data,
            )

        except requests.exceptions.Timeout:
            return TelegramResponse(
                success=False,
                status_code=408,
                message="Hết thời gian chờ kết nối tới Telegram (Timeout)",
                error_type="TIMEOUT",
            )
        except requests.exceptions.RequestException as e:
            return TelegramResponse(
                success=False,
                status_code=500,
                message=f"Lỗi mạng: {sanitize_telegram_error(e, token)}",
                error_type="NETWORK_ERROR",
            )
        except Exception as e:
            return TelegramResponse(
                success=False,
                status_code=500,
                message=f"Lỗi: {sanitize_telegram_error(e, token)}",
                error_type="UNKNOWN_ERROR",
            )

    # ──────────────────────────────────────────────
    # GỬI ẢNH BẰNG CHỨNG (SEND PHOTO)
    # ──────────────────────────────────────────────
    def send_photo(
        self,
        photo_path: Union[str, Path],
        caption: str = "",
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
        parse_mode: str = "HTML",
        timeout: Tuple[float, float] = (8.0, 20.0),
    ) -> TelegramResponse:
        """Gửi ảnh chụp bằng chứng kèm chú thích qua Telegram sendPhoto API."""
        token = (bot_token if bot_token is not None else self._bot_token).strip()
        target_chat = (chat_id if chat_id is not None else self._chat_id).strip()

        if not token or not target_chat:
            return TelegramResponse(
                success=False,
                status_code=400,
                message="Thiếu Bot Token hoặc Chat ID",
                error_type="MISSING_CREDENTIALS",
            )

        p = Path(photo_path)
        if not p.exists() or not p.is_file():
            # Nếu file ảnh không tồn tại, tự động fallback sang gửi tin nhắn văn bản thuần
            logger.warning(f"Ảnh bằng chứng '{photo_path}' không tồn tại. Fallback sang gửi tin nhắn text.")
            return self.send_message(text=caption, chat_id=target_chat, bot_token=token, parse_mode=parse_mode)

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(p, "rb") as img_file:
                files = {"photo": (p.name, img_file, "image/jpeg")}
                data = {
                    "chat_id": target_chat,
                    "caption": caption[:1024],  # Giới hạn caption của Telegram sendPhoto là 1024 ký tự
                    "parse_mode": parse_mode,
                }
                resp = requests.post(
                    url,
                    data=data,
                    files=files,
                    headers={"User-Agent": "LostAndFoundAIBus/1.0"},
                    timeout=timeout,
                )

            status_code = resp.status_code
            try:
                res_data = resp.json()
            except Exception:
                res_data = {}

            if status_code == 200 and res_data.get("ok"):
                return TelegramResponse(
                    success=True,
                    status_code=200,
                    message="Thành công",
                    raw_response=res_data,
                )

            desc = res_data.get("description", resp.reason or f"HTTP {status_code}")
            return TelegramResponse(
                success=False,
                status_code=status_code,
                message=f"Lỗi gửi ảnh Telegram ({status_code}): {desc}",
                error_type="TELEGRAM_API_ERROR",
                raw_response=res_data,
            )

        except requests.exceptions.Timeout:
            return TelegramResponse(
                success=False,
                status_code=408,
                message="Hết thời gian gửi ảnh tới Telegram (Timeout)",
                error_type="TIMEOUT",
            )
        except requests.exceptions.RequestException as e:
            return TelegramResponse(
                success=False,
                status_code=500,
                message=f"Lỗi mạng khi gửi ảnh: {sanitize_telegram_error(e, token)}",
                error_type="NETWORK_ERROR",
            )
        except Exception as e:
            return TelegramResponse(
                success=False,
                status_code=500,
                message=f"Lỗi khi gửi ảnh: {sanitize_telegram_error(e, token)}",
                error_type="UNKNOWN_ERROR",
            )

    # ──────────────────────────────────────────────
    # FORMATTING CẢNH BÁO ĐỒ BỎ QUÊN CHUẨN
    # ──────────────────────────────────────────────
    @staticmethod
    def format_forgotten_item_message(
        alert: Alert,
        event: Optional[Event] = None,
        bus_or_camera_id: str = "Tuyến xe buýt Demo 01",
    ) -> str:
        """
        Định dạng bản tin cảnh báo đồ bỏ quên theo mẫu chuẩn:
        🚨 LOST & FOUND AI BUS — CẢNH BÁO ĐỒ BỎ QUÊN
        🚌 Khu vực: [bus or camera identifier]
        💺 Ghế: [seat ID, if known]
        📦 Đồ vật: [generic item]
        🆔 Mã theo dõi: [stable track ID]
        🕒 Thời điểm: [timestamp]
        📋 Trạng thái: [confirmed by application logic / needs review]
        """
        area_safe = html.escape(bus_or_camera_id or "Khu vực khoang khách xe buýt")

        # Xác định tên ghế
        seat_raw = alert.seat_id or (event.seat_id if event else "")
        if seat_raw and seat_raw.strip() and seat_raw.strip().lower() not in ("chưa rõ", "none", ""):
            seat_display = html.escape(format_seat_label(seat_raw))
            has_seat = True
        else:
            seat_display = "Chưa rõ vị trí ghế (khu vực hành lang / sàn xe)"
            has_seat = False

        # Xác định tên đồ vật
        item_class = ""
        if event and event.class_name:
            item_class = event.class_name
        elif alert.object_id:
            item_class = alert.object_id
        item_display = html.escape(get_user_facing_label(item_class) if item_class else "Đồ vật / Hành lý")

        # Mã theo dõi ổn định
        track_id = ""
        if event and event.last_track_id not in (-1, None):
            track_id = f"T{event.last_track_id}"
        elif event and event.track_id not in (-1, None):
            track_id = f"T{event.track_id}"
        elif alert.object_id:
            track_id = alert.object_id
        else:
            track_id = "Chưa gán ID"
        track_safe = html.escape(str(track_id))

        # Thời điểm
        ts_str = alert.created_at or (event.timestamp if event else "")
        if not ts_str:
            ts_display = get_current_time().strftime("%H:%M:%S %d/%m/%Y")
        else:
            try:
                dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(APP_TIMEZONE)
                ts_display = dt.strftime("%H:%M:%S %d/%m/%Y")
            except Exception:
                ts_display = str(ts_str)
        ts_safe = html.escape(ts_display)

        # Trạng thái logic
        status_val = (event.status if event else alert.status or "").lower()
        if status_val == "suspected" and has_seat:
            status_desc = "Xác nhận bởi logic AI (Ghế trống sau khi khách rời đi)"
        elif status_val == "review_required":
            status_desc = "Cần nhân viên kiểm tra (Vật thể biến mất bất thường khỏi ghế trống)"
        elif not has_seat:
            status_desc = "Ứng viên cần rà soát (Chưa liên kết vùng ghế cố định)"
        else:
            status_desc = "Nghi ngờ bỏ quên (Đang theo dõi vị trí)"
        status_safe = html.escape(status_desc)

        message = (
            f"🚨 <b>LOST & FOUND AI BUS — CẢNH BÁO ĐỒ BỎ QUÊN</b>\n\n"
            f"🚌 <b>Khu vực:</b> {area_safe}\n"
            f"💺 <b>Ghế:</b> {seat_display}\n"
            f"📦 <b>Đồ vật:</b> {item_display}\n"
            f"🆔 <b>Mã theo dõi:</b> <code>{track_safe}</code>\n"
            f"🕒 <b>Thời điểm:</b> {ts_safe}\n"
            f"📋 <b>Trạng thái:</b> {status_safe}\n\n"
            f"<i>Vui lòng kiểm tra camera hoặc khoang hành khách để xác nhận đồ vật.</i>"
        )
        return message

    # ──────────────────────────────────────────────
    # ĐIỀU PHỐI CẢNH BÁO (DEDUPLICATION & NON-BLOCKING DISPATCH)
    # ──────────────────────────────────────────────
    def send_alert(
        self,
        alert: Alert,
        event: Optional[Event] = None,
        bus_or_camera_id: str = "Tuyến xe buýt Demo 01",
        is_async: bool = True,
        force: bool = False,
        on_complete: Optional[Callable[["TelegramResponse"], None]] = None,
    ) -> TelegramResponse:
        """
        Gửi cảnh báo phát hiện đồ bỏ quên tới Telegram Bot với cơ chế chống spam (anti-spam / deduplication).

        Args:
            alert: Đối tượng Alert
            event: Đối tượng Event tương ứng nếu có
            bus_or_camera_id: Tên xe buýt hoặc camera nguồn
            is_async: True nếu muốn gửi trong thread pool chạy ngầm (không chặn FPS xử lý video)
            force: True nếu muốn bỏ qua kiểm tra cooldown (ví dụ: nút test trên UI)
            on_complete: Callback tùy chọn được gọi sau khi hoàn tất gửi tin (nhận TelegramResponse thực tế)

        Returns:
            TelegramResponse (khi is_async=True, trả về kết quả tiếp nhận hàng đợi)
        """
        if not self._enabled:
            return TelegramResponse(
                success=False,
                status_code=200,
                message="Gửi cảnh báo Telegram đang bị vô hiệu hóa (disabled)",
                error_type="DISABLED",
            )

        if not self.is_configured():
            return TelegramResponse(
                success=False,
                status_code=400,
                message="Chưa cấu hình Telegram Bot Token hoặc Chat ID",
                error_type="NOT_CONFIGURED",
            )

        now_sec = get_current_time().timestamp()
        evt_id = (event.event_id if event else "") or alert.event_id or alert.alert_id
        seat_id = alert.seat_id or (event.seat_id if event else "")
        track_id = event.last_track_id if (event and event.last_track_id != -1) else alert.object_id
        cooldown_key = f"{evt_id}_{seat_id}_{track_id}"

        # Kiểm tra Deduplication & Cooldown
        if not force:
            with self._lock:
                # 1. Tránh gửi lại cho cùng 1 event_id đã thông báo thành công trước đó
                if evt_id and evt_id in self._notified_event_ids:
                    logger.debug(f"Bỏ qua cảnh báo lặp (Event ID {evt_id} đã được thông báo trước đó)")
                    return TelegramResponse(
                        success=True,
                        status_code=200,
                        message="Bỏ qua do sự kiện đã được gửi thông báo trước đó (Event Deduplicated)",
                    )

                # 2. Tránh spam trong khoảng thời gian cooldown
                last_time = self._last_alert_timestamps.get(cooldown_key, -9999.0)
                if (now_sec - last_time) < self._cooldown_seconds:
                    logger.debug(f"Bỏ qua cảnh báo lặp (Cooldown {self._cooldown_seconds}s): {cooldown_key}")
                    return TelegramResponse(
                        success=True,
                        status_code=200,
                        message=f"Bỏ qua do đang trong thời gian giãn cách ({self._cooldown_seconds}s cooldown)",
                    )

                # Đánh dấu thời điểm tiếp nhận để ngăn gửi trùng lặp đồng thời
                self._last_alert_timestamps[cooldown_key] = now_sec

        # Chuẩn bị nội dung
        msg_text = self.format_forgotten_item_message(
            alert=alert,
            event=event,
            bus_or_camera_id=bus_or_camera_id,
        )

        def _do_send() -> TelegramResponse:
            photo_path = alert.snapshot_path or (event.snapshot_path if event else "")
            max_retries = 2
            retry_delay = 1.5
            res: Optional[TelegramResponse] = None

            for attempt in range(max_retries + 1):
                if photo_path and Path(photo_path).exists():
                    res = self.send_photo(photo_path=photo_path, caption=msg_text)
                else:
                    res = self.send_message(text=msg_text)

                if res.success:
                    break

                # Dừng thử lại nếu gặp lỗi từ phía client không thể khắc phục (400, 401, 403, 404)
                if res.status_code in (400, 401, 403, 404) and res.error_type not in ("TIMEOUT", "NETWORK_ERROR"):
                    logger.warning(
                        f"Gửi cảnh báo Telegram gặp lỗi client ({res.status_code}): {res.message}. Không thử lại."
                    )
                    break

                # Lỗi mạng tạm thời hoặc 5xx: thử lại nếu còn lượt
                if attempt < max_retries:
                    logger.warning(
                        f"Gửi cảnh báo Telegram thất bại ({res.status_code}): {res.message}. Thử lại lần {attempt + 1}/{max_retries} sau {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

            final_res = res or TelegramResponse(
                success=False,
                status_code=500,
                message="Không nhận được phản hồi từ máy chủ Telegram",
                error_type="NO_RESPONSE",
            )

            with self._lock:
                self._last_delivery_status = {
                    "has_attempt": True,
                    "success": final_res.success,
                    "status_code": final_res.status_code,
                    "message": "Thành công" if final_res.success else final_res.message,
                    "timestamp": get_current_time().isoformat(),
                    "chat_id": self._chat_id,
                    "alert_id": alert.alert_id,
                }
                if final_res.success:
                    if evt_id:
                        self._notified_event_ids.add(evt_id)
                    logger.info(f"Đã gửi cảnh báo Telegram thành công cho Alert {alert.alert_id}")
                else:
                    # Nếu thất bại, xóa key cooldown để lần sau có thể thử lại thay vì bị khóa hoàn toàn
                    if cooldown_key in self._last_alert_timestamps:
                        del self._last_alert_timestamps[cooldown_key]
                    logger.warning(f"Gửi cảnh báo Telegram thất bại ({final_res.status_code}): {final_res.message}")

            if on_complete:
                try:
                    on_complete(final_res)
                except Exception as ex:
                    logger.warning(f"Lỗi khi thực thi on_complete callback: {ex}")

            return final_res

        if is_async:
            self._executor.submit(_do_send)
            return TelegramResponse(
                success=True,
                status_code=202,
                message="Đã đưa cảnh báo vào hàng đợi gửi Telegram ngầm (Non-blocking)",
            )
        else:
            return _do_send()


# Singleton instance để tái sử dụng thống nhất trên toàn hệ thống
_global_telegram_notifier: Optional[TelegramNotifier] = None
_global_notifier_lock = threading.Lock()


def get_telegram_notifier() -> TelegramNotifier:
    """Lấy hoặc khởi tạo instance TelegramNotifier toàn cục."""
    global _global_telegram_notifier
    with _global_notifier_lock:
        if _global_telegram_notifier is None:
            _global_telegram_notifier = TelegramNotifier()
        else:
            # Tự động đồng bộ/nâng cấp nếu instance trong memory thuộc phiên bản cũ (Streamlit Cloud hot-reload)
            if not hasattr(_global_telegram_notifier, "get_last_delivery_status"):
                _global_telegram_notifier.get_last_delivery_status = (
                    TelegramNotifier.get_last_delivery_status.__get__(_global_telegram_notifier, TelegramNotifier)
                )
            if not hasattr(_global_telegram_notifier, "_last_delivery_status"):
                _global_telegram_notifier._last_delivery_status = {
                    "has_attempt": False,
                    "success": False,
                    "status_code": 0,
                    "message": "",
                    "timestamp": "",
                    "chat_id": getattr(_global_telegram_notifier, "_chat_id", ""),
                    "alert_id": "",
                }
            # Luôn cập nhật lại token & chat_id nếu môi trường thay đổi
            if not _global_telegram_notifier.bot_token and TELEGRAM_BOT_TOKEN:
                _global_telegram_notifier.bot_token = TELEGRAM_BOT_TOKEN
            if not _global_telegram_notifier.chat_id and TELEGRAM_CHAT_ID:
                _global_telegram_notifier.chat_id = TELEGRAM_CHAT_ID
        return _global_telegram_notifier
