"""
src/services/notification_service.py
Module phụ trách gửi thông báo thực tế qua các kênh:
- Telegram Bot (API chính thức)
- Email SMTP (HTML + đính kèm ảnh bằng chứng)
- Webhook HTTP POST (Slack, Discord, custom server)
- Web Push Notification (kiểm tra trạng thái & adapter)
Đặc biệt: Ghi nhận mã lỗi thật, không giả mạo trạng thái thành công.
"""

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request
import urllib.error

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationService:
    """Xử lý gửi cảnh báo đa kênh và kiểm tra kết nối dịch vụ thực tế."""

    # ──────────────────────────────────────────────
    # TELEGRAM BOT ADAPTER
    # ──────────────────────────────────────────────
    @staticmethod
    def test_telegram_connection(bot_token: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Kiểm tra token bot Telegram bằng API getMe.
        Nếu có chat_id, thử gửi một tin nhắn kiểm tra.
        """
        if not bot_token or not bot_token.strip():
            return False, "Chưa nhập Bot Token (Trạng thái: Chưa cấu hình)"

        url = f"https://api.telegram.org/bot{bot_token.strip()}/getMe"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LostAndFoundAI/1.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                if res_data.get("ok"):
                    bot_name = res_data.get("result", {}).get("username", "Bot")
                    if chat_id and chat_id.strip():
                        # Gửi thử 1 tin nhắn test
                        ok, _, err = NotificationService.send_telegram_message(
                            bot_token=bot_token,
                            chat_id=chat_id,
                            message=f"🔔 [Lost & Found AI] Kết nối thành công tới Bot @{bot_name}!",
                        )
                        if ok:
                            return True, f"Kết nối thành công! Đã gửi tin nhắn test tới Chat ID: {chat_id}"
                        return False, f"Bot Token đúng (@{bot_name}) nhưng Chat ID lỗi: {err}"
                    return True, f"Bot Token hợp lệ! Tên bot: @{bot_name}"
                return False, f"Lỗi Telegram: {res_data.get('description', 'Không xác định')}"
        except urllib.error.HTTPError as e:
            try:
                err_json = json.loads(e.read().decode("utf-8"))
                return False, f"Lỗi HTTP {e.code}: {err_json.get('description', e.reason)}"
            except Exception:
                return False, f"Lỗi HTTP {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Lỗi kết nối mạng: {str(e)}"

    @staticmethod
    def send_telegram_message(
        bot_token: str,
        chat_id: str,
        message: str,
        image_path: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """
        Gửi tin nhắn Telegram kèm ảnh bằng chứng nếu có.
        Trả về: (thành_công, mã_lỗi, thông_điệp_lỗi)
        """
        if not bot_token or not chat_id:
            return False, 400, "Thiếu Bot Token hoặc Chat ID"

        # Nếu có ảnh và file tồn tại, dùng sendPhoto
        if image_path and Path(image_path).exists():
            return NotificationService._send_telegram_photo(bot_token, chat_id, message, image_path)

        # Ngược lại dùng sendMessage
        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
        payload = {
            "chat_id": chat_id.strip(),
            "text": message,
            "parse_mode": "HTML",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LostAndFoundAI/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    return True, 200, "Thành công"
                return False, response.status, f"HTTP Status {response.status}"
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                return False, e.code, err_data.get("description", e.reason)
            except Exception:
                return False, e.code, e.reason
        except Exception as e:
            return False, 500, str(e)

    @staticmethod
    def _send_telegram_photo(bot_token: str, chat_id: str, caption: str, image_path: str) -> Tuple[bool, int, str]:
        """Gửi ảnh bằng chứng qua Telegram Multipart API."""
        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendPhoto"
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

        try:
            with open(image_path, "rb") as f:
                img_data = f.read()

            parts = []
            # Field: chat_id
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n".encode("utf-8"))
            # Field: caption
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
            # Field: parse_mode
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
            # Field: photo file
            filename = Path(image_path).name
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"))
            parts.append(img_data)
            parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

            body = b"".join(parts)
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "LostAndFoundAI/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return True, 200, "Thành công"
                return False, response.status, f"HTTP Status {response.status}"
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                return False, e.code, err_data.get("description", e.reason)
            except Exception:
                return False, e.code, e.reason
        except Exception as e:
            return False, 500, str(e)

    # ──────────────────────────────────────────────
    # EMAIL SMTP ADAPTER
    # ──────────────────────────────────────────────
    @staticmethod
    def test_email_connection(smtp_host: str, smtp_port: int, user: str, password: str) -> Tuple[bool, str]:
        """Kiểm tra đăng nhập SMTP server."""
        if not smtp_host or not user:
            return False, "Chưa cấu hình máy chủ SMTP hoặc Email gửi"
        try:
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=8)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=8)
                server.starttls()
            server.login(user, password)
            server.quit()
            return True, f"Kết nối SMTP thành công tới {smtp_host}:{smtp_port}"
        except Exception as e:
            return False, f"Lỗi kết nối SMTP: {str(e)}"

    @staticmethod
    def send_email(
        smtp_host: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipient: str,
        subject: str,
        body_html: str,
        image_path: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """Gửi email thông báo HTML có đính kèm ảnh bằng chứng."""
        if not smtp_host or not sender or not recipient:
            return False, 400, "Thiếu thông tin cấu hình Email hoặc người nhận"

        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient

        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)
        msg_alternative.attach(MIMEText(body_html, "html"))

        if image_path and Path(image_path).exists():
            try:
                with open(image_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-ID", "<snapshot_evidence>")
                    img.add_header("Content-Disposition", "inline", filename=Path(image_path).name)
                    msg.attach(img)
            except Exception as e:
                logger.warning(f"Không thể đính kèm ảnh email: {e}")

        try:
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=12)
                server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
            server.quit()
            return True, 200, "Gửi email thành công"
        except Exception as e:
            return False, 500, str(e)

    # ──────────────────────────────────────────────
    # WEBHOOK ADAPTER (Slack / Discord / Custom)
    # ──────────────────────────────────────────────
    @staticmethod
    def send_webhook(webhook_url: str, payload: dict) -> Tuple[bool, int, str]:
        """Gửi HTTP POST payload JSON tới Webhook URL."""
        if not webhook_url or not webhook_url.startswith("http"):
            return False, 400, "Webhook URL không hợp lệ (phải bắt đầu bằng http:// hoặc https://)"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LostAndFoundAI/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return True, response.status, "Gửi webhook thành công"
        except urllib.error.HTTPError as e:
            return False, e.code, f"Lỗi HTTP {e.code}: {e.reason}"
        except Exception as e:
            return False, 500, str(e)
