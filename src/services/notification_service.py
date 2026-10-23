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
    # TELEGRAM BOT ADAPTER (Delegated to TelegramNotifier)
    # ──────────────────────────────────────────────
    @staticmethod
    def test_telegram_connection(bot_token: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Kiểm tra token bot Telegram bằng API getMe.
        Nếu có chat_id, gửi thử một tin nhắn kiểm tra.
        """
        from src.notifications.telegram_notifier import get_telegram_notifier
        notifier = get_telegram_notifier()
        return notifier.test_connection(bot_token=bot_token, chat_id=chat_id)

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
        from src.notifications.telegram_notifier import get_telegram_notifier
        notifier = get_telegram_notifier()
        if image_path and Path(image_path).exists():
            res = notifier.send_photo(photo_path=image_path, caption=message, chat_id=chat_id, bot_token=bot_token)
        else:
            res = notifier.send_message(text=message, chat_id=chat_id, bot_token=bot_token)
        return res.success, res.status_code, res.message

    @staticmethod
    def get_last_delivery_status() -> Dict:
        """Lấy thông tin trạng thái phân phối Telegram gần nhất từ TelegramNotifier."""
        from src.notifications.telegram_notifier import get_telegram_notifier
        notifier = get_telegram_notifier()
        if hasattr(notifier, "get_last_delivery_status"):
            return notifier.get_last_delivery_status()
        return {
            "has_attempt": False,
            "success": False,
            "status_code": 0,
            "message": "",
            "timestamp": "",
            "chat_id": getattr(notifier, "chat_id", ""),
            "alert_id": "",
        }


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
