"""
src/services/alert_service.py
Module quản lý vòng đời Alert, chống gửi cảnh báo trùng lặp (cooldown)
và điều phối gửi thông báo đa kênh (In-app, Telegram, Email, Webhook).
"""

from datetime import datetime
import os
from typing import Dict, List, Optional
import uuid

from src.models import Alert, Event
from src.database.alert_repository import AlertRepository
from src.services.notification_service import NotificationService
from src.notifications.telegram_notifier import get_telegram_notifier, TelegramNotifier
from src.utils.logger import get_logger
from config.settings import (
    ALERT_COOLDOWN_SECONDS,
    AlertPriority,
    AlertStatus,
    NotificationChannel,
    CLASS_NAMES_VI,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ALERTS_ENABLED,
    get_current_time,
)
from src.detection.class_mapper import get_user_facing_label, format_seat_label

logger = get_logger(__name__)


class AlertService:
    """Quản lý cảnh báo và điều phối thông báo."""

    def __init__(self, alert_repo: Optional[AlertRepository] = None):
        self._repo = alert_repo
        # Dict lưu thời điểm gửi cảnh báo gần nhất để chống spam: key (seat_id, class_name) -> timestamp_sec
        self._last_alert_times: Dict[str, float] = {}

    def set_repository(self, alert_repo: AlertRepository):
        self._repo = alert_repo

    def process_event_for_alert(
        self,
        event: Event,
        current_time_sec: float,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
    ) -> Optional[Alert]:
        """
        Đánh giá sự kiện theo quy tắc:
        - Chỉ phát sinh alert khi event ở trạng thái SUSPECTED hoặc REVIEW_REQUIRED.
        - Kiểm tra cooldown (tránh spam cùng 1 ghế / vật trong 60 giây).
        - Lưu vào Database.
        - Gửi thông báo tới các kênh được cấu hình.
        """
        if event.status not in ("suspected", "review_required"):
            return None

        # Key định danh duy nhất cho ghế và loại vật thể
        cooldown_key = f"{event.seat_id}_{event.class_name}_{event.last_track_id}"
        last_sent = self._last_alert_times.get(cooldown_key, -9999.0)
        if (current_time_sec - last_sent) < ALERT_COOLDOWN_SECONDS:
            logger.debug(f"Bỏ qua cảnh báo trùng lặp (cooldown): {cooldown_key}")
            return None

        self._last_alert_times[cooldown_key] = current_time_sec

        # Xác định mức độ ưu tiên
        priority = AlertPriority.HIGH if event.status == "suspected" else AlertPriority.MEDIUM
        now_iso = get_current_time().isoformat()
        vi_class = get_user_facing_label(event.class_name)
        seat_desc = format_seat_label(event.seat_id) if event.seat_id else "khu vực xe buýt"

        reason = (
            f"Cảnh báo: Đồ vật có thể bị bỏ quên trên {seat_desc} "
            f"sau {event.vacant_duration:.1f}s ghế trống."
        )

        alert_id = f"ALT_{uuid.uuid4().hex[:8].upper()}"
        alert = Alert(
            alert_id=alert_id,
            event_id=event.event_id or f"EVT_{event.track_id}",
            seat_id=event.seat_id or "Chưa rõ",
            object_id=f"O{event.last_track_id if event.last_track_id != -1 else event.track_id}",
            person_id="",
            priority=priority,
            status=AlertStatus.PENDING,
            reason=reason,
            snapshot_path=event.snapshot_path,
            created_at=now_iso,
            sent_channels=[NotificationChannel.IN_APP],
        )

        # Lưu Alert vào DB
        if self._repo:
            self._repo.save_alert(alert)

        # Điều phối gửi thông báo đa kênh
        self._dispatch_notifications(
            alert=alert,
            event=event,
            telegram_config=telegram_config,
            email_config=email_config,
            webhook_config=webhook_config,
        )

        return alert

    def _dispatch_notifications(
        self,
        alert: Alert,
        event: Event,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
        is_test: bool = False,
    ):
        """Gửi thông báo tới các kênh được cấu hình và ghi nhận log thực tế."""
        vi_class = get_user_facing_label(event.class_name)
        msg_title = f"⚠️ [LOST & FOUND AI] CẢNH BÁO ĐỒ BỎ QUÊN"
        msg_body = (
            f"<b>Mã cảnh báo:</b> {alert.alert_id}\n"
            f"<b>Vật thể:</b> {vi_class} ({alert.object_id})\n"
            f"<b>Vị trí:</b> {format_seat_label(alert.seat_id)}\n"
            f"<b>Mức độ:</b> {alert.priority}\n"
            f"<b>Lý do:</b> {alert.reason}\n"
            f"<b>Thời điểm:</b> {get_current_time().strftime('%H:%M:%S %d/%m/%Y')}"
        )

        # 1. Kênh Telegram
        # Nếu không truyền cấu hình riêng, tự động dùng biến môi trường hệ thống
        t_cfg = telegram_config
        if t_cfg is None:
            t_cfg = {
                "enabled": TELEGRAM_ALERTS_ENABLED,
                "bot_token": TELEGRAM_BOT_TOKEN,
                "chat_id": TELEGRAM_CHAT_ID,
            }

        if t_cfg and t_cfg.get("enabled", True):
            token = t_cfg.get("bot_token") or TELEGRAM_BOT_TOKEN
            chat_id = t_cfg.get("chat_id") or TELEGRAM_CHAT_ID
            if token and chat_id:
                notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)
                bus_name = getattr(event, "video_name", "Xe buýt Tuyến 01") or "Xe buýt Tuyến 01"
                res = notifier.send_alert(
                    alert=alert,
                    event=event,
                    bus_or_camera_id=bus_name,
                    is_async=not is_test,
                    force=is_test,
                )
                if res.success:
                    if NotificationChannel.TELEGRAM not in alert.sent_channels:
                        alert.sent_channels.append(NotificationChannel.TELEGRAM)
                    if self._repo:
                        self._repo.save_alert(alert)

                if self._repo:
                    self._repo.log_notification(
                        alert_id=alert.alert_id,
                        channel=NotificationChannel.TELEGRAM,
                        recipient=chat_id,
                        status="SUCCESS" if res.success else "FAILED",
                        status_code=res.status_code,
                        error_message="" if res.success else res.message,
                        payload={"message": msg_title, "object_id": alert.object_id},
                    )

        # 2. Kênh Email SMTP
        if email_config and email_config.get("enabled"):
            smtp_host = email_config.get("host")
            smtp_port = email_config.get("port", 587)
            sender = email_config.get("sender")
            pwd = email_config.get("password")
            recipient = email_config.get("recipient")

            if smtp_host and sender and recipient:
                html_body = f"""
                <div style="font-family: sans-serif; padding: 20px; background: #0B132B; color: #FFFFFF; border-radius: 8px;">
                    <h2 style="color: #F59E0B;">⚠️ Lost & Found AI - Cảnh báo đồ bỏ quên</h2>
                    <p><b>Mã cảnh báo:</b> {alert.alert_id}</p>
                    <p><b>Vật thể:</b> {vi_class} ({alert.object_id})</p>
                    <p><b>Vị trí:</b> Ghế {alert.seat_id}</p>
                    <p><b>Mức độ ưu tiên:</b> <span style="color: #EF4444;">{alert.priority}</span></p>
                    <p><b>Chi tiết:</b> {alert.reason}</p>
                    <p style="color: #94A3B8; font-size: 12px;">Hệ thống camera giám sát trần xe buýt Lost & Found AI.</p>
                </div>
                """
                ok, code, err = NotificationService.send_email(
                    smtp_host=smtp_host,
                    smtp_port=smtp_port,
                    sender=sender,
                    password=pwd,
                    recipient=recipient,
                    subject=f"[Lost & Found AI] Cảnh báo đồ bỏ quên trên ghế {alert.seat_id}",
                    body_html=html_body,
                    image_path=alert.snapshot_path,
                )
                if self._repo:
                    self._repo.log_notification(
                        alert_id=alert.alert_id,
                        channel=NotificationChannel.EMAIL,
                        recipient=recipient,
                        status="SUCCESS" if ok else "FAILED",
                        status_code=code,
                        error_message="" if ok else err,
                    )

        # 3. Kênh Webhook
        if webhook_config and webhook_config.get("enabled"):
            hook_url = webhook_config.get("url")
            if hook_url:
                payload = {
                    "event": "item_left_behind",
                    "alert_id": alert.alert_id,
                    "seat_id": alert.seat_id,
                    "object": alert.object_id,
                    "priority": alert.priority,
                    "reason": alert.reason,
                    "timestamp": alert.created_at,
                }
                ok, code, err = NotificationService.send_webhook(hook_url, payload)
                if self._repo:
                    self._repo.log_notification(
                        alert_id=alert.alert_id,
                        channel=NotificationChannel.WEBHOOK,
                        recipient=hook_url,
                        status="SUCCESS" if ok else "FAILED",
                        status_code=code,
                        error_message="" if ok else err,
                        payload=payload,
                    )

    def send_test_alert(
        self,
        telegram_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
        seat_id: str = "S01",
    ) -> Alert:
        """Gửi một cảnh báo thử nghiệm thực tế ra các kênh."""
        now_iso = get_current_time().isoformat()
        alert = Alert(
            alert_id=f"TEST_{uuid.uuid4().hex[:6].upper()}",
            event_id="EVT_TEST",
            seat_id=seat_id,
            object_id="O_TEST",
            priority=AlertPriority.HIGH,
            status=AlertStatus.PENDING,
            reason=f"Cảnh báo thử nghiệm hệ thống từ Notification Center trên ghế {seat_id}.",
            created_at=now_iso,
            sent_channels=[NotificationChannel.IN_APP],
        )
        if self._repo:
            self._repo.save_alert(alert)

        dummy_event = Event(
            event_id="EVT_TEST",
            class_name="backpack",
            seat_id=seat_id,
            status="suspected",
            vacant_duration=12.5,
        )
        self._dispatch_notifications(
            alert=alert,
            event=dummy_event,
            telegram_config=telegram_config,
            email_config=email_config,
            webhook_config=webhook_config,
            is_test=True,
        )
        return alert
