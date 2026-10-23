#!/usr/bin/env python3
"""
scripts/get_telegram_chat_id.py
Công cụ hỗ trợ đăng ký và lấy TELEGRAM_CHAT_ID an toàn qua Telegram Bot API.

Hướng dẫn sử dụng:
1. Đảm bảo bạn đã có biến TELEGRAM_BOT_TOKEN trong file .env
2. Mở Telegram và truy cập bot: https://t.me/DucLostFoundAIBot
3. Nhấn "Start" (hoặc gửi tin nhắn bất kỳ như "/start" cho bot)
4. Chạy script này:
   python scripts/get_telegram_chat_id.py
5. Lưu Chat ID nhận được vào file .env dưới dạng:
   TELEGRAM_CHAT_ID=<chat_id_của_bạn>
"""

import os
from pathlib import Path
import sys

# Đảm bảo đường dẫn root nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import requests


def get_chat_id():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    print("=" * 65)
    print("🤖 LOST & FOUND AI BUS — TELEGRAM CHAT ID SETUP HELPER")
    print("=" * 65)

    if not token or "placeholder" in token.lower() or token == "your_bot_token_here":
        print("⚠️  CHƯA CẤU HÌNH BOT TOKEN:")
        print("   Vui lòng mở file .env và điền TELEGRAM_BOT_TOKEN của bạn.")
        print("   Ví dụ: TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
        print("\n   Sau khi lưu file .env, vui lòng chạy lại script này.")
        print("=" * 65)
        sys.exit(1)

    # Hiển thị token dưới dạng che mờ bảo mật
    masked_token = f"{token[:4]}••••••••{token[-4:]}" if len(token) > 8 else "••••••••"
    print(f"🔑 Đã nạp Bot Token: {masked_token} (từ file .env)")
    print("📡 Đang truy vấn máy chủ Telegram Bot API...")

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except requests.exceptions.Timeout:
        print("❌ LỖI: Hết thời gian chờ kết nối tới api.telegram.org. Vui lòng kiểm tra lại mạng internet.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ LỖI KẾT NỐI MẠNG: {e}")
        sys.exit(1)

    if resp.status_code == 401:
        print("❌ LỖI: Bot Token không hợp lệ hoặc đã bị thu hồi bởi Telegram.")
        print("   Vui lòng kiểm tra lại giá trị TELEGRAM_BOT_TOKEN trong file .env.")
        sys.exit(1)

    if not data.get("ok"):
        print(f"❌ LỖI TỪ TELEGRAM: {data.get('description', 'Không xác định')}")
        sys.exit(1)

    results = data.get("result", [])
    if not results:
        print("\nℹ️  CHƯA NHẬN ĐƯỢC TIN NHẮN TỪ BẠN!")
        print("   Để Telegram cấp quyền cho Bot gửi tin nhắn, bạn cần kích hoạt Bot trước:")
        print("   1. Mở liên kết Bot: https://t.me/DucLostFoundAIBot")
        print("   2. Nhấn nút 'Start' (hoặc gửi '/start')")
        print("   3. Chạy lại script này: python scripts/get_telegram_chat_id.py")
        print("=" * 65)
        sys.exit(0)

    # Trích xuất các Chat ID duy nhất
    seen_chats = {}
    for update in results:
        msg = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
            or update.get("my_chat_member")
        )
        if msg and "chat" in msg:
            chat = msg["chat"]
            cid = str(chat.get("id"))
            c_type = chat.get("type", "private")
            c_title = chat.get("title") or chat.get("first_name", "") or chat.get("username", "Người dùng")
            username = chat.get("username", "")
            seen_chats[cid] = {
                "name": c_title,
                "username": f"@{username}" if username else "N/A",
                "type": c_type,
            }

    if not seen_chats:
        print("\nℹ️  Chưa tìm thấy tin nhắn trò chuyện phù hợp. Vui lòng gửi lại tin nhắn '/start' cho bot.")
        sys.exit(0)

    print("\n✅ TÌM THẤY THÔNG TIN CHAT ID THÀNH CÔNG:")
    print("-" * 65)
    for cid, info in seen_chats.items():
        print(f"👤 Tên:     {info['name']} ({info['username']})")
        print(f"📱 Loại:    {info['type']}")
        print(f"🆔 CHAT ID: {cid}")
        print("-" * 65)

    # Lấy chat ID đầu tiên
    first_cid = list(seen_chats.keys())[-1]
    print(f"\n👉 CÁCH CẤU HÌNH TIẾP THEO:")
    print(f"   Thêm hoặc cập nhật dòng sau trong file .env:")
    print(f"   TELEGRAM_CHAT_ID={first_cid}")
    print(f"\n   Sau đó khởi động lại ứng dụng Streamlit (streamlit run app.py)")
    print("=" * 65)


if __name__ == "__main__":
    get_chat_id()
