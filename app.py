"""
app.py
Lost & Found AI - Hệ thống AI giám sát trần xe buýt phát hiện đồ bỏ quên
Kiến trúc: Occlusion-Aware, Fixed Seat Entities, Multi-Channel Alerts,
Dataset Annotation & YOLO Export, Model Versioning & Deployment.
"""

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc của repository luôn nằm đầu sys.path trên Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
import json
import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.services.analysis_service import AnalysisService
from src.tracking.object_tracker import ObjectTracker
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.seats.seat_config import SeatConfigManager
from src.models import SeatConfig, SeatState
from src.utils.file_utils import format_file_size, format_duration
from src.video.camera_service import CameraSourceType, ensure_demo_video_exists
from src.video.video_writer import VideoWriter
from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_STATIONARY_TIME_SEC,
    STATIONARY_MOVEMENT_THRESHOLD,
    ITEM_CLASSES,
    CLASS_NAMES_VI,
    SUPPORTED_VIDEO_EXTENSIONS,
    EventStatus,
    EVENT_STATUS_META,
    SystemMode,
    COLOR_SEAT_HEX,
    COLOR_PERSON_HEX,
    COLOR_OBJECT_HEX,
    COLOR_ALERT_HEX,
)

# ──────────────────────────────────────────────
# Cấu hình giao diện Streamlit
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lost & Found AI - Bus Surveillance",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Giao diện Dark Navy chuyên nghiệp & Hiện đại
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Dark Navy Theme Variables */
    :root {{
        --bg-main: #0B132B;
        --bg-panel: #1C2541;
        --bg-card: #141E33;
        --border-color: #3A506B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --color-seat: {COLOR_SEAT_HEX};
        --color-person: {COLOR_PERSON_HEX};
        --color-object: {COLOR_OBJECT_HEX};
        --color-alert: {COLOR_ALERT_HEX};
    }}

    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }}

    /* Panel container */
    .metric-panel {{
        background-color: #1C2541;
        border: 1px solid #3A506B;
        border-radius: 8px;
        padding: 14px 18px;
        color: #F8FAFC;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .metric-panel h3 {{
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .metric-panel p {{
        margin: 2px 0 0 0;
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
    }}

    /* Top Navigation Header */
    .header-bar {{
        background: #1C2541;
        border: 1px solid #3A506B;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .header-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .header-subtitle {{
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 2px;
    }}

    /* Status Badges */
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .badge-live {{ background-color: #065F46; color: #34D399; border: 1px solid #059669; }}
    .badge-processing {{ background-color: #1E3A8A; color: #60A5FA; border: 1px solid #2563EB; }}
    .badge-demo {{ background-color: #78350F; color: #FBBF24; border: 1px solid #D97706; }}
    .badge-alert {{ background-color: #7F1D1D; color: #F87171; border: 1px solid #DC2626; }}

    /* Legend Bar */
    .legend-box {{
        background: #141E33;
        border: 1px solid #3A506B;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 12px;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .legend-color {{
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }}

    /* Audio Chime Script Hook */
    .audio-btn {{ display: none; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service() -> AnalysisService:
    """Khởi tạo AnalysisService một lần duy nhất."""
    service = AnalysisService()
    service.initialize()
    return service


def render_header(title: str, subtitle: str, mode: str = SystemMode.DEMO, alert_count: int = 0):
    """Render thanh header chuẩn hóa trên đầu mỗi trang."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div class="header-title">🚌 {title}</div>
            <div class="header-subtitle">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        now_str = datetime.now().strftime("%H:%M:%S")
        if mode == SystemMode.LIVE:
            mode_badge = '<span class="badge badge-live">● LIVE STREAM</span>'
        elif mode == SystemMode.PROCESSING:
            mode_badge = '<span class="badge badge-processing">● PROCESSING</span>'
        else:
            mode_badge = '<span class="badge badge-demo">● DEMO MODE</span>'

        alert_badge = (
            f'<span class="badge badge-alert" style="margin-left: 6px;">⚠️ {alert_count} CẢNH BÁO</span>'
            if alert_count > 0 else ""
        )

        st.markdown(f"""
        <div style="text-align: right; padding-top: 4px;">
            {mode_badge}{alert_badge}
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">⏱️ {now_str}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 4px 0 16px 0; border-color: #3A506B;'>", unsafe_allow_html=True)


def play_audio_alert():
    """Phát âm thanh cảnh báo ngắn (Web Audio API) trực tiếp trên trình duyệt."""
    st.components.v1.html("""
    <script>
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.25, ctx.currentTime);
            osc.start();
            osc.stop(ctx.currentTime + 0.3);
        } catch(e) { console.log(e); }
    </script>
    """, height=0, width=0)


def render_legend():
    """Hiển thị bảng màu chuẩn hóa cho 3 nhóm đối tượng."""
    st.markdown(f"""
    <div class="legend-box">
        <span style="font-weight: 600; color: #94A3B8;">CHÚ GIẢI MÀU:</span>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLOR_SEAT_HEX};"></div>
            <span><b>GHẾ (Seats):</b> S01, S02... (Cyan / Blue)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLOR_PERSON_HEX};"></div>
            <span><b>NGƯỜI (Persons):</b> P01, P12... (Green)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLOR_OBJECT_HEX};"></div>
            <span><b>ĐỒ VẬT (Objects):</b> O01, O31... (Amber)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLOR_ALERT_HEX};"></div>
            <span><b>CẢNH BÁO BỎ QUÊN:</b> ⚠ Nghi ngờ (Red)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN APP ENTRY POINT
# ══════════════════════════════════════════════
def main():
    service = get_service()

    # ── Sidebar Navigation ──
    with st.sidebar:
        st.markdown("### 🚌 LOST & FOUND AI")
        st.caption("Giám sát xe buýt • Nhận biết che khuất • Báo cáo tức thì")
        st.markdown("---")

        page = st.radio(
            "CHỨC NĂNG HỆ THỐNG",
            [
                "🏠 Tổng quan",
                "🎥 Giám sát Camera",
                "⚠️ Trung tâm Cảnh báo",
                "🪑 Quản lý Ghế",
                "🏷️ AI Training & Dataset",
                "🤖 Quản lý Model",
                "🔔 Cấu hình Thông báo",
                "📊 Lịch sử Phân tích",
                "⚙️ Cài đặt & Bảo mật",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        # System Health & Status Indicators
        model_info = service.get_model_info()
        if model_info.get("loaded"):
            st.success(f"🤖 Model: `{model_info['model_name']}`")
        else:
            st.error("❌ Model AI chưa tải")

        seat_count = len(service.get_seats())
        if seat_count > 0:
            st.info(f"🪑 Sơ đồ xe: **{seat_count} ghế**")
        else:
            st.warning("⚠️ Chưa cấu hình ghế")

        unresolved_alerts = service.get_unresolved_alert_count()
        if unresolved_alerts > 0:
            st.warning(f"🔔 **{unresolved_alerts}** cảnh báo chưa xử lý")

    # ── Routing Pages ──
    if "Tổng quan" in page:
        page_dashboard(service)
    elif "Giám sát Camera" in page:
        page_camera_monitor(service)
    elif "Trung tâm Cảnh báo" in page:
        page_alert_center(service)
    elif "Quản lý Ghế" in page:
        page_seats(service)
    elif "AI Training" in page:
        page_dataset(service)
    elif "Quản lý Model" in page:
        page_model_management(service)
    elif "Thông báo" in page:
        page_notifications(service)
    elif "Lịch sử" in page:
        page_history(service)
    elif "Cài đặt" in page:
        page_settings(service)


# ══════════════════════════════════════════════
# TRANG 1: TỔNG QUAN (DASHBOARD)
# ══════════════════════════════════════════════
def page_dashboard(service: AnalysisService):
    alert_count = service.get_unresolved_alert_count()
    render_header(
        title="Tổng quan Hệ thống Giám sát Xe buýt",
        subtitle="Trung tâm điều khiển AI phát hiện đồ bỏ quên & Theo dõi che khuất",
        mode=SystemMode.DEMO,
        alert_count=alert_count,
    )

    stats = service.get_dashboard_stats()
    seat_count = len(service.get_seats())

    # 4 Metric Panels
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-panel">
            <p>🪑 VÙNG GHẾ ĐÃ ĐỊNH DANH</p>
            <h3>{seat_count} <span style="font-size: 0.9rem; color: #94A3B8;">ghế</span></h3>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-panel">
            <p>⚠️ CẢNH BÁO CHỜ XỬ LÝ</p>
            <h3 style="color: {'#EF4444' if alert_count > 0 else '#10B981'};">{alert_count}</h3>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-panel">
            <p>👁️ SỰ KIỆN GHI NHẬN</p>
            <h3>{stats['total_events']}</h3>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-panel">
            <p>🎬 PHIÊN PHÂN TÍCH</p>
            <h3>{stats['total_analyses']}</h3>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### ⚡ Thao tác nhanh")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("🎥 Chạy Giám sát Camera", use_container_width=True):
                st.session_state["nav_to"] = "Giám sát Camera"
                st.rerun()
        with sc2:
            if st.button("⚠️ Mở Alert Center", use_container_width=True):
                st.session_state["nav_to"] = "Trung tâm Cảnh báo"
                st.rerun()
        with sc3:
            if st.button("🪑 Sơ đồ Ghế (ROI)", use_container_width=True):
                st.session_state["nav_to"] = "Quản lý Ghế"
                st.rerun()

        st.markdown("#### 🔔 Cảnh báo Mới nhất")
        recent_alerts = service.get_recent_alerts(limit=5)
        if recent_alerts:
            for alt in recent_alerts:
                p_color = "#EF4444" if alt["priority"] == "HIGH" else "#F59E0B"
                st.markdown(f"""
                <div style="background: #141E33; border-left: 4px solid {p_color}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; color: #F8FAFC;">{alt['alert_id']} • Ghế {alt['seat_id']} ({alt['object_id']})</span>
                        <span style="font-size: 0.8rem; color: #94A3B8;">{alt['created_at'][:19].replace('T', ' ')}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">{alt['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ Hiện tại không có cảnh báo bỏ quên nào chưa xử lý.")

    with col_right:
        st.markdown("#### 🪑 Trạng thái Ghế thời gian thực")
        if seat_count > 0:
            statuses = service.seat_manager.get_all_statuses()
            for s in service.get_seats():
                st_info = statuses.get(s.seat_id)
                st_state = st_info.state if st_info else SeatState.UNKNOWN
                if st_state == SeatState.OCCUPIED:
                    s_badge = '<span class="badge" style="background: #1E3A8A; color: #60A5FA;">CÓ NGƯỜI</span>'
                elif st_state == SeatState.VACANT:
                    s_badge = '<span class="badge" style="background: #065F46; color: #34D399;">TRỐNG</span>'
                else:
                    s_badge = '<span class="badge" style="background: #374151; color: #9CA3AF;">CHƯA RÕ</span>'

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: #141E33; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border: 1px solid #3A506B;">
                    <span><b>{s.seat_id}</b> {f'({s.description})' if s.description else ''}</span>
                    {s_badge}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Chưa thiết lập vùng ghế.")


# ══════════════════════════════════════════════
# TRANG 2: GIÁM SÁT CAMERA & VIDEO (CAMERA MONITOR)
# ══════════════════════════════════════════════
def page_camera_monitor(service: AnalysisService):
    render_header(
        title="Giám sát Camera & Phân tích Video Trực tiếp",
        subtitle="Hỗ trợ Video tải lên, Demo xe buýt mẫu, Live Webcam máy tính và RTSP",
        mode=SystemMode.LIVE,
        alert_count=service.get_unresolved_alert_count(),
    )

    render_legend()

    # Nguồn dữ liệu & Tùy chọn điều khiển
    st.markdown("##### ⚙️ Nguồn dữ liệu & Cấu hình luồng")
    src_col1, src_col2, src_col3 = st.columns([2, 1, 1])

    with src_col1:
        source_mode = st.radio(
            "Chọn nguồn video:",
            [
                "🚌 Video Mẫu Xe Buýt (Demo Mode - Khuyên dùng trên Cloud)",
                "📁 Tải lên Video từ máy tính",
                "🎥 Live Webcam máy tính (FaceTime HD / USB)",
                "📸 Chụp quét từ Webcam trình duyệt",
                "🌐 Luồng IP / RTSP Camera",
            ],
            horizontal=False,
        )

    with src_col2:
        conf_thresh = st.slider("Ngưỡng tin cậy (Confidence)", 0.15, 0.90, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
        min_stat_time = st.number_input("Thời gian nghi ngờ bỏ quên (giây)", 2.0, 60.0, 4.0, 1.0)

    with src_col3:
        st.markdown("**Bộ lọc lớp hiển thị:**")
        show_seats = st.checkbox("🪑 Vùng ghế (Seats)", value=True)
        show_persons = st.checkbox("👤 Người (Persons)", value=True)
        show_objects = st.checkbox("🎒 Đồ vật (Objects)", value=True)
        privacy_blur = st.checkbox("🛡️ Làm mờ mặt (Privacy Blur)", value=False)

    st.markdown("---")

    # Xử lý theo từng nguồn
    if "Demo Mode" in source_mode:
        st.info("💡 **Chế độ Demo mẫu:** Hệ thống sử dụng video xe buýt mô phỏng chuẩn (người ngồi, đặt ba lô, đứng dậy rời đi, để lại ba lô trên ghế trống).")
        demo_path = ensure_demo_video_exists()

        dcol1, dcol2 = st.columns([3, 1])
        with dcol2:
            st.markdown("##### Điều khiển Demo")
            start_demo = st.button("▶️ Chạy Phân tích Demo Mẫu", type="primary", use_container_width=True)
            auto_alert = st.checkbox("Kích hoạt âm thanh cảnh báo", value=True)

        with dcol1:
            frame_view = st.empty()
            metric_view = st.empty()

        if start_demo:
            cap = cv2.VideoCapture(demo_path)
            tracker = ObjectTracker()
            event_manager = EventManager()
            event_detector = ForgottenItemDetector(
                min_stationary_time=min_stat_time,
                movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
                watched_classes=ITEM_CLASSES,
                seat_manager=service.seat_manager,
                event_manager=event_manager,
            )

            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            alert_fired = False

            while cap.isOpened():
                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                annotated, events, summary = service.process_webcam_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    fps=fps,
                    tracker=tracker,
                    event_detector=event_detector,
                    event_manager=event_manager,
                    confidence_threshold=conf_thresh,
                    show_seats=show_seats,
                    show_persons=show_persons,
                    show_objects=show_objects,
                    show_track_ids=True,
                    show_confidence=True,
                    show_event_markers=True,
                    privacy_blur_faces=privacy_blur,
                )

                latency_ms = (time.time() - t0) * 1000.0
                calc_fps = 1000.0 / max(1.0, latency_ms)

                rgb_f = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_view.image(rgb_f, channels="RGB", use_container_width=True)

                metric_view.markdown(
                    f"**⏱️ Frame:** `{frame_idx}` | "
                    f"**FPS Suy luận:** `{calc_fps:.1f}` | "
                    f"**Độ trễ:** `{latency_ms:.1f}ms` | "
                    f"**Đang theo dõi:** `{len(events)}` | "
                    f"**⚠️ Nghi ngờ:** `{summary.get('suspected', 0)}` | "
                    f"**🛡️ Che khuất:** `{summary.get('occluded', 0)}`"
                )

                if summary.get("suspected", 0) > 0 and not alert_fired and auto_alert:
                    play_audio_alert()
                    alert_fired = True

                frame_idx += 1
                time.sleep(0.03)

            cap.release()
            st.success("✅ Phiên phân tích Demo mẫu hoàn tất!")

    elif "Tải lên Video" in source_mode:
        uploaded = st.file_uploader("Chọn file video MP4, AVI, MOV:", type=["mp4", "avi", "mov"])
        if uploaded:
            video_path = service.save_uploaded_video(uploaded)
            st.success(f"Đã tải lên: {uploaded.name}")
            if st.button("🚀 Bắt đầu Phân tích Video này", type="primary"):
                with st.spinner("AI đang phân tích video từng frame..."):
                    result = service.analyze_video(video_path, confidence_threshold=conf_thresh)
                    if result.success:
                        st.success(f"✅ Hoàn tất! Phát hiện {len(result.events)} sự kiện.")
                        if result.output_video_path and Path(result.output_video_path).exists():
                            st.video(result.output_video_path)
                    else:
                        st.error(f"Lỗi: {result.error_message}")

    elif "Live Webcam máy tính" in source_mode:
        wcol1, wcol2 = st.columns([2, 1])
        with wcol1:
            cam_idx = st.number_input("Cổng Camera (0 = FaceTime HD / Mặc định, 1, 2 = USB)", 0, 5, 0)
            run_cam = st.toggle("🔴 BẬT / TẮT CAMERA THỜI GIAN THỰC", value=False)
        with wcol2:
            st.caption("⚡ Trên macOS: Đảm bảo cấp quyền Camera cho Terminal hoặc Trình duyệt.")

        if run_cam:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_AVFOUNDATION)

            if not cap.isOpened():
                st.error(f"❌ Không thể mở camera cổng {cam_idx}. Hãy kiểm tra kết nối phần cứng.")
            else:
                cam_view = st.empty()
                cam_metrics = st.empty()
                tracker = ObjectTracker()
                event_manager = EventManager()
                event_detector = ForgottenItemDetector(
                    min_stationary_time=min_stat_time,
                    movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
                    watched_classes=ITEM_CLASSES,
                    seat_manager=service.seat_manager,
                    event_manager=event_manager,
                )

                frame_idx = 0
                while run_cam:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        st.warning("Mất tín hiệu camera.")
                        break

                    annotated, events, summary = service.process_webcam_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        fps=15.0,
                        tracker=tracker,
                        event_detector=event_detector,
                        event_manager=event_manager,
                        confidence_threshold=conf_thresh,
                        show_seats=show_seats,
                        show_persons=show_persons,
                        show_objects=show_objects,
                        privacy_blur_faces=privacy_blur,
                    )

                    lat_ms = (time.time() - t0) * 1000.0
                    rgb_f = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cam_view.image(rgb_f, channels="RGB", use_container_width=True)

                    cam_metrics.markdown(
                        f"**⏱️ Frame:** `{frame_idx:,}` | "
                        f"**Độ trễ:** `{lat_ms:.1f}ms` | "
                        f"**FPS:** `{1000.0/max(1.0, lat_ms):.1f}` | "
                        f"**Đang theo dõi:** `{len(events)}` | "
                        f"**⚠️ Nghi ngờ:** `{summary.get('suspected', 0)}` | "
                        f"**🛡️ Che khuất:** `{summary.get('occluded', 0)}`"
                    )
                    frame_idx += 1

                cap.release()

    elif "Chụp quét từ Webcam trình duyệt" in source_mode:
        st.markdown("##### 📸 Chụp ảnh từ Camera MacBook / Trình duyệt:")
        photo = st.camera_input("Chụp ảnh phân tích tức thì")
        if photo:
            np_arr = np.frombuffer(photo.getvalue(), np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            tracker = ObjectTracker()
            event_manager = EventManager()
            event_detector = ForgottenItemDetector(
                min_stationary_time=min_stat_time,
                movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
                watched_classes=ITEM_CLASSES,
                seat_manager=service.seat_manager,
                event_manager=event_manager,
            )

            with st.spinner("AI đang phân tích ảnh..."):
                annotated, events, summary = service.process_webcam_frame(
                    frame=cv_img,
                    frame_idx=1,
                    fps=1.0,
                    tracker=tracker,
                    event_detector=event_detector,
                    event_manager=event_manager,
                    confidence_threshold=conf_thresh,
                    show_seats=show_seats,
                    show_persons=show_persons,
                    show_objects=show_objects,
                    privacy_blur_faces=privacy_blur,
                )

            rgb_res = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb_res, caption="Kết quả phân tích", channels="RGB", use_container_width=True)
            st.success(f"Phân tích hoàn tất! Số đối tượng: {len(events)} | Nghi ngờ: {summary.get('suspected', 0)}")

    elif "Luồng IP / RTSP" in source_mode:
        rtsp_url = st.text_input("Nhập URL luồng RTSP (ví dụ: rtsp://admin:123456@192.168.1.100:554/stream1):")
        if st.button("Kết nối luồng RTSP"):
            if not rtsp_url:
                st.warning("Vui lòng nhập URL hợp lệ.")
            else:
                ok, msg = service.camera_manager.start_stream(CameraSourceType.RTSP, rtsp_url)
                if ok:
                    st.success("Đã kết nối RTSP thành công!")
                else:
                    st.error(f"Lỗi kết nối RTSP: {msg}")


# ══════════════════════════════════════════════
# TRANG 3: TRUNG TÂM CẢNH BÁO (ALERT CENTER)
# ══════════════════════════════════════════════
def page_alert_center(service: AnalysisService):
    render_header(
        title="Trung tâm Cảnh báo Đồ thất lạc (Alert Center)",
        subtitle="Quản lý các cảnh báo thời gian thực, xem ảnh bằng chứng và xác minh",
        mode=SystemMode.LIVE,
        alert_count=service.get_unresolved_alert_count(),
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn2:
        if st.button("🔊 Phát thử chuông cảnh báo", use_container_width=True):
            play_audio_alert()
            st.success("Đã phát chuông audio!")

    # Bộ lọc trạng thái Alert
    filter_status = st.selectbox(
        "Lọc theo trạng thái:",
        ["Tất cả", "PENDING (Chờ xử lý)", "CONFIRMED (Đã xác nhận)", "REVIEW_REQUIRED (Cần duyệt)", "DISMISSED (Đã loại bỏ)"],
    )

    status_query = None
    if "PENDING" in filter_status:
        status_query = "PENDING"
    elif "CONFIRMED" in filter_status:
        status_query = "CONFIRMED"
    elif "REVIEW_REQUIRED" in filter_status:
        status_query = "REVIEW_REQUIRED"
    elif "DISMISSED" in filter_status:
        status_query = "DISMISSED"

    alerts = service.get_recent_alerts(limit=50, status=status_query)

    if not alerts:
        st.info("Hiện không có cảnh báo nào phù hợp với bộ lọc.")
    else:
        for alt in alerts:
            with st.container():
                p_badge = "🔴 KHẨN CẤP" if alt["priority"] == "HIGH" else "🟡 TRUNG BÌNH"
                st.markdown(f"""
                <div style="background: #1C2541; border: 1px solid #3A506B; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">
                            {alt['alert_id']} — Ghế {alt['seat_id']} ({alt['object_id']})
                        </span>
                        <span>
                            <span class="badge" style="background: #1E293B; color: #E2E8F0;">{p_badge}</span>
                            <span class="badge" style="background: #0F172A; color: #94A3B8; margin-left: 6px;">{alt['status']}</span>
                        </span>
                    </div>
                    <p style="color: #CBD5E1; margin: 8px 0 6px 0; font-size: 0.9rem;"><b>Lý do:</b> {alt['reason']}</p>
                    <div style="font-size: 0.8rem; color: #94A3B8;">⏱️ Phát hiện lúc: {alt['created_at'][:19].replace('T', ' ')}</div>
                </div>
                """, unsafe_allow_html=True)

                act1, act2, act3 = st.columns([1, 1, 3])
                with act1:
                    if alt["status"] != "CONFIRMED" and st.button("✅ Xác nhận đúng", key=f"conf_{alt['alert_id']}"):
                        service.update_alert_status(alt["alert_id"], "CONFIRMED")
                        st.rerun()
                with act2:
                    if alt["status"] != "DISMISSED" and st.button("❌ Loại bỏ (Báo sai)", key=f"dism_{alt['alert_id']}"):
                        service.update_alert_status(alt["alert_id"], "DISMISSED")
                        st.rerun()


# ══════════════════════════════════════════════
# TRANG 4: QUẢN LÝ GHẾ (SEAT CALIBRATION & GRID)
# ══════════════════════════════════════════════
def page_seats(service: AnalysisService):
    render_header(
        title="Quản lý Sơ đồ Ghế xe buýt (Fixed Seat Entities)",
        subtitle="Định danh cố định từng ghế (S01, S02...), hiệu chuẩn polygon và sinh sơ đồ tự động",
        mode=SystemMode.DEMO,
    )

    tab_grid, tab_list = st.tabs(["⚡ Sinh sơ đồ ghế tự động", "📋 Danh sách & Tọa độ Polygon"])

    with tab_grid:
        st.markdown("##### 🚌 Trình sinh nhanh sơ đồ xe buýt (Bus Grid Generator)")
        st.caption("Ghế xe buýt là các thực thể cố định, giữ ID vĩnh viễn và không phụ thuộc vào vùng chuyển động.")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            rows = st.number_input("Số hàng ghế", 1, 12, 4, 1)
        with gcol2:
            left_cols = st.number_input("Số ghế bên trái lối đi", 1, 3, 2, 1)
        with gcol3:
            right_cols = st.number_input("Số ghế bên phải lối đi", 1, 3, 2, 1)

        name_style = st.selectbox("Định dạng tên ghế:", ["S01, S02, S03...", "Ghế 1A, 1B, 2A..."])

        total_seats = rows * (left_cols + right_cols)
        st.info(f"Sơ đồ sẽ tạo tổng cộng **{total_seats} ghế cố định**.")

        if st.button("💾 Tạo và Lưu Sơ đồ Ghế mới", type="primary"):
            from src.seats.seat_manager import SeatManager
            style_code = "numeric" if "S01" in name_style else "alphanumeric"
            new_grid = SeatManager.generate_bus_grid(
                rows=rows,
                left_cols=left_cols,
                right_cols=right_cols,
                naming_style=style_code,
            )
            service.save_seats(new_grid)
            st.success(f"✅ Đã tạo thành công {len(new_grid)} ghế cố định!")
            st.rerun()

    with tab_list:
        seats = service.get_seats()
        if not seats:
            st.warning("Chưa có cấu hình vùng ghế nào.")
        else:
            seat_data = []
            for s in seats:
                seat_data.append({
                    "Seat ID": s.seat_id,
                    "Mô tả": s.description,
                    "Chuẩn hóa [0-1]": s.is_normalized,
                    "Số đỉnh polygon": len(s.polygon),
                })
            st.dataframe(pd.DataFrame(seat_data), use_container_width=True)


# ══════════════════════════════════════════════
# TRANG 5: AI TRAINING & DATASET
# ══════════════════════════════════════════════
def page_dataset(service: AnalysisService):
    render_header(
        title="Dữ liệu Huấn luyện & Gán nhãn AI (Dataset Manager)",
        subtitle="Quản lý tập ảnh, trích xuất frame từ video, gán nhãn bbox và xuất chuẩn YOLOv8",
        mode=SystemMode.DEMO,
    )

    dstats = service.get_dataset_stats()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Tổng số ảnh", dstats.get("total_images", 0))
    with m2:
        st.metric("Đã gán nhãn", dstats.get("labeled_images", 0))
    with m3:
        st.metric("Tập Train", dstats.get("splits", {}).get("train", 0))
    with m4:
        st.metric("Tập Val / Test", f"{dstats.get('splits', {}).get('val', 0)} / {dstats.get('splits', {}).get('test', 0)}")

    tab_upload, tab_extract, tab_annotate, tab_export = st.tabs([
        "📤 Tải lên ảnh mới", "🎞️ Trích xuất frame từ Video", "🏷️ Gán nhãn Bounding Box", "📦 Xuất chuẩn YOLO (.zip)"
    ])

    with tab_upload:
        st.markdown("##### Tải lên ảnh đơn hoặc nhiều ảnh xe buýt:")
        up_files = st.file_uploader("Chọn file ảnh (JPG, PNG):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        split_choice = st.selectbox("Chọn tập dữ liệu gán cho ảnh này:", ["train", "val", "test"])
        if up_files and st.button("Lưu ảnh vào Dataset"):
            count = 0
            for f in up_files:
                res = service.add_dataset_image(f.getvalue(), f.name, split=split_choice)
                if res:
                    count += 1
            st.success(f"Đã lưu thành công {count} ảnh vào tập {split_choice}!")
            st.rerun()

    with tab_extract:
        st.markdown("##### Trích xuất frame từ video giám sát trần xe:")
        vid_file = st.file_uploader("Tải video lên để trích xuất frame:", type=["mp4", "avi", "mov"], key="vid_extract")
        interval = st.slider("Khoảng cách trích xuất (giây/frame):", 1.0, 5.0, 2.0, 0.5)
        if vid_file and st.button("Bắt đầu Trích xuất Frame"):
            temp_path = service.save_uploaded_video(vid_file)
            ids = service.extract_frames_from_video(temp_path, interval_seconds=interval)
            st.success(f"Đã trích xuất thành công {len(ids)} frame và thêm vào dataset!")
            st.rerun()

    with tab_annotate:
        st.markdown("##### Gán nhãn Bounding Box:")
        imgs = service.get_dataset_images()
        if not imgs:
            st.info("Chưa có ảnh nào trong dataset để gán nhãn.")
        else:
            img_options = {f"{img['image_id']} - {img['filename']}": img for img in imgs}
            chosen = st.selectbox("Chọn ảnh để xem/gán nhãn:", list(img_options.keys()))
            cur_img = img_options[chosen]

            p = Path(cur_img["image_path"])
            if p.exists():
                st.image(str(p), caption=f"ID: {cur_img['image_id']} | Kích thước: {cur_img['width']}x{cur_img['height']}", use_container_width=True)

                st.markdown("###### Thêm nhãn mới:")
                ac1, ac2, ac3, ac4 = st.columns(4)
                with ac1:
                    cls_name = st.selectbox("Lớp đối tượng:", ["person", "backpack", "handbag", "umbrella", "bottle", "cell phone", "laptop", "book", "wallet", "unknown_object"])
                with ac2:
                    bx1 = st.number_input("Tọa độ X1 (px)", 0, cur_img["width"], 50)
                    by1 = st.number_input("Tọa độ Y1 (px)", 0, cur_img["height"], 50)
                with ac3:
                    bx2 = st.number_input("Tọa độ X2 (px)", 0, cur_img["width"], 150)
                    by2 = st.number_input("Tọa độ Y2 (px)", 0, cur_img["height"], 150)
                with ac4:
                    is_occ = st.checkbox("Bị che khuất một phần")
                    if st.button("Lưu nhãn"):
                        service.save_image_annotations(cur_img["image_id"], [{
                            "class_name": cls_name,
                            "bbox_x1": bx1, "bbox_y1": by1,
                            "bbox_x2": bx2, "bbox_y2": by2,
                            "is_occluded": is_occ,
                        }])
                        st.success("Đã lưu nhãn thành công!")

    with tab_export:
        st.markdown("##### Xuất Dataset chuẩn YOLOv8")
        st.caption("Cấu trúc xuất ra gồm file `data.yaml`, thư mục `images/` và `labels/` tương ứng.")
        if st.button("📦 Nén và Xuất Dataset YOLO (.zip)", type="primary"):
            ok, zip_p, msg = service.export_yolo_dataset()
            if ok and Path(zip_p).exists():
                st.success(msg)
                with open(zip_p, "rb") as zf:
                    st.download_button(
                        "⬇️ Tải file Dataset ZIP về máy",
                        data=zf.read(),
                        file_name=Path(zip_p).name,
                        mime="application/zip",
                        use_container_width=True,
                    )
            else:
                st.error(msg)


# ══════════════════════════════════════════════
# TRANG 6: QUẢN LÝ MODEL (MODEL VERSIONING)
# ══════════════════════════════════════════════
def page_model_management(service: AnalysisService):
    render_header(
        title="Quản lý Phiên bản Model AI (Model Lifecycle)",
        subtitle="Theo dõi chỉ số mAP50, Precision, Recall, phê duyệt Candidate và Rollback",
        mode=SystemMode.DEMO,
    )

    models = service.get_all_models()
    active_m = service.get_active_model()

    if active_m:
        st.markdown(f"""
        <div style="background: #1C2541; border-left: 4px solid #10B981; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">
                ⭐ MODEL ĐANG HOẠT ĐỘNG: {active_m['model_id']} ({active_m['version']})
            </div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                Tập dữ liệu: {active_m['dataset_version']} | mAP50: <b>{active_m.get('metrics', {}).get('map50', 'N/A')}</b> |
                Precision: <b>{active_m.get('metrics', {}).get('precision', 'N/A')}</b> |
                Recall: <b>{active_m.get('metrics', {}).get('recall', 'N/A')}</b>
            </div>
            <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 4px;">{active_m.get('notes', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    tab_list, tab_train = st.tabs(["📋 Danh sách các Model", "⚙️ Huấn luyện & Đăng ký Candidate Model"])

    with tab_list:
        if not models:
            st.info("Chưa có model nào được ghi nhận.")
        else:
            for m in models:
                is_active = (m["status"] == "active")
                s_color = "#10B981" if is_active else "#64748B"
                st.markdown(f"""
                <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 700; color: #F8FAFC;">{m['model_id']}</span>
                        <span style="color: #94A3B8; font-size: 0.85rem; margin-left: 8px;">({m['dataset_version']})</span>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                            mAP50: {m.get('metrics', {}).get('map50', 'N/A')} • Precision: {m.get('metrics', {}).get('precision', 'N/A')} • Recall: {m.get('metrics', {}).get('recall', 'N/A')}
                        </div>
                    </div>
                    <div>
                        <span class="badge" style="background: {s_color}22; color: {s_color}; border: 1px solid {s_color};">{m['status'].upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if not is_active:
                    if st.button(f"Phê duyệt & Kích hoạt Model này", key=f"act_{m['model_id']}"):
                        service.activate_model(m["model_id"])
                        st.rerun()

    with tab_train:
        st.markdown("##### 🚀 Huấn luyện Model mới (Training Job)")
        st.caption("Sinh lệnh huấn luyện cho GPU server hoặc chạy giả lập đăng ký Candidate Model.")

        tc1, tc2 = st.columns(2)
        with tc1:
            new_ver = st.text_input("Tên phiên bản mới:", value="v1.1-bus-custom")
            dver = st.text_input("Phiên bản Dataset:", value="bus-dataset-v1.0")
        with tc2:
            epochs = st.slider("Số Epochs:", 10, 100, 30, 5)
            batch = st.selectbox("Batch size:", [8, 16, 32], index=1)

        cmd = service.model_service.generate_training_command("data/dataset/exports/data.yaml", epochs=epochs, batch_size=batch)
        st.code(cmd, language="bash")

        if st.button("Bắt đầu Huấn luyện Candidate Model"):
            with st.spinner("Đang chạy validation và đăng ký candidate model..."):
                ok, met, msg = service.simulate_training_job(new_ver, dver, epochs=epochs)
                if ok:
                    st.success(f"✅ {msg}")
                    st.json(met)
                    st.rerun()
                else:
                    st.error(msg)


# ══════════════════════════════════════════════
# TRANG 7: CẤU HÌNH THÔNG BÁO (NOTIFICATIONS)
# ══════════════════════════════════════════════
def page_notifications(service: AnalysisService):
    render_header(
        title="Cấu hình Thông báo & Cảnh báo (Notification Center)",
        subtitle="Kết nối Telegram Bot, Email SMTP, Webhook; kiểm tra kết nối và gửi thử nghiệm",
        mode=SystemMode.DEMO,
    )

    tab_tele, tab_email, tab_webhook, tab_logs = st.tabs([
        "📱 Telegram Bot", "📧 Email SMTP", "🌐 Webhook", "📜 Lịch sử gửi thông báo"
    ])

    with tab_tele:
        st.markdown("##### Cấu hình Telegram Bot:")
        t_token = st.text_input("Telegram Bot Token:", value=os.environ.get("TELEGRAM_BOT_TOKEN", ""), type="password")
        t_chat_id = st.text_input("Telegram Chat ID:", value=os.environ.get("TELEGRAM_CHAT_ID", ""))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔍 Kiểm tra Kết nối Telegram"):
                from src.services.notification_service import NotificationService
                ok, msg = NotificationService.test_telegram_connection(t_token, t_chat_id)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        with b2:
            if st.button("🚀 Bắn thử Cảnh báo qua Telegram"):
                alt = service.send_test_alert(telegram_config={"enabled": True, "bot_token": t_token, "chat_id": t_chat_id})
                st.success(f"Đã phát lệnh gửi cảnh báo {alt.alert_id}! Kiểm tra ứng dụng Telegram.")

    with tab_email:
        st.markdown("##### Cấu hình Email SMTP:")
        e_host = st.text_input("SMTP Host:", value=os.environ.get("SMTP_HOST", "smtp.gmail.com"))
        e_port = st.number_input("SMTP Port:", value=int(os.environ.get("SMTP_PORT", 587)))
        e_sender = st.text_input("Email gửi:", value=os.environ.get("SMTP_SENDER", ""))
        e_pwd = st.text_input("Mật khẩu ứng dụng (App Password):", value=os.environ.get("SMTP_PASSWORD", ""), type="password")
        e_recip = st.text_input("Email người nhận cảnh báo:", value=os.environ.get("SMTP_RECIPIENT", ""))

        if st.button("🔍 Kiểm tra Kết nối SMTP"):
            from src.services.notification_service import NotificationService
            ok, msg = NotificationService.test_email_connection(e_host, e_port, e_sender, e_pwd)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    with tab_webhook:
        st.markdown("##### Cấu hình Webhook HTTP:")
        w_url = st.text_input("Webhook URL (Slack, Discord, Server):", value=os.environ.get("WEBHOOK_URL", ""))
        if st.button("🔍 Kiểm tra Webhook"):
            from src.services.notification_service import NotificationService
            ok, code, msg = NotificationService.send_webhook(w_url, {"event": "test_connection", "time": datetime.now().isoformat()})
            if ok:
                st.success(f"✅ {msg} (Mã HTTP: {code})")
            else:
                st.error(f"❌ {msg} (Mã HTTP: {code})")

    with tab_logs:
        st.markdown("##### Lịch sử gửi thông báo thực tế:")
        logs = service.get_notification_logs(limit=50)
        if not logs:
            st.info("Chưa có nhật ký gửi thông báo nào.")
        else:
            st.dataframe(pd.DataFrame(logs), use_container_width=True)


# ══════════════════════════════════════════════
# TRANG 8: LỊCH SỬ PHÂN TÍCH (ANALYSIS HISTORY)
# ══════════════════════════════════════════════
def page_history(service: AnalysisService):
    render_header(
        title="Lịch sử Phân tích Video (Audit & Logs)",
        subtitle="Chi tiết các phiên xử lý video, số lượng đối tượng và sự kiện ghi nhận",
        mode=SystemMode.DEMO,
    )

    analyses = service.get_all_analyses()
    if not analyses:
        st.info("Chưa có lịch sử phân tích nào.")
    else:
        st.dataframe(pd.DataFrame(analyses), use_container_width=True)


# ══════════════════════════════════════════════
# TRANG 9: CÀI ĐẶT HỆ THỐNG & BẢO MẬT (SETTINGS)
# ══════════════════════════════════════════════
def page_settings(service: AnalysisService):
    render_header(
        title="Cài đặt Hệ thống & Chính sách Bảo mật (Settings & Privacy)",
        subtitle="Tùy chỉnh thông số AI, làm mờ mặt (Face Privacy) và lưu trữ dữ liệu",
        mode=SystemMode.DEMO,
    )

    st.markdown("##### 🛡️ Quyền riêng tư & Bảo mật dữ liệu trên xe buýt")
    p1, p2 = st.columns(2)
    with p1:
        st.checkbox("Bật làm mờ khuôn mặt hành khách (Face Blurring)", value=True)
        st.caption("Tự động nhận diện và làm mờ khuôn mặt người để bảo vệ quyền riêng tư nơi công cộng.")
    with p2:
        st.selectbox("Thời hạn lưu trữ snapshot (Data Retention):", ["7 ngày", "14 ngày", "30 ngày", "Vĩnh viễn"])

    st.markdown("---")
    st.markdown("##### ⏱️ Tham số Heuristics phát hiện đồ bỏ quên")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Thời gian đứng yên tối thiểu (giây)", 2.0, 30.0, float(MIN_STATIONARY_TIME_SEC))
    with c2:
        st.slider("Ngưỡng di chuyển coi là đứng yên (pixels)", 10.0, 80.0, float(STATIONARY_MOVEMENT_THRESHOLD))


if __name__ == "__main__":
    main()
