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
from src.seats import SeatDetectionMethod
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

    # Tự động xóa cache cũ của Streamlit nếu instance service lưu trong cache từ phiên trước thiếu method
    if not hasattr(service, "get_training_hardware_info") or not hasattr(service, "start_real_training_job"):
        st.cache_resource.clear()
        service = AnalysisService()
        service.initialize()

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
    statuses = service.seat_manager.get_all_statuses()
    occ_count = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
    vac_count = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)
    amb_count = sum(1 for s in statuses.values() if s.state in (SeatState.AMBIGUOUS, SeatState.UNKNOWN))

    # Phương thức nhận diện ghế đang hoạt động
    active_method = service.seat_manager.active_detection_method
    method_val = active_method.value if hasattr(active_method, "value") else str(active_method)

    if method_val == "MODEL_DETECTION":
        method_badge = '<span class="badge" style="background: #065F46; color: #34D399; border: 1px solid #10B981;">AI MODEL DETECTION (Custom Deep Learning)</span>'
        method_desc = "Ghế được nhận diện trực tiếp từ hình ảnh bằng mô hình Computer Vision chuyên biệt."
    elif method_val == "GEOMETRY_FALLBACK":
        method_badge = '<span class="badge" style="background: #78350F; color: #FBBF24; border: 1px solid #F59E0B;">GEOMETRY FALLBACK (Dự phòng kiểm thử, KHÔNG PHẢI AI)</span>'
        method_desc = "Tọa độ ghế được sinh từ lưới hình học dự phòng để kiểm thử luồng. Không phải kết quả nhận diện của AI."
    elif method_val == "MANUAL_CONFIGURATION":
        method_badge = '<span class="badge" style="background: #1E3A8A; color: #60A5FA; border: 1px solid #3B82F6;">MANUAL CONFIGURATION</span>'
        method_desc = "Sơ đồ ghế do người quản trị thiết lập thủ công."
    else:
        method_badge = '<span class="badge" style="background: #374151; color: #9CA3AF; border: 1px solid #6B7280;">CHƯA CẤU HÌNH</span>'
        method_desc = "Chưa có cấu hình sơ đồ ghế xe buýt."

    # Model & Detector info
    model_info = service.get_model_info()
    model_name = model_info.get("model_name", "YOLOv8")
    classes_supported = model_info.get("classes", [])
    classes_str = ", ".join(classes_supported[:8]) + ("..." if len(classes_supported) > 8 else "")

    st.markdown(f"""
    <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Cơ chế nhận diện ghế hiện tại:</span>
                <span style="margin-left: 8px;">{method_badge}</span>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">{method_desc}</div>
            </div>
            <div style="text-align: right; font-size: 0.8rem; color: #94A3B8;">
                <div>🧠 Model chính: <b style="color: #F8FAFC;">{model_name}</b></div>
                <div>Lớp hỗ trợ: <span style="color: #60A5FA;">{classes_str if classes_str else 'person, items'}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if method_val == "GEOMETRY_FALLBACK":
        st.warning(
            "⚠️ **CẢNH BÁO MINH BẠCH**: Hệ thống đang chạy ở chế độ **GEOMETRY FALLBACK** phục vụ kiểm thử. "
            "Tọa độ các ghế hiện tại KHÔNG PHẢI do AI nhận diện trực tiếp từ hình ảnh. "
            "Để nhận diện thực tế, cần huấn luyện hoặc nạp weights chuyên biệt vào `models/bus_seat_yolo.pt`."
        )

    # 4 Metric Panels
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-panel">
            <p>🪑 GHẾ ĐÃ ĐỊNH DANH</p>
            <h3>{seat_count} <span style="font-size: 0.85rem; color: #94A3B8;">(👤 {occ_count} | 🟢 {vac_count} | ⚪ {amb_count})</span></h3>
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
            <h3>{stats.get('total_events', 0)}</h3>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-panel">
            <p>🎬 PHIÊN PHÂN TÍCH</p>
            <h3>{stats.get('total_analyses', 0)}</h3>
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
                elif st_state == SeatState.AMBIGUOUS:
                    s_badge = '<span class="badge" style="background: #78350F; color: #FBBF24;">TRANH CHẤP / CHƯA RÕ</span>'
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

                total_seats = len(service.seat_manager.seats)
                statuses = service.seat_manager.get_all_statuses()
                occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
                vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)

                metric_view.markdown(
                    f"🪑 **Ghế:** `{total_seats}` (👤 Có người: `{occ_seats}` | 🟢 Trống: `{vac_seats}`) | "
                    f"🎒 **Đồ theo dõi:** `{len(events)}` | "
                    f"⚠️ **Nghi ngờ:** `{summary.get('suspected', 0)}` | "
                    f"🛡️ **Che khuất:** `{summary.get('occluded', 0)}` | "
                    f"⏱️ **FPS:** `{calc_fps:.1f}` (`{latency_ms:.1f}ms`)"
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

                    total_seats = len(service.seat_manager.seats)
                    statuses = service.seat_manager.get_all_statuses()
                    occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
                    vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)

                    cam_metrics.markdown(
                        f"🪑 **Ghế:** `{total_seats}` (👤 Có người: `{occ_seats}` | 🟢 Trống: `{vac_seats}`) | "
                        f"🎒 **Đồ theo dõi:** `{len(events)}` | "
                        f"⚠️ **Nghi ngờ:** `{summary.get('suspected', 0)}` | "
                        f"🛡️ **Che khuất:** `{summary.get('occluded', 0)}` | "
                        f"⏱️ **FPS:** `{1000.0/max(1.0, lat_ms):.1f}` (`{lat_ms:.1f}ms`)"
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
        subtitle="Tự động nhận diện từng ghế, phân tách ghế đôi (Ghế A & Ghế B), gán ID ổn định",
        mode=SystemMode.DEMO,
    )

    tab_auto, tab_grid, tab_list, tab_diag = st.tabs([
        "🔍 Tự động Nhận diện Ghế (AI / Vision)",
        "⚡ Sinh sơ đồ tham số xe buýt",
        "📋 Danh sách & Trạng thái Ghế",
        "🧠 Kiểm tra Khả năng Model (Diagnostics)",
    ])

    with tab_auto:
        st.markdown("##### 🚌 Nhận diện & Phân Tách Từng Ghế (Twin-Seat Decomposition)")
        st.caption(
            "Phân tách độc lập từng chiếc ghế trong hàng ghế đôi (Ghế A & Ghế B) từ camera nhìn xuống, không gộp cả hàng thành 1 vùng lớn."
        )

        diag = service.get_seat_diagnostics()
        custom_ready = diag.get("custom_adapter_ready", False)

        if not custom_ready:
            st.markdown("""
            <div style="background: #1C2541; border-left: 4px solid #F59E0B; padding: 14px 16px; border-radius: 6px; margin-bottom: 14px;">
                <div style="font-weight: 700; color: #FBBF24; font-size: 0.95rem;">
                    ⚠️ CUSTOM SEAT MODEL NOT CONFIGURED
                </div>
                <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 6px; line-height: 1.5;">
                    Chưa nạp file weights mô hình chuyên biệt cho ghế xe buýt (mặc định tìm: <code>models/bus_seat_yolo.pt</code>).<br>
                    • Model YOLO COCO chuẩn (<code>yolov8n.pt</code>) chỉ có lớp <code>chair</code> (ghế tựa văn phòng nhìn ngang), 
                    <b>tuyệt đối không nhận diện được</b> đệm ghế xe buýt nhìn từ trên xuống (top-down view).<br>
                    • <b>Minh bạch kỹ thuật:</b> Hệ thống không tự ý sinh sơ đồ hình học rồi mạo nhận là AI. 
                    Để kiểm thử luồng hệ thống khi chưa có weights, hãy tích chọn cho phép phương án dự phòng bên dưới.
                </div>
            </div>
            """, unsafe_allow_html=True)
            allow_fallback = st.checkbox(
                "Cho phép dùng GEOMETRY FALLBACK để kiểm thử luồng (Dự phòng kiểm thử, KHÔNG PHẢI AI)",
                value=False,
                key="seat_allow_fallback"
            )
        else:
            allow_fallback = False
            st.success(f"✅ Đã kết nối Custom Seat Model chuyên biệt (`{diag.get('model_weights')}`). Sẵn sàng nhận diện trực tiếp từ ảnh!")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            auto_rows = st.number_input("Số hàng ghế trong góc nhìn:", 1, 8, 4, 1, key="auto_rows")
        with col_cfg2:
            auto_left = st.number_input("Ghế bên trái lối đi:", 1, 2, 2, 1, help="2 = Ghế đôi kề nhau (Cửa sổ + Lối đi)", key="auto_left")
        with col_cfg3:
            auto_right = st.number_input("Ghế bên phải lối đi:", 1, 2, 2, 1, help="2 = Ghế đôi kề nhau (Lối đi + Cửa sổ)", key="auto_right")

        auto_naming = st.selectbox(
            "Quy tắc đặt Seat ID:",
            ["SEAT_1A, SEAT_1B... (Khuyên dùng chuẩn hàng không/xe khách)", "SEAT_001, SEAT_002..."],
            key="auto_naming"
        )
        naming_format = "SEAT" if "1A" in auto_naming else "NUM"

        st.markdown("###### 📸 Chọn frame hình ảnh để quét:")
        img_source = st.radio(
            "Nguồn ảnh mẫu:",
            ["Trích xuất từ Video Demo mẫu xe buýt", "Tải lên ảnh chụp từ trần xe buýt (JPG/PNG)"],
            horizontal=True,
            key="img_source"
        )

        target_frame = None
        if "Demo mẫu" in img_source:
            demo_p = ensure_demo_video_exists()
            cap = cv2.VideoCapture(demo_p)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                target_frame = frame
        else:
            up_img = st.file_uploader("Tải lên ảnh chụp từ trần xe buýt:", type=["jpg", "jpeg", "png"], key="up_seat_frame")
            if up_img:
                file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
                target_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if target_frame is None:
            target_frame = np.full((720, 1280, 3), (35, 30, 25), dtype=np.uint8)
            cv2.putText(target_frame, "OVERHEAD BUS CABIN SIMULATION", (340, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        if st.button("🚀 Bắt đầu Quét & Nhận diện Từng Ghế", type="primary", use_container_width=True):
            if not custom_ready and not allow_fallback:
                st.error("❌ CUSTOM SEAT MODEL NOT CONFIGURED: Chưa nạp mô hình ghế chuyên biệt và bạn chưa kích hoạt tùy chọn GEOMETRY FALLBACK.")
            else:
                with st.spinner("Đang chạy phân tích thị giác..."):
                    res = service.auto_detect_seats(
                        frame=target_frame,
                        allow_fallback=allow_fallback,
                        rows=auto_rows,
                        left_cols=auto_left,
                        right_cols=auto_right,
                        naming_format=naming_format,
                        save_to_disk=True,
                    )
                    st.session_state["last_seat_detection_result"] = res
                    if res.method == SeatDetectionMethod.MODEL_DETECTION:
                        st.success(f"✅ MODEL DETECTION: Đã nhận diện thành công {res.total_seats} ghế bằng mô hình AI chuyên biệt!")
                    elif res.method == SeatDetectionMethod.GEOMETRY_FALLBACK:
                        st.warning(f"⚠️ GEOMETRY FALLBACK: Đã sinh {res.total_seats} ghế bằng lưới hình học dự phòng để kiểm thử luồng (KHÔNG PHẢI AI)!")
                    else:
                        st.error(f"❌ {res.message}")

        if "last_seat_detection_result" in st.session_state:
            res = st.session_state["last_seat_detection_result"]
            cur_h, cur_w = target_frame.shape[:2]

            preview_img = target_frame.copy()
            overlay = preview_img.copy()

            for s in res.seats:
                poly = s.get_pixel_polygon(cur_w, cur_h)
                pts_arr = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                
                # Màu sắc: Phân biệt ghế trái (Cyan) và ghế phải (Teal)
                is_left = "A" in s.seat_id or "B" in s.seat_id or (s.seat_id.replace("SEAT_", "").isdigit() and int(s.seat_id.replace("SEAT_", "")) % 4 in (1, 2))
                color = (255, 200, 0) if is_left else (0, 215, 255)

                cv2.fillPoly(overlay, [pts_arr], color)
                cv2.polylines(preview_img, [pts_arr], isClosed=True, color=(255, 255, 255), thickness=2)

                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                cv2.putText(preview_img, s.seat_id, (cx - 35, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.addWeighted(overlay, 0.25, preview_img, 0.75, 0, preview_img)
            method_caption = (
                f"🎯 [MODEL DETECTION - AI CHUYÊN BIỆT] {res.total_seats} ghế được phát hiện trực tiếp từ hình ảnh"
                if res.method == SeatDetectionMethod.MODEL_DETECTION
                else f"⚠️ [GEOMETRY FALLBACK - DỰ PHÒNG KIỂM THỬ, KHÔNG PHẢI AI] {res.total_seats} ghế sinh từ lưới tham số"
            )
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption=method_caption, use_container_width=True)

            seat_rows_data = []
            for s in res.seats:
                poly = s.get_pixel_polygon(cur_w, cur_h)
                seat_rows_data.append({
                    "Seat ID": s.seat_id,
                    "Vị trí / Loại": s.description,
                    "Cơ chế phát hiện": "🧠 AI Model" if res.method == SeatDetectionMethod.MODEL_DETECTION else "⚙️ Geometry Fallback (Test Only)",
                    "Tọa độ Pixel (Top-Left → Bottom-Right)": f"({poly[0][0]}, {poly[0][1]}) → ({poly[2][0]}, {poly[2][1]})",
                    "Phân tách độc lập": "✅ Tách biệt hoàn toàn",
                })
            st.dataframe(pd.DataFrame(seat_rows_data), use_container_width=True)

    with tab_grid:
        st.markdown("##### 🚌 Trình sinh nhanh sơ đồ xe buýt (Bus Grid Generator)")
        st.caption("Ghế xe buýt là các thực thể cố định, giữ ID vĩnh viễn và không phụ thuộc vào vùng chuyển động.")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            rows = st.number_input("Số hàng ghế", 1, 12, 4, 1, key="grid_rows")
        with gcol2:
            left_cols = st.number_input("Số ghế bên trái lối đi", 1, 3, 2, 1, key="grid_left")
        with gcol3:
            right_cols = st.number_input("Số ghế bên phải lối đi", 1, 3, 2, 1, key="grid_right")

        name_style = st.selectbox("Định dạng tên ghế:", ["SEAT_1A, SEAT_1B...", "S01, S02, S03..."], key="grid_name_style")

        total_seats = rows * (left_cols + right_cols)
        st.info(f"Sơ đồ sẽ tạo tổng cộng **{total_seats} ghế cố định**.")

        if st.button("💾 Tạo và Lưu Sơ đồ Ghế mới", type="primary"):
            from src.seats.seat_manager import SeatManager
            style_code = "SEAT" if "1A" in name_style else "NUM"
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
            statuses = service.seat_manager.get_all_statuses()
            seat_data = []
            for s in seats:
                st_info = statuses.get(s.seat_id)
                st_state = st_info.state.value if st_info else "unknown"
                notes = st_info.notes if st_info else ""
                seat_data.append({
                    "Seat ID": s.seat_id,
                    "Trạng thái": st_state.upper(),
                    "Mô tả": s.description,
                    "Chuẩn hóa [0-1]": s.is_normalized,
                    "Số đỉnh polygon": len(s.polygon),
                    "Ghi chú": notes,
                })
            st.dataframe(pd.DataFrame(seat_data), use_container_width=True)

    with tab_diag:
        st.markdown("##### 🧠 Báo cáo Khả năng Model & Hỗ trợ Nhận diện Ghế")
        diag = service.get_seat_diagnostics()
        custom_ready = diag.get("custom_adapter_ready", False)
        weights_path = diag.get("model_weights", "models/bus_seat_yolo.pt")

        status_color = "#10B981" if custom_ready else "#EF4444"
        status_text = "ĐÃ SẴN SÀNG (Custom Weights Found)" if custom_ready else "CUSTOM SEAT MODEL NOT CONFIGURED"

        st.markdown(f"""
        <div style="background: #141E33; border: 1px solid #3A506B; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
            <h4 style="color: #F8FAFC; margin-top: 0;">📊 Trạng thái Adapter Nhận diện Ghế Chuyên biệt</h4>
            <div style="margin-bottom: 10px;">
                <span style="color: #94A3B8;">Trạng thái Model Ghế:</span> 
                <span class="badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; font-weight: bold;">{status_text}</span>
            </div>
            <ul style="color: #CBD5E1; line-height: 1.6;">
                <li><b>Đường dẫn kiểm tra trọng số:</b> <code>{weights_path}</code></li>
                <li><b>Lớp COCO 56 (chair):</b> <span style="color: #EF4444; font-weight: bold;">TỪ CHỐI TỰ ĐỘNG</span> (ghế tựa văn phòng nhìn ngang, không nhận diện được đệm ghế trần xe buýt).</li>
                <li><b>Các lớp ghế xe buýt hợp lệ:</b> <code>bus_seat</code>, <code>seat</code>, <code>seat_cushion</code>, <code>seat_window</code>, <code>seat_aisle</code>.</li>
                <li><b>Cơ chế fallback hiện hành:</b> <code>{service.seat_manager.active_detection_method.value}</code></li>
                <li><b>Phân tách ghế đôi (Twin-Seat):</b> <span style="color: #10B981; font-weight: bold;">TÍCH HỢP (Độc lập Ghế A & Ghế B)</span></li>
            </ul>
            <div style="background: #0F172A; padding: 12px; border-radius: 6px; border-left: 4px solid #3B82F6; margin-top: 10px;">
                <p style="margin: 0; color: #94A3B8; font-size: 0.9rem;">
                    💡 <b>Hướng dẫn nạp model chuyên biệt:</b> Đặt file trọng số <code>models/bus_seat_yolo.pt</code> được huấn luyện trên 
                    tập dữ liệu camera trần xe buýt vào thư mục dự án. Hệ thống sẽ tự động đọc danh sách class từ file weights và chuyển sang 
                    nhận diện hoàn toàn bằng AI thực thụ.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


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
        st.markdown("##### 🚀 Huấn luyện Model mới (Real YOLO Training Pipeline)")
        st.caption("Khởi chạy huấn luyện thật bằng Ultralytics YOLO trong tiến trình nền, theo dõi nhật ký thực tế và đọc chỉ số mAP50 từ results.csv.")

        from src.services.model_service import ModelService
        ms = getattr(service, "model_service", None) or getattr(service, "_model_service", None)

        if hasattr(service, "get_training_hardware_info"):
            hw = service.get_training_hardware_info()
        elif ms and hasattr(ms, "get_training_hardware_info"):
            hw = ms.get_training_hardware_info()
        else:
            hw = ModelService.get_training_hardware_info()

        dev_type = hw.get("device", "cpu")
        icon = "⚡" if dev_type in ("cuda", "mps") else "⚠️"
        st.markdown(f"""
        <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="font-weight: 600; color: #F8FAFC;">{icon} Thiết bị tính toán khả dụng: {hw.get('device_name')} ({dev_type.upper()})</span>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 2px;">{hw.get('description')}</div>
        </div>
        """, unsafe_allow_html=True)

        yaml_col, val_btn_col = st.columns([3, 1])
        with yaml_col:
            dataset_yaml = st.text_input("Đường dẫn file cấu hình data.yaml:", value="data/dataset/exports/data.yaml")
        with val_btn_col:
            st.write("")
            st.write("")
            run_check = st.button("🔍 Kiểm tra Dataset", use_container_width=True)

        if run_check:
            if hasattr(service, "validate_training_dataset"):
                ok_val, val_msg, val_details = service.validate_training_dataset(dataset_yaml)
            elif ms and hasattr(ms, "validate_training_dataset"):
                ok_val, val_msg, val_details = ms.validate_training_dataset(dataset_yaml)
            else:
                ok_val, val_msg, val_details = ModelService.validate_training_dataset(dataset_yaml)

            if ok_val:
                st.success(f"✅ {val_msg}")
                st.json(val_details)
            else:
                st.error(f"❌ {val_msg}")
                if val_details:
                    st.json(val_details)

        tc1, tc2 = st.columns(2)
        with tc1:
            new_ver = st.text_input("Tên phiên bản model mới:", value="v1.1-bus-seats")
            base_m = st.selectbox("Kiến trúc base model YOLO:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        with tc2:
            epochs = st.slider("Số Epochs:", 5, 100, 30, 5)
            batch = st.selectbox("Batch size:", [4, 8, 16, 32], index=1)

        if ms and hasattr(ms, "generate_training_command"):
            cmd = ms.generate_training_command(dataset_yaml, epochs=epochs, batch_size=batch, model=base_m)
        else:
            cmd = ModelService(getattr(service, "_model_repo", None)).generate_training_command(dataset_yaml, epochs=epochs, batch_size=batch, model=base_m)
        st.caption("Lệnh Ultralytics CLI thực tế được sinh:")
        st.code(cmd, language="bash")

        if st.button("🚀 Bắt đầu Huấn luyện Thật trong Tiến trình Nền (Background Process)", type="primary", use_container_width=True):
            if hasattr(service, "validate_training_dataset"):
                ok_val, val_msg, _ = service.validate_training_dataset(dataset_yaml)
            elif ms and hasattr(ms, "validate_training_dataset"):
                ok_val, val_msg, _ = ms.validate_training_dataset(dataset_yaml)
            else:
                ok_val, val_msg, _ = ModelService.validate_training_dataset(dataset_yaml)

            if not ok_val:
                st.error(f"❌ Từ chối khởi chạy: Dataset không hợp lệ hoặc thiếu dữ liệu. {val_msg}. Vui lòng kiểm tra lại dataset trước khi train.")
            else:
                if hasattr(service, "start_real_training_job"):
                    ok_launch, job_id, launch_info = service.start_real_training_job(
                        dataset_yaml_path=dataset_yaml,
                        new_version=new_ver,
                        base_model=base_m,
                        epochs=epochs,
                        batch_size=batch,
                    )
                elif ms and hasattr(ms, "start_real_training_job"):
                    ok_launch, job_id, launch_info = ms.start_real_training_job(
                        dataset_yaml_path=dataset_yaml,
                        new_version=new_ver,
                        base_model=base_m,
                        epochs=epochs,
                        batch_size=batch,
                    )
                else:
                    m_temp = ModelService(getattr(service, "_model_repo", None))
                    ok_launch, job_id, launch_info = m_temp.start_real_training_job(
                        dataset_yaml_path=dataset_yaml,
                        new_version=new_ver,
                        base_model=base_m,
                        epochs=epochs,
                        batch_size=batch,
                    )

                if ok_launch:
                    st.session_state["active_training_job_id"] = job_id
                    st.session_state["active_training_dataset_ver"] = new_ver
                    st.success(f"✅ Đã khởi chạy tiến trình huấn luyện ngầm! Job ID: `{job_id}` (PID: {launch_info.get('pid')})")
                    st.rerun()
                else:
                    st.error(f"❌ Lỗi khởi chạy: {launch_info.get('error', 'Không xác định')}")

        active_job_id = st.session_state.get("active_training_job_id")
        if active_job_id:
            st.markdown("---")
            st.markdown("##### 📡 Theo dõi Tiến trình Huấn luyện Thật (Live Job Monitor)")
            if hasattr(service, "get_training_job_status"):
                job_status = service.get_training_job_status(active_job_id)
            elif ms and hasattr(ms, "get_training_job_status"):
                job_status = ms.get_training_job_status(active_job_id)
            else:
                job_status = ModelService(getattr(service, "_model_repo", None)).get_training_job_status(active_job_id)

            st_code = job_status.get("status", "UNKNOWN")
            stat_color = "#3B82F6" if st_code == "running" else ("#10B981" if st_code == "completed" else "#EF4444")

            st.markdown(f"""
            <div style="background: #1C2541; border: 1px solid #3A506B; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 700; color: #F8FAFC;">Job ID: {active_job_id}</span>
                        <span style="color: #94A3B8; font-size: 0.85rem; margin-left: 8px;">(Phiên bản: {job_status.get('version')})</span>
                    </div>
                    <div>
                        <span class="badge" style="background: {stat_color}22; color: {stat_color}; border: 1px solid {stat_color}; font-weight: bold;">
                            {st_code.upper()}
                        </span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 8px;">
                    PID: <code>{job_status.get('pid', 'N/A')}</code> • Thời gian chạy: <b>{job_status.get('elapsed_seconds', 0):.1f}s</b> • File Log: <code>{job_status.get('log_file', '')}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if st.button("🔄 Cập nhật Tiến độ", use_container_width=True):
                    st.rerun()

            if st_code == "completed":
                metrics = job_status.get("metrics", {})
                st.success("🎉 Huấn luyện và Validation thực tế đã HOÀN TẤT!")
                st.markdown(f"""
                <b>Kết quả Metrics thực tế từ results.csv:</b><br>
                • mAP50: <code>{metrics.get('map50', 'N/A')}</code><br>
                • Precision: <code>{metrics.get('precision', 'N/A')}</code><br>
                • Recall: <code>{metrics.get('recall', 'N/A')}</code><br>
                • mAP50-95: <code>{metrics.get('map50_95', 'N/A')}</code><br>
                • Weights: <code>{job_status.get('best_weights_path')}</code>
                """, unsafe_allow_html=True)

                if st.button("⭐ Phê duyệt & Đăng ký Candidate Model vào Hệ thống", type="primary"):
                    if hasattr(service, "approve_and_register_completed_job"):
                        ok_app, app_msg = service.approve_and_register_completed_job(
                            job_id=active_job_id,
                            dataset_version=st.session_state.get("active_training_dataset_ver", "v1.0"),
                            notes=f"Huấn luyện thật từ {job_status.get('dataset_yaml')}"
                        )
                    elif ms and hasattr(ms, "approve_and_register_completed_job"):
                        ok_app, app_msg = ms.approve_and_register_completed_job(
                            job_id=active_job_id,
                            dataset_version=st.session_state.get("active_training_dataset_ver", "v1.0"),
                            notes=f"Huấn luyện thật từ {job_status.get('dataset_yaml')}"
                        )
                    else:
                        ok_app, app_msg = ModelService(getattr(service, "_model_repo", None)).approve_and_register_completed_job(
                            job_id=active_job_id,
                            dataset_version=st.session_state.get("active_training_dataset_ver", "v1.0"),
                            notes=f"Huấn luyện thật từ {job_status.get('dataset_yaml')}"
                        )

                    if ok_app:
                        st.success(f"✅ {app_msg}")
                        st.session_state.pop("active_training_job_id", None)
                        st.rerun()
                    else:
                        st.error(f"❌ {app_msg}")

            elif st_code == "failed":
                st.error(f"❌ Quá trình huấn luyện thất bại: {job_status.get('error', 'Lỗi không xác định')}")

            log_tail = job_status.get("log_tail", [])
            if log_tail:
                with st.expander("📜 Xem nhật ký huấn luyện (Log Tail)", expanded=(st_code == "running")):
                    st.code("".join(log_tail), language="text")


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
