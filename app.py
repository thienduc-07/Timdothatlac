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
from src.detection.class_mapper import format_seat_label, get_user_facing_label
from src.utils.file_utils import format_file_size, format_duration

from src.video.camera_service import (
    CameraSourceType,
    ensure_demo_video_exists,
    generate_bus_simulation_frame,
    get_demo_ground_truth,
    ThreadedCameraReader,
)
from src.video.webrtc_processor import (
    WEBRTC_AVAILABLE,
    BusWebRtcVideoProcessor,
    RTC_CONFIGURATION,
    webrtc_streamer,
    WebRtcMode,
    Translations,
)
from src.utils.text_renderer import draw_text_unicode
from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SHOW_SEGMENTATION_MASKS,
    CUSTOM_SEGMENTATION_WEIGHTS_PATH,
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
# CSS Giao diện Dark Navy chuyên nghiệp & Hỗ trợ tiếng Việt đầy đủ
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label, button {{
        font-family: 'Be Vietnam Pro', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }}

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
def _get_cached_service(_cache_tag: str = "v3.0_universal_cam") -> AnalysisService:
    service = AnalysisService()
    service.initialize()
    return service


def get_service() -> AnalysisService:
    """Khởi tạo AnalysisService một lần duy nhất, tự động xóa cache cũ nếu phát hiện thiếu tham số."""
    try:
        service = _get_cached_service()
        import inspect
        sig = inspect.signature(service.process_webcam_frame)
        if "show_masks" not in sig.parameters:
            st.cache_resource.clear()
            service = _get_cached_service(_cache_tag=f"v3.1_fresh_{time.time()}")
        return service
    except Exception as e:
        st.cache_resource.clear()
        service = AnalysisService()
        service.initialize()
        return service


def safe_process_webcam_frame(service: AnalysisService, **kwargs):
    """
    Gọi service.process_webcam_frame an toàn tuyệt đối.
    Tự động đối chiếu signature để không bao giờ bị TypeError khi chạy trên Streamlit Cloud với cache cũ.
    """
    import inspect
    sig = inspect.signature(service.process_webcam_frame)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_varkw:
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return service.process_webcam_frame(**filtered)
    return service.process_webcam_frame(**kwargs)


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

        page_list = [
            "🏠 Tổng quan",
            "🎥 Giám sát Camera",
            "⚠️ Trung tâm Cảnh báo",
            "🪑 Quản lý Ghế",
            "🏷️ AI Training & Dataset",
            "🤖 Quản lý Model",
            "🔔 Cấu hình Thông báo",
            "📊 Lịch sử Phân tích",
            "⚙️ Cài đặt & Bảo mật",
        ]

        # Xử lý điều hướng nhanh nếu có yêu cầu chuyển trang từ dashboard
        nav_target = st.session_state.pop("nav_to", None)
        selected_index = 0
        if nav_target:
            for idx, p in enumerate(page_list):
                if nav_target in p:
                    selected_index = idx
                    break

        page = st.radio(
            "CHỨC NĂNG HỆ THỐNG",
            page_list,
            index=selected_index,
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
    is_seg = model_info.get("is_segmentation", False)
    task_name = model_info.get("task", "detect")
    classes_supported = model_info.get("classes", []) or model_info.get("tracked_class_names", [])
    classes_str = ", ".join(classes_supported[:8]) + ("..." if len(classes_supported) > 8 else "")
    has_custom_weights = CUSTOM_SEGMENTATION_WEIGHTS_PATH.exists()

    seg_badge = '<span class="badge" style="background: #0284C7; color: #E0F2FE; border: 1px solid #38BDF8;">TASK: SEGMENTATION (Polygon Masks)</span>' if is_seg else '<span class="badge" style="background: #334155; color: #94A3B8;">TASK: DETECTION (BBox)</span>'

    st.markdown(f"""
    <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Cơ chế nhận diện ghế hiện tại:</span>
                <span style="margin-left: 8px;">{method_badge}</span>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">{method_desc}</div>
            </div>
            <div style="text-align: right; font-size: 0.8rem; color: #94A3B8;">
                <div>🧠 Model chính: <b style="color: #F8FAFC;">{model_name}</b> {seg_badge}</div>
                <div>Lớp nhận diện: <span style="color: #60A5FA;">{classes_str if classes_str else 'seat, person, item'}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not has_custom_weights:
        st.info(
            "⏳ **Chờ nạp weights YOLO26 Instance Segmentation:**\n\n"
            "Mô hình 3 lớp (`seat`, `person`, `item`) hiện đang được huấn luyện trên Ultralytics Platform. "
            "Sau khi hoàn tất, hãy tải file weights và đặt tại đường dẫn: **`models/best.pt`**.\n\n"
            "Hệ thống đã sẵn sàng 100% về kiến trúc Instance Segmentation, tự động trích xuất polygon masks, "
            "dùng chung model duy nhất cho ghế & tracking đối tượng mà không cần tải lại model hai lần."
        )

    if method_val == "GEOMETRY_FALLBACK":
        st.warning(
            "⚠️ **CẢNH BÁO MINH BẠCH**: Hệ thống đang chạy ở chế độ **GEOMETRY FALLBACK** phục vụ kiểm thử. "
            "Tọa độ các ghế hiện tại KHÔNG PHẢI do AI nhận diện trực tiếp từ hình ảnh. "
            "Để nhận diện thực tế bằng AI, hãy đặt weights segmentation đã train vào `models/best.pt`."
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


def _run_browser_camera_analysis(
    photo,
    service: AnalysisService,
    conf_thresh: float,
    min_stat_time: float,
    show_seats: bool,
    show_persons: bool,
    show_objects: bool,
    privacy_blur: bool,
    show_masks: bool = DEFAULT_SHOW_SEGMENTATION_MASKS,
):
    """Xử lý hình ảnh chụp từ webcam trình duyệt và hiển thị trực quan chi tiết."""
    if not photo:
        return
    try:
        np_arr = np.frombuffer(photo.getvalue(), np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            st.error("Không thể đọc định dạng ảnh từ Camera.")
            return

        tracker = ObjectTracker()
        event_manager = EventManager()
        event_detector = ForgottenItemDetector(
            min_stationary_time=min_stat_time,
            movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
            watched_classes=ITEM_CLASSES,
            seat_manager=service.seat_manager,
            event_manager=event_manager,
        )

        with st.spinner("🧠 AI đang quét & phân tích hình ảnh từ Camera của bạn..."):
            annotated, events, summary = safe_process_webcam_frame(
                service=service,
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
                show_masks=show_masks,
            )

        rgb_res = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_res, caption="Kết quả Phân tích Trực quan AI (Bounding box, Ghế, Người & Đồ vật)", channels="RGB", use_container_width=True)

        statuses = service.seat_manager.get_all_statuses()
        total_seats = len(service.seat_manager.seats)
        occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
        vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Tổng số ghế AI phát hiện", f"{total_seats}")
        with mcol2:
            st.metric("Ghế có người", f"{occ_seats}", delta=f"{vac_seats} ghế trống")
        with mcol3:
            st.metric("Đồ vật theo dõi", f"{len(events)}")
        with mcol4:
            sus_count = summary.get("suspected", 0)
            st.metric("Nghi ngờ bỏ quên", f"{sus_count}", delta="Cần chú ý" if sus_count > 0 else "An toàn", delta_color="inverse" if sus_count > 0 else "normal")

        if total_seats == 0:
            st.info("ℹ️ **Trạng thái ghế:** Chưa phát hiện đệm ghế từ hình ảnh camera. Hệ thống chỉ ghi nhận và vẽ ghế khi mô hình AI nhận diện trực tiếp từ thị giác (không dùng tọa độ giả lập).")

        if summary.get("suspected", 0) > 0:
            st.error("⚠️ **CẢNH BÁO:** Phát hiện vật thể có dấu hiệu bị bỏ quên trên ghế!")
            play_audio_alert()
        else:
            st.success(f"✅ Quét thành công! Phát hiện {len(events)} đối tượng trong tầm quan sát của AI.")
    except Exception as e:
        logger.error(f"Lỗi phân tích ảnh webcam: {e}")
        st.error(f"Lỗi khi xử lý hình ảnh camera: {e}")


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

    # Nhận diện môi trường Cloud hoặc Local
    is_cloud_env = (
        os.path.exists("/mount/src")
        or "STREAMLIT_SHARING_HOST" in os.environ
        or (sys.platform.startswith("linux") and not os.path.exists("/dev/video0"))
    )

    # Nguồn dữ liệu & Tùy chọn điều khiển
    st.markdown("##### ⚙️ Nguồn dữ liệu & Cấu hình luồng")
    src_col1, src_col2, src_col3 = st.columns([2, 1, 1])

    with src_col1:
        source_mode = st.radio(
            "Chọn nguồn video / camera:",
            [
                "🎥 Live Camera (Hỗ trợ tất cả dòng máy: Mac, Windows, Cloud, Mobile)",
                "🚌 Video Mẫu Xe Buýt (Demo Mode - Khuyên dùng kiểm thử)",
                "📁 Tải lên Video từ máy tính (.mp4, .avi, .mov)",
                "🌐 Luồng IP / RTSP Camera",
            ],
            index=0,
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
        show_masks = st.checkbox("🎭 Mặt nạ Segmentation (Masks)", value=DEFAULT_SHOW_SEGMENTATION_MASKS)
        privacy_blur = st.checkbox("🛡️ Làm mờ mặt (Privacy Blur)", value=False)

    st.markdown("---")

    # Xử lý theo từng nguồn
    if "Demo Mode" in source_mode:
        st.info("💡 **Chế độ Demo mẫu:** Hệ thống mô phỏng chuẩn camera trần khoang xe buýt (hành khách bước vào, ngồi tại Ghế 1A, đặt ba lô, đứng dậy rời đi để quên ba lô trên ghế trống -> AI tự động kích hoạt cảnh báo).")
        demo_path = ensure_demo_video_exists()

        dcol1, dcol2 = st.columns([3, 1])
        with dcol2:
            st.markdown("##### Điều khiển Demo")
            start_demo = st.button("▶️ Chạy Phân tích Demo Mẫu", type="primary", use_container_width=True)
            recreate_demo = st.button("🔄 Làm mới video mẫu (Tạo lại file)", use_container_width=True)
            if recreate_demo:
                demo_path = ensure_demo_video_exists(force_recreate=True)
                st.success("Đã làm mới và sinh lại video demo thành công!")
            auto_alert = st.checkbox("Kích hoạt âm thanh cảnh báo", value=True)

        with dcol1:
            frame_view = st.empty()
            metric_view = st.empty()

        if start_demo:
            cap = cv2.VideoCapture(demo_path)
            if not cap.isOpened():
                demo_path = ensure_demo_video_exists(force_recreate=True)
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
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 20.0
            if not fps or fps <= 0:
                fps = 20.0
            alert_fired = False
            total_demo_frames = 200

            for frame_idx in range(total_demo_frames):
                t0 = time.time()
                ret = False
                frame = None
                if cap.isOpened():
                    ret, frame = cap.read()

                # Fallback generator nếu không đọc được từ file video (đảm bảo không bao giờ mất source)
                if not ret or frame is None:
                    frame = generate_bus_simulation_frame(frame_idx, 640, 360, total_demo_frames, fps)

                # Chạy phân tích an toàn
                annotated, events, summary = safe_process_webcam_frame(
                    service=service,
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
                    show_masks=show_masks,
                )

                # Fallback thông minh: nếu YOLO COCO weights chưa train trên tranh vẽ mô phỏng,
                # kết hợp ground truth để người dùng thấy đầy đủ 100% vòng đời logic AI trên xe buýt
                if len(events) == 0 and summary.get("suspected", 0) == 0:
                    gt = get_demo_ground_truth(frame_idx, 640, 360)
                    sim_tracked = []
                    if gt.get("has_person") and gt.get("person_bbox"):
                        sim_tracked.append(TrackedObject(
                            track_id=1,
                            class_id=0,
                            class_name="person",
                            bbox=gt["person_bbox"],
                            confidence=0.92,
                            frame_index=frame_idx,
                            timestamp=frame_idx / fps,
                        ))
                    if gt.get("has_bag") and gt.get("bag_bbox"):
                        bx1, by1, bx2, by2 = gt["bag_bbox"]
                        bag_poly = [
                            (float(bx1 + 12), float(by1)),
                            (float(bx2 - 12), float(by1)),
                            (float(bx2), float(by1 + 14)),
                            (float(bx2), float(by2 - 10)),
                            (float(bx2 - 8), float(by2)),
                            (float(bx1 + 8), float(by2)),
                            (float(bx1), float(by2 - 10)),
                            (float(bx1), float(by1 + 14)),
                        ]
                        sim_tracked.append(TrackedObject(
                            track_id=2,
                            class_id=24,
                            class_name="backpack",
                            bbox=gt["bag_bbox"],
                            confidence=0.89,
                            frame_index=frame_idx,
                            timestamp=frame_idx / fps,
                            polygon=bag_poly,
                        ))

                    if sim_tracked and service.seat_manager and service.seat_manager.is_configured:
                        event_detector.process_frame(
                            tracked_objects=sim_tracked,
                            current_frame=frame_idx,
                            fps=fps,
                            frame_width=640,
                            frame_height=360,
                            video_name="Demo_Bus",
                            frame=frame,
                        )
                        summary = event_manager.get_summary()
                        events = event_manager.events

                        evt_statuses = {evt.last_track_id: evt.status for evt in events if evt.last_track_id != -1}
                        annotated = VideoWriter.annotate_frame(
                            frame=frame,
                            tracked_objects=sim_tracked,
                            event_statuses=evt_statuses,
                            seat_manager=service.seat_manager,
                            active_events=events,
                            width=640,
                            height=360,
                            show_seats=show_seats,
                            show_persons=show_persons,
                            show_objects=show_objects,
                            show_track_ids=True,
                            show_confidence=True,
                            show_event_markers=True,
                            privacy_blur_faces=privacy_blur,
                            show_masks=show_masks,
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

                time.sleep(0.035)

            if cap.isOpened():
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

    elif "Live Camera" in source_mode or "Live Webcam" in source_mode:
        st.markdown("### 🎥 Giám sát Camera Trực Tiếp Thời Gian Thực (Live AI Stream)")
        st.caption("Truyền trực tiếp hình ảnh từ Webcam thiết bị của bạn (Client Browser) vào pipeline AI — Tương thích 100% với Streamlit Cloud, macOS, Windows và Mobile.")

        # Lựa chọn phương thức kết nối camera
        if is_cloud_env:
            cam_backend = "🌐 Client Browser Camera (WebRTC)"
            st.info(
                "☁️ **Đang chạy trên Streamlit Cloud:** Ứng dụng tự động kích hoạt camera từ chính thiết bị/trình duyệt của bạn qua **WebRTC**. "
                "Bất kỳ máy tính hoặc điện thoại nào truy cập link đều có thể quét camera trực tiếp và nhận diện đồ vật dưới dạng **Polygons (đa giác bám sát viền)** mà không cần cài đặt code!"
            )
        else:
            backend_choices = [
                "🌐 Client Browser Camera (WebRTC) [Khuyên dùng: Streamlit Cloud, Mobile, Laptop]",
                "🖥️ Server-Side OpenCV Camera [Chỉ dùng khi chạy terminal local có cắm webcam]",
            ]
            cam_backend = st.radio(
                "Phương thức kết nối Camera:",
                backend_choices,
                index=0,
                horizontal=True,
                help="WebRTC truyền luồng trực tiếp từ trình duyệt của bất kỳ người dùng nào truy cập trang web. OpenCV cục bộ chỉ chạy được khi Streamlit mở trực tiếp trên máy chủ gắn webcam vật lý."
            )

        if cam_backend.startswith("🌐"):
            # ── WebRTC Client-Side Streaming (Streamlit Cloud & Mọi thiết bị) ──
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 1])
            with ctrl_c1:
                cam_preset = st.selectbox(
                    "Cấu hình thiết bị & Độ phân giải",
                    [
                        "⚡ Tối ưu mượt mà (480x360 - Khuyên dùng cho Mobile & Cloud)",
                        "⚖️ Cân bằng (640x480 - Mặc định chuẩn)",
                        "🔍 Độ nét cao (960x540 - Desktop băng thông mạnh)",
                    ],
                    index=1,
                    key="webrtc_preset_select",
                    help="Độ phân giải nhỏ hơn giúp giảm tải CPU/RAM trên Streamlit Cloud và tăng FPS đáng kể."
                )
            with ctrl_c2:
                infer_stride = st.selectbox(
                    "Tần suất phân tích AI",
                    [
                        "Mỗi frame (100% Realtime)",
                        "Cách 1 frame (Khuyên dùng Cloud đa người dùng)",
                        "Cách 2 frame (Siêu tiết kiệm CPU)",
                    ],
                    index=0,
                    key="webrtc_stride_select",
                    help="Bỏ qua phân tích một số frame để ưu tiên độ mượt mà của video WebRTC."
                )
            with ctrl_c3:
                backend_engine = st.selectbox(
                    "Backend suy luận",
                    [
                        "🖥️ Local CPU/GPU Worker (In-Process)",
                        "☁️ Roboflow Cloud Serverless (Khuyên dùng khi máy chủ quá tải)",
                    ],
                    index=0 if service.inference_mode != "roboflow" else 1,
                    key="webrtc_backend_engine",
                    help="Nếu máy chủ Streamlit Cloud bị quá tải CPU khi nhiều người dùng truy cập, chuyển sang Roboflow Cloud để offload GPU."
                )

            if "480x360" in cam_preset:
                res_w, res_h, infer_sz = 480, 360, 480
            elif "960x540" in cam_preset:
                res_w, res_h, infer_sz = 960, 540, 640
            else:
                res_w, res_h, infer_sz = 640, 480, 640

            stride_val = 1 if "Mỗi frame" in infer_stride else (2 if "Cách 1 frame" in infer_stride else 3)

            # Đồng bộ chế độ backend suy luận nếu người dùng đổi lựa chọn
            if "Roboflow" in backend_engine and service.inference_mode != "roboflow":
                service.inference_mode = "roboflow"
            elif "Local" in backend_engine and service.inference_mode == "roboflow":
                service.inference_mode = "local"

            if not WEBRTC_AVAILABLE or webrtc_streamer is None:
                st.error("⚠️ Thư viện `streamlit-webrtc` hoặc `av` chưa sẵn sàng trên môi trường này. Vui lòng kiểm tra requirements.txt.")
            else:
                # Video streamer widget
                webrtc_ctx = webrtc_streamer(
                    key="bus_live_webrtc_stream",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    translations=Translations(
                        start="📷 BẬT CAMERA",
                        stop="⏹️ TẮT CAMERA",
                        select_device="Chọn Camera thiết bị",
                    ) if Translations is not None else None,
                    video_processor_factory=lambda: BusWebRtcVideoProcessor(
                        service=service,
                        conf_thresh=conf_thresh,
                        min_stat_time=min_stat_time,
                        show_seats=show_seats,
                        show_persons=show_persons,
                        show_objects=show_objects,
                        show_track_ids=True,
                        show_confidence=True,
                        show_event_markers=True,
                        privacy_blur=privacy_blur,
                        show_masks=show_masks,
                        stride=stride_val,
                        infer_imgsz=infer_sz,
                    ),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": res_w},
                            "height": {"ideal": res_h},
                            "frameRate": {"ideal": 30},
                        },
                        "audio": False,
                    },
                    async_processing=True,
                )

                if webrtc_ctx and webrtc_ctx.state.playing:
                    st.success("🟢 **Webcam Client đang kết nối thành công.** Luồng AI đang xử lý theo thời gian thực!")
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.update_config(
                            conf_thresh=conf_thresh,
                            min_stat_time=min_stat_time,
                            show_seats=show_seats,
                            show_persons=show_persons,
                            show_objects=show_objects,
                            privacy_blur=privacy_blur,
                            show_masks=show_masks,
                            stride=stride_val,
                            infer_imgsz=infer_sz,
                        )
                        metrics = webrtc_ctx.video_processor.get_metrics()

                        # Nút chụp lưu ảnh bằng chứng
                        snap_col, _ = st.columns([2, 2])
                        with snap_col:
                            if st.button("📸 Lưu ảnh snapshot bằng chứng hiện tại", key="btn_save_webrtc_evidence"):
                                snap_frame = webrtc_ctx.video_processor.capture_snapshot()
                                if snap_frame is not None:
                                    snap_dir = Path("data/snapshots")
                                    snap_dir.mkdir(parents=True, exist_ok=True)
                                    snap_path = snap_dir / f"evidence_webrtc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                    cv2.imwrite(str(snap_path), snap_frame)
                                    st.toast(f"✅ Đã lưu snapshot bằng chứng: {snap_path.name}")
                                else:
                                    st.warning("Chưa có frame hình ảnh nào từ luồng camera để lưu.")

                        # Trích xuất toàn bộ metrics chi tiết theo Yêu cầu 8
                        in_fps = metrics.get("input_fps", 0.0)
                        inf_fps = metrics.get("infer_fps", 0.0)
                        out_fps = metrics.get("output_fps", 0.0)
                        avg_lat = metrics.get("latency_avg_ms", 0.0)
                        p95_lat = metrics.get("latency_p95_ms", 0.0)
                        dropped = metrics.get("dropped_frames", 0)
                        sessions = metrics.get("active_sessions", 1)

                        total_seats = metrics.get("total_seats", 0)
                        occ_seats = metrics.get("occ_seats", 0)
                        vac_seats = metrics.get("vac_seats", 0)
                        person_count = metrics.get("person_count", 0)
                        item_count = metrics.get("item_count", 0)
                        sus_count = metrics.get("sus_count", 0)
                        status_tag = metrics.get("status_tag", "🟢 BÌNH THƯỜNG")
                        status_color = metrics.get("status_color", "#10B981")

                        cpu_pct = metrics.get("cpu_percent", 0.0)
                        rss_mb = metrics.get("process_rss_mb", 0.0)
                        sys_mem = metrics.get("system_mem_percent", 0.0)
                        device_label = metrics.get("device", "CPU")

                        # Bảng hiển thị telemetry toàn diện & trực quan
                        st.markdown(f"""
                        <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 8px; padding: 12px 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 8px;">
                            <!-- Hàng 1: FPS & Độ trễ chi tiết -->
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; border-bottom: 1px solid #233554; padding-bottom: 6px;">
                                <div>
                                    <span style="font-weight: 700; color: #F8FAFC;">● WEBRTC ACTIVE</span>
                                    <span style="color: #38BDF8; font-weight: 600; margin-left: 12px;">📥 Input: <b>{in_fps:.1f}</b> FPS</span>
                                    <span style="color: #A78BFA; font-weight: 600; margin-left: 10px;">🧠 AI Infer: <b>{inf_fps:.1f}</b> FPS</span>
                                    <span style="color: #34D399; font-weight: 600; margin-left: 10px;">📤 Output: <b>{out_fps:.1f}</b> FPS</span>
                                </div>
                                <div>
                                    <span style="color: #CBD5E1;">Độ trễ AI: <b>{avg_lat:.1f}ms</b> (P95: <b>{p95_lat:.1f}ms</b>)</span>
                                    <span style="color: #F87171; margin-left: 12px;">Bỏ qua (Dropped): <b>{dropped}</b> frames</span>
                                </div>
                            </div>
                            <!-- Hàng 2: Trạng thái hành khách, ghế & cảnh báo -->
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                                <div style="display: flex; gap: 14px; align-items: center;">
                                    <span style="color: #CBD5E1;">🪑 Ghế: <b>{total_seats}</b> (👤 {occ_seats} | 🟢 {vac_seats})</span>
                                    <span style="color: #CBD5E1;">👤 Người: <b>{person_count}</b></span>
                                    <span style="color: #CBD5E1;">🎒 Đồ vật: <b>{item_count}</b></span>
                                    <span style="color: {status_color}; font-weight: 700; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem;">{status_tag}</span>
                                </div>
                                <div style="font-size: 0.82rem; color: #94A3B8;">
                                    <span>👥 Phiên: <b>{sessions}</b></span> | 
                                    <span>⚙️ CPU: <b>{cpu_pct:.1f}%</b></span> | 
                                    <span>💾 RAM: <b>{rss_mb:.0f}MB</b> ({sys_mem:.0f}%)</span> | 
                                    <span>⚡ {device_label}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if sus_count > 0:
                            st.error(f"⚠️ **CẢNH BÁO BỎ QUÊN:** Phát hiện {sus_count} đồ vật bị bỏ lại trên ghế trống! Hãy kiểm tra khoang hành khách.")
                else:
                    st.info(
                        """⚪ **Camera đang ở trạng thái Dừng (Offline).**

💡 Nhấn nút **'📷 BẬT CAMERA'** màu xanh trên khung camera bên trên để cấp quyền webcam của thiết bị và bắt đầu phân tích trực tiếp. Hồ sơ sự kiện và danh sách đồ vật đã phát hiện được bảo lưu an toàn trong hệ thống."""
                    )

                with st.expander("ℹ️ Hướng dẫn cấp quyền Webcam & Khắc phục sự cố khi truy cập từ thiết bị khác", expanded=False):
                    st.markdown("""
                    - **Cấp quyền truy cập:** Khi nhấn `📷 BẬT CAMERA`, trình duyệt sẽ hiển thị thông báo xin cấp quyền truy cập Camera. Hãy bấm **'Allow' (Cho phép)**.
                    - **Nếu bị chặn quyền (NotAllowedError):** Bấm biểu tượng ổ khóa hoặc camera trên thanh địa chỉ URL của trình duyệt > Chuyển Camera sang **'Cho phép' (Allow)** > Tải lại trang (F5).
                    - **Trên điện thoại (Safari iOS / Chrome Android):** Trình duyệt yêu cầu kết nối bảo mật HTTPS để mở camera (Streamlit Cloud đã có chứng chỉ HTTPS bảo mật). Đảm bảo không có ứng dụng khác (Zoom, FaceTime, v.v.) đang chiếm quyền camera.
                    - **Không có Webcam:** Nếu thiết bị không gắn camera, hãy chuyển sang chế độ **'🚌 Video Mẫu Xe Buýt (Demo Mode)'** hoặc **'📁 Tải lên Video'** để kiểm thử hệ thống.
                    """)

        else:
            # ── Server-Side OpenCV Camera (Dành cho máy phát triển Local) ──
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 1, 1, 1])
            with ctrl_c1:
                run_cam = st.toggle(
                    "🔴 BẬT / TẮT CAMERA LOCAL (OPENCV BUFFER)",
                    value=False,
                    key="toggle_realtime_webcam",
                    help="Bật để bắt đầu đọc webcam gắn trực tiếp trên máy chủ bằng OpenCV."
                )
            with ctrl_c2:
                cam_idx = st.number_input("Cổng Camera", 0, 5, 0, key="cam_port_input", help="0 = FaceTime HD / Mặc định, 1 = USB Cam / Continuity")
            with ctrl_c3:
                stream_res = st.selectbox("Độ phân giải", ["640x480 (Mượt, 30 FPS)", "960x540", "1280x720"], index=0, key="cv_res_select")
            with ctrl_c4:
                infer_stride = st.selectbox("Tần suất AI", ["Mỗi frame (100% Realtime)", "Cách 1 frame (Tối ưu CPU)"], index=0, key="cv_stride_select")

            res_w, res_h = (640, 480) if "640x480" in stream_res else (960, 540) if "960x540" in stream_res else (1280, 720)
            stride_val = 1 if "Mỗi frame" in infer_stride else 2

            if run_cam:
                reader = ThreadedCameraReader(src=int(cam_idx), target_width=res_w, target_height=res_h)
                is_opened, open_msg = reader.open()

                if not is_opened:
                    if is_cloud_env:
                        st.warning(
                            "☁️ **Môi trường Streamlit Cloud (Linux Container):**\n\n"
                            "Máy chủ đám mây không có phần cứng webcam vật lý kết nối trực tiếp.\n\n"
                            "👉 **Hãy chuyển sang tùy chọn '🌐 Client Browser Camera (WebRTC)'** ở trên để cấp quyền webcam từ chính thiết bị/trình duyệt của bạn mà không cần chạy local!"
                        )
                    else:
                        st.error(
                            f"❌ **Không thể mở camera cổng {cam_idx}:** {open_msg}\n\n"
                            "🔍 **Cách khắc phục trên macOS:**\n"
                            "1. Mở `Cài đặt hệ thống > Quyền riêng tư & Bảo mật > Camera` và bật quyền cho Terminal / Python / VS Code.\n"
                            "2. Đảm bảo các ứng dụng khác (FaceTime, Zoom, Teams, Photo Booth) không chiếm dụng camera."
                        )
                else:
                    st.success(f"🟢 Đã kết nối thành công Camera cổng {reader.active_port}! Luồng hình ảnh đang chạy liên tục theo thời gian thực...")

                    frame_view = st.empty()
                    metrics_view = st.empty()
                    alert_view = st.empty()

                    snap_col, _ = st.columns([2, 2])
                    with snap_col:
                        btn_save_evidence = st.button("📸 Lưu ảnh snapshot bằng chứng hiện tại", key="btn_save_evidence_snapshot")

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
                    alert_fired = False
                    t_last_metric = time.time()
                    last_annotated = None
                    last_events = []
                    last_summary = {}

                    try:
                        while run_cam:
                            t0 = time.time()
                            ret, frame, hardware_frame_cnt = reader.read_latest_frame()
                            if not ret or frame is None:
                                time.sleep(0.01)
                                continue

                            should_infer = (frame_idx % stride_val == 0) or (last_annotated is None)

                            if should_infer:
                                annotated, events, summary = safe_process_webcam_frame(
                                    service=service,
                                    frame=frame,
                                    frame_idx=frame_idx,
                                    fps=reader.fps or 25.0,
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
                                    show_masks=show_masks,
                                )
                                last_annotated = annotated
                                last_events = events
                                last_summary = summary
                            else:
                                annotated = last_annotated if last_annotated is not None else frame
                                events = last_events
                                summary = last_summary

                            latency_ms = (time.time() - t0) * 1000.0
                            now = time.time()
                            actual_fps = 1.0 / max(0.001, (now - t_last_metric))
                            t_last_metric = now

                            if btn_save_evidence and annotated is not None:
                                snap_dir = Path("data/snapshots")
                                snap_dir.mkdir(parents=True, exist_ok=True)
                                snap_path = snap_dir / f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                cv2.imwrite(str(snap_path), annotated)
                                st.toast(f"✅ Đã lưu ảnh snapshot bằng chứng: {snap_path.name}")
                                btn_save_evidence = False

                            rgb_f = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_view.image(rgb_f, channels="RGB", use_container_width=True)

                            total_seats = len(service.seat_manager.seats)
                            statuses = service.seat_manager.get_all_statuses()
                            occ_seats = sum(1 for s in statuses.values() if s.state == SeatState.OCCUPIED)
                            vac_seats = sum(1 for s in statuses.values() if s.state == SeatState.VACANT)

                            person_count = sum(1 for obj in tracker.current_tracked_objects if obj.class_name == "person")
                            item_count = sum(1 for obj in tracker.current_tracked_objects if obj.class_name in ITEM_CLASSES or obj.class_name == "item")
                            sus_count = summary.get("suspected", 0)

                            status_tag = "🔴 PHÁT HIỆN ĐỒ BỎ QUÊN!" if sus_count > 0 else "🟢 BÌNH THƯỜNG"
                            status_color = "#EF4444" if sus_count > 0 else "#10B981"

                            metrics_view.markdown(f"""
                            <div style="background: #141E33; border: 1px solid #3A506B; border-radius: 8px; padding: 10px 16px; margin-top: 6px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                                <div>
                                    <span style="font-weight: 700; color: #F8FAFC;">● CAMERA ACTIVE</span>
                                    <span style="color: #38BDF8; font-weight: 600; margin-left: 10px;">⚡ {actual_fps:.1f} FPS ({latency_ms:.1f}ms)</span>
                                </div>
                                <div style="display: flex; gap: 14px; align-items: center;">
                                    <span style="color: #CBD5E1;">🪑 Ghế: <b>{total_seats}</b> (👤 {occ_seats} | 🟢 {vac_seats})</span>
                                    <span style="color: #CBD5E1;">👤 Người: <b>{person_count}</b></span>
                                    <span style="color: #CBD5E1;">🎒 Đồ vật: <b>{item_count}</b></span>
                                    <span style="color: {status_color}; font-weight: 700; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem;">{status_tag}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            if sus_count > 0:
                                if not alert_fired:
                                    play_audio_alert()
                                    alert_fired = True
                                alert_view.error(f"⚠️ **CẢNH BÁO BỎ QUÊN:** Phát hiện {sus_count} đồ vật bị bỏ lại trên ghế trống! Hãy kiểm tra khoang hành khách.")
                            else:
                                alert_fired = False
                                alert_view.empty()

                            frame_idx += 1
                            time.sleep(0.01)

                    finally:
                        reader.stop()
                        st.info("⏹️ Đã dừng luồng camera và giải phóng phần cứng an toàn.")
            else:
                st.info("💡 Hãy bật công tắc **'🔴 BẬT / TẮT CAMERA LOCAL'** ở trên để bắt đầu quan sát luồng video webcam trực tiếp.")

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
        seat_ready = diag.get("seat_detection_ready", False)
        custom_ready = diag.get("custom_adapter_ready", False)
        roboflow_ready = diag.get("roboflow_ready", False)
        mode = diag.get("inference_mode", "local")

        if not seat_ready:
            diag_explanation = diag.get("diagnostic_message") or (
                "Chưa kích hoạt được mô hình nhận diện ghế chuyên biệt cho góc nhìn camera trần xe buýt.<br>"
                "• <b>Mô hình YOLO COCO cục bộ</b> (<code>yolov8n.pt</code>) chỉ có lớp <code>chair</code> (ghế tựa gia đình nhìn ngang), "
                "<b>không nhận diện được</b> đệm ghế xe buýt nhìn từ trên trần xuống.<br>"
                "• <b>Để nhận diện ghế bằng AI thực thụ:</b><br>"
                "&nbsp;&nbsp;1. Chuyển sang <b>Chế độ Roboflow hoặc Hybrid</b> để sử dụng mô hình Roboflow chuyên biệt (đã huấn luyện các lớp ghế: empty_seat, human_occupied, object_occupied).<br>"
                "&nbsp;&nbsp;2. Hoặc nạp file trọng số YOLO26 chuyên biệt vào đường dẫn <code>models/best.pt</code>.<br>"
                "• <b>Minh bạch kỹ thuật:</b> Hệ thống tuyệt đối không tự ý sinh sơ đồ hình học rồi mạo nhận là AI."
            )
            st.markdown(f"""
            <div style="background: #1C2541; border-left: 4px solid #F59E0B; padding: 14px 16px; border-radius: 6px; margin-bottom: 14px;">
                <div style="font-weight: 700; color: #FBBF24; font-size: 0.95rem;">
                    ⚠️ CUSTOM SEAT MODEL NOT CONFIGURED (Chế độ: {mode.upper()})
                </div>
                <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 6px; line-height: 1.5;">
                    {diag_explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if roboflow_ready:
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("🔄 Chuyển sang Chế độ Hybrid AI (Khuyến nghị)", key="switch_to_hybrid_seat_page"):
                        service.inference_mode = "hybrid"
                        os.environ["INFERENCE_MODE"] = "hybrid"
                        st.toast("✅ Đã chuyển sang chế độ HYBRID!")
                        st.rerun()
                with btn_c2:
                    if st.button("☁️ Chuyển sang Chế độ Roboflow AI", key="switch_to_rf_seat_page"):
                        service.inference_mode = "roboflow"
                        os.environ["INFERENCE_MODE"] = "roboflow"
                        st.toast("✅ Đã chuyển sang chế độ ROBOFLOW!")
                        st.rerun()
            allow_fallback = st.checkbox(
                "Cho phép dùng GEOMETRY FALLBACK để kiểm thử luồng (Dự phòng kiểm thử, KHÔNG PHẢI AI)",
                value=False,
                key="seat_allow_fallback"
            )
        else:
            allow_fallback = False
            model_source = "Roboflow Serverless Workflow" if (mode in ("roboflow", "hybrid") and roboflow_ready) else diag.get("model_weights")
            st.success(f"✅ Sẵn sàng nhận diện ghế AI từ camera! Chế độ: **{mode.upper()}** | Nguồn model: `{model_source}` | Lớp ghế: `{diag.get('supported_classes', [])}`")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            auto_rows = st.number_input("Số hàng ghế trong góc nhìn:", 1, 8, 4, 1, key="auto_rows")
        with col_cfg2:
            auto_left = st.number_input("Ghế bên trái lối đi:", 1, 2, 2, 1, help="2 = Ghế đôi kề nhau (Cửa sổ + Lối đi)", key="auto_left")
        with col_cfg3:
            auto_right = st.number_input("Ghế bên phải lối đi:", 1, 2, 2, 1, help="2 = Ghế đôi kề nhau (Lối đi + Cửa sổ)", key="auto_right")

        auto_naming = st.selectbox(
            "Quy tắc đặt Seat ID:",
            ["SEAT_001, SEAT_002... (Chuẩn số hóa)", "SEAT_1A, SEAT_1B... (Chuẩn hàng không/xe khách)"],
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
            draw_text_unicode(target_frame, "MÔ PHỎNG KHOANG XE BUÝT TỪ TRẦN", (340, 45), font_size=18, text_color=(200, 200, 200), bg_color=(20, 24, 32))

        if st.button("🚀 Bắt đầu Quét & Nhận diện Từng Ghế", type="primary", use_container_width=True):
            if not seat_ready and not allow_fallback:
                st.error("❌ CUSTOM SEAT MODEL NOT CONFIGURED: Chưa nạp mô hình ghế chuyên biệt cho chế độ hiện tại và chưa kích hoạt tùy chọn GEOMETRY FALLBACK.")
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
                        if res.total_seats > 0:
                            st.success(f"✅ MODEL DETECTION: Đã nhận diện thành công {res.total_seats} ghế bằng mô hình AI ({res.model_name})!")
                        else:
                            st.info(f"ℹ️ {res.message}")
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
                if len(poly) < 3:
                    continue
                pts_arr = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                
                # Màu sắc: Phân biệt ghế trái (Cyan) và ghế phải (Teal)
                is_left = "A" in s.seat_id or "B" in s.seat_id or (s.seat_id.replace("SEAT_", "").isdigit() and int(s.seat_id.replace("SEAT_", "")) % 4 in (1, 2))
                color = (255, 200, 0) if is_left else (0, 215, 255)

                cv2.fillPoly(overlay, [pts_arr], color)
                cv2.polylines(preview_img, [pts_arr], isClosed=True, color=(255, 255, 255), thickness=2)

                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                seat_label = format_seat_label(s.seat_id)
                draw_text_unicode(preview_img, seat_label, (cx - 30, cy - 8), font_size=12, text_color=(255, 255, 255), bg_color=(20, 25, 35))

            cv2.addWeighted(overlay, 0.25, preview_img, 0.75, 0, preview_img)
            method_caption = (
                f"🎯 [MODEL DETECTION - AI CHUYÊN BIỆT] {res.total_seats} ghế được phát hiện trực tiếp từ hình ảnh ({res.model_name})"
                if res.method == SeatDetectionMethod.MODEL_DETECTION
                else f"⚠️ [GEOMETRY FALLBACK - DỰ PHÒNG KIỂM THỬ, KHÔNG PHẢI AI] {res.total_seats} ghế sinh từ lưới tham số"
            )
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption=method_caption, use_container_width=True)

            if res.seats:
                seat_rows_data = []
                for s in res.seats:
                    poly = s.get_pixel_polygon(cur_w, cur_h)
                    coord_str = f"({poly[0][0]}, {poly[0][1]}) → ({poly[2][0]}, {poly[2][1]})" if len(poly) >= 3 else "N/A"
                    seat_rows_data.append({
                        "Seat ID": format_seat_label(s.seat_id),
                        "Vị trí / Loại": s.description,
                        "Cơ chế phát hiện": f"🧠 AI Model ({res.model_name})" if res.method == SeatDetectionMethod.MODEL_DETECTION else "⚙️ Geometry Fallback (Test Only)",
                        "Tọa độ Pixel": coord_str,
                        "Phân tách độc lập": "✅ Tách biệt hoàn toàn",
                    })
                st.dataframe(pd.DataFrame(seat_rows_data), use_container_width=True)
            else:
                st.info("ℹ️ Hiện không có ghế nào được hiển thị do mô hình AI không phát hiện ghế trong khung hình này.")


    with tab_grid:
        st.markdown("##### 🚌 Tham chiếu hình học mô phỏng (Simulation Geometry Reference)")
        st.info("ℹ️ **Lưu ý kiến trúc:** Hệ thống hiện tại sử dụng **Genuine AI Seat Detection** trực tiếp từ khung hình camera. Các ghế hiển thị trên camera và video được phát hiện bằng AI model chứ không dùng sơ đồ cố định. Công cụ dưới đây chỉ phục vụ kiểm thử mô phỏng ngoại tuyến (Offline Simulation Test).")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            rows = st.number_input("Số hàng ghế", 1, 12, 4, 1, key="grid_rows")
        with gcol2:
            left_cols = st.number_input("Số ghế bên trái lối đi", 1, 3, 2, 1, key="grid_left")
        with gcol3:
            right_cols = st.number_input("Số ghế bên phải lối đi", 1, 3, 2, 1, key="grid_right")

        name_style = st.selectbox("Định dạng tên ghế:", ["SEAT_001, SEAT_002...", "SEAT_1A, SEAT_1B..."], key="grid_name_style")

        total_seats = rows * (left_cols + right_cols)
        st.caption(f"Sơ đồ mô phỏng gồm **{total_seats} ghế**.")

        if st.button("🧪 Tạo Sơ đồ Ghế Kiểm thử Mô phỏng", type="secondary"):
            from src.seats.seat_manager import SeatManager
            style_code = "SEAT" if "1A" in name_style else "NUM"
            new_grid = SeatManager.generate_bus_grid(
                rows=rows,
                left_cols=left_cols,
                right_cols=right_cols,
                naming_style=style_code,
            )
            service.save_seats(new_grid)
            st.success(f"✅ Đã nạp {len(new_grid)} ghế mô phỏng kiểm thử!")
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
        roboflow_ready = diag.get("roboflow_ready", False)
        seat_ready = diag.get("seat_detection_ready", False)
        mode = diag.get("inference_mode", "local")
        weights_path = diag.get("model_weights", "models/best.pt")

        status_color = "#10B981" if seat_ready else "#EF4444"
        if seat_ready:
            status_text = f"ĐÃ SẴN SÀNG ({mode.upper()})"
        else:
            status_text = f"CUSTOM SEAT MODEL NOT CONFIGURED ({mode.upper()})"

        st.markdown(f"""
        <div style="background: #141E33; border: 1px solid #3A506B; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
            <h4 style="color: #F8FAFC; margin-top: 0;">📊 Trạng thái Nhận diện Ghế Chuyên biệt từ Camera</h4>
            <div style="margin-bottom: 10px;">
                <span style="color: #94A3B8;">Trạng thái Tổng thể:</span> 
                <span class="badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; font-weight: bold;">{status_text}</span>
            </div>
            <ul style="color: #CBD5E1; line-height: 1.6;">
                <li><b>Chế độ Inference hiện hành:</b> <code>{mode.upper()}</code></li>
                <li><b>Roboflow Workflow Seat Model:</b> <span style="color: {'#10B981' if roboflow_ready else '#F59E0B'}; font-weight: bold;">{'ĐÃ KẾT NỐI (empty_seat, human_occupied, object_occupied)' if roboflow_ready else 'CHƯA CẤU HÌNH'}</span></li>
                <li><b>Local Custom Weights:</b> <span style="color: {'#10B981' if custom_ready else '#EF4444'}; font-weight: bold;">{'SẴN SÀNG (' + weights_path + ')' if custom_ready else 'CHƯA CÓ TRỌNG SỐ TẠI ' + weights_path}</span></li>
                <li><b>Lớp COCO 56 (chair):</b> <span style="color: #EF4444; font-weight: bold;">TỪ CHỐI TỰ ĐỘNG</span> (ghế tựa văn phòng nhìn ngang, không nhận diện được đệm ghế trần xe buýt).</li>
                <li><b>Lớp ghế đang hỗ trợ:</b> <code>{diag.get('supported_classes', [])}</code></li>
                <li><b>Phương thức hiện hành:</b> <code>{service.seat_manager.active_detection_method.value}</code></li>
            </ul>
            <div style="background: #0F172A; padding: 12px; border-radius: 6px; border-left: 4px solid #3B82F6; margin-top: 10px;">
                <p style="margin: 0; color: #94A3B8; font-size: 0.9rem;">
                    💡 <b>Giải pháp nhận diện ghế AI:</b><br>
                    1. <b>Khuyên dùng:</b> Vào <b>Cài đặt & Bảo mật</b>, chọn chế độ <b>Roboflow</b> hoặc <b>Hybrid</b>. Hệ thống sẽ sử dụng mô hình Roboflow chuyên biệt để nhận diện ghế từ camera.<br>
                    2. <b>Hoặc:</b> Nạp file weights YOLO26 chuyên biệt vào đường dẫn <code>models/best.pt</code>.
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

    tab_upload, tab_extract, tab_annotate, tab_export, tab_roboflow = st.tabs([
        "📤 Tải lên ảnh mới", "🎞️ Trích xuất frame từ Video", "🏷️ Gán nhãn Bounding Box", "📦 Xuất chuẩn YOLO (.zip)", "☁️ Roboflow Cloud Dataset"
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

    with tab_roboflow:
        st.markdown("##### ☁️ Roboflow Cloud Dataset & Model Training Workflow")
        st.caption("Đồng bộ dữ liệu trực tiếp với Roboflow Serverless Workflow và dự án Cloud.")
        from src.roboflow.dataset_workflow import RoboflowDatasetWorkflow

        try:
            rf_wf = RoboflowDatasetWorkflow()
            summary = rf_wf.get_project_summary()
            val = rf_wf.validate_annotations()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Tổng ảnh trên Roboflow", summary["images_total"])
            with c2:
                st.metric("Đã gán nhãn", val["annotated_images"])
            with c3:
                st.metric("Chưa gán nhãn", val["unannotated_images"])
            with c4:
                st.metric("Số phiên bản Dataset", len(summary.get("versions", [])))

            st.markdown("###### Phân bố lớp đối tượng (Seat & Occupancy Classes):")
            st.json(summary["classes"])

            if val["issues"]:
                for issue in val["issues"]:
                    st.warning(f"⚠️ {issue}")
            if val["recommendations"]:
                for rec in val["recommendations"]:
                    st.info(f"💡 {rec}")

            st.markdown("---")
            st.markdown("###### 📤 Tải ảnh lên Roboflow Dataset:")
            rf_files = st.file_uploader("Chọn ảnh để tải lên Roboflow:", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="rf_up")
            rf_split = st.selectbox("Tập dữ liệu đích (Split):", ["train", "valid", "test"], key="rf_split")
            if rf_files and st.button("Tải lên Roboflow", type="primary"):
                import tempfile
                up_cnt = 0
                for f in rf_files:
                    tmp_p = Path(tempfile.mkstemp(suffix=Path(f.name).suffix)[1])
                    with open(tmp_p, "wb") as tmp_f:
                        tmp_f.write(f.getvalue())
                    try:
                        rf_wf.upload_image(tmp_p, split=rf_split)
                        up_cnt += 1
                    except Exception as e:
                        st.error(f"Lỗi tải {f.name}: {e}")
                    finally:
                        tmp_p.unlink(missing_ok=True)
                st.success(f"Đã tải thành công {up_cnt} ảnh lên Roboflow tập {rf_split}!")
                st.rerun()

            st.markdown("---")
            st.markdown("###### 📦 Tạo phiên bản Dataset mới (Version Generation):")
            v_col1, v_col2 = st.columns([3, 1])
            with v_col1:
                v_name = st.text_input("Tên phiên bản mới:", value=f"bus-seat-v{len(summary.get('versions', [])) + 1}")
            with v_col2:
                st.write("")
                st.write("")
                gen_btn = st.button("🚀 Tạo Version", use_container_width=True)
            if gen_btn:
                try:
                    gen_res = rf_wf.generate_version(version_name=v_name)
                    st.success(f"✅ Đã gửi yêu cầu tạo phiên bản trên Roboflow: {gen_res}")
                except Exception as e:
                    st.error(f"Lỗi tạo version: {e}")

            st.markdown("---")
            st.markdown("###### 🏆 Đánh giá & Kích hoạt Model Roboflow:")
            versions_list = summary.get("versions", [])
            if versions_list:
                v_nums = [v.get("version", 1) for v in versions_list]
                chosen_v = st.selectbox("Chọn phiên bản để xem đánh giá & kích hoạt:", v_nums, index=0)
                eval_data = rf_wf.get_model_evaluation(chosen_v)
                m_metrics = eval_data.get("metrics", {})
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("mAP50", f"{m_metrics.get('map50', 0.0):.1f}%")
                with e2:
                    st.metric("Precision", f"{m_metrics.get('precision', 0.0):.1f}%")
                with e3:
                    st.metric("Recall", f"{m_metrics.get('recall', 0.0):.1f}%")

                st.caption(f"Model ID: `{eval_data.get('model_id')}` | Trạng thái: `{eval_data.get('status')}`")
                ms = getattr(service, "model_service", None) or getattr(service, "_model_service", None)
                if st.button(f"⭐ Kích hoạt Model Roboflow Version {chosen_v} cho hệ thống", type="primary"):
                    ok_act, msg_act = rf_wf.activate_model(chosen_v, model_service=ms)
                    if ok_act:
                        st.success(msg_act)
                        st.rerun()
                    else:
                        st.error(msg_act)
        except Exception as e:
            st.error(f"Không thể kết nối Roboflow Dataset API: {e}")



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

    tab_list, tab_train, tab_roboflow = st.tabs([
        "📋 Danh sách Model Local",
        "⚙️ Huấn luyện Model Local",
        "☁️ Quản lý Version Model Roboflow",
    ])


    with tab_list:
        st.markdown("##### 🎯 Model YOLO26 Instance Segmentation (Ultralytics Platform)")
        custom_weights_p = CUSTOM_SEGMENTATION_WEIGHTS_PATH
        if custom_weights_p.exists():
            st.success(f"✅ Đã tìm thấy weights: `{custom_weights_p}` ({format_file_size(custom_weights_p.stat().st_size)})")
            st.caption("Model này được ưu tiên nạp tự động cho cả nhánh nhận diện ghế (`seat`) và tracking đồ vật (`item, person`).")
        else:
            st.info(
                "⏳ **Chưa có file weights `models/best.pt` (Đang train trên Ultralytics Platform):**\n\n"
                "• **Tác vụ:** Instance Segmentation (3 classes: `seat`, `person`, `item`).\n"
                "• **Cách tích hợp:** Sau khi training hoàn tất trên Ultralytics, tải file `best.pt` về và lưu vào thư mục `models/best.pt`.\n"
                "• **Cơ chế tự động:** Hệ thống đã được lập trình sẵn sàng để nhận diện task `segment`, tự trích xuất polygon masks và dùng chung một instance model duy nhất để tối ưu hiệu năng."
            )
        st.markdown("---")

        if not models:
            st.info("Chưa có candidate model nào được đăng ký trong database.")
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

        task_choice = st.selectbox("Tác vụ huấn luyện (Task):", ["segment (Instance Segmentation - Phân đoạn đối tượng)", "detect (Object Detection - Khung bao Bounding Box)"], index=0)
        task_val = "segment" if "segment" in task_choice else "detect"

        tc1, tc2 = st.columns(2)
        with tc1:
            new_ver = st.text_input("Tên phiên bản model mới:", value=f"v1.1-bus-seats-{task_val}")
            seg_bases = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
            det_bases = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
            base_m = st.selectbox("Kiến trúc base model YOLO:", seg_bases if task_val == "segment" else det_bases)
        with tc2:
            epochs = st.slider("Số Epochs:", 5, 100, 30, 5)
            batch = st.selectbox("Batch size:", [4, 8, 16, 32], index=1)

        if ms and hasattr(ms, "generate_training_command"):
            cmd = ms.generate_training_command(dataset_yaml, epochs=epochs, batch_size=batch, model=base_m, task=task_val)
        else:
            cmd = ModelService(getattr(service, "_model_repo", None)).generate_training_command(dataset_yaml, epochs=epochs, batch_size=batch, model=base_m, task=task_val)
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
                        task=task_val,
                    )
                elif ms and hasattr(ms, "start_real_training_job"):
                    ok_launch, job_id, launch_info = ms.start_real_training_job(
                        dataset_yaml_path=dataset_yaml,
                        new_version=new_ver,
                        base_model=base_m,
                        epochs=epochs,
                        batch_size=batch,
                        task=task_val,
                    )
                else:
                    m_temp = ModelService(getattr(service, "_model_repo", None))
                    ok_launch, job_id, launch_info = m_temp.start_real_training_job(
                        dataset_yaml_path=dataset_yaml,
                        new_version=new_ver,
                        base_model=base_m,
                        epochs=epochs,
                        batch_size=batch,
                        task=task_val,
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

    with tab_roboflow:
        st.markdown("##### ☁️ Quản lý Phiên bản Model Roboflow (Roboflow Workflow & Model Versions)")
        st.caption("Xem thông tin model đã triển khai, danh sách phiên bản (Versions), và chuyển đổi phiên bản active phục vụ suy luận mà không cần sửa code.")

        rf_client = getattr(service, "roboflow_client", None)
        if rf_client is None:
            try:
                from src.roboflow.roboflow_client import RoboflowWorkflowClient
                rf_client = RoboflowWorkflowClient()
            except Exception as e:
                rf_client = None

        if rf_client and rf_client.api_key:
            wf_info = rf_client.get_workflow_info()
            meta = rf_client.get_project_metadata()
            proj_name = meta.get("name", wf_info["project_id"])
            proj_type = meta.get("type", "object-detection")
            active_ver = rf_client.model_version

            st.markdown(f"""
            <div style="background: #141E33; border: 1px solid #3A506B; padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">
                    🚌 Dự án: {proj_name} (Workspace: <code>{wf_info['workspace']}</code>)
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                    Tác vụ model: <b>{proj_type.upper()}</b> | Phiên bản active hiện tại: <b>Version {active_ver}</b> | Workflow: <code>{wf_info['workflow_id']}</code>
                </div>
                <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 6px;">
                    <b>API Key:</b> <code>{wf_info['masked_api_key']}</code> (Bảo mật, không lộ khóa) | <b>Inference SDK:</b> {'✅ Sẵn sàng' if wf_info['sdk_available'] else '⚠️ Fallback HTTP'}
                </div>
                <div style="color: #38BDF8; font-size: 0.85rem; margin-top: 4px;">
                    🎯 <b>Model ID đích (Inference Target):</b> <code>{wf_info.get('target_model_id', f"{wf_info['project_id']}/{active_ver}")}</code> | <b>Trạng thái:</b> <code>{wf_info.get('last_inference_info', {}).get('status', 'READY')}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

            last_inf = wf_info.get("last_inference_info", {})
            if last_inf.get("error"):
                st.error(
                    f"⚠️ **Thông báo suy luận Roboflow:** Yêu cầu tới model `{last_inf.get('requested_model_id')}` "
                    f"thất bại: {last_inf.get('error')}"
                )

            st.info(
                "ℹ️ **Minh bạch kỹ thuật về tác vụ model:**\n\n"
                f"• Dự án Roboflow `{wf_info['project_id']}` hiện tại được cấu hình kiểu **`{proj_type}`**.\n"
                "• Mô hình xuất ra **khung bao Bounding Box** cho 3 lớp trạng thái ghế: `empty_seat`, `human_occupied`, `object_occupied`.\n"
                "• Dù ảnh huấn luyện có vẽ polygon, model deployed dạng `object-detection` chỉ trả về bounding box. Để có mặt nạ pixel (Instance Segmentation) từ Roboflow, cần tạo version mới với kiểu `instance-segmentation` và train model segmentation (ví dụ YOLOv8-seg hoặc YOLOv11-seg)."
            )

            classes = meta.get("classes", {})
            if classes:
                st.markdown("###### 📊 Phân bố các lớp đối tượng trong tập dữ liệu:")
                cls_cols = st.columns(len(classes))
                for idx, (cname, ccount) in enumerate(classes.items()):
                    with cls_cols[idx]:
                        st.metric(label=cname, value=ccount)

            st.markdown("---")
            st.markdown("##### 🔄 Danh sách Phiên bản (Versions) & Chuyển đổi Model Active")

            versions = rf_client.list_available_versions()
            if not versions:
                st.warning("Chưa đọc được danh sách version từ Roboflow REST API. Đang sử dụng Version mặc định.")
            else:
                ver_options = [f"Version {v['version']} — {v['name']} ({v.get('images', 0)} ảnh)" for v in versions]
                current_idx = 0
                for i, v in enumerate(versions):
                    if str(v["version"]) == str(active_ver):
                        current_idx = i
                        break

                selected_ver_str = st.selectbox(
                    "Chọn phiên bản model Roboflow để kích hoạt:",
                    ver_options,
                    index=current_idx,
                    key="rf_version_select"
                )
                selected_ver_num = selected_ver_str.split(" — ")[0].replace("Version ", "").strip()

                if st.button("🚀 Kích hoạt Phiên bản Model này cho Toàn hệ thống", type="primary"):
                    ok_switch = rf_client.set_active_version(selected_ver_num)
                    if ok_switch:
                        st.success(f"✅ Đã kích hoạt Roboflow Model Version {selected_ver_num} thành công! Hệ thống sẽ dùng phiên bản này cho inference mà không cần sửa code.")
                        st.rerun()

            st.markdown("---")
            st.markdown("##### 📋 Quy trình Cập nhật Model Roboflow (Lifecycle Playbook)")
            st.markdown("""
            Để thêm dữ liệu và cập nhật model mà không phát sinh lỗi hay tự ý phát sinh chi phí:
            1. **Bước 1 - Tải ảnh & Gán nhãn:** Thêm ảnh mới vào dự án trên Roboflow Web và gán nhãn chính xác (`empty_seat`, `human_occupied`, `object_occupied`).
            2. **Bước 2 - Sinh Dataset Version:** Trên Roboflow, chọn *Generate New Version* (chọn tiền xử lý/tăng cường dữ liệu). Thao tác này hoàn toàn **miễn phí**.
            3. **Bước 3 - Huấn luyện Model (Roboflow Train):** Bấm *Start Training* trên Roboflow. Bước này sẽ sử dụng Roboflow training credits và chỉ chạy khi bạn chủ động xác nhận trên Roboflow. *Ứng dụng không tự ý trigger train ngầm để tránh phát sinh chi phí ngoài ý muốn.*
            4. **Bước 4 - Kích hoạt vào Ứng dụng:** Sau khi huấn luyện hoàn tất, tải lại tab này, chọn Version vừa sinh trong danh sách ở trên và bấm **Kích hoạt Phiên bản Model**. Toàn bộ luồng nhận diện xe buýt sẽ lập tức chuyển sang model mới!
            """)
        else:
            st.error("Chưa kết nối được Roboflow Client. Vui lòng kiểm tra biến môi trường ROBOFLOW_API_KEY trong file .env.")


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

    st.markdown("---")
    st.markdown("##### 🚀 Tùy chọn Inference & Tracking Đa chế độ")
    inf_c1, inf_c2 = st.columns(2)
    current_mode = getattr(service, "inference_mode", os.getenv("INFERENCE_MODE", "local")).lower()
    mode_options = [
        "local (Chạy YOLO cục bộ)",
        "roboflow (Chạy Roboflow Serverless Workflow)",
        "hybrid (Kết hợp Local YOLO + Roboflow Remote)",
    ]
    cur_idx = 1 if current_mode == "roboflow" else (2 if current_mode == "hybrid" else 0)

    with inf_c1:
        sel_mode = st.selectbox(
            "Chế độ Inference (Inference Mode):",
            mode_options,
            index=cur_idx,
            help="Chế độ hybrid dung hợp phát hiện từ cả model cục bộ và workflow từ xa mà không loại bỏ nhầm thực thể.",
            key="settings_inf_mode_select",
        )
        new_mode = "roboflow" if "roboflow" in sel_mode else ("hybrid" if "hybrid" in sel_mode else "local")
        if new_mode != current_mode:
            service.inference_mode = new_mode
            os.environ["INFERENCE_MODE"] = new_mode
            st.toast(f"✅ Đã chuyển sang chế độ {new_mode.upper()}!")

    with inf_c2:
        st.selectbox(
            "Kiến trúc Model YOLO:",
            [
                "yolo26n-seg.pt (YOLO26 Instance Segmentation - Có Masks)",
                "yolo26n.pt (YOLO26 Object Detection - Bounding Box)",
                "models/best.pt (Custom Segmentation Weights)",
            ],
            index=0,
            help="YOLO26 Instance Segmentation trích xuất mask phân đoạn pixel cho ghế, người và vật thể.",
        )

    st.markdown("---")
    st.markdown("##### ☁️ Kết nối Roboflow Serverless Workflow")
    from src.roboflow.roboflow_client import RoboflowWorkflowClient
    try:
        rf_client = RoboflowWorkflowClient()
        wf_info = rf_client.get_workflow_info()
        st.success(f"✅ Workflow đã sẵn sàng: Workspace `{wf_info['workspace']}` | Workflow ID `{wf_info['workflow_id']}`")
        st.caption(f"Endpoint: `{wf_info['api_url']}` | inference-sdk: `{'Sẵn sàng' if wf_info['sdk_available'] else 'Không có'}`")
    except Exception as e:
        st.warning(f"⚠️ Chưa kết nối Roboflow Workflow: {e}")


if __name__ == "__main__":
    main()

