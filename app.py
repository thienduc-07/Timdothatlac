"""
app.py
Giao diện Streamlit cho hệ thống Lost & Found AI trên xe buýt:
- Tích hợp nhận biết che khuất (Occlusion-Aware).
- Hỗ trợ Live Webcam trực tiếp + Video upload file.
- Quản lý sơ đồ ghế xe buýt linh hoạt (8, 12, 16, 24... ghế) và hiển thị trực tiếp số ghế trên cam.
- Quản lý State Machine vòng đời sự kiện: OBSERVED, ITEM_LEFT, SUSPECTED,
  OCCLUDED, RECHECK, REVIEW_REQUIRED, RESOLVED, DISMISSED.
"""

import cv2
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.services.analysis_service import AnalysisService
from src.tracking.object_tracker import ObjectTracker
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.seats.seat_config import SeatConfigManager
from src.models import SeatConfig
from src.utils.file_utils import format_file_size, format_duration
from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_STATIONARY_TIME_SEC,
    STATIONARY_MOVEMENT_THRESHOLD,
    ITEM_CLASSES,
    CLASS_NAMES_VI,
    SUPPORTED_VIDEO_EXTENSIONS,
    EventStatus,
    EVENT_STATUS_META,
)

# ──────────────────────────────────────────────
# Cấu hình trang
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lost & Found AI (Bus)",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS tùy chỉnh thẩm mỹ cao
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }

    /* Stat Cards */
    .stat-card {
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stat-card h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .stat-card p { margin: 4px 0 0 0; font-size: 0.85rem; opacity: 0.95; }

    .stat-card.blue { background: linear-gradient(135deg, #4338ca 0%, #3b82f6 100%); }
    .stat-card.green { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .stat-card.orange { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); }
    .stat-card.purple { background: linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%); }
    .stat-card.red { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
    .stat-card.pink { background: linear-gradient(135deg, #be185d 0%, #ec4899 100%); }

    /* Tiêu đề trang */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }

    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    """Khởi tạo AnalysisService một lần duy nhất."""
    service = AnalysisService()
    service.initialize()
    return service


def get_status_badge_html(status: str) -> str:
    """Trả về HTML badge hiển thị trạng thái."""
    meta = EVENT_STATUS_META.get(status, {"label": status, "color": "#6b7280", "badge": "•"})
    color = meta["color"]
    label = f"{meta['badge']} {meta['label']}"
    return f'<span class="status-badge" style="background-color: {color};">{label}</span>'


def main():
    service = get_service()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🚌 Bus Lost & Found AI")
        st.markdown("*Phát hiện đồ bỏ quên & Nhận biết che khuất*")
        st.markdown("---")

        page = st.radio(
            "📋 Điều hướng",
            [
                "🏠 Tổng quan",
                "🎬 Phân tích video & Webcam",
                "⚠️ Quản lý sự kiện",
                "🪑 Vùng ghế (Seat ROI)",
                "📊 Lịch sử phân tích",
                "⚙️ Cấu hình hệ thống",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        # Trạng thái Model & Vùng ghế
        model_info = service.get_model_info()
        if model_info.get("loaded"):
            st.success(f"✅ AI Model: {model_info['model_name']}")
        else:
            st.error("❌ Model AI chưa tải")

        if service.is_seat_configured():
            st.info(f"🪑 Sơ đồ: {len(service.get_seats())} ghế")
        else:
            st.warning("⚠️ Chưa cấu hình vùng ghế (Seat ROI)")

    # ── Routing ──
    if "Tổng quan" in page:
        page_dashboard(service)
    elif "Phân tích" in page:
        page_analyze(service)
    elif "Quản lý sự kiện" in page:
        page_events(service)
    elif "Vùng ghế" in page:
        page_seats(service)
    elif "Lịch sử" in page:
        page_history(service)
    elif "Cấu hình" in page:
        page_settings(service)


# ══════════════════════════════════════════════
# TRANG 1: TỔNG QUAN
# ══════════════════════════════════════════════
def page_dashboard(service: AnalysisService):
    st.markdown('<div class="page-title">🏠 Tổng quan hệ thống xe buýt</div>', unsafe_allow_html=True)

    if not service.is_seat_configured():
        st.warning(
            "⚠️ **Chưa cấu hình vùng ghế (Seat ROI)!** Hệ thống camera trần cần biết vị trí các ghế để "
            "phát hiện đồ bỏ quên và theo dõi hiện tượng che khuất chính xác. Hãy vào mục **'Vùng ghế (Seat ROI)'** "
            "để thiết lập sơ đồ ghế xe buýt."
        )

    stats = service.get_dashboard_stats()
    seat_count = len(service.get_seats())

    # Thẻ thống kê
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="stat-card blue">'
            f'<h2>{stats.get("total_analyses", 0)}</h2>'
            f'<p>Video đã phân tích</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card green">'
            f'<h2>{seat_count}</h2>'
            f'<p>Ghế đã cấu hình</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card red">'
            f'<h2>{stats.get("suspected_events", 0)}</h2>'
            f'<p>Nghi ngờ bỏ quên</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card purple">'
            f'<h2>{stats.get("occluded_events", 0)}</h2>'
            f'<p>Đang bị che khuất</p></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="stat-card pink">'
            f'<h2>{stats.get("review_required_events", 0)}</h2>'
            f'<p>Cần kiểm tra</p></div>',
            unsafe_allow_html=True,
        )

    # Hiển thị danh sách ghế hiện tại
    if seat_count > 0:
        st.markdown(f"### 🪑 Trạng thái {seat_count} vùng ghế xe buýt")
        seats = service.get_seats()
        seat_cols = st.columns(min(4, max(1, len(seats))))
        statuses = service.seat_manager.get_all_statuses()

        for idx, seat in enumerate(seats):
            with seat_cols[idx % len(seat_cols)]:
                status = statuses.get(seat.seat_id)
                state_str = status.state.value if status else "chưa phân tích"
                state_icon = {"vacant": "🟢 Trống", "occupied": "🔴 Có người", "unknown": "⚪ Chưa rõ"}.get(
                    state_str, "⚪ Chưa rõ"
                )
                st.info(f"**{seat.seat_id}**\n\n{seat.description or 'Ghế hành khách'}\n\n*Trạng thái: {state_icon}*")

    # Sự kiện gần đây
    st.markdown("### 📋 Sự kiện gần đây")
    recent = service.get_recent_events(10)
    if not recent:
        st.info("Chưa có sự kiện nào được ghi nhận. Hãy phân tích video hoặc bật live webcam để bắt đầu.")
    else:
        for evt in recent:
            vi_name = CLASS_NAMES_VI.get(evt["class_name"], evt["class_name"])
            seat_info = f"tại **{evt['seat_id']}**" if evt.get("seat_id") else ""
            badge_html = get_status_badge_html(evt["status"])
            occlusion_text = (
                f" — Đã che: {evt.get('occlusion_duration', 0.0):.1f}s"
                if evt.get("occlusion_duration", 0.0) > 0
                else ""
            )

            st.markdown(
                f"{badge_html} &nbsp; **{vi_name}** {seat_info} (ID: `{evt['event_id']}`) — "
                f"Nguồn: *{evt['video_name']}*{occlusion_text} — "
                f"*{evt['timestamp'][:19]}*",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TRANG 2: PHÂN TÍCH VIDEO & LIVE WEBCAM
# ══════════════════════════════════════════════
def page_analyze(service: AnalysisService):
    st.markdown('<div class="page-title">🎬 Phân tích Video & Live Webcam</div>', unsafe_allow_html=True)

    if not service.is_ready:
        st.error("❌ Model AI chưa được tải.")
        return

    # Cảnh báo nếu chưa có Seat ROI
    if not service.is_seat_configured():
        st.warning(
            "⚠️ Chưa có tọa độ vùng ghế (Seat ROI) nào được thiết lập! "
            "Bạn có thể vào trang 'Vùng ghế (Seat ROI)' để tự động sinh số ghế xe buýt trước."
        )

    tab_video, tab_webcam = st.tabs(["📁 Phân tích từ File Video", "🎥 Live Camera / Webcam"])

    # ──── TAB 1: FILE VIDEO ────
    with tab_video:
        extensions = [ext.replace(".", "") for ext in SUPPORTED_VIDEO_EXTENSIONS]
        uploaded = st.file_uploader(
            "📁 Chọn file video camera trần xe buýt",
            type=extensions,
            help=f"Định dạng hỗ trợ: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}",
        )

        if uploaded is not None:
            video_path = service.save_uploaded_video(uploaded)
            if video_path:
                meta = service.get_video_metadata(video_path)
                if meta.is_valid:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tên file:** {uploaded.name}")
                        st.write(f"**Dung lượng:** {format_file_size(uploaded.size)}")
                    with col2:
                        st.write(f"**Độ phân giải:** {meta.width} × {meta.height}")
                        st.write(f"**Thời lượng:** {format_duration(meta.duration_seconds)} ({meta.fps:.1f} FPS)")

                    st.video(video_path)

                    st.markdown("### ⚙️ Tham số phân tích")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        conf_threshold = st.slider("Ngưỡng confidence YOLO", 0.1, 0.9, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, key="file_conf")
                        min_time = st.number_input("Thời gian đứng yên trên ghế trống để nghi ngờ (giây)", 1.0, 120.0, MIN_STATIONARY_TIME_SEC, 1.0, key="file_time")
                    with col_b:
                        move_thresh = st.number_input("Ngưỡng di chuyển coi là đứng yên (pixel)", 5.0, 200.0, STATIONARY_MOVEMENT_THRESHOLD, 5.0, key="file_move")
                        selected_classes = st.multiselect(
                            "Lớp đồ vật cần theo dõi",
                            options=ITEM_CLASSES,
                            default=ITEM_CLASSES,
                            format_func=lambda x: CLASS_NAMES_VI.get(x, x),
                            key="file_classes",
                        )

                    st.markdown("---")
                    if "analyzing" not in st.session_state:
                        st.session_state.analyzing = False

                    if st.button("🚀 Bắt đầu phân tích video file", disabled=st.session_state.analyzing, type="primary", use_container_width=True):
                        st.session_state.analyzing = True
                        progress_bar = st.progress(0, text="Đang bắt đầu phân tích...")

                        def update_progress(current, total):
                            if total > 0:
                                pct = min(current / total, 1.0)
                                progress_bar.progress(pct, text=f"Đang xử lý frame {current}/{total}...")

                        try:
                            result = service.analyze_video(
                                video_path=video_path,
                                confidence_threshold=conf_threshold,
                                min_stationary_time=min_time,
                                movement_threshold=move_thresh,
                                watched_classes=selected_classes or ITEM_CLASSES,
                                progress_callback=update_progress,
                            )

                            progress_bar.progress(1.0, text="Hoàn thành phân tích!")

                            if result.success:
                                st.success("✅ Phân tích video hoàn tất!")

                                res_c1, res_c2, res_c3 = st.columns(3)
                                with res_c1:
                                    st.metric("Frame đã xử lý", f"{result.total_frames_processed:,}")
                                with res_c2:
                                    st.metric("Vật thể phát hiện", f"{result.total_objects_detected:,}")
                                with res_c3:
                                    st.metric("Sự kiện ghi nhận", f"{result.total_events_created}")

                                if result.output_video_path and Path(result.output_video_path).exists():
                                    st.markdown("### 🎥 Video đã gắn nhãn số ghế & trạng thái")
                                    st.video(result.output_video_path)

                                    with open(result.output_video_path, "rb") as f:
                                        st.download_button(
                                            "📥 Tải video kết quả",
                                            data=f,
                                            file_name=Path(result.output_video_path).name,
                                            mime="video/mp4",
                                        )

                                if result.events:
                                    st.markdown("### ⚠️ Danh sách sự kiện phát hiện")
                                    for evt in result.events:
                                        vi_name = CLASS_NAMES_VI.get(evt.class_name, evt.class_name)
                                        badge = get_status_badge_html(evt.status)
                                        st.markdown(
                                            f"{badge} &nbsp; **{vi_name}** | Vị trí: **{evt.seat_id or 'Chưa gán'}** | "
                                            f"Event ID: `{evt.event_id}` | Bị che: `{evt.occlusion_duration:.1f}s` | "
                                            f"*{evt.notes}*",
                                            unsafe_allow_html=True,
                                        )
                                        if evt.snapshot_path and Path(evt.snapshot_path).exists():
                                            st.image(evt.snapshot_path, caption=f"Snapshot {evt.event_id}", width=400)
                            else:
                                st.error(f"❌ Phân tích thất bại: {result.error_message}")
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
                        finally:
                            st.session_state.analyzing = False

    # ──── TAB 2: LIVE WEBCAM ────
    with tab_webcam:
        st.markdown("### 🎥 Phân tích Trực tiếp qua Camera / Webcam")
        st.info("💡 Hệ thống hỗ trợ nhận diện trực tiếp từ **Webcam máy tính (FaceTime HD / USB Camera)** và **Camera trình duyệt web**, tự động gán nhãn số ghế và theo dõi che khuất theo thời gian thực!")

        webcam_mode = st.radio(
            "Chọn chế độ Camera:",
            ["🔴 Live Stream liên tục (Khuyên dùng khi chạy trên MacBook / PC)", "📸 Chụp & Quét qua Camera trình duyệt (Hỗ trợ tốt nhất khi deploy Online)"],
            horizontal=True,
        )

        wcol1, wcol2 = st.columns(2)
        with wcol1:
            if "Live Stream" in webcam_mode:
                cam_idx = st.number_input("Cổng Camera (0 = Webcam máy tính / FaceTime HD, 1, 2 = Camera ngoài)", min_value=0, max_value=10, value=0)
            webcam_conf = st.slider("Ngưỡng confidence YOLO", 0.1, 0.9, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, key="cam_conf")
        with wcol2:
            webcam_min_time = st.number_input("Thời gian đứng yên để nghi ngờ bỏ quên (giây)", 1.0, 60.0, 4.0, 1.0, key="cam_time")
            webcam_classes = st.multiselect(
                "Lớp đồ vật theo dõi",
                options=ITEM_CLASSES,
                default=ITEM_CLASSES,
                format_func=lambda x: CLASS_NAMES_VI.get(x, x),
                key="cam_classes",
            )

        if "Live Stream" in webcam_mode:
            st.caption("⚡ Mẹo trên MacBook: Cổng `0` là camera FaceTime HD có sẵn của máy. Đảm bảo bạn đã cấp quyền Camera cho Terminal hoặc Trình duyệt.")
            run_webcam = st.toggle("🔴 BẬT / TẮT LIVE WEBCAM", value=False)

            if run_webcam:
                # Thử mở camera với cổng đã chọn, kèm fallback AVFoundation cho macOS
                cap = cv2.VideoCapture(int(cam_idx))
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_AVFOUNDATION)

                if not cap.isOpened():
                    st.error(f"❌ Không thể kết nối với Camera cổng {cam_idx}. Vui lòng kiểm tra quyền Camera trong System Settings > Privacy & Security > Camera.")
                else:
                    frame_placeholder = st.empty()
                    status_placeholder = st.empty()

                    tracker = ObjectTracker()
                    event_manager = EventManager()
                    event_detector = ForgottenItemDetector(
                        min_stationary_time=webcam_min_time,
                        movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
                        watched_classes=webcam_classes or ITEM_CLASSES,
                        seat_manager=service.seat_manager,
                        event_manager=event_manager,
                    )

                    frame_idx = 0
                    while run_webcam:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("⚠️ Mất tín hiệu từ camera.")
                            break

                        annotated, active_events, summary = service.process_webcam_frame(
                            frame=frame,
                            frame_idx=frame_idx,
                            fps=15.0,
                            tracker=tracker,
                            event_detector=event_detector,
                            event_manager=event_manager,
                            confidence_threshold=webcam_conf,
                        )

                        # Đổi BGR sang RGB để hiển thị trên web
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        status_placeholder.markdown(
                            f"**⏱️ Frame:** `{frame_idx:,}` | "
                            f"**👁️ Đang theo dõi:** `{len(active_events)}` | "
                            f"**⚠️ Nghi ngờ:** `{summary.get('suspected', 0)}` | "
                            f"**🛡️ Bị che khuất:** `{summary.get('occluded', 0)}` | "
                            f"**📋 Cần kiểm tra:** `{summary.get('review_required', 0)}`"
                        )

                        frame_idx += 1

                    cap.release()
        else:
            # Chế độ Browser Camera Input (Chuẩn web Streamlit, dùng trực tiếp camera MacBook trong trình duyệt)
            st.markdown("##### 📸 Chụp ảnh từ Camera MacBook / Trình duyệt để phân tích ngay:")
            camera_photo = st.camera_input("Chụp ảnh từ Camera")

            if camera_photo is not None:
                import numpy as np
                bytes_data = camera_photo.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                tracker = ObjectTracker()
                event_manager = EventManager()
                event_detector = ForgottenItemDetector(
                    min_stationary_time=webcam_min_time,
                    movement_threshold=STATIONARY_MOVEMENT_THRESHOLD,
                    watched_classes=webcam_classes or ITEM_CLASSES,
                    seat_manager=service.seat_manager,
                    event_manager=event_manager,
                )

                with st.spinner("🔍 AI đang phân tích ảnh từ camera..."):
                    annotated, active_events, summary = service.process_webcam_frame(
                        frame=cv2_img,
                        frame_idx=1,
                        fps=1.0,
                        tracker=tracker,
                        event_detector=event_detector,
                        event_manager=event_manager,
                        confidence_threshold=webcam_conf,
                    )

                rgb_result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb_result, caption="Kết quả phân tích từ Camera", channels="RGB", use_container_width=True)

                st.success(
                    f"✅ Đã phân tích xong! Đang theo dõi: **{len(active_events)}** vật thể | "
                    f"Nghi ngờ bỏ quên: **{summary.get('suspected', 0)}** | "
                    f"Che khuất: **{summary.get('occluded', 0)}**"
                )


# ══════════════════════════════════════════════
# TRANG 3: QUẢN LÝ SỰ KIỆN
# ══════════════════════════════════════════════
def page_events(service: AnalysisService):
    st.markdown('<div class="page-title">⚠️ Quản lý vòng đời sự kiện đồ vật</div>', unsafe_allow_html=True)

    events = service.get_all_events()
    if not events:
        st.info("Chưa có sự kiện nào. Hãy phân tích video hoặc chạy webcam để phát hiện sự kiện.")
        return

    # Bộ lọc trạng thái
    status_filter = st.selectbox(
        "Lọc theo trạng thái sự kiện",
        [
            "Tất cả",
            "⚠️ Nghi ngờ bỏ quên (suspected)",
            "🛡️ Đang bị che khuất (occluded)",
            "📋 Cần nhân viên kiểm tra (review_required)",
            "🪑 Ghế trống còn đồ (item_left)",
            "👁️ Đang quan sát (observed/monitoring)",
            "✅ Đã xác nhận (confirmed/resolved)",
            "❌ Đã loại bỏ (dismissed)",
        ],
    )

    status_code_map = {
        "⚠️ Nghi ngờ bỏ quên (suspected)": "suspected",
        "🛡️ Đang bị che khuất (occluded)": "occluded",
        "📋 Cần nhân viên kiểm tra (review_required)": "review_required",
        "🪑 Ghế trống còn đồ (item_left)": "item_left",
        "👁️ Đang quan sát (observed/monitoring)": "observed",
        "✅ Đã xác nhận (confirmed/resolved)": "confirmed",
        "❌ Đã loại bỏ (dismissed)": "dismissed",
    }

    if status_filter != "Tất cả":
        target = status_code_map.get(status_filter, "")
        if target == "confirmed":
            events = [e for e in events if e["status"] in ("confirmed", "resolved")]
        elif target == "observed":
            events = [e for e in events if e["status"] in ("observed", "monitoring")]
        else:
            events = [e for e in events if e["status"] == target]

    st.markdown(f"**Tổng: {len(events)} sự kiện**")

    for evt in events:
        vi_name = CLASS_NAMES_VI.get(evt["class_name"], evt["class_name"])
        seat_str = f"| Vị trí: {evt.get('seat_id')}" if evt.get("seat_id") else ""
        badge_html = get_status_badge_html(evt["status"])

        with st.expander(f"**{evt['event_id']}** — {vi_name} {seat_str} — [{evt['status'].upper()}]"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Trạng thái:** {badge_html}", unsafe_allow_html=True)
                st.write(f"**Mã sự kiện (Event ID):** `{evt['event_id']}`")
                st.write(f"**Số ghế (Seat ID):** `{evt.get('seat_id', 'Chưa rõ')}`")
                st.write(f"**Loại vật thể:** {vi_name} ({evt['class_name']})")
                st.write(f"**Confidence:** {evt['confidence']:.0%}")
                st.write(f"**Thời gian quan sát:** {evt.get('observed_duration', evt.get('duration_seconds', 0.0)):.1f}s")
            with col2:
                st.write(f"**Thời gian bị che khuất:** {evt.get('occlusion_duration', 0.0):.1f}s")
                st.write(f"**Phát hiện lần đầu:** {evt.get('first_seen_timestamp', evt['timestamp'])[:19]}")
                st.write(f"**Nhìn thấy lần cuối:** {evt.get('last_seen_timestamp', evt['timestamp'])[:19]}")
                st.write(f"**Nguồn:** {evt['video_name']}")
                st.write(f"**Thời điểm video:** {format_duration(evt['video_time_sec'])}")

            if evt.get("notes"):
                st.info(f"📝 {evt['notes']}")

            # Lịch sử chuyển trạng thái
            history_json = evt.get("event_history")
            if history_json:
                try:
                    history = json.loads(history_json) if isinstance(history_json, str) else history_json
                    if history:
                        st.markdown("**📜 Lịch sử chuyển đổi trạng thái:**")
                        for h in history:
                            st.write(f"- `{h.get('from_state')}` ➔ `{h.get('to_state')}` (t={h.get('time_sec', 0):.1f}s): *{h.get('reason')}*")
                except Exception:
                    pass

            # Snapshot
            if evt.get("snapshot_path") and Path(evt["snapshot_path"]).exists():
                st.image(evt["snapshot_path"], width=480)

            # Nút hành động cho nhân viên
            b1, b2, b3 = st.columns(3)
            with b1:
                if evt["status"] not in ("confirmed", "resolved"):
                    if st.button("✅ Xác nhận đồ bỏ quên", key=f"conf_{evt['event_id']}"):
                        service.update_event_status(evt["event_id"], "confirmed")
                        st.success("Đã xác nhận sự kiện!")
                        st.rerun()
            with b2:
                if evt["status"] not in ("review_required",):
                    if st.button("📋 Yêu cầu kiểm tra", key=f"rev_{evt['event_id']}"):
                        service.update_event_status(evt["event_id"], "review_required")
                        st.info("Đã chuyển sang cần kiểm tra.")
                        st.rerun()
            with b3:
                if evt["status"] not in ("dismissed",):
                    if st.button("❌ Loại bỏ cảnh báo", key=f"dism_{evt['event_id']}"):
                        service.update_event_status(evt["event_id"], "dismissed")
                        st.warning("Đã loại bỏ cảnh báo.")
                        st.rerun()


# ══════════════════════════════════════════════
# TRANG 4: VÙNG GHẾ (SEAT ROI)
# ══════════════════════════════════════════════
def page_seats(service: AnalysisService):
    st.markdown('<div class="page-title">🪑 Quản lý Vùng ghế & Đánh số ghế xe buýt</div>', unsafe_allow_html=True)

    st.markdown("""
    Hệ thống hỗ trợ cấu hình số lượng ghế linh hoạt cho **mọi loại xe buýt** (16 chỗ, 29 chỗ, 45 chỗ...).
    Khi nhận diện người hay đồ vật, hệ thống sẽ **gán số ghế và vẽ ký hiệu số ghế trực tiếp lên camera và video**.
    """)

    # Bộ sinh nhanh sơ đồ ghế
    st.markdown("### 🚌 Bộ sinh nhanh sơ đồ ghế xe buýt")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        num_rows = st.number_input("Số hàng ghế (tầm nhìn camera)", min_value=1, max_value=12, value=4)
    with gc2:
        col_layout = st.selectbox("Cách bố trí ghế mỗi hàng", [
            "1 ghế trái - 1 ghế phải (2 ghế/hàng)",
            "2 ghế trái - 1 ghế phải (3 ghế/hàng)",
            "2 ghế trái - 2 ghế phải (4 ghế/hàng)",
        ])
    with gc3:
        naming_choice = st.selectbox("Ký hiệu số ghế hiển thị trên cam/video", [
            "Ghế 1A, 1B, 2A, 2B... (Khuyên dùng)",
            "SEAT_01, SEAT_02, SEAT_03...",
        ])

    left_c = 2 if "2 ghế trái" in col_layout else 1
    right_c = 2 if "2 ghế phải" in col_layout else 1
    naming_style = "GHE" if "Ghế" in naming_choice else "SEAT"
    total_generated = num_rows * (left_c + right_c)

    if st.button(f"⚡ Tạo & Lưu ngay sơ đồ {total_generated} ghế xe buýt", type="primary", use_container_width=True):
        new_grid = SeatConfigManager.generate_bus_grid(
            rows=num_rows,
            left_cols=left_c,
            right_cols=right_c,
            naming_style=naming_style,
        )
        service.save_seats(new_grid)
        st.success(f"Đã tạo thành công sơ đồ {total_generated} ghế xe buýt!")
        st.rerun()

    st.markdown("---")
    seats = service.get_seats()

    if not seats:
        st.warning("⚠️ **Chưa có vùng ghế nào được cấu hình.** Hãy nhấn nút tạo nhanh ở trên.")
    else:
        st.markdown(f"### 📋 Danh sách {len(seats)} ghế đang hoạt động trên hệ thống")
        for seat in seats:
            with st.expander(f"**{seat.seat_id}** — {seat.description}"):
                st.write(f"**Ký hiệu số ghế:** `{seat.seat_id}`")
                st.write(f"**Tọa độ chuẩn hóa:** {'Có [0.0 - 1.0]' if seat.is_normalized else 'Pixel tuyệt đối'}")
                st.write(f"**Tọa độ polygon:** {json.dumps([[round(pt[0], 3), round(pt[1], 3)] for pt in seat.polygon])}")

    st.markdown("---")
    st.markdown("### ✏️ Tùy chỉnh chi tiết tọa độ (JSON Editor)")

    current_dict = [
        {
            "seat_id": s.seat_id,
            "description": s.description,
            "is_normalized": s.is_normalized,
            "polygon": [[round(pt[0], 4), round(pt[1], 4)] for pt in s.polygon],
        }
        for s in seats
    ]
    raw_json = st.text_area(
        "Cấu hình các ghế (JSON):",
        value=json.dumps(current_dict, indent=2, ensure_ascii=False),
        height=240,
    )

    if st.button("💾 Lưu cấu hình JSON thủ công"):
        try:
            parsed = json.loads(raw_json)
            new_seats = []
            for item in parsed:
                seat_id = str(item.get("seat_id", "")).strip()
                poly_raw = item.get("polygon", [])
                if seat_id and len(poly_raw) >= 3:
                    new_seats.append(SeatConfig(
                        seat_id=seat_id,
                        polygon=[(float(pt[0]), float(pt[1])) for pt in poly_raw],
                        is_normalized=bool(item.get("is_normalized", False)),
                        description=str(item.get("description", "")),
                    ))
            service.save_seats(new_seats)
            st.success(f"Đã lưu thành công {len(new_seats)} vùng ghế!")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi cú pháp JSON: {e}")


# ══════════════════════════════════════════════
# TRANG 5: LỊCH SỬ PHÂN TÍCH
# ══════════════════════════════════════════════
def page_history(service: AnalysisService):
    st.markdown('<div class="page-title">📊 Lịch sử phân tích video</div>', unsafe_allow_html=True)

    analyses = service.get_all_analyses()
    if not analyses:
        st.info("Chưa có video nào được phân tích.")
        return

    for item in analyses:
        with st.expander(f"📹 {item['video_name']} — {item['created_at'][:19]}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tổng frame", f"{item['total_frames']:,}")
            with c2:
                st.metric("Vật thể phát hiện", f"{item['total_objects']:,}")
            with c3:
                st.metric("Sự kiện ghi nhận", f"{item['total_events']}")

            if item.get("output_video_path") and Path(item["output_video_path"]).exists():
                st.video(item["output_video_path"])


# ══════════════════════════════════════════════
# TRANG 6: CẤU HÌNH
# ══════════════════════════════════════════════
def page_settings(service: AnalysisService):
    st.markdown('<div class="page-title">⚙️ Cấu hình hệ thống</div>', unsafe_allow_html=True)

    model_info = service.get_model_info()

    st.markdown("### 🤖 Thông tin mô hình AI")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Tên model:** {model_info.get('model_name', 'N/A')}")
        st.write(f"**Trạng thái:** {'✅ Đã tải' if model_info.get('loaded') else '❌ Chưa tải'}")
        st.write(f"**Thiết bị:** {model_info.get('device', 'N/A')}")
    with c2:
        st.write(f"**Số lớp hỗ trợ:** {model_info.get('total_classes', 0)}")
        tracked = model_info.get("tracked_class_names", [])
        vi_tracked = [CLASS_NAMES_VI.get(c, c) for c in tracked]
        st.write(f"**Lớp đồ vật theo dõi:** {', '.join(vi_tracked)}")

    st.markdown("### ⚠️ Giới hạn hệ thống & Tuyên bố trách nhiệm")
    st.warning("""
    1. **Không xác định chủ sở hữu:** Hệ thống chỉ phát hiện sự tồn tại của vật thể đứng yên trên ghế;
       tuyệt đối không thể xác định chính xác ai là chủ nhân của món đồ từ camera thông thường.
    2. **Xử lý che khuất (Occlusion-aware):** Khi hành khách ngồi đè hoặc che khuất món đồ,
       hệ thống duy trì sự kiện ở trạng thái `OCCLUDED` hoặc `RECHECK`, tuyệt đối không tự ý xóa cảnh báo.
    3. **Giới hạn góc nhìn camera:** Nếu hành động lấy đồ xảy ra hoàn toàn trong điểm mù hoặc bị che khuất
       100% thời gian, hệ thống sẽ đưa sự kiện về trạng thái `REVIEW_REQUIRED` để nhân viên nhà xe kiểm tra thực tế.
    4. **Cấu hình tham số:** Các ngưỡng thời gian (đứng yên, xác nhận có người/trống, kiểm tra lại)
       là các tham số ban đầu cần được tinh chỉnh theo video thực tế của từng dòng xe buýt.
    """)


if __name__ == "__main__":
    main()
