"""
src/components/hybrid_camera/__init__.py
Streamlit Custom Component cho Hybrid AI Camera:
- Chạy trực tiếp trên trình duyệt của người dùng (Client-Side WebGPU / WASM).
- Tích hợp điều khiển camera 1 nút duy nhất (Single-Button Toggle: BẬT CAMERA <-> TẮT CAMERA).
- Giao tiếp 2 chiều với Streamlit qua Streamlit Component API.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "frontend"

# Khai báo component với Streamlit
_hybrid_camera_func = components.declare_component(
    "hybrid_camera",
    path=str(_COMPONENT_DIR),
)


def hybrid_camera(
    mode: str = "auto",
    model_url: str = "/app/static/models/yolo26n-seg.onnx",
    fallback_model_url: str = "https://raw.githubusercontent.com/thienduc-07/Timdothatlac/main/static/models/yolo26n-seg.onnx",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    show_masks: bool = True,
    show_seats: bool = True,
    show_persons: bool = True,
    show_objects: bool = True,
    target_width: int = 640,
    target_height: int = 480,
    height: int = 720,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hiển thị component Hybrid Camera Client-Side.

    Parameters:
    - mode: "auto", "on_device", "server"
    - model_url: Đường dẫn tải model ONNX chính (được phục vụ qua static serving)
    - fallback_model_url: Đường dẫn tải dự phòng có CORS toàn cầu (GitHub Raw)
    - conf_threshold: Ngưỡng tin cậy (0.1 - 1.0)
    - iou_threshold: Ngưỡng NMS IoU
    - show_masks: Bật/tắt hiển thị polygon segmentation masks
    - height: Chiều cao iframe tính bằng pixel (mặc định 720px để tránh bị collapse)
    - key: Streamlit widget key

    Returns:
    - Dict chứa telemetry thời gian thực từ client (fps, latency, active_backend, detections, v.v.)
    """
    default_val = {
        "status": "idle",
        "active_backend": "client_webgpu",
        "is_running": False,
        "input_fps": 0.0,
        "infer_fps": 0.0,
        "display_fps": 0.0,
        "latency_ms": 0.0,
        "person_count": 0,
        "item_count": 0,
        "error": None,
    }

    component_value = _hybrid_camera_func(
        mode=mode,
        modelUrl=model_url,
        fallbackModelUrl=fallback_model_url,
        modelSize=11078627,
        modelSha256="02d840d18222fdfbe32e93879f5b4d520f1b6b6af87222cc7b03ca1cce0372b3",
        confThreshold=conf_threshold,
        iouThreshold=iou_threshold,
        showMasks=show_masks,
        showSeats=show_seats,
        showPersons=show_persons,
        showObjects=show_objects,
        targetWidth=target_width,
        targetHeight=target_height,
        height=height,
        key=key,
        default=default_val,
    )

    if not isinstance(component_value, dict):
        return default_val
    return component_value
