"""
src/video package
"""

from src.video.camera_service import (
    CameraSourceType,
    CameraStreamManager,
    ThreadedCameraReader,
    ensure_demo_video_exists,
    generate_bus_simulation_frame,
    get_demo_ground_truth,
)

__all__ = [
    "CameraSourceType",
    "CameraStreamManager",
    "ThreadedCameraReader",
    "ensure_demo_video_exists",
    "generate_bus_simulation_frame",
    "get_demo_ground_truth",
]
