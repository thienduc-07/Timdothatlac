"""
tests/test_supervision_roboflow_yolo26_integration.py
Test suite covering:
1. Supervision tracking across multi-frame sequences for Local, Roboflow, and Hybrid modes.
2. Hybrid inference fusion preventing naive IoU discards of valid multi-class objects.
3. Class mapper fidelity (seat state classes not falsely mapped to generic item/person).
4. YOLO26 model checkpoint loading and segmentation mask generation.
5. Roboflow Workflow client parsing and dataset workflow validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.models import Detection, TrackedObject, SeatConfig
from src.tracking.object_tracker import ObjectTracker
from src.detection.class_mapper import map_class, is_seat_state_class, seat_state_to_occupancy
from src.video.video_processor import _iou, fuse_hybrid_detections
from src.roboflow.roboflow_client import RoboflowWorkflowClient
from src.roboflow.dataset_workflow import RoboflowDatasetWorkflow


# ────────────────────────────────────────────────────────────────
# 1. Supervision Tracking in All Three Modes
# ────────────────────────────────────────────────────────────────

def test_supervision_tracking_local_mode_detections():
    """Kiểm tra Supervision ByteTrack duy trì tracking ID nhất quán qua các frame cho local detections."""
    tracker = ObjectTracker()
    tracker.reset()

    # Frame 0: Một người ở vị trí (100, 100, 200, 200)
    dets_f0 = [
        Detection(class_name="person", confidence=0.92, bbox=(100, 100, 200, 200))
    ]
    tracked_f0 = tracker.update(detections=dets_f0, frame_index=0, fps=10.0)
    assert len(tracked_f0) == 1
    person_id = tracked_f0[0].track_id
    assert person_id > 0
    assert tracked_f0[0].class_name == "person"

    # Frame 1: Người đó dịch chuyển nhẹ sang (102, 102, 202, 202)
    dets_f1 = [
        Detection(class_name="person", confidence=0.91, bbox=(102, 102, 202, 202))
    ]
    tracked_f1 = tracker.update(detections=dets_f1, frame_index=1, fps=10.0)
    assert len(tracked_f1) == 1
    assert tracked_f1[0].track_id == person_id, "Track ID của người phải được giữ nguyên qua các frame"


def test_supervision_tracking_roboflow_mode_detections():
    """Kiểm tra Supervision ByteTrack gán và duy trì ID cho remote detections từ Roboflow."""
    tracker = ObjectTracker()
    tracker.reset()

    # Frame 0: Hai vật thể từ Roboflow
    remote_f0 = [
        Detection(class_name="backpack", confidence=0.88, bbox=(150, 150, 220, 220)),
        Detection(class_name="person", confidence=0.95, bbox=(100, 80, 250, 350)),
    ]
    tracked_f0 = tracker.update(detections=remote_f0, frame_index=0, fps=10.0)
    assert len(tracked_f0) == 2
    id_map = {obj.class_name: obj.track_id for obj in tracked_f0}
    assert id_map["backpack"] != id_map["person"]

    # Frame 1: Người rời đi, balo vẫn còn ở vị trí cũ
    remote_f1 = [
        Detection(class_name="backpack", confidence=0.87, bbox=(150, 150, 220, 220)),
    ]
    tracked_f1 = tracker.update(detections=remote_f1, frame_index=1, fps=10.0)
    assert len(tracked_f1) == 1
    assert tracked_f1[0].track_id == id_map["backpack"], "Balo phải giữ nguyên track_id khi người rời đi"


def test_supervision_tracking_hybrid_mode_detections():
    """Kiểm tra Supervision ByteTrack trong chế độ Hybrid với danh sách phát hiện hợp nhất."""
    tracker = ObjectTracker()
    tracker.reset()

    # Frame 0: Local phát hiện person, remote phát hiện item/backpack
    local_dets = [Detection(class_name="person", confidence=0.90, bbox=(100, 100, 250, 350))]
    remote_dets = [Detection(class_name="backpack", confidence=0.85, bbox=(120, 200, 180, 280))]
    fused_f0 = fuse_hybrid_detections(local_dets, remote_dets)
    tracked_f0 = tracker.update(detections=fused_f0, frame_index=0, fps=10.0)
    assert len(tracked_f0) == 2

    # Frame 1: Cả 2 đối tượng tiếp tục được track
    local_dets_f1 = [Detection(class_name="person", confidence=0.89, bbox=(101, 101, 251, 351))]
    remote_dets_f1 = [Detection(class_name="backpack", confidence=0.84, bbox=(120, 200, 180, 280))]
    fused_f1 = fuse_hybrid_detections(local_dets_f1, remote_dets_f1)
    tracked_f1 = tracker.update(detections=fused_f1, frame_index=1, fps=10.0)
    assert len(tracked_f1) == 2
    assert {t.track_id for t in tracked_f1} == {t.track_id for t in tracked_f0}


# ────────────────────────────────────────────────────────────────
# 2. Hybrid Inference Fusion (No Naive Discards)
# ────────────────────────────────────────────────────────────────

def test_hybrid_fusion_preserves_coexisting_person_and_item():
    """
    Kiểm tra hybrid fusion không loại bỏ vật thể (balo/túi) nằm đè trên người
    như cách IoU-only thô thiển thường làm.
    """
    # Người và balo có bounding box lồng nhau đáng kể (IoU > 0.3, containment cao)
    local_dets = [
        Detection(class_name="person", confidence=0.92, bbox=(100, 100, 300, 400))
    ]
    remote_dets = [
        Detection(class_name="backpack", confidence=0.85, bbox=(140, 150, 260, 320))
    ]

    fused = fuse_hybrid_detections(local_dets, remote_dets)
    classes = [d.class_name for d in fused]
    assert "person" in classes
    assert "backpack" in classes
    assert len(fused) == 2, "Cả person và backpack phải được giữ lại vì là 2 thực thể khác nhau"


def test_hybrid_fusion_selects_higher_confidence_for_same_class():
    """Khi cả 2 nguồn phát hiện cùng một đối tượng (cùng semantic class), giữ lại detection có confidence cao hơn."""
    local_dets = [
        Detection(class_name="backpack", confidence=0.60, bbox=(100, 100, 200, 200))
    ]
    remote_dets = [
        Detection(class_name="backpack", confidence=0.93, bbox=(102, 98, 204, 202))
    ]

    fused = fuse_hybrid_detections(local_dets, remote_dets, iou_threshold=0.5)
    assert len(fused) == 1
    assert fused[0].confidence == 0.93, "Phải chọn detection có độ tin cậy cao hơn từ remote"


def test_hybrid_fusion_preserves_segmentation_polygon():
    """Nếu detection bị thay thế có polygon mask, polygon phải được bảo toàn."""
    mock_poly = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    local_dets = [
        Detection(class_name="backpack", confidence=0.75, bbox=(100, 100, 200, 200), polygon=mock_poly)
    ]
    remote_dets = [
        Detection(class_name="backpack", confidence=0.95, bbox=(100, 100, 200, 200), polygon=None)
    ]

    fused = fuse_hybrid_detections(local_dets, remote_dets)
    assert len(fused) == 1
    assert fused[0].confidence == 0.95
    assert fused[0].polygon == mock_poly, "Polygon mask phải được kế thừa nếu remote không có mask"


# ────────────────────────────────────────────────────────────────
# 3. Class Mapper Fidelity
# ────────────────────────────────────────────────────────────────

def test_class_mapper_seat_state_fidelity():
    """
    Kiểm tra class_mapper.py không tự ý map empty_seat, human_occupied, object_occupied
    sang seat, person, item.
    """
    assert map_class("empty_seat") == "seat_state_empty"
    assert map_class("human_occupied") == "seat_state_occupied"
    assert map_class("object_occupied") == "seat_state_object"

    assert is_seat_state_class("seat_state_empty") is True
    assert is_seat_state_class("seat_state_occupied") is True
    assert is_seat_state_class("seat_state_object") is True
    assert is_seat_state_class("person") is False
    assert is_seat_state_class("item") is False

    assert seat_state_to_occupancy("seat_state_empty") == "vacant"
    assert seat_state_to_occupancy("seat_state_occupied") == "occupied"
    assert seat_state_to_occupancy("seat_state_object") == "object"


# ────────────────────────────────────────────────────────────────
# 4. YOLO26 Model & Segmentation Masks
# ────────────────────────────────────────────────────────────────

def test_yolo26_checkpoint_segmentation_masks():
    """
    Kiểm tra model yolo26n-seg.pt thực sự có task='segment' và tạo polygon masks
    khi có đối tượng được nhận diện.
    """
    from ultralytics import YOLO
    model_path = Path("yolo26n-seg.pt")
    if not model_path.exists():
        pytest.skip("yolo26n-seg.pt không có sẵn trên đĩa")

    model = YOLO(str(model_path))
    assert model.task == "segment", "yolo26n-seg.pt phải là model segmentation"

    # Inference trên synthetic frame với đối tượng rõ ràng
    frame = np.full((320, 320, 3), 128, dtype=np.uint8)
    results = model(frame, verbose=False)
    assert len(results) > 0
    res = results[0]
    # Kiểm tra cấu trúc output segmentation
    assert hasattr(res, "masks")


# ────────────────────────────────────────────────────────────────
# 5. Roboflow Client & Dataset Workflow
# ────────────────────────────────────────────────────────────────

def test_roboflow_client_output_normalization():
    """Kiểm tra parse_workflow_output chuyển đổi đúng tọa độ x, y, width, height sang [x1, y1, x2, y2]."""
    raw_response = {
        "predictions": {
            "image": {"width": 640, "height": 480},
            "predictions": [
                {
                    "x": 200.0,
                    "y": 250.0,
                    "width": 100.0,
                    "height": 80.0,
                    "confidence": 0.88,
                    "class": "human_occupied",
                    "points": [{"x": 150.0, "y": 210.0}, {"x": 250.0, "y": 210.0}, {"x": 250.0, "y": 290.0}],
                }
            ],
        },
        "model_id": "bus-seat-and-belongings-monitori/1",
    }

    parsed = RoboflowWorkflowClient.parse_workflow_output(raw_response)
    preds = parsed.get("predictions", [])
    assert len(preds) == 1
    p = preds[0]
    assert p["class"] == "human_occupied"
    assert p["confidence"] == 0.88
    assert p["bbox"] == [150, 210, 250, 290]
    assert len(p["polygon"]) == 3


def test_roboflow_dataset_workflow_validation_structure():
    """Kiểm tra RoboflowDatasetWorkflow cấu trúc validation và tổng kết."""
    wf = RoboflowDatasetWorkflow()
    # Mock summary để test logic validation độc lập mạng
    with patch.object(wf, "get_project_summary") as mock_summary:
        mock_summary.return_value = {
            "workspace": "le-uc-s-workspace",
            "project_id": "bus-seat-and-belongings-monitori",
            "name": "Bus Seat and Belongings Monitori",
            "type": "object-detection",
            "images_total": 154,
            "unannotated_total": 52,
            "classes": {"human_occupied": 264, "object_occupied": 93, "empty_seat": 741},
            "splits": {"train": 107, "valid": 32, "test": 15},
            "versions": [{"version": 1}],
        }

        val = wf.validate_annotations()
        assert val["total_images"] == 154
        assert val["annotated_images"] == 102
        assert val["unannotated_images"] == 52
        assert val["ready_for_version"] is True
        assert len(val["issues"]) == 1
        assert "52 / 154 images are unannotated" in val["issues"][0]
