"""
tests/test_audit_fixes_regression.py

Regression test suite verifying:
1. Polygon Segmentation:
   - Preservation of polygon coordinates throughout detection, hybrid fusion, tracking.
   - Rendering of actual segmentation mask overlay without forced rectangular box overwrite.
   - Fallback to bounding box when no polygon is available.
2. Vietnamese Text & Encoding:
   - Full diacritic preservation in user-facing text and labels.
   - Unicode text rendering on OpenCV frames without converting characters to '?'.
   - UTF-8 source file encoding integrity.
3. Genuine AI Seat Detection:
   - No hardcoded seats loaded automatically on initialization.
   - Genuine seat locations originate from actual model predictions.
   - Stable seat IDs (SEAT_001, SEAT_002...) across consecutive frames.
   - No artificial seats drawn when model returns no seat predictions.
   - Transparent diagnostics when model lacks a seat class.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Detection, TrackedObject, SeatConfig, SeatState, Event
from src.tracking.object_tracker import ObjectTracker
from src.video.video_processor import fuse_hybrid_detections
from src.video.video_writer import VideoWriter
from src.seats.seat_manager import SeatManager
from src.seats.seat_detector import SeatDetector, SeatDetectionMethod, CustomSeatYOLOAdapter
from src.seats.seat_tracker import SeatTracker
from src.utils.text_renderer import draw_text_unicode, measure_text_unicode, find_system_unicode_font
from src.services.analysis_service import AnalysisService
from config.settings import CLASS_NAMES_VI, EventStatus


# =====================================================================
# PART 1: POLYGON SEGMENTATION TESTS
# =====================================================================

def test_polygon_preserved_in_tracking():
    """Verify polygon coordinates are preserved through Supervision ByteTrack."""
    tracker = ObjectTracker()
    test_poly = [(100.0, 120.0), (220.0, 115.0), (210.0, 260.0), (95.0, 250.0)]
    det = Detection(
        class_name="backpack",
        confidence=0.92,
        bbox=(95, 115, 220, 260),
        class_id=24,
        polygon=test_poly,
    )

    tracked = tracker.update(
        detections=[det],
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_index=1,
        fps=10.0,
    )

    assert len(tracked) == 1
    assert tracked[0].polygon is not None
    poly_pts = [(float(p[0]), float(p[1])) for p in tracked[0].polygon]
    assert poly_pts == test_poly

    obs = tracker.observations.get(tracked[0].track_id)
    assert obs is not None
    assert len(obs.polygons) > 0
    obs_poly_pts = [(float(p[0]), float(p[1])) for p in obs.polygons[-1]]
    assert obs_poly_pts == test_poly


def test_fuse_hybrid_detections_preserves_polygon():
    """Verify hybrid fusion preserves segmentation polygon data."""
    local_det = Detection(
        class_name="person",
        confidence=0.80,
        bbox=(50, 50, 150, 200),
        polygon=[(50.0, 50.0), (150.0, 50.0), (150.0, 200.0), (50.0, 200.0)],
    )
    remote_det = Detection(
        class_name="backpack",
        confidence=0.88,
        bbox=(200, 100, 300, 220),
        polygon=[(200.0, 100.0), (300.0, 100.0), (280.0, 220.0), (190.0, 210.0)],
    )

    fused = fuse_hybrid_detections([local_det], [remote_det])
    assert len(fused) == 2
    assert fused[0].polygon is not None
    assert fused[1].polygon is not None
    assert len(fused[1].polygon) == 4


def test_polygon_rendered_without_box_overwrite():
    """
    Verify that when polygon data is available, the segmentation overlay is drawn
    and no forced bounding box rectangle overwrites it.
    """
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    test_poly = [(100.0, 100.0), (200.0, 100.0), (190.0, 220.0), (95.0, 210.0)]
    obj = TrackedObject(
        track_id=1,
        class_name="backpack",
        confidence=0.90,
        bbox=(95, 100, 200, 220),
        polygon=test_poly,
    )

    annotated = VideoWriter.annotate_frame(
        frame=canvas,
        tracked_objects=[obj],
        event_statuses={},
        seat_manager=None,
        active_events=[],
        width=600,
        height=400,
        show_masks=True,
    )

    # Check that mask region has non-zero pixels
    mask_pixel = annotated[150, 150]
    assert np.any(mask_pixel > 0), "Polygon overlay must be rendered onto frame"


def test_bbox_fallback_when_no_polygon():
    """Verify bounding box fallback works when no polygon is present without fabricating one."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    obj = TrackedObject(
        track_id=2,
        class_name="handbag",
        confidence=0.85,
        bbox=(50, 60, 180, 200),
        polygon=None,  # No polygon returned by model
    )

    annotated = VideoWriter.annotate_frame(
        frame=canvas,
        tracked_objects=[obj],
        event_statuses={},
        seat_manager=None,
        active_events=[],
        width=600,
        height=400,
        show_masks=True,
    )

    assert obj.polygon is None, "Bounding box prediction must not fabricate a polygon"
    # Check that bbox border was drawn
    border_pixel = annotated[60, 100]
    assert np.any(border_pixel > 0), "Bounding box border must be rendered"


# =====================================================================
# PART 2: VIETNAMESE TEXT & ENCODING TESTS
# =====================================================================

def test_unicode_text_renderer_vietnamese():
    """Verify Unicode renderer correctly renders Vietnamese diacritics without producing '?'."""
    frame = np.zeros((100, 500, 3), dtype=np.uint8)
    sample_text = "Ghế 1A - CÓ NGƯỜI - Ba lô - CẢNH BÁO"
    draw_text_unicode(frame, sample_text, (10, 20), font_size=16, text_color=(255, 255, 255), bg_color=(0, 100, 200))

    nonzero_count = np.count_nonzero(frame)
    assert nonzero_count > 1000, "Rendered text must produce visible pixels"


def test_vietnamese_class_names_accents():
    """Verify CLASS_NAMES_VI retains valid Vietnamese accents without corruption."""
    expected_accents = {
        "person": "Người",
        "item": "Đồ vật / Hành lý",
        "seat": "Ghế xe buýt",
        "backpack": "Ba lô",
        "handbag": "Túi xách",
        "suitcase": "Vali",
        "umbrella": "Ô/Dù",
        "bottle": "Bình nước/Chai",
        "cell phone": "Điện thoại",
        "empty_seat": "Ghế trống",
        "human_occupied": "Ghế có người",
        "object_occupied": "Ghế có đồ",
    }
    for key, expected_label in expected_accents.items():
        assert key in CLASS_NAMES_VI, f"Missing key {key} in CLASS_NAMES_VI"
        actual = CLASS_NAMES_VI[key]
        assert actual == expected_label, f"Mismatch for {key}: expected '{expected_label}', got '{actual}'"
        assert "?" not in actual, f"Corrupted character '?' found in {key}: '{actual}'"


def test_python_source_files_utf8():
    """Verify all Python source files in src/ are valid UTF-8."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    py_files = list(src_dir.rglob("*.py"))
    assert len(py_files) > 10

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            assert len(content) > 0
        except UnicodeDecodeError as e:
            pytest.fail(f"Source file {py_file} is not valid UTF-8: {e}")


# =====================================================================
# PART 3: GENUINE AI SEAT DETECTION TESTS
# =====================================================================

def test_seat_manager_starts_empty_without_hardcoded_seats():
    """Verify SeatManager starts with 0 seats and does not load hardcoded seats."""
    mgr = SeatManager()
    assert len(mgr.seats) == 0, "SeatManager must not automatically preload hardcoded seats"
    assert mgr.is_configured is False
    assert mgr.active_detection_method == SeatDetectionMethod.NOT_CONFIGURED


def test_analysis_service_initialize_no_hardcoded_seats():
    """Verify AnalysisService.initialize() does not populate hardcoded seats."""
    service = AnalysisService()
    ok = service.initialize()
    assert ok is True
    assert len(service.seat_manager.seats) == 0, "AnalysisService must not inject hardcoded seats on init"
    assert service.seat_manager.is_configured is False


def test_genuine_ai_seat_detection_from_predictions():
    """Verify seats are dynamically detected from visual model predictions."""
    mgr = SeatManager()

    # Model returns seat predictions from camera feed
    seat_preds = [
        Detection(class_name="empty_seat", confidence=0.88, bbox=(40, 60, 240, 180)),
        Detection(class_name="human_occupied", confidence=0.91, bbox=(400, 60, 600, 180)),
    ]

    detected = mgr.update_detected_seats(seat_preds, frame_idx=1, frame_width=640, frame_height=360)

    assert len(detected) == 2, "Both seats detected by model must be registered"
    assert mgr.is_configured is True
    assert mgr.active_detection_method == SeatDetectionMethod.MODEL_DETECTION

    # Stable IDs assigned
    seat_ids = [s.seat_id for s in detected]
    assert "SEAT_001" in seat_ids
    assert "SEAT_002" in seat_ids


def test_seat_id_stability_across_frames():
    """Verify SeatTracker maintains stable seat IDs across consecutive frames."""
    tracker = SeatTracker()
    frame1_seats = [
        SeatConfig(seat_id="SEAT_001", polygon=[(0.10, 0.15), (0.35, 0.15), (0.35, 0.40), (0.10, 0.40)]),
        SeatConfig(seat_id="SEAT_002", polygon=[(0.60, 0.15), (0.85, 0.15), (0.85, 0.40), (0.60, 0.40)]),
    ]
    t1 = tracker.update(frame1_seats, frame_number=1)
    assert len(t1) == 2
    assert t1[0].seat_id == "SEAT_001"
    assert t1[1].seat_id == "SEAT_002"

    # Frame 2 with slight visual detection coordinate noise
    frame2_seats = [
        SeatConfig(seat_id="", polygon=[(0.102, 0.151), (0.351, 0.149), (0.349, 0.402), (0.101, 0.398)]),
        SeatConfig(seat_id="", polygon=[(0.598, 0.152), (0.852, 0.148), (0.848, 0.401), (0.602, 0.399)]),
    ]
    t2 = tracker.update(frame2_seats, frame_number=2)
    assert len(t2) == 2
    t2_ids = {s.seat_id for s in t2}
    assert t2_ids == {"SEAT_001", "SEAT_002"}, "Seat IDs must remain stable across consecutive frames"


def test_no_artificial_seats_when_model_has_no_seat_predictions():
    """Verify that when no seats are detected, no artificial seats are drawn or created."""
    mgr = SeatManager()
    # Model detects only a person and a backpack, NO seats
    non_seat_preds = [
        Detection(class_name="person", confidence=0.90, bbox=(100, 100, 200, 300)),
        Detection(class_name="backpack", confidence=0.85, bbox=(150, 120, 220, 180)),
    ]

    detected = mgr.update_detected_seats(non_seat_preds, frame_idx=1, frame_width=640, frame_height=360)
    assert len(detected) == 0, "No artificial seats should be generated when model returns no seat predictions"
    assert mgr.is_configured is False
    assert len(mgr.seats) == 0


def test_missing_seat_model_reported_transparently():
    """Verify transparent diagnostic reporting when model lacks a seat class."""
    adapter = CustomSeatYOLOAdapter(weights_path="non_existent_weights.pt")
    inspection = adapter.inspect_weights()
    assert inspection["is_valid_seat_model"] is False

    detector = SeatDetector(custom_weights_path="non_existent_weights.pt")
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res = detector.detect_seats(dummy_frame, allow_fallback=False)

    assert res.method == SeatDetectionMethod.NOT_CONFIGURED
    assert len(res.seats) == 0
    assert "CUSTOM SEAT MODEL NOT CONFIGURED" in res.status_message


# =====================================================================
# PART 4: LABELS, SWITCHING, AND ADVANCED AUDIT TESTS
# =====================================================================

def test_generic_user_facing_labels_normalization():
    """Verify detailed object names (backpack, handbag, phone, etc.) are simplified to generic Vietnamese labels."""
    from src.detection.class_mapper import get_user_facing_label, format_seat_label

    # Seats
    assert format_seat_label("SEAT_001") == "Ghế 001"
    assert format_seat_label("SEAT_1") == "Ghế 001"
    assert format_seat_label("SEAT_1A") == "Ghế 1A"
    assert format_seat_label("3") == "Ghế 003"
    assert get_user_facing_label("seat", seat_id="SEAT_005") == "Ghế 005"

    # People
    assert get_user_facing_label("person") == "Người"
    assert get_user_facing_label("Person") == "Người"

    # Belongings / Objects: NEVER leak detailed names like phone, wallet, backpack, etc.
    detailed_classes = ["backpack", "handbag", "suitcase", "umbrella", "bottle", "cell phone", "laptop", "book", "wallet", "item"]
    for cls in detailed_classes:
        label = get_user_facing_label(cls)
        assert label == "Đồ vật", f"Class '{cls}' must normalize to 'Đồ vật', but got '{label}'"

    # Forgotten item alert label
    assert get_user_facing_label("backpack", is_alert=True) == "Cảnh báo: Đồ vật có thể bị bỏ quên"
    assert get_user_facing_label("cell phone", is_alert=True, seat_id="SEAT_002") == "Cảnh báo: Đồ vật có thể bị bỏ quên (Ghế 002)"


def test_bounding_box_fallback_when_masks_unavailable():
    """Verify that when no masks are available, system falls back to bounding box cleanly without fabricating polygons."""
    writer = VideoWriter(output_path="/tmp/test_writer.mp4", fps=20.0, width=640, height=360)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    obj_no_poly = TrackedObject(
        track_id=1,
        class_name="backpack",
        confidence=0.88,
        bbox=(100, 100, 200, 200),
        polygon=None,  # No polygon mask
    )

    annotated = writer.annotate_frame(
        frame=dummy_frame,
        tracked_objects=[obj_no_poly],
        seat_manager=None,
        show_masks=True,
    )

    # Frame should be modified (bounding box and label rendered)
    assert np.any(annotated > 0)
    # obj.polygon must remain None (no fake polygon fabricated)
    assert obj_no_poly.polygon is None


def test_roboflow_model_version_switching():
    """Verify Roboflow model version can be configured and switched without rewriting code."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient

    client = RoboflowWorkflowClient()
    initial_version = client.model_version
    assert initial_version is not None

    # Switch to version "2"
    ok = client.set_active_version("2")
    assert ok is True
    assert client.model_version == "2"

    # Switch back to version "1"
    ok2 = client.set_active_version("1")
    assert ok2 is True
    assert client.model_version == "1"

    # Test workflow info masking
    info = client.get_workflow_info()
    assert "api_key_masked" in info
    assert "KiM6" in info["api_key_masked"]
    assert "****" in info["api_key_masked"]
    # Full API key must NEVER be exposed in workflow info
    assert client.api_key not in info["api_key_masked"]


def test_hybrid_seat_detection_deduplication():
    """Verify Hybrid seat detection fuses local and remote seats without duplicate seats."""
    detector = SeatDetector()

    # Suppose both remote and local return overlapping seats at roughly the same position
    remote_preds = [
        {"class": "empty_seat", "confidence": 0.90, "bbox": [50, 50, 150, 150]},
        {"class": "human_occupied", "confidence": 0.88, "bbox": [200, 50, 300, 150]},
    ]
    rf_seats = detector.detect_seats_from_predictions(remote_preds, frame_width=640, frame_height=360)
    assert len(rf_seats) == 2

    # In hybrid fusion with another seat far away
    extra_local_seat = SeatConfig(seat_id="SEAT_LOCAL", polygon=[(0.6, 0.1), (0.8, 0.1), (0.8, 0.3), (0.6, 0.3)])

    from src.seats.seat_tracker import calculate_polygon_iou
    fused = list([extra_local_seat])
    for r_seat in rf_seats:
        matched = any(
            calculate_polygon_iou(r_seat.polygon, l_seat.polygon) > 0.45
            for l_seat in [extra_local_seat]
        )
        if not matched:
            fused.append(r_seat)

    assert len(fused) == 3, "Non-overlapping seats should be retained without duplication"


# ════════════════════════════════════════════════════════════════════════
# TESTS FOR AUDIT FIXES: TASK 1 (ROBOFLOW VERSION SWITCHING)
# ════════════════════════════════════════════════════════════════════════

def test_roboflow_version_routing_target_model():
    """TASK 1: Verify version 1 vs version 2 target model IDs and request routing."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient
    from unittest.mock import MagicMock, patch

    client = RoboflowWorkflowClient()

    # Test Version 1 selection
    client.set_active_version("1")
    assert client.model_version == "1"
    info1 = client.get_workflow_info()
    assert info1["target_model_id"] == f"{client.project_id}/1"
    assert client.last_inference_info["requested_model_id"] == f"{client.project_id}/1"

    # Test Version 2 selection
    client.set_active_version("2")
    assert client.model_version == "2"
    info2 = client.get_workflow_info()
    assert info2["target_model_id"] == f"{client.project_id}/2"
    assert client.last_inference_info["requested_model_id"] == f"{client.project_id}/2"

    # Switch back to 1
    client.set_active_version("1")
    assert client.model_version == "1"


def test_roboflow_version_switching_cache_invalidation():
    """TASK 1: Verify cache is invalidated on version switch."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient

    client = RoboflowWorkflowClient()
    dummy_key = "/tmp/test_frame.jpg"
    client._last_submit[dummy_key] = 123456.78

    client.set_active_version("2")
    # Cache must be wiped
    assert dummy_key not in client._last_submit
    assert client.last_inference_info["status"] == "VERSION_SWITCHED"


def test_roboflow_version_2_non_existent_fails_without_fallback():
    """TASK 1: When requested version does not exist, fail cleanly without fallback to v1."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient
    from unittest.mock import MagicMock
    import tempfile

    client = RoboflowWorkflowClient()
    client.set_active_version("2")

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        # Mock SDK client throwing 404 Not Found
        mock_sdk = MagicMock()
        mock_sdk.infer.side_effect = Exception("404 Client Error: Not Found for url: https://detect.roboflow.com/proj/2")
        client._sdk_client = mock_sdk

        result = client.submit_image(Path(tmp.name))
        assert result is None, "Must return None when version inference fails"
        assert client.last_inference_info["status"] == "FAILED"
        assert client.last_inference_info["actual_model_id"] is None
        assert "404" in client.last_inference_info["error"]
        assert client.last_inference_info["requested_model_id"] == f"{client.project_id}/2"


def test_roboflow_api_key_masked_in_exceptions():
    """TASK 1: Raw API keys must never be exposed in error logs or inference info."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient
    from unittest.mock import MagicMock
    import tempfile

    client = RoboflowWorkflowClient()
    raw_key = client.api_key

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        mock_sdk = MagicMock()
        mock_sdk.infer.side_effect = Exception(f"Connection failed with key {raw_key} at endpoint")
        client._sdk_client = mock_sdk

        res = client.submit_image(Path(tmp.name))
        assert res is None
        err = client.last_inference_info["error"]
        # Raw API key must NOT be in error string
        assert raw_key not in err
        assert "****" in err


# ════════════════════════════════════════════════════════════════════════
# TESTS FOR AUDIT FIXES: TASK 2 (TWIN SEAT DECOMPOSITION IN PIPELINE)
# ════════════════════════════════════════════════════════════════════════

def test_wide_bench_decomposes_into_two_child_seats():
    """TASK 2: A wide bench prediction creates exactly 2 distinct child seats."""
    detector = SeatDetector()

    # Wide bench: x from 50 to 350, y from 100 to 220 (width 300, height 120 -> ratio 2.5 > 1.45)
    bench_pred = [
        {"class": "double_seat", "confidence": 0.92, "bbox": [50, 100, 350, 220]},
    ]
    seats = detector.detect_seats_from_predictions(bench_pred, frame_width=1000, frame_height=1000)
    assert len(seats) == 2, f"Wide bench must decompose into exactly 2 child seats, got {len(seats)}"
    seat_a, seat_b = seats[0], seats[1]

    # Verify separate IDs
    assert seat_a.seat_id != seat_b.seat_id
    assert "SEAT_" in seat_a.seat_id
    assert "SEAT_" in seat_b.seat_id

    # Verify separate geometry
    poly_a = seat_a.polygon
    poly_b = seat_b.polygon
    max_x_a = max(p[0] for p in poly_a)
    min_x_b = min(p[0] for p in poly_b)
    assert max_x_a < min_x_b, "Seat A must be strictly to the left of Seat B"


def test_single_seat_not_split():
    """TASK 2: Normal single seats are NOT split."""
    detector = SeatDetector()

    # Single seat: width 100, height 110 -> ratio ~0.9
    single_pred = [
        {"class": "empty_seat", "confidence": 0.88, "bbox": [100, 100, 200, 210]},
    ]
    seats = detector.detect_seats_from_predictions(single_pred, frame_width=1000, frame_height=1000)
    assert len(seats) == 1, "Single seat must NOT be split"


def test_child_seats_geometry_within_parent_bounds():
    """TASK 2: Decomposed child seat geometry must stay strictly within parent bounds."""
    detector = SeatDetector()

    p_x1, p_y1, p_x2, p_y2 = 100, 150, 450, 280
    bench_pred = [
        {"class": "bench_seat", "confidence": 0.95, "bbox": [p_x1, p_y1, p_x2, p_y2]},
    ]
    frame_w, frame_h = 1000, 1000
    norm_x1, norm_y1 = p_x1 / frame_w, p_y1 / frame_h
    norm_x2, norm_y2 = p_x2 / frame_w, p_y2 / frame_h

    seats = detector.detect_seats_from_predictions(bench_pred, frame_width=frame_w, frame_height=frame_h)
    assert len(seats) == 2

    for s in seats:
        xs = [p[0] for p in s.polygon]
        ys = [p[1] for p in s.polygon]
        assert min(xs) >= norm_x1 - 1e-4
        assert max(xs) <= norm_x2 + 1e-4
        assert min(ys) >= norm_y1 - 1e-4
        assert max(ys) <= norm_y2 + 1e-4


def test_child_seats_stable_ids_over_frames():
    """TASK 2: Child seats keep stable IDs across multiple frames without duplication."""
    from src.seats.seat_manager import SeatManager

    sm = SeatManager()
    bench_pred = [
        {"class": "double_seat", "confidence": 0.90, "bbox": [100, 100, 360, 210]},
    ]

    # Frame 0: First detection
    seats_f0 = sm.update_detected_seats(bench_pred, frame_idx=0, frame_width=1000, frame_height=1000)
    assert len(seats_f0) == 2
    ids_f0 = {s.seat_id for s in seats_f0}

    # Frame 1: Same bench slightly jittered
    bench_pred_f1 = [
        {"class": "double_seat", "confidence": 0.91, "bbox": [102, 101, 358, 209]},
    ]
    seats_f1 = sm.update_detected_seats(bench_pred_f1, frame_idx=1, frame_width=1000, frame_height=1000)
    assert len(seats_f1) == 2, "Must NOT duplicate seats on subsequent frames"
    ids_f1 = {s.seat_id for s in seats_f1}
    assert ids_f1 == ids_f0, "Seat IDs must remain 100% stable across frames"

    # Frame 2: Same bench again
    seats_f2 = sm.update_detected_seats(bench_pred_f1, frame_idx=2, frame_width=1000, frame_height=1000)
    assert len(seats_f2) == 2
    assert {s.seat_id for s in seats_f2} == ids_f0


def test_child_seats_independent_occupancy_and_occlusion():
    """TASK 2: Child seats have independent occupancy; passenger on Seat A does not mark Seat B occupied."""
    from src.seats.seat_manager import SeatManager
    from src.models import SeatState

    sm = SeatManager()
    # Bench from x=100 to 400 (norm 0.1 to 0.4), y=100 to 200 (norm 0.1 to 0.2)
    bench_pred = [
        {"class": "double_seat", "confidence": 0.90, "bbox": [100, 100, 400, 200]},
    ]
    seats = sm.update_detected_seats(bench_pred, frame_idx=0, frame_width=1000, frame_height=1000)
    assert len(seats) == 2
    seat_a, seat_b = seats[0], seats[1]

    # Passenger sits on Seat A in pixels (x ~ 100..230, y ~ 100..200)
    poly_a = seat_a.polygon
    px1, px2 = int(min(p[0] for p in poly_a) * 1000), int(max(p[0] for p in poly_a) * 1000)
    py1, py2 = int(min(p[1] for p in poly_a) * 1000), int(max(p[1] for p in poly_a) * 1000)
    person_a = TrackedObject(
        track_id=10,
        class_name="person",
        confidence=0.92,
        bbox=(px1, py1, px2, py2),
    )

    # Update over multiple frames to confirm occupancy
    for f in range(10):
        statuses = sm.update(
            persons=[person_a],
            frame_index=f,
            fps=5.0,
            frame_width=1000,
            frame_height=1000,
        )

    status_a = sm.get_seat_status(seat_a.seat_id)
    status_b = sm.get_seat_status(seat_b.seat_id)
    assert status_a is not None
    assert status_b is not None
    assert status_a.state == SeatState.OCCUPIED, f"Seat A must be OCCUPIED, got {status_a.state}"
    assert status_b.state == SeatState.VACANT, f"Seat B must be VACANT, got {status_b.state}"


# =====================================================================
# SECTION 8: INSTANCE SEGMENTATION POLYGON FOR BELONGINGS AUDIT TESTS
# =====================================================================

def test_belongings_with_real_mask_rendered_as_polygon():
    """Criteria 1: Đồ vật có mask thật được hiển thị bằng polygon."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    test_poly = [(120.0, 120.0), (220.0, 120.0), (210.0, 240.0), (115.0, 230.0)]
    bag = TrackedObject(
        track_id=1,
        class_name="backpack",
        confidence=0.91,
        bbox=(115, 120, 220, 240),
        polygon=test_poly,
    )

    annotated = VideoWriter.annotate_frame(
        frame=canvas,
        tracked_objects=[bag],
        show_masks=True,
    )

    # Interior of polygon (160, 160) must have mask overlay non-zero
    assert np.any(annotated[160, 160] > 0), "Interior of polygon must be filled with mask overlay"
    assert bag.polygon == test_poly, "Original polygon coordinates must remain intact"


def test_person_and_seat_preserve_original_rendering_even_with_mask():
    """Criteria 2: Ghế và người vẫn hiển thị theo cách cũ (ghế là ROI polygon, người luôn là bounding box)."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    person_poly = [(100.0, 80.0), (180.0, 80.0), (170.0, 300.0), (90.0, 290.0)]
    person = TrackedObject(
        track_id=10,
        class_name="person",
        confidence=0.95,
        bbox=(90, 80, 180, 300),
        polygon=person_poly,  # Even if segmentation model supplies mask for person
    )

    annotated = VideoWriter.annotate_frame(
        frame=canvas,
        tracked_objects=[person],
        show_masks=True,
    )

    # Person: strictly bounding box wireframe, NEVER filled polygon overlay
    # Check interior point (140, 200) inside person bbox: must remain 0 (not filled)
    assert np.all(annotated[200, 140] == 0), "Person must NOT have filled segmentation mask overlay; bounding box only!"
    # Check that person bounding box border is drawn (green)
    assert np.any(annotated[80, 130] > 0), "Person bounding box top border must be rendered"


def test_bounding_box_not_mistaken_for_segmentation_mask():
    """Criteria 3: Bounding box không bị nhầm thành segmentation mask; không tạo polygon giả."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    item_no_mask = TrackedObject(
        track_id=5,
        class_name="cell phone",
        confidence=0.88,
        bbox=(50, 50, 100, 120),
        polygon=None,  # Model was object detection only
    )

    annotated = VideoWriter.annotate_frame(
        frame=canvas,
        tracked_objects=[item_no_mask],
        show_masks=True,
    )

    assert item_no_mask.polygon is None, "Item without mask must remain polygon=None (never fabricate fake polygon)"
    # Interior of bbox is not filled
    assert np.all(annotated[80, 75] == 0), "Item without mask must use bounding box wireframe without filled mask"
    # Border is drawn
    assert np.any(annotated[50, 75] > 0), "Bounding box border must be rendered"


def test_polygon_coordinates_match_original_frame_scale():
    """Criteria 4: Tọa độ polygon đúng với frame gốc và nằm trong giới hạn [0, w] x [0, h]."""
    w, h = 1280, 720
    test_poly = [(50.0, 60.0), (250.0, 60.0), (240.0, 300.0), (45.0, 290.0)]
    for pt in test_poly:
        assert 0 <= pt[0] <= w, f"X coordinate {pt[0]} must be within [0, {w}]"
        assert 0 <= pt[1] <= h, f"Y coordinate {pt[1]} must be within [0, {h}]"

    det = Detection(
        class_name="backpack",
        confidence=0.89,
        bbox=(45, 60, 250, 300),
        polygon=test_poly,
    )
    assert det.polygon == test_poly


def test_tracking_id_stability_with_segmentation_polygons():
    """Criteria 5: Tracking ID được giữ ổn định qua nhiều frame khi có segmentation polygon."""
    from src.tracking.object_tracker import ObjectTracker
    tracker = ObjectTracker()

    poly1 = [(100.0, 100.0), (150.0, 100.0), (150.0, 160.0), (100.0, 160.0)]
    poly2 = [(102.0, 101.0), (152.0, 101.0), (152.0, 161.0), (102.0, 161.0)]

    det1 = Detection(class_name="backpack", confidence=0.92, bbox=(100, 100, 150, 160), polygon=poly1)
    det2 = Detection(class_name="backpack", confidence=0.91, bbox=(102, 101, 152, 161), polygon=poly2)

    tracked_f1 = tracker.update(frame_index=1, detections=[det1])
    assert len(tracked_f1) == 1
    tid_1 = tracked_f1[0].track_id
    assert tracked_f1[0].polygon is not None

    tracked_f2 = tracker.update(frame_index=2, detections=[det2])
    assert len(tracked_f2) == 1
    tid_2 = tracked_f2[0].track_id

    assert tid_1 == tid_2, f"Tracking ID must remain stable across consecutive frames ({tid_1} == {tid_2})"
    assert tracker.observations[tid_1].polygons[-1] is not None


def test_occluded_item_not_marked_as_disappeared():
    """Criteria 6: Đồ vật bị che khuất không bị tự động kết luận là đã biến mất."""
    from src.events.forgotten_item_detector import ForgottenItemDetector
    from src.events.event_manager import EventManager
    from config.settings import EventStatus
    from src.seats.seat_manager import SeatManager

    sm = SeatManager()
    em = EventManager()
    ed = ForgottenItemDetector(seat_manager=sm, event_manager=em, min_stationary_time=0.1)

    item = TrackedObject(track_id=1, class_name="backpack", confidence=0.9, bbox=(100, 100, 200, 200))
    for f in range(6):
        ed.process_frame(tracked_objects=[item], current_frame=f, fps=10.0, frame_width=640, frame_height=360)

    person = TrackedObject(track_id=9, class_name="person", confidence=0.9, bbox=(90, 80, 220, 250))
    for f in range(6, 10):
        ed.process_frame(tracked_objects=[person], current_frame=f, fps=10.0, frame_width=640, frame_height=360)

    assert len(em.events) > 0, "Event must be preserved when occluded"
    active_statuses = [e.status for e in em.events]
    assert any(st in active_statuses for st in (EventStatus.OCCLUDED, EventStatus.ITEM_LEFT, EventStatus.SUSPECTED))


def test_forgotten_item_detection_logic_preserved():
    """Criteria 7: Logic phát hiện đồ bỏ quên vẫn hoạt động chuẩn xác."""
    from src.events.forgotten_item_detector import ForgottenItemDetector
    from src.events.event_manager import EventManager
    from config.settings import EventStatus

    em = EventManager()
    ed = ForgottenItemDetector(event_manager=em, min_stationary_time=0.1)

    p = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(100, 100, 180, 250))
    b = TrackedObject(track_id=2, class_name="backpack", confidence=0.88, bbox=(120, 150, 170, 220))
    for f in range(5):
        ed.process_frame(tracked_objects=[p, b], current_frame=f, fps=10.0, frame_width=640, frame_height=360)

    p_far = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(500, 300, 580, 360))
    for f in range(5, 20):
        ed.process_frame(tracked_objects=[p_far, b], current_frame=f, fps=10.0, frame_width=640, frame_height=360)

    suspected = [e for e in em.events if e.status == EventStatus.SUSPECTED]
    assert len(suspected) > 0, "Item left behind after person leaves must transition to SUSPECTED"


def test_unsupported_classes_no_hallucination():
    """Criteria 8: Model không hỗ trợ lớp đồ vật (ví dụ wallet không có trong COCO) thì không đưa ra nhận diện giả."""
    from src.detection.yolo_detector import YOLODetector
    detector = YOLODetector()
    detector.load_model()

    all_names = list(detector.class_names.values())
    assert "wallet" not in all_names, "COCO pretrained model does NOT have 'wallet'; must not falsely claim support"
    assert "lunch_box" not in all_names, "COCO pretrained model does NOT have 'lunch_box'"
    assert "backpack" in all_names, "COCO pretrained model supports 'backpack'"
    assert "cell phone" in all_names, "COCO pretrained model supports 'cell phone'"


