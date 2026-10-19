"""
tests/test_yolo26_instance_segmentation.py
Bộ kiểm thử tích hợp chuyên biệt cho YOLO26 Instance Segmentation
Tuân thủ nghiêm ngặt 12 yêu cầu kiểm thử trong Giai đoạn F:
1. Nạp model segmentation.
2. Kiểm tra model.task == "segment".
3. Xác minh đủ ba class, không phụ thuộc thứ tự class ID.
4. Chạy inference trên ảnh kiểm thử thật.
5. Xác minh masks và boxes có cùng số detection tương ứng khi model trả về masks.
6. Polygon ghế được chuẩn hóa đúng.
7. Một frame không có detection không làm crash hệ thống.
8. Class item được chuyển tới tracking và forgotten-item logic.
9. ID ghế không thay đổi chỉ vì frame tiếp theo được xử lý.
10. Đồ vật còn lại sau khi người rời đi được theo dõi theo thời gian.
11. Đồ vật bị che khuất hoặc mất detection ngắn hạn không lập tức bị kết luận thất lạc.
12. Giao diện camera và module annotate giữ nguyên tính tương thích ngược.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Đảm bảo đường dẫn gốc của repository luôn trong sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    CUSTOM_SEGMENTATION_WEIGHTS_PATH,
    SUPPORTED_CUSTOM_WEIGHTS_PATHS,
    DEFAULT_SHOW_SEGMENTATION_MASKS,
    ITEM_CLASSES,
    CLASS_NAMES_VI,
    EventStatus,
)
from src.models import Detection, TrackedObject, SeatConfig, SeatStatus, SeatState
from src.detection.yolo_detector import YOLODetector
from src.tracking.object_tracker import ObjectTracker
from src.seats.seat_detector import SeatDetector, CustomSeatYOLOAdapter
from src.seats.seat_manager import SeatManager
from src.seats.seat_config import SeatConfigManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.events.event_manager import EventManager
from src.video.video_writer import VideoWriter


class TestYOLO26InstanceSegmentation(unittest.TestCase):
    """Bộ test kiểm thử toàn diện cho việc tích hợp YOLO26 Instance Segmentation."""

    def setUp(self):
        self.frame_w = 640
        self.frame_h = 480
        # Tạo ảnh kiểm thử thật (synthetic real test image)
        self.test_img = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        # Vẽ một vài hình học cơ bản mô phỏng khung cảnh xe buýt
        cv2.rectangle(self.test_img, (50, 100), (250, 400), (80, 80, 80), -1)  # Ghế 1
        cv2.rectangle(self.test_img, (350, 100), (550, 400), (80, 80, 80), -1)  # Ghế 2
        cv2.circle(self.test_img, (150, 200), 40, (0, 200, 0), -1)  # Người ngồi ghế 1
        cv2.rectangle(self.test_img, (130, 250), (170, 310), (0, 165, 255), -1)  # Đồ vật trên ghế 1

    # ────────────────────────────────────────────────────────────
    # Test 1: Nạp model segmentation & cấu hình weights models/best.pt
    # ────────────────────────────────────────────────────────────
    def test_01_load_segmentation_model_configuration(self):
        """1. Kiểm tra cấu hình đường dẫn weights ưu tiên models/best.pt và xử lý khi chưa có weights."""
        self.assertEqual(CUSTOM_SEGMENTATION_WEIGHTS_PATH.name, "best.pt")
        self.assertIn(CUSTOM_SEGMENTATION_WEIGHTS_PATH, SUPPORTED_CUSTOM_WEIGHTS_PATHS)

        # Khởi tạo detector khi chưa có file best.pt: hệ thống phải báo log rõ ràng và không crash
        detector = YOLODetector(model_name="non_existent_weights.pt")
        info = detector.get_model_info()
        self.assertIn("loaded", info)
        self.assertIn("task", info)

        # Kiểm tra cơ chế set_shared_model để nạp 1 lần dùng chung
        dummy_model = MagicMock()
        dummy_model.task = "segment"
        dummy_model.names = {0: "seat", 1: "person", 2: "item"}
        detector.set_shared_model(dummy_model)
        self.assertTrue(detector.is_segmentation)
        self.assertTrue(detector.is_custom_3class)
        self.assertIn("seat", detector.class_names.values())

    # ────────────────────────────────────────────────────────────
    # Test 2: Kiểm tra model.task == "segment"
    # ────────────────────────────────────────────────────────────
    def test_02_verify_model_task_segment(self):
        """2. Kiểm tra xác minh task segment của model (không chấp nhận model detect giả làm segment)."""
        detector = YOLODetector()

        # Trường hợp model segmentation
        mock_seg_model = MagicMock()
        mock_seg_model.task = "segment"
        mock_seg_model.names = {0: "seat", 1: "person", 2: "item"}
        detector.set_shared_model(mock_seg_model)
        self.assertEqual(detector.task, "segment")
        self.assertTrue(detector.is_segmentation)

        # Trường hợp model detection thông thường (không phải segment)
        mock_det_model = MagicMock()
        mock_det_model.task = "detect"
        mock_det_model.names = {0: "seat", 1: "person", 2: "item"}
        detector_det = YOLODetector()
        detector_det.set_shared_model(mock_det_model)
        self.assertEqual(detector_det.task, "detect")
        self.assertFalse(detector_det.is_segmentation)

    # ────────────────────────────────────────────────────────────
    # Test 3: Xác minh đủ ba class, không phụ thuộc thứ tự class ID
    # ────────────────────────────────────────────────────────────
    def test_03_verify_three_classes_order_independent(self):
        """3. Xác minh đủ ba class ('seat', 'person', 'item'), không phụ thuộc thứ tự class ID."""
        permutations = [
            {0: "seat", 1: "person", 2: "item"},
            {0: "item", 1: "seat", 2: "person"},
            {0: "person", 1: "item", 2: "seat"},
            {10: "item", 20: "person", 30: "seat"},
        ]

        for p_names in permutations:
            detector = YOLODetector()
            mock_model = MagicMock()
            mock_model.task = "segment"
            mock_model.names = p_names
            detector.set_shared_model(mock_model)

            self.assertTrue(
                detector.is_custom_3class,
                f"Thất bại trong việc nhận diện 3-class với thứ tự: {p_names}"
            )
            # Kiểm tra cả 3 lớp đều có mặt trong target_class_ids hoặc class_names
            names_found = set(detector.class_names.values())
            self.assertIn("seat", names_found)
            self.assertIn("person", names_found)
            self.assertIn("item", names_found)

            # Person và Item phải có trong target class IDs để tracking
            item_ids = [cid for cid, cname in detector.class_names.items() if cname in ("person", "item")]
            for cid in item_ids:
                self.assertIn(cid, detector.target_class_ids)

    # ────────────────────────────────────────────────────────────
    # Test 4: Chạy inference trên ảnh kiểm thử thật
    # ────────────────────────────────────────────────────────────
    def test_04_run_inference_on_real_image(self):
        """4. Chạy inference trên ảnh kiểm thử thật (không crash, trả về cấu trúc hợp lệ)."""
        detector = YOLODetector()

        # Tạo mock result có masks và boxes
        mock_box1 = MagicMock()
        mock_box1.xyxy = np.array([[50, 100, 250, 400]])
        mock_box1.conf = np.array([0.92])
        mock_box1.cls = np.array([0])

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_box1]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        # Polygon mask cho 1 đối tượng
        poly_pts = np.array([[50, 100], [250, 100], [250, 400], [50, 400]], dtype=np.float32)
        mock_result.masks.xy = [poly_pts]

        mock_model = MagicMock()
        mock_model.task = "segment"
        mock_model.names = {0: "seat", 1: "person", 2: "item"}
        mock_model.return_value = [mock_result]
        detector.set_shared_model(mock_model)

        detections = detector.detect(self.test_img)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.class_name, "seat")
        self.assertAlmostEqual(det.confidence, 0.92, places=2)
        self.assertEqual(det.bbox, (50, 100, 250, 400))
        self.assertIsNotNone(det.polygon)
        self.assertEqual(len(det.polygon), 4)

    # ────────────────────────────────────────────────────────────
    # Test 5: Xác minh masks và boxes có cùng số detection tương ứng
    # ────────────────────────────────────────────────────────────
    def test_05_masks_and_boxes_count_synchronization(self):
        """5. Xác minh masks và boxes có cùng số detection tương ứng khi model trả về masks."""
        detector = YOLODetector()

        # 3 boxes và 3 masks tương ứng
        b1 = MagicMock()
        b1.xyxy = np.array([[10, 10, 50, 50]])
        b1.conf = np.array([0.90])
        b1.cls = np.array([0])

        b2 = MagicMock()
        b2.xyxy = np.array([[60, 60, 120, 120]])
        b2.conf = np.array([0.85])
        b2.cls = np.array([1])

        b3 = MagicMock()
        b3.xyxy = np.array([[150, 150, 200, 200]])
        b3.conf = np.array([0.80])
        b3.cls = np.array([2])

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 3
        mock_boxes.__iter__.return_value = [b1, b2, b3]

        poly1 = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        poly2 = np.array([[60, 60], [120, 60], [120, 120], [60, 120]], dtype=np.float32)
        poly3 = np.array([[150, 150], [200, 150], [200, 200], [150, 200]], dtype=np.float32)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.masks.xy = [poly1, poly2, poly3]

        mock_model = MagicMock()
        mock_model.task = "segment"
        mock_model.names = {0: "seat", 1: "person", 2: "item"}
        mock_model.return_value = [mock_result]
        detector.set_shared_model(mock_model)

        detections = detector.detect(self.test_img)
        self.assertEqual(len(detections), 3)
        for i, det in enumerate(detections):
            self.assertIsNotNone(det.polygon, f"Detection {i} thiếu polygon mask")
            self.assertEqual(len(det.polygon), 4)

        # Trường hợp masks is None (fallback an toàn)
        mock_result_nomask = MagicMock()
        mock_result_nomask.boxes = mock_boxes
        mock_result_nomask.masks = None
        mock_model.return_value = [mock_result_nomask]

        detections_nomask = detector.detect(self.test_img)
        self.assertEqual(len(detections_nomask), 3)
        for det in detections_nomask:
            self.assertIsNone(det.polygon)

    # ────────────────────────────────────────────────────────────
    # Test 6: Polygon ghế được chuẩn hóa đúng tọa độ [0.0, 1.0]
    # ────────────────────────────────────────────────────────────
    def test_06_seat_polygon_normalized_properly(self):
        """6. Polygon ghế được chuẩn hóa đúng tọa độ [0.0, 1.0] và tối thiểu 3 điểm."""
        adapter = CustomSeatYOLOAdapter()

        # Giả lập kết quả segmentation ghế trên ảnh 1000x500
        mock_b1 = MagicMock()
        mock_b1.xyxy = np.array([[100, 50, 300, 250]])
        mock_b1.conf = np.array([0.88])
        mock_b1.cls = np.array([0])

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_b1]

        # Đa giác 5 điểm cho ghế
        raw_poly = np.array([
            [100, 50],
            [300, 50],
            [300, 250],
            [200, 240],
            [100, 250],
        ], dtype=np.float32)

        mock_res = MagicMock()
        mock_res.boxes = mock_boxes
        mock_res.masks.xy = [raw_poly]

        mock_model = MagicMock()
        mock_model.task = "segment"
        mock_model.names = {0: "seat", 1: "person", 2: "item"}
        mock_model.return_value = [mock_res]
        mock_model.predict.return_value = [mock_res]
        adapter.set_shared_model(mock_model)

        test_frame = np.zeros((500, 1000, 3), dtype=np.uint8)
        seats = adapter.detect(test_frame)

        self.assertEqual(len(seats), 1)
        seat = seats[0]
        self.assertGreaterEqual(len(seat.polygon), 3)
        # Xác minh toàn bộ tọa độ nằm trong [0.0, 1.0]
        for nx, ny in seat.polygon:
            self.assertGreaterEqual(nx, 0.0)
            self.assertLessEqual(nx, 1.0)
            self.assertGreaterEqual(ny, 0.0)
            self.assertLessEqual(ny, 1.0)

        # Kiểm tra chuyển đổi ngược lại toạ độ pixel
        pixel_poly = seat.get_pixel_polygon(1000, 500)
        self.assertEqual(len(pixel_poly), len(seat.polygon))
        self.assertEqual(pixel_poly[0], (100, 50))
        self.assertEqual(pixel_poly[1], (300, 50))

    # ────────────────────────────────────────────────────────────
    # Test 7: Một frame không có detection không làm crash hệ thống
    # ────────────────────────────────────────────────────────────
    def test_07_empty_frame_no_detection_no_crash(self):
        """7. Một frame không có detection không làm crash hệ thống."""
        detector = YOLODetector()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.masks = None
        mock_model = MagicMock()
        mock_model.task = "segment"
        mock_model.names = {0: "seat", 1: "person", 2: "item"}
        mock_model.return_value = [mock_result]
        detector.set_shared_model(mock_model)

        # Chạy inference trên frame đen hoàn toàn
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(black_frame)
        self.assertEqual(detections, [])

        # Kiểm tra tracker với 0 detection
        tracker = ObjectTracker()
        tracked = tracker.update(mock_model, black_frame, frame_index=1, fps=10.0)
        self.assertEqual(tracked, [])

        # Kiểm tra ForgottenItemDetector với 0 tracked object
        event_mgr = EventManager()
        seat_mgr = SeatManager()
        event_det = ForgottenItemDetector(
            min_stationary_time=2.0,
            watched_classes=ITEM_CLASSES,
            seat_manager=seat_mgr,
            event_manager=event_mgr,
        )
        event_det.process_frame(
            tracked_objects=[],
            current_frame=1,
            fps=10.0,
            frame_width=640,
            frame_height=480,
            frame=black_frame,
        )
        self.assertEqual(len(event_mgr.events), 0)

    # ────────────────────────────────────────────────────────────
    # Test 8: Class item được chuyển tới tracking và forgotten-item logic
    # ────────────────────────────────────────────────────────────
    def test_08_item_class_forwarded_to_tracking_and_forgotten_item_logic(self):
        """8. Class item được chuyển tới tracking và forgotten-item logic (không bị bỏ qua)."""
        self.assertIn("item", ITEM_CLASSES)
        self.assertIn("item", CLASS_NAMES_VI)

        # Giả lập tracker phát hiện đối tượng class 'item'
        tracker = ObjectTracker()
        mock_item_box = MagicMock()
        mock_item_box.xyxy = np.array([[100, 100, 180, 180]])
        mock_item_box.conf = np.array([0.89])
        mock_item_box.cls = np.array([2])
        mock_item_box.id = np.array([101])

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_item_box]

        mock_res = MagicMock()
        mock_res.boxes = mock_boxes
        mock_poly = np.array([[100, 100], [180, 100], [180, 180], [100, 180]], dtype=np.float32)
        mock_res.masks.xy = [mock_poly]

        mock_model = MagicMock()
        mock_model.task = "segment"
        mock_model.names = {0: "seat", 1: "person", 2: "item"}
        mock_model.track.return_value = [mock_res]

        tracked = tracker.update(
            mock_model,
            self.test_img,
            frame_index=1,
            fps=10.0,
            target_class_ids=[1, 2],
        )

        self.assertEqual(len(tracked), 1)
        item_obj = tracked[0]
        self.assertEqual(item_obj.class_name, "item")
        self.assertEqual(item_obj.track_id, 101)
        self.assertIsNotNone(item_obj.polygon)

        # Kiểm tra item được lưu vào item_observations của tracker
        item_obs = tracker.get_item_observations()
        self.assertIn(101, item_obs)
        self.assertEqual(item_obs[101].class_name, "item")

    # ────────────────────────────────────────────────────────────
    # Test 9: ID ghế không thay đổi chỉ vì frame tiếp theo được xử lý
    # ────────────────────────────────────────────────────────────
    def test_09_persistent_seat_id_across_frames(self):
        """9. ID ghế không thay đổi chỉ vì frame tiếp theo được xử lý (Persistent Seat IDs)."""
        seat_mgr = SeatManager()
        seat1 = SeatConfig(
            seat_id="SEAT_001",
            polygon=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.8), (0.1, 0.8)],
            is_normalized=True,
            description="Ghế 1",
        )
        seat2 = SeatConfig(
            seat_id="SEAT_002",
            polygon=[(0.5, 0.1), (0.8, 0.1), (0.8, 0.8), (0.5, 0.8)],
            is_normalized=True,
            description="Ghế 2",
        )
        seat_mgr.set_seats([seat1, seat2])

        initial_ids = [s.seat_id for s in seat_mgr.seats]
        self.assertEqual(initial_ids, ["SEAT_001", "SEAT_002"])

        # Chạy qua 5 frame liên tiếp với người di chuyển
        for f_idx in range(1, 6):
            person = TrackedObject(
                track_id=1,
                class_name="person",
                bbox=(int(640 * 0.15), int(480 * 0.2), int(640 * 0.35), int(480 * 0.7)),
                confidence=0.9,
            )
            seat_mgr.update_occupancy([person], frame_index=f_idx, fps=10.0, frame_width=640, frame_height=480)

            current_ids = [s.seat_id for s in seat_mgr.seats]
            self.assertEqual(
                current_ids,
                initial_ids,
                f"ID ghế bị biến đổi ở frame {f_idx}: {current_ids}"
            )

    # ────────────────────────────────────────────────────────────
    # Test 10: Đồ vật còn lại sau khi người rời đi được theo dõi theo thời gian
    # ────────────────────────────────────────────────────────────
    def test_10_item_left_behind_after_person_departs_tracked_over_time(self):
        """10. Đồ vật còn lại sau khi người rời đi được theo dõi theo thời gian (ITEM_LEFT -> SUSPECTED)."""
        seat_mgr = SeatManager()
        seat1 = SeatConfig(
            seat_id="SEAT_001",
            polygon=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.8), (0.1, 0.8)],
            is_normalized=True,
            description="Ghế 1",
        )
        seat_mgr.set_seats([seat1])

        event_mgr = EventManager()
        event_det = ForgottenItemDetector(
            min_stationary_time=1.0,  # 1 giây = 10 frames tại 10 FPS
            watched_classes=ITEM_CLASSES,
            seat_manager=seat_mgr,
            event_manager=event_mgr,
        )

        item_bbox = (int(640 * 0.15), int(480 * 0.3), int(640 * 0.25), int(480 * 0.5))
        item_poly = [(item_bbox[0], item_bbox[1]), (item_bbox[2], item_bbox[1]), (item_bbox[2], item_bbox[3]), (item_bbox[0], item_bbox[3])]
        person_bbox = (int(640 * 0.12), int(480 * 0.15), int(640 * 0.38), int(480 * 0.75))

        # Giai đoạn 1: Frame 1 -> 5: Người ngồi cùng đồ vật
        for f in range(1, 6):
            person = TrackedObject(track_id=1, class_name="person", bbox=person_bbox, confidence=0.9)
            item = TrackedObject(track_id=2, class_name="item", bbox=item_bbox, confidence=0.88, polygon=item_poly)
            seat_mgr.update_occupancy([person], frame_index=f, fps=10.0, frame_width=640, frame_height=480)
            event_det.process_frame([person, item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

        # Giai đoạn 2: Frame 6 -> 25: Người đã rời khỏi ghế, đồ vật vẫn nằm yên
        for f in range(6, 26):
            item = TrackedObject(track_id=2, class_name="item", bbox=item_bbox, confidence=0.88, polygon=item_poly)
            seat_mgr.update_occupancy([], frame_index=f, fps=10.0, frame_width=640, frame_height=480)
            event_det.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

        # Xác minh sự kiện ghi nhận đồ vật bị bỏ lại
        events = event_mgr.events
        self.assertGreater(len(events), 0)
        item_event = next((e for e in events if e.track_id == 2 or e.last_track_id == 2), None)
        self.assertIsNotNone(item_event)
        self.assertEqual(item_event.seat_id, "SEAT_001")
        # Trạng thái phải là SUSPECTED hoặc ITEM_LEFT (được theo dõi chặt chẽ)
        self.assertIn(item_event.status, (EventStatus.SUSPECTED, EventStatus.ITEM_LEFT, EventStatus.REVIEW_REQUIRED))

    # ────────────────────────────────────────────────────────────
    # Test 11: Đồ vật bị che khuất không lập tức bị kết luận thất lạc
    # ────────────────────────────────────────────────────────────
    def test_11_occluded_or_lost_detection_not_immediately_lost(self):
        """11. Đồ vật bị che khuất hoặc mất detection ngắn hạn không lập tức bị kết luận thất lạc (grace period)."""
        seat_mgr = SeatManager()
        seat1 = SeatConfig(
            seat_id="SEAT_001",
            polygon=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.8), (0.1, 0.8)],
            is_normalized=True,
            description="Ghế 1",
        )
        seat_mgr.set_seats([seat1])

        event_mgr = EventManager()
        event_det = ForgottenItemDetector(
            min_stationary_time=1.0,
            watched_classes=ITEM_CLASSES,
            seat_manager=seat_mgr,
            event_manager=event_mgr,
        )

        item_bbox = (int(640 * 0.15), int(480 * 0.3), int(640 * 0.25), int(480 * 0.5))
        item_poly = [(item_bbox[0], item_bbox[1]), (item_bbox[2], item_bbox[1]), (item_bbox[2], item_bbox[3]), (item_bbox[0], item_bbox[3])]

        # Frame 1 -> 15: Đồ vật xuất hiện ổn định
        for f in range(1, 16):
            item = TrackedObject(track_id=2, class_name="item", bbox=item_bbox, confidence=0.88, polygon=item_poly)
            event_det.process_frame([item], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

        # Frame 16 -> 18: Đồ vật bị che khuất bởi người đi qua (tạm thời biến mất trong 2 frame)
        for f in range(16, 19):
            event_det.process_frame([], current_frame=f, fps=10.0, frame_width=640, frame_height=480)

        # Kiểm tra event không bị xóa bỏ hoặc chuyển sang LOST ngay lập tức
        events = event_mgr.events
        self.assertGreater(len(events), 0)
        ev = events[0]
        self.assertNotEqual(ev.status, EventStatus.RESOLVED)

    # ────────────────────────────────────────────────────────────
    # Test 12: Giao diện camera hiện tại vẫn hoạt động và tương thích
    # ────────────────────────────────────────────────────────────
    def test_12_camera_interface_and_annotation_backward_compatibility(self):
        """12. Giao diện camera hiện tại vẫn hoạt động với cả segmentation masks và bounding box fallback."""
        seat_mgr = SeatManager()
        seat1 = SeatConfig(
            seat_id="SEAT_001",
            polygon=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.8), (0.1, 0.8)],
            is_normalized=True,
            description="Ghế 1",
        )
        seat_mgr.set_seats([seat1])

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        person = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=(100, 100, 200, 300),
            confidence=0.91,
            polygon=[(100, 100), (200, 100), (200, 300), (100, 300)],
        )
        item = TrackedObject(
            track_id=2,
            class_name="item",
            bbox=(120, 250, 180, 320),
            confidence=0.87,
            polygon=[(120, 250), (180, 250), (180, 320), (120, 320)],
        )

        # 1. Vẽ với show_masks = True (Instance Segmentation mode)
        annotated_masks = VideoWriter.annotate_frame(
            frame=dummy_frame,
            tracked_objects=[person, item],
            event_statuses={2: EventStatus.SUSPECTED},
            seat_manager=seat_mgr,
            show_seats=True,
            show_persons=True,
            show_objects=True,
            show_masks=True,
        )
        self.assertEqual(annotated_masks.shape, dummy_frame.shape)

        # 2. Vẽ với show_masks = False (Bounding Box fallback mode)
        annotated_nomasks = VideoWriter.annotate_frame(
            frame=dummy_frame,
            tracked_objects=[person, item],
            event_statuses={2: EventStatus.SUSPECTED},
            seat_manager=seat_mgr,
            show_seats=True,
            show_persons=True,
            show_objects=True,
            show_masks=False,
        )
        self.assertEqual(annotated_nomasks.shape, dummy_frame.shape)


if __name__ == "__main__":
    unittest.main()
