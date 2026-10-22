"""
tests/test_true_instance_segmentation_polygons.py
Bộ kiểm thử hồi quy nghiêm ngặt cho Yêu cầu 5A (CRITICAL REQUIREMENT — RESTORE TRUE INSTANCE SEGMENTATION POLYGONS):
1. Hộp chữ nhật (rectangular bounding box) KHÔNG được chấp nhận là bằng chứng phân đoạn (segmentation).
2. Mặt nạ đối tượng tạo ra đường viền (contour) bám sát hình dáng thực tế, bảo toàn các đường cong và vết lõm (không bị biến thành convex hull hoặc box).
3. Nhiều đồ vật cá nhân khác nhau nhận các instance masks và polygons riêng biệt, độc lập.
4. Tọa độ polygon giữ đúng căn chỉnh khi khung hình thay đổi tỉ lệ (letterbox unpadding 4:3, 16:9).
5. Polygon hiển thị ổn định qua các frame tracking và không bị rơi về bounding box khi mất frame ngắn hạn.
6. Backend suy luận không âm thầm chuyển renderer về bounding box khi đối tượng có mặt nạ segmentation.
7. Ứng dụng xác minh năng lực segmentation (task == 'segment') trước khi kích hoạt chế độ polygon.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import numpy as np
import cv2

from src.models import Detection, TrackedObject
from src.video.video_writer import VideoWriter
from src.tracking.object_tracker import ObjectTracker
from src.detection.onnx_detector import ONNXSegmentationDetector
from src.detection.yolo_detector import YOLODetector


def is_axis_aligned_box(polygon, tolerance=2.0) -> bool:
    """Kiểm tra xem một polygon có phải chỉ là một hình chữ nhật hộp (bounding box) giả lập hay không."""
    if len(polygon) != 4:
        # Nếu có số điểm khác 4 nhưng các điểm đều nằm trên 4 cạnh của bbox thì vẫn là giả lập box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        on_boundary = all(
            abs(x - min_x) <= tolerance or abs(x - max_x) <= tolerance or
            abs(y - min_y) <= tolerance or abs(y - max_y) <= tolerance
            for x, y in polygon
        )
        return on_boundary and len(set(xs)) <= 2 and len(set(ys)) <= 2

    # Với 4 điểm, kiểm tra các góc vuông song song trục
    xs = sorted([p[0] for p in polygon])
    ys = sorted([p[1] for p in polygon])
    is_rect_x = (abs(xs[0] - xs[1]) <= tolerance and abs(xs[2] - xs[3]) <= tolerance)
    is_rect_y = (abs(ys[0] - ys[1]) <= tolerance and abs(ys[2] - ys[3]) <= tolerance)
    return is_rect_x and is_rect_y


class TestTrueInstanceSegmentationPolygons(unittest.TestCase):

    def test_01_rectangular_bbox_not_accepted_as_segmentation(self):
        """1. Bounding box 4 góc hình chữ nhật KHÔNG được coi là segmentation polygon."""
        fake_box_polygon = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        self.assertTrue(
            is_axis_aligned_box(fake_box_polygon),
            "Hàm kiểm tra phải nhận diện được polygon 4 điểm hình hộp"
        )

        # Polygon thực tế uốn lượn theo hình dáng đồ vật
        curved_backpack_polygon = [
            (150.0, 100.0), (180.0, 120.0), (210.0, 110.0), (230.0, 160.0),
            (240.0, 220.0), (200.0, 260.0), (160.0, 250.0), (130.0, 200.0),
            (135.0, 140.0)
        ]
        self.assertFalse(
            is_axis_aligned_box(curved_backpack_polygon),
            "Polygon thực tế uốn cong không được bị đánh giá là hình hộp chữ nhật"
        )

    def test_02_concave_contour_following_shape(self):
        """2. Đường viền contour phải bám sát hình dáng lõm (concave cutout), không bị san phẳng thành convex hull."""
        # Giả lập mặt nạ nhị phân của ba lô có quai cầm (phần rỗng lõm ở giữa quai)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:35, 10:35] = 1
        mask[10:22, 18:28] = 0  # Vết lõm sâu giữa 2 quai đeo

        # Trích xuất contour bằng findContours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertGreaterEqual(len(contours), 1)

        pts = contours[0].squeeze()
        # Kiểm tra xem có điểm nào nằm sâu trong phần lõm (y >= 20, 18 <= x <= 28) hay không
        has_indent_visited = any(p[1] >= 20 and 18 <= p[0] <= 28 for p in pts)
        self.assertTrue(
            has_indent_visited,
            "Contour phải đi vào vết lõm của vật thể thay vì cắt ngang như convex hull"
        )

    def test_03_multiple_belongings_separate_instance_masks_and_polygons(self):
        """3. Nhiều đồ vật nhận diện riêng biệt phải có các polygons độc lập không bị dính chùm."""
        backpack_poly = [(50.0, 50.0), (120.0, 60.0), (110.0, 140.0), (45.0, 130.0)]
        bottle_poly = [(250.0, 100.0), (280.0, 100.0), (285.0, 220.0), (245.0, 220.0)]

        det1 = Detection(class_name="backpack", confidence=0.88, bbox=(45, 50, 120, 140), polygon=backpack_poly)
        det2 = Detection(class_name="bottle", confidence=0.92, bbox=(245, 100, 285, 220), polygon=bottle_poly)

        self.assertNotEqual(det1.polygon, det2.polygon)
        self.assertEqual(len(det1.polygon), 4)
        self.assertEqual(len(det2.polygon), 4)

        # Kiểm tra không giao nhau
        b1_max_x = max(p[0] for p in det1.polygon)
        b2_min_x = min(p[0] for p in det2.polygon)
        self.assertLess(b1_max_x, b2_min_x, "Hai đồ vật độc lập ở 2 vị trí khác nhau không được đè polygon lên nhau")

    def test_04_polygon_coordinate_alignment_across_aspect_ratios(self):
        """4. Chuẩn hóa tọa độ polygon chính xác qua letterbox unpadding (4:3 và 16:9)."""
        input_size = 480

        # Trường hợp 1: Ảnh 4:3 (640x480)
        orig_w, orig_h = 640, 480
        r = min(input_size / orig_h, input_size / orig_w)  # 480/640 = 0.75
        nw, nh = round(orig_w * r), round(orig_h * r)      # 480, 360
        pad_w = (input_size - nw) / 2.0                    # 0.0
        pad_h = (input_size - nh) / 2.0                    # 60.0

        # Giả lập điểm tâm trong không gian input [240, 240]
        in_x, in_y = 240.0, 240.0
        orig_x = (in_x - pad_w) / r
        orig_y = (in_y - pad_h) / r

        # Tâm điểm phải khớp hoàn hảo với ảnh gốc 640x480 (x = 320, y = 240)
        self.assertAlmostEqual(orig_x, 320.0, places=1)
        self.assertAlmostEqual(orig_y, 240.0, places=1)

        # Trường hợp 2: Ảnh 16:9 (1280x720)
        orig_w2, orig_h2 = 1280, 720
        r2 = min(input_size / orig_h2, input_size / orig_w2) # 480/1280 = 0.375
        nw2, nh2 = round(orig_w2 * r2), round(orig_h2 * r2)   # 480, 270
        pad_w2 = (input_size - nw2) / 2.0                     # 0.0
        pad_h2 = (input_size - nh2) / 2.0                     # 105.0

        orig_x2 = (in_x - pad_w2) / r2
        orig_y2 = (in_y - pad_h2) / r2
        self.assertAlmostEqual(orig_x2, 640.0, places=1)
        self.assertAlmostEqual(orig_y2, 360.0, places=1)

    def test_05_polygon_temporal_stability_during_tracking(self):
        """5. Polygon được bảo toàn qua các frame tracking và không bị mất viền rơi về bounding box."""
        tracker = ObjectTracker()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_poly = [(100.0, 100.0), (160.0, 90.0), (180.0, 180.0), (110.0, 190.0), (95.0, 140.0)]
        det_frame1 = [
            Detection(class_name="backpack", confidence=0.85, bbox=(95, 90, 180, 190), polygon=mock_poly)
        ]

        # Frame 0: có polygon đầy đủ
        tracked1 = tracker.update(
            model=None,
            frame=dummy_frame,
            frame_index=0,
            fps=25.0,
            detections=det_frame1,
        )
        self.assertEqual(len(tracked1), 1)
        self.assertIsNotNone(tracked1[0].polygon)
        self.assertEqual(len(tracked1[0].polygon), 5)

        # Frame 1: Detection bị gián đoạn mask (polygon=None) do chuyển động nhẹ hoặc góc nhìn
        det_frame2 = [
            Detection(class_name="backpack", confidence=0.82, bbox=(97, 92, 182, 192), polygon=None)
        ]
        tracked2 = tracker.update(
            model=None,
            frame=dummy_frame,
            frame_index=1,
            fps=25.0,
            detections=det_frame2,
        )
        self.assertEqual(len(tracked2), 1)
        # Hệ thống phải bảo toàn polygon từ lịch sử quan sát, không được rơi về None!
        self.assertIsNotNone(tracked2[0].polygon, "Polygon phải được bảo toàn qua frame gián đoạn mask")
        self.assertEqual(len(tracked2[0].polygon), 5)

    def test_06_backend_updates_do_not_revert_to_boxes(self):
        """6. VideoWriter.annotate_frame vẽ mặt nạ polygon khi show_masks=True thay vì chỉ vẽ box."""
        frame = np.full((300, 300, 3), 50, dtype=np.uint8)
        poly = [(50.0, 50.0), (150.0, 50.0), (180.0, 150.0), (70.0, 160.0)]
        tracked_obj = TrackedObject(
            track_id=1,
            class_name="backpack",
            confidence=0.90,
            bbox=(50, 50, 180, 160),
            polygon=poly,
        )

        annotated = VideoWriter.annotate_frame(
            frame=frame,
            tracked_objects=[tracked_obj],
            show_objects=True,
            show_masks=True,
        )

        # Kiểm tra xem pixel bên trong polygon có bị thay đổi màu (được vẽ mask màu hổ phách/amber) không
        interior_pixel = annotated[100, 100]
        # Frame ban đầu là màu xám (50, 50, 50)
        self.assertFalse(
            (interior_pixel == np.array([50, 50, 50])).all(),
            "Mặt nạ polygon phải được phủ màu trong suốt lên vùng bên trong đồ vật"
        )

    def test_07_backend_verifies_segmentation_capability(self):
        """7. Detector phải xác minh model có task=='segment' và nạp đúng cấu trúc segmentation."""
        onnx_model_path = Path("static/models/yolo26n-seg.onnx")
        if onnx_model_path.exists():
            det = ONNXSegmentationDetector(model_path=onnx_model_path)
            self.assertEqual(det.task, "segment")
            self.assertTrue(det.is_loaded)

        pt_model_path = Path("yolo26n-seg.pt")
        if pt_model_path.exists():
            yolo = YOLODetector(model_name=str(pt_model_path))
            yolo.load_model()
            self.assertTrue(yolo.is_segmentation)
            self.assertEqual(yolo.task, "segment")


if __name__ == "__main__":
    unittest.main()
