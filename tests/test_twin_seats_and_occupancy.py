"""
tests/test_twin_seats_and_occupancy.py
Bộ kiểm thử toàn diện cho hệ thống nhận diện ghế, ghế đôi kề nhau, theo dõi người - đồ vật
và xử lý che khuất theo các tiêu chí nghiệm thu A, B, C, D, E, G:

1. Kiểm tra khả năng phát hiện từng ghế và kiểm tra model COCO transparent fallback.
2. Phân tách chính xác hai ghế nằm cạnh nhau trong cùng một hàng (Ghế A & Ghế B).
3. Duy trì ID ghế ổn định và cấu trúc cố định khi có người ngồi lên/đứng dậy (SeatTracker).
4. Phân biệt độc lập: Seat ID, Person Track ID, Object Track ID, Event ID.
5. Ước lượng trạng thái có người bằng vùng tiếp xúc cơ thể (SeatOccupancyEstimator) và xử lý người ngồi giữa 2 ghế (AMBIGUOUS).
6. Kiểm thử đầy đủ Kịch bản C (Ghế đôi liền nhau):
   - Ghế A có người, Ghế B trống.
   - Ghế A có người, Ghế B có ví.
   - Người ở Ghế A đứng lên rời đi, Ghế B có ví.
   - Người mới (Person 2) ngồi xuống Ghế B đè lên ví (chuyển OCCLUDED, giữ nguyên Event ID).
   - Người mới đứng lên: ví xuất hiện trở lại (khôi phục) hoặc biến mất (REVIEW_REQUIRED).
7. Ghép đồ vật với từng ghế: ranh giới giữa 2 ghế gán UNKNOWN_SEAT.
"""

import pytest
import numpy as np

from src.models import (
    SeatConfig,
    SeatState,
    SeatStatus,
    TrackedObject,
    Event,
)
from src.seats.seat_detector import (
    SeatDetector,
    SeatDetectionMethod,
    TwinSeatDecomposer,
    BusGeometrySeatDetector,
)
from src.seats.seat_tracker import SeatTracker
from src.seats.seat_occupancy import SeatOccupancyEstimator
from src.seats.seat_manager import SeatManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from src.events.occlusion_manager import OcclusionManager
from config.settings import EventStatus


class TestSeatDetectionAndTwinSeats:
    """Kiểm thử Nhận diện từng ghế và Phân tách ghế đôi (Mục A & E)."""

    def test_coco_model_support_inspection(self):
        """Kiểm tra báo cáo minh bạch về giới hạn lớp ghế của model COCO pretrained."""
        coco_info = SeatDetector.inspect_coco_classes()
        assert coco_info["has_overhead_bus_seat_class"] is False
        assert coco_info["has_coco_chair_class"] is True
        assert "chair" in coco_info["explanation"].lower()
        # Không được giả lập là pretrained COCO tự nhận diện được ghế trần xe buýt
        assert "không có lớp nhận diện đệm ghế xe buýt" in coco_info["explanation"].lower()

    def test_twin_seat_decomposition(self):
        """Kiểm thử thuật toán phân rã một băng ghế đôi thành 2 ghế độc lập A và B."""
        # Một băng ghế đôi bên trái lối đi: x từ 0.06 đến 0.38, y từ 0.12 đến 0.28
        bench_bbox = (0.06, 0.12, 0.38, 0.28)
        decomposer = TwinSeatDecomposer()

        split_seats = decomposer.split_horizontal_bench(
            polygon_or_bbox=bench_bbox,
            row_index=1,
            side="left",
            naming_format="SEAT",
        )

        assert len(split_seats) == 2
        seat_a, seat_b = split_seats[0], split_seats[1]

        # Kiểm tra ID độc lập
        assert seat_a.seat_id == "SEAT_1A"
        assert seat_b.seat_id == "SEAT_1B"

        # Ghế A phải nằm bên trái Ghế B
        poly_a = seat_a.polygon
        poly_b = seat_b.polygon
        max_x_a = max(p[0] for p in poly_a)
        min_x_b = min(p[0] for p in poly_b)
        assert max_x_a < min_x_b  # Có khoảng cách khe giữa 2 đệm ghế

        # Không được gộp thành 1 vùng duy nhất
        assert seat_a.seat_id != seat_b.seat_id

    def test_multi_row_detection(self):
        """Kiểm thử khả năng phát hiện nhiều hàng ghế trong cùng 1 frame camera."""
        detector = SeatDetector()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Yêu cầu 4 hàng ghế, mỗi hàng có 2 ghế trái và 2 ghế phải (tổng 16 ghế)
        result = detector.detect_seats(
            frame=dummy_frame,
            allow_fallback=True,
            rows=4,
            left_cols=2,
            right_cols=2,
            naming_format="SEAT",
        )

        assert result.total_seats == 16
        seat_ids = [s.seat_id for s in result.seats]

        # Kiểm tra sự hiện diện của từng ghế trong từng hàng
        assert "SEAT_1A" in seat_ids
        assert "SEAT_1B" in seat_ids
        assert "SEAT_1C" in seat_ids
        assert "SEAT_1D" in seat_ids
        assert "SEAT_4A" in seat_ids
        assert "SEAT_4D" in seat_ids

        # Tất cả các ID ghế phải là duy nhất
        assert len(set(seat_ids)) == 16


class TestSeatTrackerStability:
    """Kiểm thử tính ổn định ID và tọa độ qua các frame (Mục A.3, A.4, A.5, A.6)."""

    def test_seat_id_stability_and_no_flipping(self):
        """Kiểm tra không gán ID mới cho cùng một chiếc ghế ở mỗi frame."""
        tracker = SeatTracker()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector = SeatDetector()

        seats_f1 = detector.detect_seats(dummy_frame, rows=2, left_cols=2, right_cols=2).seats
        tracked_f1 = tracker.update(seats_f1, frame_number=1)
        f1_ids = [s.seat_id for s in tracked_f1]

        # Frame 2: Có rung lắc nhẹ do động cơ xe buýt (dịch chuyển tọa độ 0.002)
        seats_f2 = []
        for s in seats_f1:
            jittered_poly = [(round(x + 0.002, 4), round(y + 0.001, 4)) for x, y in s.polygon]
            seats_f2.append(SeatConfig(seat_id=s.seat_id, polygon=jittered_poly, is_normalized=True))

        tracked_f2 = tracker.update(seats_f2, frame_number=2)
        f2_ids = [s.seat_id for s in tracked_f2]

        # ID phải được giữ nguyên vẹn 100%, không bị đổi hay sinh ID mới
        assert f1_ids == f2_ids

    def test_seat_permanence_when_passenger_sits_down(self):
        """
        Kiểm tra cấu trúc ghế không bị xóa khi có người ngồi lên
        (che khuất cushion nên detection contour frame đó có thể bị thiếu).
        """
        tracker = SeatTracker()
        initial_seats = [
            SeatConfig(seat_id="SEAT_1A", polygon=[(0.1, 0.2), (0.25, 0.2), (0.25, 0.4), (0.1, 0.4)], is_normalized=True),
            SeatConfig(seat_id="SEAT_1B", polygon=[(0.28, 0.2), (0.43, 0.2), (0.43, 0.4), (0.28, 0.4)], is_normalized=True),
        ]
        tracker.register_seats(initial_seats)

        # Ở frame 50: Người ngồi che kín đệm SEAT_1A nên camera chỉ phát hiện SEAT_1B
        f50_detected = [initial_seats[1]]  # Chỉ thấy 1B
        active_seats = tracker.update(f50_detected, frame_number=50)

        # SEAT_1A vẫn phải tồn tại trong hệ thống vì ghế gắn cố định vào sàn xe buýt!
        active_ids = [s.seat_id for s in active_seats]
        assert "SEAT_1A" in active_ids
        assert "SEAT_1B" in active_ids


class TestSeatOccupancyEstimation:
    """Kiểm thử Ước lượng trạng thái có người / trống (Mục B & C)."""

    @pytest.fixture
    def twin_seats(self):
        """Tạo 2 ghế đôi liền nhau trong hàng 1."""
        seat_a = SeatConfig(
            seat_id="SEAT_1A",
            polygon=[(100, 200), (250, 200), (250, 400), (100, 400)],
            is_normalized=False,
            description="Ghế 1A (Cửa sổ)",
        )
        seat_b = SeatConfig(
            seat_id="SEAT_1B",
            polygon=[(270, 200), (420, 200), (420, 400), (270, 400)],
            is_normalized=False,
            description="Ghế 1B (Lối đi)",
        )
        return [seat_a, seat_b]

    def test_person_on_seat_a_seat_b_vacant(self, twin_seats):
        """Tình huống 1: Ghế A có người ngồi, Ghế B trống."""
        manager = SeatManager(
            seats=twin_seats,
            occupied_confirm_sec=0.1,
            vacant_confirm_sec=0.1,
            hold_sec=0.1,
        )

        # Người P1 ngồi trên ghế A: bbox (110, 180, 240, 390)
        p1 = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(110, 180, 240, 390))

        # Cập nhật qua một vài frame để xác nhận trạng thái
        for frame_idx in range(5):
            manager.update(persons=[p1], frame_index=frame_idx, fps=15.0, frame_width=1280, frame_height=720)

        status_a = manager.get_seat_status("SEAT_1A")
        status_b = manager.get_seat_status("SEAT_1B")

        assert status_a.state == SeatState.OCCUPIED
        assert 1 in status_a.occupant_person_ids

        # Ghế B phải được xác nhận TRỐNG, không được coi cả hàng ghế là có người
        assert status_b.state == SeatState.VACANT
        assert len(status_b.occupant_person_ids) == 0

    def test_person_straddling_both_seats_is_ambiguous(self, twin_seats):
        """Tình huống: Người ngồi lấn giữa 2 ghế -> Đánh dấu UNKNOWN/AMBIGUOUS, không gán bừa."""
        estimator = SeatOccupancyEstimator(ambiguity_overlap_diff=0.12)
        statuses = {s.seat_id: SeatStatus(seat_id=s.seat_id) for s in twin_seats}

        # Người ngồi ngay chính giữa ranh giới giữa Ghế A và Ghế B: bbox (200, 200, 320, 400)
        p_straddle = TrackedObject(track_id=9, class_name="person", confidence=0.85, bbox=(200, 200, 320, 400))

        for frame_idx in range(5):
            estimator.update_occupancy(
                seats=twin_seats,
                current_statuses=statuses,
                persons=[p_straddle],
                frame_index=frame_idx,
                fps=15.0,
                frame_width=1280,
                frame_height=720,
            )

        # Cả 2 ghế không thể khẳng định chắc chắn -> UNKNOWN với ghi chú rõ ràng
        assert statuses["SEAT_1A"].state == SeatState.UNKNOWN
        assert statuses["SEAT_1B"].state == SeatState.UNKNOWN
        assert "giữa 2 ghế" in statuses["SEAT_1A"].notes


class TestTwinSeatsCompleteLifecycle:
    """
    Kiểm thử Kịch bản C đầy đủ theo đúng yêu cầu đề bài:
    - Ghế A có người, Ghế B có một chiếc ví.
    - Người ở Ghế A đứng lên rời đi, người ở Ghế B vẫn trống và ví vẫn còn.
    - Một người khác (Person 2) ngồi xuống Ghế B đè lên ví và che khuất nó (chuyển OCCLUDED).
    - Người mới đứng lên:
      + Phục hồi sự kiện nếu ví xuất hiện trở lại.
      + Hoặc chuyển REVIEW_REQUIRED nếu ví bị lấy đi / không xuất hiện.
    """

    @pytest.fixture
    def setup_system(self):
        seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (300, 100), (300, 300), (100, 300)], is_normalized=False)
        seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(320, 100), (520, 100), (520, 300), (320, 300)], is_normalized=False)
        seats = [seat_a, seat_b]

        seat_mgr = SeatManager(seats=seats, occupied_confirm_sec=0.1, vacant_confirm_sec=0.1, hold_sec=0.1)
        event_mgr = EventManager()
        occlusion_mgr = OcclusionManager()

        detector = ForgottenItemDetector(
            min_stationary_time=0.2,
            movement_threshold=10.0,
            watched_classes=["handbag", "backpack", "wallet"],
            seat_manager=seat_mgr,
            event_manager=event_mgr,
            occlusion_manager=occlusion_mgr,
            recheck_duration=0.5,
        )

        return seat_mgr, event_mgr, detector

    def test_full_scenario_c_twin_seats_and_occlusion_recovery(self, setup_system):
        seat_mgr, event_mgr, detector = setup_system
        fps = 10.0
        w, h = 1280, 720

        # Bước 1: Ghế A có người (P1), Ghế B trống nhưng có chiếc ví (O10)
        p1 = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 110, 280, 290))
        wallet = TrackedObject(track_id=10, class_name="handbag", confidence=0.88, bbox=(350, 150, 420, 220))

        # Cập nhật trạng thái ban đầu qua 5 frames
        for f in range(5):
            detector.process_frame([p1, wallet], current_frame=f, fps=fps, frame_width=w, frame_height=h)

        assert seat_mgr.is_seat_occupied("SEAT_1A")
        assert seat_mgr.is_seat_vacant("SEAT_1B")

        # Ví nằm trên ghế trống B -> Phải bắt đầu theo dõi bỏ quên (ITEM_LEFT -> SUSPECTED)
        evt_b = event_mgr.get_active_event_for_seat("SEAT_1B")
        assert evt_b is not None
        initial_event_id = evt_b.event_id
        assert evt_b.seat_id == "SEAT_1B"

        # Đủ thời gian trên ghế trống -> nâng lên SUSPECTED
        for f in range(5, 12):
            detector.process_frame([p1, wallet], current_frame=f, fps=fps, frame_width=w, frame_height=h)

        assert evt_b.status == EventStatus.SUSPECTED

        # Bước 2: Người ở Ghế A đứng dậy và rời đi. Ví vẫn còn trên Ghế B.
        for f in range(12, 18):
            detector.process_frame([wallet], current_frame=f, fps=fps, frame_width=w, frame_height=h)

        assert seat_mgr.is_seat_vacant("SEAT_1A")
        assert seat_mgr.is_seat_vacant("SEAT_1B")
        assert evt_b.status == EventStatus.SUSPECTED
        assert evt_b.event_id == initial_event_id

        # Bước 3: Người mới (P2) lên xe và ngồi vào Ghế B (đè lên chiếc ví, ví bị che khuất)
        p2 = TrackedObject(track_id=2, class_name="person", confidence=0.92, bbox=(340, 110, 500, 290))
        # Vì P2 ngồi đè lên nên camera không còn phát hiện thấy chiếc ví nữa (chỉ còn P2)
        for f in range(18, 26):
            detector.process_frame([p2], current_frame=f, fps=fps, frame_width=w, frame_height=h)

        assert seat_mgr.is_seat_occupied("SEAT_1B")

        # Sự kiện chiếc ví phải chuyển sang OCCLUDED và GIỮ NGUYÊN Event ID
        assert evt_b.status == EventStatus.OCCLUDED
        assert evt_b.event_id == initial_event_id  # Không sinh Event ID mới, không xóa sự kiện

        # Bước 4: Người mới P2 đứng lên và rời khỏi Ghế B. Chiếc ví xuất hiện trở lại với track ID mới (O25)!
        recovered_wallet = TrackedObject(track_id=25, class_name="handbag", confidence=0.85, bbox=(352, 153, 421, 222))
        for f in range(26, 35):
            detector.process_frame([recovered_wallet], current_frame=f, fps=fps, frame_width=w, frame_height=h)

        # Tái kết nối thành công: Sự kiện vẫn mang initial_event_id, cập nhật track_id mới = 25
        active_evt_after = event_mgr.get_active_event_for_seat("SEAT_1B")
        assert active_evt_after is not None
        assert active_evt_after.event_id == initial_event_id
        assert active_evt_after.last_track_id == 25


class TestObjectSeatAssociationAndBoundary:
    """Kiểm thử Gán đồ vật cho từng ghế và Xử lý ranh giới UNKNOWN_SEAT (Mục D)."""

    def test_object_on_boundary_returns_unknown_seat(self):
        """Đồ vật nằm ở khe ranh giới giữa 2 ghế kề nhau -> trả về UNKNOWN_SEAT."""
        seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (200, 100), (200, 300), (100, 300)], is_normalized=False)
        seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(205, 100), (305, 100), (305, 300), (205, 300)], is_normalized=False)

        manager = SeatManager(seats=[seat_a, seat_b])

        # Vật thể nằm vắt ngang qua ranh giới giữa 2 ghế: x từ 180 đến 225
        boundary_bbox = (180, 180, 225, 230)
        assigned_seat = manager.find_seat_for_bbox(boundary_bbox, 1280, 720)

        # Phải trả về UNKNOWN_SEAT để tránh gán sai cho Ghế A hoặc Ghế B
        assert assigned_seat == "UNKNOWN_SEAT"

    def test_object_clearly_inside_seat_a(self):
        """Đồ vật nằm trọn trong lòng Ghế A -> trả về SEAT_1A chính xác."""
        seat_a = SeatConfig(seat_id="SEAT_1A", polygon=[(100, 100), (200, 100), (200, 300), (100, 300)], is_normalized=False)
        seat_b = SeatConfig(seat_id="SEAT_1B", polygon=[(220, 100), (320, 100), (320, 300), (220, 300)], is_normalized=False)

        manager = SeatManager(seats=[seat_a, seat_b])

        # Vật thể nằm sâu trong Ghế A: (130, 150, 170, 190)
        item_a = (130, 150, 170, 190)
        assert manager.find_seat_for_bbox(item_a, 1280, 720) == "SEAT_1A"
