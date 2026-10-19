"""
tests/test_occlusion_and_state_machine.py
Kiểm thử toàn diện 10 tình huống bắt buộc theo yêu cầu hệ thống:
1. Đồ vật trên ghế có người: chưa tạo cảnh báo bỏ quên (OBSERVED).
2. Người đứng lên, đồ vật còn trên ghế trống: bắt đầu theo dõi (ITEM_LEFT).
3. Đồ vật tồn tại trên ghế trống đủ thời gian: tạo sự kiện nghi ngờ (SUSPECTED).
4. Người mới ngồi xuống che đồ: chuyển sang OCCLUDED, giữ nguyên Event ID.
5. Người ngồi che đồ rồi đứng lên, đồ xuất hiện lại: khôi phục sự kiện cũ nếu bằng chứng phù hợp.
6. Người ngồi che đồ rồi đứng lên, đồ không xuất hiện lại: không tự kết luận đồ đã bị lấy, chuyển REVIEW_REQUIRED.
7. Detection mất trong một frame do nhiễu: không tạo cảnh báo mới.
8. Vùng ghế không bị che nhưng đồ biến mất: chuyển sang xác minh phù hợp (RECHECK -> REVIEW_REQUIRED).
9. Một vật thể bị mất tracking ID rồi xuất hiện lại: không tạo sự kiện trùng lặp nếu có thể nối lại bằng chứng.
10. Nhiều ghế cùng có sự kiện: các sự kiện phải được quản lý hoàn toàn độc lập.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SeatConfig, TrackedObject, Event
from src.seats.seat_manager import SeatManager
from src.events.occlusion_manager import OcclusionManager
from src.events.event_manager import EventManager
from src.events.forgotten_item_detector import ForgottenItemDetector
from config.settings import EventStatus


@pytest.fixture
def test_setup():
    """Tạo môi trường giả lập chuẩn cho xe buýt gồm 2 ghế."""
    seat1 = SeatConfig(
        seat_id="SEAT_01",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        is_normalized=False,
        description="Ghế 1",
    )
    seat2 = SeatConfig(
        seat_id="SEAT_02",
        polygon=[(500, 100), (700, 100), (700, 300), (500, 300)],
        is_normalized=False,
        description="Ghế 2",
    )

    seat_mgr = SeatManager(
        seats=[seat1, seat2],
        occupied_confirm_sec=0.2,
        vacant_confirm_sec=0.3,
        hold_sec=0.3,
    )
    event_mgr = EventManager()
    occlusion_mgr = OcclusionManager(max_reassoc_dist=120.0, max_occlusion_time=60.0)

    detector = ForgottenItemDetector(
        min_stationary_time=2.0,  # 2 giây trên ghế trống -> SUSPECTED
        movement_threshold=30.0,
        recheck_duration=1.0,     # 1 giây RECHECK trước khi sang REVIEW_REQUIRED
        seat_manager=seat_mgr,
        occlusion_manager=occlusion_mgr,
        event_manager=event_mgr,
    )

    return {
        "seat_mgr": seat_mgr,
        "event_mgr": event_mgr,
        "detector": detector,
    }


class TestMandatoryScenarios:
    """Kiểm thử 10 kịch bản bắt buộc theo đặc tả nghiệp vụ."""

    def test_scenario_01_item_on_occupied_seat_no_alert(self, test_setup):
        """
        Kịch bản 1: Đồ vật trên ghế có người ngồi.
        Yêu cầu: Không tạo cảnh báo bỏ quên (chỉ ở trạng thái OBSERVED, chưa SUSPECTED).
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # Người và chiếc ví cùng xuất hiện trên SEAT_01
        person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 110, 280, 290))
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))

        # Giả lập 20 frames (2.0 giây ở 10fps)
        for f in range(20):
            detector.process_frame(
                tracked_objects=[person, wallet],
                current_frame=f,
                fps=10.0,
                frame_width=1000,
                frame_height=1000,
            )

        events = event_mgr.events
        assert len(events) == 1
        # Trạng thái phải là OBSERVED, tuyệt đối chưa được là SUSPECTED vì người vẫn ở đó
        assert events[0].status == EventStatus.OBSERVED
        assert events[0].status != EventStatus.SUSPECTED

    def test_scenario_02_person_leaves_item_left_tracked(self, test_setup):
        """
        Kịch bản 2: Người đứng lên rời ghế, chiếc ví vẫn còn trên ghế trống.
        Yêu cầu: Bắt đầu theo dõi, chuyển trạng thái sang ITEM_LEFT.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 110, 280, 290))
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))

        # Ban đầu có người
        for f in range(5):
            detector.process_frame([person, wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Người đứng dậy (không còn person trong detection), chiếc ví vẫn còn
        for f in range(5, 12):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        events = event_mgr.events
        assert len(events) == 1
        assert events[0].status == EventStatus.ITEM_LEFT
        assert events[0].seat_id == "SEAT_01"

    def test_scenario_03_item_on_vacant_seat_becomes_suspected(self, test_setup):
        """
        Kịch bản 3: Đồ vật tồn tại trên ghế trống đủ thời gian quy định (min_stationary_time = 2.0s).
        Yêu cầu: Tạo sự kiện nghi ngờ (SUSPECTED).
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        person = TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=(120, 110, 280, 290))
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))

        # Frame 0-4: có người
        for f in range(5):
            detector.process_frame([person, wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Frame 5-35: người đi, ví ở lại trên ghế trống 3.0s (> 2.0s min_stationary_time)
        for f in range(5, 35):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        events = event_mgr.events
        assert len(events) == 1
        assert events[0].status == EventStatus.SUSPECTED
        assert events[0].suspected_timestamp != ""

    def test_scenario_04_new_passenger_sits_down_causes_occlusion(self, test_setup):
        """
        Kịch bản 4: Người mới ngồi xuống che khuất chiếc ví.
        Yêu cầu: Chuyển sang OCCLUDED, giữ nguyên Event ID, không xóa sự kiện.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # Ví đang ở trạng thái SUSPECTED trên SEAT_01
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))
        for f in range(30):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert event_mgr.events[0].status == EventStatus.SUSPECTED
        original_event_id = event_mgr.events[0].event_id

        # Hành khách mới (person 99) ngồi xuống SEAT_01, che khuất ví -> YOLO không còn thấy ví
        new_person = TrackedObject(track_id=99, class_name="person", confidence=0.95, bbox=(120, 110, 280, 290))
        for f in range(30, 40):
            detector.process_frame([new_person], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        events = event_mgr.events
        assert len(events) == 1
        event = events[0]
        # Không được xóa sự kiện
        assert event.event_id == original_event_id
        # Chuyển sang OCCLUDED
        assert event.status == EventStatus.OCCLUDED
        # Có ghi nhận thời gian bắt đầu che khuất
        assert event.occlusion_duration > 0.0

    def test_scenario_05_person_leaves_item_reappears_reassociated(self, test_setup):
        """
        Kịch bản 5: Người ngồi che đứng lên, chiếc ví xuất hiện trở lại.
        Yêu cầu: Khôi phục sự kiện cũ nếu bằng chứng phù hợp (giữ nguyên Event ID).
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # 1. Ví xuất hiện và trở thành SUSPECTED
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))
        for f in range(30):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        orig_id = event_mgr.events[0].event_id

        # 2. Người mới ngồi lên che
        new_person = TrackedObject(track_id=99, class_name="person", confidence=0.95, bbox=(120, 110, 280, 290))
        for f in range(30, 45):
            detector.process_frame([new_person], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert event_mgr.events[0].status == EventStatus.OCCLUDED

        # 3. Người đứng lên, ví xuất hiện lại (có thể với track_id mới do ByteTrack bị đứt: track_id=105)
        reappeared_wallet = TrackedObject(
            track_id=105,  # ByteTrack ID mới
            class_name="handbag",
            confidence=0.88,
            bbox=(152, 148, 202, 198),  # Gần vị trí cũ (sai số vài pixel)
        )

        for f in range(45, 50):
            detector.process_frame([reappeared_wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Sự kiện phải được nối lại, không tạo sự kiện thứ 2
        events = event_mgr.events
        assert len(events) == 1
        assert events[0].event_id == orig_id
        # Trạng thái khôi phục lại (SUSPECTED)
        assert events[0].status == EventStatus.SUSPECTED
        assert events[0].last_track_id == 105

    def test_scenario_06_person_leaves_item_not_reappearing_needs_review(self, test_setup):
        """
        Kịch bản 6: Người ngồi che đứng lên, nhưng đồ vật không xuất hiện lại.
        Yêu cầu: Không tự kết luận đồ đã bị lấy đi, chuyển sang REVIEW_REQUIRED.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # Ví ban đầu -> SUSPECTED
        wallet = TrackedObject(track_id=2, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))
        for f in range(30):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Người ngồi che
        person = TrackedObject(track_id=99, class_name="person", confidence=0.95, bbox=(120, 110, 280, 290))
        for f in range(30, 45):
            detector.process_frame([person], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Người đứng lên, ghế trống hoàn toàn (cả người lẫn ví đều không còn)
        # Chạy trong 2 giây (20 frames, vượt quá recheck_duration = 1.0s)
        for f in range(45, 70):
            detector.process_frame([], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        events = event_mgr.events
        assert len(events) == 1
        # Phải chuyển sang REVIEW_REQUIRED để nhân viên kiểm tra, không tự tiện xoá hay kết luận
        assert events[0].status == EventStatus.REVIEW_REQUIRED

    def test_scenario_07_single_frame_noise_does_not_break_state(self, test_setup):
        """
        Kịch bản 7: Detection bị mất trong 1 frame do nhiễu sensor/ánh sáng.
        Yêu cầu: Không tạo cảnh báo mới, duy trì trạng thái ổn định.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        wallet = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(150, 150, 200, 200))

        # Quan sát 10 frame
        for f in range(10):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        orig_id = event_mgr.events[0].event_id

        # Mất 1 frame duy nhất tại frame 10
        detector.process_frame([], current_frame=10, fps=10.0, frame_width=1000, frame_height=1000)

        # Frame 11: Ví xuất hiện trở lại
        detector.process_frame([wallet], current_frame=11, fps=10.0, frame_width=1000, frame_height=1000)

        # Vẫn chỉ có 1 sự kiện duy nhất
        assert len(event_mgr.events) == 1
        assert event_mgr.events[0].event_id == orig_id

    def test_scenario_08_unoccluded_item_disappears_triggers_recheck(self, test_setup):
        """
        Kịch bản 8: Vùng ghế không bị che nhưng đồ biến mất.
        Yêu cầu: Chuyển sang RECHECK, nếu quá thời hạn chuyển REVIEW_REQUIRED.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # Ví trên ghế trống -> SUSPECTED
        wallet = TrackedObject(track_id=2, class_name="backpack", confidence=0.85, bbox=(150, 150, 200, 200))
        for f in range(30):
            detector.process_frame([wallet], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert event_mgr.events[0].status == EventStatus.SUSPECTED

        # Đồ biến mất đột ngột mà không có ai che ghế
        # Chạy 8 frames -> vào RECHECK
        for f in range(30, 38):
            detector.process_frame([], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert event_mgr.events[0].status in (EventStatus.RECHECK, EventStatus.REVIEW_REQUIRED)

        # Chạy thêm 20 frames nữa (vượt quá recheck_duration = 1.0s)
        for f in range(38, 60):
            detector.process_frame([], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        assert event_mgr.events[0].status == EventStatus.REVIEW_REQUIRED

    def test_scenario_09_lost_tracking_id_reconnected_by_evidence(self, test_setup):
        """
        Kịch bản 9: Vật thể bị mất ByteTrack ID rồi xuất hiện lại với ID khác.
        Yêu cầu: Không tạo sự kiện trùng lặp nếu có thể nối lại bằng chứng.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        item_old = TrackedObject(track_id=5, class_name="umbrella", confidence=0.8, bbox=(140, 140, 180, 180))
        for f in range(15):
            detector.process_frame([item_old], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        orig_id = event_mgr.events[0].event_id

        # Mất 3 frame do người đi ngang qua che
        person = TrackedObject(track_id=88, class_name="person", confidence=0.9, bbox=(120, 110, 280, 290))
        for f in range(15, 20):
            detector.process_frame([person], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Xuất hiện lại với track_id mới 999 nhưng cùng ghế SEAT_01 và vị trí gần kề
        item_new = TrackedObject(track_id=999, class_name="umbrella", confidence=0.82, bbox=(142, 141, 182, 181))
        for f in range(20, 25):
            detector.process_frame([item_new], current_frame=f, fps=10.0, frame_width=1000, frame_height=1000)

        # Tổng số sự kiện vẫn chỉ là 1 (không bị nhân bản)
        assert len(event_mgr.events) == 1
        assert event_mgr.events[0].event_id == orig_id
        assert event_mgr.events[0].last_track_id == 999

    def test_scenario_10_multiple_seats_independent_events(self, test_setup):
        """
        Kịch bản 10: Nhiều ghế cùng có sự kiện đồng thời.
        Yêu cầu: Các sự kiện phải được quản lý độc lập trên từng ghế.
        """
        detector = test_setup["detector"]
        event_mgr = test_setup["event_mgr"]

        # Ví trên SEAT_01 và ba lô trên SEAT_02
        wallet_seat1 = TrackedObject(track_id=10, class_name="handbag", confidence=0.85, bbox=(150, 150, 200, 200))
        backpack_seat2 = TrackedObject(track_id=20, class_name="backpack", confidence=0.90, bbox=(550, 150, 650, 250))

        # Cả 2 đều đứng yên trên ghế trống đủ thời gian -> đều trở thành SUSPECTED
        for f in range(30):
            detector.process_frame(
                [wallet_seat1, backpack_seat2],
                current_frame=f,
                fps=10.0,
                frame_width=1000,
                frame_height=1000,
            )

        events = event_mgr.events
        assert len(events) == 2

        seat_ids = {e.seat_id for e in events}
        assert seat_ids == {"SEAT_01", "SEAT_02"}

        # Bây giờ một người ngồi lên SEAT_01 che ví, nhưng SEAT_02 không có người
        person_seat1 = TrackedObject(track_id=77, class_name="person", confidence=0.95, bbox=(120, 110, 280, 290))
        for f in range(30, 40):
            detector.process_frame(
                [person_seat1, backpack_seat2],
                current_frame=f,
                fps=10.0,
                frame_width=1000,
                frame_height=1000,
            )

        evt_seat1 = next(e for e in event_mgr.events if e.seat_id == "SEAT_01")
        evt_seat2 = next(e for e in event_mgr.events if e.seat_id == "SEAT_02")

        # Sự kiện ở SEAT_01 bị che khuất (OCCLUDED)
        assert evt_seat1.status == EventStatus.OCCLUDED
        # Sự kiện ở SEAT_02 vẫn tiếp tục SUSPECTED bình thường, hoàn toàn độc lập
        assert evt_seat2.status == EventStatus.SUSPECTED
