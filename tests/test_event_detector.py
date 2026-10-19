"""
tests/test_event_detector.py
Kiểm thử logic phát hiện vật thể đứng yên và sự kiện nghi ngờ bỏ quên.
Sử dụng dữ liệu giả lập — không cần model AI hoặc video thật.
"""

import pytest
import sys
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TrackObservation
from src.events.forgotten_item_detector import ForgottenItemDetector
from config.settings import EventStatus


class TestStationaryDetection:
    """Test logic xác định vật thể đứng yên."""

    def test_stationary_object_detected(self):
        """Vật thể đứng yên trong ngưỡng → is_stationary = True."""
        positions = [(100, 100, 150, 150)] * 20  # Không di chuyển
        result = ForgottenItemDetector.is_stationary_from_positions(
            positions, threshold=30.0
        )
        assert result is True

    def test_moving_object_not_stationary(self):
        """Vật thể di chuyển nhiều → is_stationary = False."""
        positions = [(100 + i * 50, 100, 150 + i * 50, 150) for i in range(10)]
        result = ForgottenItemDetector.is_stationary_from_positions(
            positions, threshold=30.0
        )
        assert result is False

    def test_slight_movement_still_stationary(self):
        """Vật thể di chuyển nhẹ (< threshold) → vẫn đứng yên."""
        # Di chuyển 10 pixel — dưới ngưỡng 30
        positions = [
            (100, 100, 150, 150),
            (105, 102, 155, 152),
            (103, 101, 153, 151),
            (108, 104, 158, 154),
        ]
        result = ForgottenItemDetector.is_stationary_from_positions(
            positions, threshold=30.0
        )
        assert result is True

    def test_single_position_is_stationary(self):
        """Chỉ có 1 vị trí → coi như đứng yên (chưa đủ dữ liệu)."""
        positions = [(100, 100, 150, 150)]
        result = ForgottenItemDetector.is_stationary_from_positions(
            positions, threshold=30.0
        )
        assert result is True

    def test_empty_positions(self):
        """Danh sách rỗng → đứng yên (không có chuyển động)."""
        positions = []
        result = ForgottenItemDetector.is_stationary_from_positions(
            positions, threshold=30.0
        )
        assert result is True


class TestForgottenItemAnalysis:
    """Test logic phân tích sự kiện nghi ngờ bỏ quên."""

    def _make_observation(
        self,
        track_id=1,
        class_name="backpack",
        first_frame=0,
        last_frame=300,
        fps=30.0,
        n_positions=20,
        stationary=True,
        nearby_person=True,
        person_last_frame=-1,
    ):
        """Hàm tiện ích tạo TrackObservation giả lập."""
        if stationary:
            positions = [(100, 100, 200, 200)] * n_positions
        else:
            positions = [(100 + i * 50, 100, 200 + i * 50, 200) for i in range(n_positions)]

        obs = TrackObservation(
            track_id=track_id,
            class_name=class_name,
            first_seen_frame=first_frame,
            last_seen_frame=last_frame,
            first_seen_time=first_frame / fps,
            last_seen_time=last_frame / fps,
            positions=positions,
            confidence_history=[0.8] * n_positions,
        )

        if nearby_person:
            obs.nearby_person_track_ids = [99]
            obs.person_last_seen_frame = person_last_frame if person_last_frame >= 0 else first_frame

        return obs

    def test_stationary_item_creates_monitoring_event(self):
        """Vật thể đứng yên đủ lâu → tạo sự kiện monitoring."""
        detector = ForgottenItemDetector(
            min_stationary_time=5.0,
            movement_threshold=30.0,
        )

        obs = self._make_observation(
            first_frame=0, last_frame=300, fps=30.0,
            stationary=True, nearby_person=False,
        )

        events = detector.analyze(
            item_observations={1: obs},
            current_frame=300,
            fps=30.0,
        )

        assert len(events) >= 1
        assert events[0]["status"] in (EventStatus.MONITORING, EventStatus.SUSPECTED)

    def test_moving_item_no_event(self):
        """Vật thể di chuyển → không tạo sự kiện."""
        detector = ForgottenItemDetector(
            min_stationary_time=5.0,
            movement_threshold=30.0,
        )

        obs = self._make_observation(
            first_frame=0, last_frame=300, fps=30.0,
            stationary=False, nearby_person=False,
        )

        events = detector.analyze(
            item_observations={1: obs},
            current_frame=300,
            fps=30.0,
        )

        # Vật thể di chuyển → không có sự kiện
        assert len(events) == 0

    def test_person_left_creates_suspected_event(self):
        """Người rời đi + đồ vật đứng yên → nghi ngờ bỏ quên."""
        detector = ForgottenItemDetector(
            min_stationary_time=5.0,
            movement_threshold=30.0,
            alert_delay=3.0,
        )

        # Người rời đi từ frame 100, hiện tại frame 300 (fps=30 → >3s)
        obs = self._make_observation(
            first_frame=0, last_frame=300, fps=30.0,
            stationary=True, nearby_person=True,
            person_last_frame=100,
        )

        events = detector.analyze(
            item_observations={1: obs},
            current_frame=300,
            fps=30.0,
        )

        assert len(events) >= 1
        assert events[0]["status"] == EventStatus.SUSPECTED

    def test_non_watched_class_ignored(self):
        """Lớp không trong danh sách theo dõi → bỏ qua."""
        detector = ForgottenItemDetector(watched_classes=["backpack"])

        obs = self._make_observation(
            class_name="bottle",  # Không trong watched
            first_frame=0, last_frame=300, fps=30.0,
            stationary=True,
        )

        events = detector.analyze(
            item_observations={1: obs},
            current_frame=300,
            fps=30.0,
        )

        assert len(events) == 0

    def test_short_duration_no_event(self):
        """Vật thể xuất hiện quá ngắn → không tạo sự kiện."""
        detector = ForgottenItemDetector(min_stationary_time=10.0)

        obs = self._make_observation(
            first_frame=0, last_frame=60, fps=30.0,  # 2 giây
            stationary=True,
        )

        events = detector.analyze(
            item_observations={1: obs},
            current_frame=60,
            fps=30.0,
        )

        assert len(events) == 0


class TestDuplicateEventPrevention:
    """Test chống tạo sự kiện trùng lặp."""

    def test_same_track_no_duplicate(self):
        """Cùng track_id, gọi analyze nhiều lần → không tạo trùng."""
        detector = ForgottenItemDetector(min_stationary_time=5.0)

        obs = self._make_observation(
            first_frame=0, last_frame=300, fps=30.0,
            stationary=True, nearby_person=False,
        )

        # Lần 1
        events1 = detector.analyze({1: obs}, current_frame=300, fps=30.0)
        # Lần 2 (cùng track)
        events2 = detector.analyze({1: obs}, current_frame=301, fps=30.0)

        # Lần 2 không nên tạo lại monitoring cho cùng track
        # (đã có trạng thái trong _track_status)
        monitoring_events_2 = [
            e for e in events2 if e["status"] == EventStatus.MONITORING
        ]
        assert len(monitoring_events_2) == 0

    def _make_observation(self, **kwargs):
        """Hàm tiện ích giống TestForgottenItemAnalysis."""
        return TestForgottenItemAnalysis._make_observation(self, **kwargs)


class TestEventStatusUpdate:
    """Test cập nhật trạng thái sự kiện."""

    def test_update_monitoring_to_suspected(self):
        """Chuyển từ monitoring lên suspected."""
        detector = ForgottenItemDetector(
            min_stationary_time=5.0,
            movement_threshold=30.0,
        )

        obs = TrackObservation(
            track_id=1, class_name="backpack",
            first_seen_frame=0, last_seen_frame=1000,
            first_seen_time=0.0, last_seen_time=33.3,
            positions=[(100, 100, 200, 200)] * 30,
            confidence_history=[0.8] * 30,
        )

        # Frame ban đầu → monitoring
        events1 = detector.analyze({1: obs}, current_frame=300, fps=30.0)

        # Thời gian đứng yên tăng lên (> 3x min_time) → nâng cấp suspected
        obs.last_seen_frame = 1000
        obs.last_seen_time = 33.3  # > 15s = 3 * 5s
        events2 = detector.analyze({1: obs}, current_frame=1000, fps=30.0)

        suspected = [e for e in events2 if e["status"] == EventStatus.SUSPECTED]
        assert len(suspected) == 1
