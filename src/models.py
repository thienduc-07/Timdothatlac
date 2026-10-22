"""
src/models.py
Các dataclass dùng chung giữa các module.
Đảm bảo cấu trúc dữ liệu thống nhất trong toàn bộ pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class AlertPriority:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus:
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    RESOLVED = "RESOLVED"


@dataclass
class Detection:
    """Kết quả nhận diện một vật thể trong một frame."""
    class_name: str          # Tên lớp (ví dụ: "backpack", "person", "item", "seat")
    confidence: float        # Điểm tin cậy (0.0 - 1.0)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) - tọa độ bounding box
    class_id: int = -1       # ID lớp trong model
    polygon: Optional[List[Tuple[float, float]]] = None  # Tọa độ đa giác phân đoạn (segmentation mask polygon)


@dataclass
class TrackedObject:
    """Vật thể đang được theo dõi qua nhiều frame."""
    track_id: int            # ID tracking (do ByteTrack gán)
    class_name: str          # Tên lớp vật thể
    confidence: float        # Điểm tin cậy gần nhất
    bbox: Tuple[int, int, int, int]  # Bounding box hiện tại
    class_id: int = -1
    polygon: Optional[List[Tuple[float, float]]] = None  # Segmentation polygon gần nhất nếu có


@dataclass
class TrackObservation:
    """
    Lịch sử quan sát của một vật thể (ghi lại qua nhiều frame).
    Dùng để phân tích xem vật thể có đứng yên hay không.
    """
    track_id: int
    class_name: str
    first_seen_frame: int       # Frame đầu tiên nhìn thấy
    last_seen_frame: int        # Frame cuối cùng nhìn thấy
    first_seen_time: float      # Thời gian (giây) trong video khi xuất hiện lần đầu
    last_seen_time: float       # Thời gian (giây) trong video khi xuất hiện gần nhất
    positions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Danh sách bbox qua thời gian, dùng để tính chuyển động
    polygons: List[Optional[List[Tuple[float, float]]]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    nearby_person_track_ids: List[int] = field(default_factory=list)
    # Các track ID của người từng ở gần vật thể này
    person_last_seen_frame: int = -1
    # Frame cuối cùng mà có người ở gần vật thể

    @property
    def duration_seconds(self) -> float:
        """Thời gian tồn tại trong video (giây)."""
        return self.last_seen_time - self.first_seen_time

    @property
    def avg_confidence(self) -> float:
        """Confidence trung bình."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def center(self) -> Tuple[float, float]:
        """Tâm hiện tại của bounding box."""
        if not self.positions:
            return (0.0, 0.0)
        x1, y1, x2, y2 = self.positions[-1]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def max_displacement(self, n_recent: int = 10) -> float:
        """
        Tính khoảng di chuyển lớn nhất của vật thể trong n frame gần nhất.
        Dùng để xác định xem vật thể có đứng yên hay không.
        """
        recent = self.positions[-n_recent:] if len(self.positions) >= 2 else self.positions
        if len(recent) < 2:
            return 0.0

        max_disp = 0.0
        first_center = ((recent[0][0] + recent[0][2]) / 2,
                        (recent[0][1] + recent[0][3]) / 2)
        for bbox in recent[1:]:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            disp = ((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2) ** 0.5
            max_disp = max(max_disp, disp)
        return max_disp


class SeatState(str, Enum):
    """Trạng thái ước lượng của một ghế."""
    VACANT = "vacant"                  # Ghế trống
    EMPTY = "vacant"                   # Alias cho VACANT
    OCCUPIED = "occupied"              # Ghế có người ngồi / che
    PARTIALLY_OCCLUDED = "partially_occluded" # Ghế bị che một phần
    AMBIGUOUS = "ambiguous"            # Trạng thái mơ hồ / tranh chấp giữa 2 ghế
    UNKNOWN = "unknown"                # Chưa xác định / chưa đủ dữ liệu


@dataclass
class Alert:
    """Cảnh báo được sinh ra từ Event khi đủ điều kiện nghi ngờ bỏ quên."""
    alert_id: str                      # Mã cảnh báo duy nhất (UUID)
    event_id: str                      # Mã sự kiện tương ứng
    seat_id: str                       # Ghế liên quan (S01, S02, ...)
    object_id: str                     # Mã track / định danh vật thể (O01, O02, ...)
    person_id: str = ""                # Mã track người liên quan (P01, ...) nếu có
    priority: str = "MEDIUM"           # HIGH, MEDIUM, LOW
    status: str = "PENDING"            # PENDING, CONFIRMED, DISMISSED, RESOLVED
    reason: str = ""                   # Lý do sinh cảnh báo
    snapshot_path: str = ""            # Đường dẫn ảnh bằng chứng
    created_at: str = ""               # ISO timestamp
    sent_channels: List[str] = field(default_factory=list) # Danh sách kênh đã gửi thành công


@dataclass
class NotificationDelivery:
    """Nhật ký gửi thông báo qua các kênh (Telegram, Email, Webhook, Web Push)."""
    delivery_id: Optional[int] = None
    alert_id: str = ""
    channel: str = ""                  # in_app, telegram, email, webhook, web_push
    recipient: str = ""                # Chat ID, Email nhận, Webhook URL
    status: str = "PENDING"            # SUCCESS, FAILED, PENDING
    status_code: int = 0               # Mã phản hồi từ API nhà cung cấp
    error_message: str = ""            # Chi tiết lỗi nếu thất bại
    payload: dict = field(default_factory=dict)
    sent_at: str = ""


@dataclass
class DatasetImage:
    """Một ảnh trong dataset dùng để huấn luyện/cải thiện AI."""
    image_id: str
    image_path: str
    filename: str
    split: str = "train"               # train, val, test
    width: int = 0
    height: int = 0
    source_type: str = "upload"        # upload, video_extracted
    is_labeled: bool = False
    quality_tag: str = "good"          # good, low_quality, duplicate
    created_at: str = ""


@dataclass
class Annotation:
    """Nhãn bounding box của một đối tượng trong DatasetImage."""
    annotation_id: str
    image_id: str
    class_name: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    is_occluded: bool = False
    created_at: str = ""


@dataclass
class ModelVersion:
    """Thông tin một phiên bản model trong hệ thống."""
    model_id: str                      # Mã định danh model
    version: str                       # v1.0, v1.1...
    dataset_version: str = "v1.0"
    classes: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict) # {"map50": 0.85, "precision": 0.82, "recall": 0.80}
    confidence_threshold: float = 0.35
    status: str = "archived"           # active, candidate, archived
    artifact_path: str = ""
    notes: str = ""
    created_at: str = ""


@dataclass
class CameraConfig:
    """Cấu hình nguồn camera giám sát."""
    camera_id: str
    name: str
    source_type: str = "webcam"        # webcam, video_file, demo_sample, rtsp
    stream_url: str = ""
    fps: float = 15.0
    is_active: bool = True


@dataclass
class SeatConfig:
    """
    Cấu hình một vùng ghế (Seat ROI) trên frame camera.
    Tọa độ polygon có thể là tọa độ pixel hoặc normalized [0, 1].
    """
    seat_id: str                              # Mã ghế, ví dụ: "SEAT_01"
    polygon: List[Tuple[float, float]]        # Danh sách đỉnh [(x1, y1), (x2, y2), ...]
    is_normalized: bool = False               # True nếu tọa độ trong khoảng [0.0, 1.0]
    description: str = ""                     # Mô tả vị trí ghế (hàng ghế trái, phải...)

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """Chuyển đổi tọa độ polygon về pixel trên frame cụ thể."""
        if self.is_normalized:
            return [
                (int(pt[0] * frame_width), int(pt[1] * frame_height))
                for pt in self.polygon
            ]
        return [(int(pt[0]), int(pt[1])) for pt in self.polygon]


@dataclass
class SeatStatus:
    """Trạng thái thời gian thực của một ghế trong quá trình phân tích video."""
    seat_id: str
    state: SeatState = SeatState.UNKNOWN
    occupant_person_ids: List[int] = field(default_factory=list)
    overlap_ratio: float = 0.0                # Tỷ lệ diện tích giao nhau giữa person và ghế
    occupied_since_sec: float = -1.0          # Thời điểm bắt đầu có người (giây)
    vacant_since_sec: float = -1.0            # Thời điểm bắt đầu trống (giây)
    last_person_seen_time: float = -1.0       # Thời điểm gần nhất phát hiện người trên ghế
    consecutive_occupied_frames: int = 0      # Số frame liên tiếp phát hiện có người
    consecutive_vacant_frames: int = 0        # Số frame liên tiếp không có người
    notes: str = ""                           # Ghi chú trạng thái


@dataclass
class StateTransition:
    """Ghi nhận một lần chuyển đổi trạng thái trong State Machine."""
    from_state: str
    to_state: str
    timestamp: str                            # ISO timestamp hoặc video time
    frame_number: int
    reason: str


@dataclass
class Event:
    """
    Một sự kiện nghi ngờ bỏ quên đồ vật được quản lý theo vòng đời (State Machine).
    Lưu trữ đầy đủ lịch sử, vị trí ghế và bằng chứng che khuất.
    """
    event_id: Optional[str] = None            # Mã sự kiện ổn định (UUID)
    timestamp: str = ""                       # Thời điểm phát hiện (ISO format)
    class_name: str = ""                      # Tên lớp vật thể (backpack, handbag, ...)
    track_id: int = -1                        # Tracking ID ban đầu (tương thích ngược)
    last_track_id: int = -1                   # Tracking ID gần nhất
    seat_id: str = ""                         # Mã ghế liên kết (ví dụ: SEAT_01)
    confidence: float = 0.0                   # Điểm tin cậy quan sát
    duration_seconds: float = 0.0             # Thời gian đứng yên/tồn tại (tương thích ngược)
    status: str = "monitoring"                # Trạng thái hiện tại theo State Machine
    video_name: str = ""                      # Tên video nguồn
    video_time_sec: float = 0.0               # Thời điểm phát hiện trong video (giây)
    frame_number: int = 0                     # Số frame đầu tiên phát hiện
    snapshot_path: str = ""                   # Đường dẫn ảnh chụp frame bằng chứng
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0
    notes: str = ""                           # Ghi chú chi tiết

    # Các trường bổ sung cho Occlusion & Lifecyle:
    first_seen_timestamp: str = ""            # Thời điểm đầu tiên nhìn thấy
    last_seen_timestamp: str = ""             # Thời điểm gần nhất nhìn thấy
    last_seen_frame: int = 0                  # Frame cuối cùng nhìn thấy
    last_seen_time_sec: float = 0.0           # Giây video cuối cùng nhìn thấy
    suspected_timestamp: str = ""             # Thời điểm chuyển sang SUSPECTED
    suspected_time_sec: float = 0.0           # Giây video khi bắt đầu SUSPECTED
    occlusion_start_timestamp: str = ""       # Thời điểm bắt đầu bị che khuất
    occlusion_start_time_sec: float = 0.0     # Giây video khi bắt đầu bị che khuất
    occlusion_duration: float = 0.0           # Tổng thời gian bị che khuất (giây)
    observed_duration: float = 0.0            # Thời gian thực sự quan sát được vật thể (giây)
    vacant_duration: float = 0.0              # Thời gian ghế trống và vật thể đứng yên (giây)
    last_known_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    evidence_metadata: dict = field(default_factory=dict)
    state_history: List[dict] = field(default_factory=list)

    def add_transition(self, from_state: str, to_state: str, frame: int, time_sec: float, reason: str):
        """Ghi nhận một bước chuyển trạng thái vào lịch sử."""
        self.state_history.append({
            "from_state": from_state,
            "to_state": to_state,
            "frame": frame,
            "time_sec": time_sec,
            "reason": reason,
        })



@dataclass
class VideoMetadata:
    """Thông tin metadata của video đầu vào."""
    filename: str = ""
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    codec: str = ""
    is_valid: bool = False
    error: str = ""


@dataclass
class AnalysisResult:
    """Kết quả tổng hợp sau khi phân tích xong video."""
    video_name: str = ""
    total_frames_processed: int = 0
    total_objects_detected: int = 0
    total_events_created: int = 0
    output_video_path: str = ""
    events: List[Event] = field(default_factory=list)
    success: bool = False
    error_message: str = ""
    analysis_id: Optional[str] = None

    @property
    def total_frames(self) -> int:
        """Alias for total_frames_processed."""
        return self.total_frames_processed
