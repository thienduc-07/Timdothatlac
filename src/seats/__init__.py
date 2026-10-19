# src/seats package
from src.seats.seat_config import SeatConfigManager, calculate_overlap_ratio, is_point_in_seat
from src.seats.seat_detector import (
    SeatDetector,
    SeatDetectionResult,
    SeatDetectionMethod,
    CustomSeatYOLOAdapter,
    CustomYOLOSeatDetector,
    BusGeometrySeatDetector,
    TwinSeatDecomposer,
    VALID_SEAT_CLASS_NAMES,
)
from src.seats.seat_tracker import SeatTracker, TrackedSeat
from src.seats.seat_occupancy import SeatOccupancyEstimator
from src.seats.seat_manager import SeatManager

__all__ = [
    "SeatConfigManager",
    "SeatManager",
    "SeatDetector",
    "SeatDetectionResult",
    "SeatDetectionMethod",
    "TwinSeatDecomposer",
    "CustomSeatYOLOAdapter",
    "CustomYOLOSeatDetector",
    "BusGeometrySeatDetector",
    "VALID_SEAT_CLASS_NAMES",
    "SeatTracker",
    "TrackedSeat",
    "SeatOccupancyEstimator",
    "calculate_overlap_ratio",
    "is_point_in_seat",
]
