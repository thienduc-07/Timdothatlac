"""
src/detection/class_mapper.py

Map raw model class names (COCO or custom) to the semantic entities used
by the business logic.

Semantic entities
-----------------
* seat            – physical seat geometry (ROI calculations).
* person          – human passenger.
* item            – personal belongings.
* seat_state_empty        – seat occupancy: vacant (from custom model).
* seat_state_occupied     – seat occupancy: occupied (from custom model).
* seat_state_object       – seat occupancy: object on seat (from custom model).

Roboflow project classes
------------------------
The project *bus-seat-and-belongings-monitori* uses three classes that
describe seat **state**, not independent objects:

    empty_seat       → seat_state_empty
    human_occupied   → seat_state_occupied
    object_occupied  → seat_state_object

These classes annotate the seat region according to what is on it.  They
are NOT equivalent to detecting a person or an item independently.

The COCO pretrained model (YOLO26n) detects ``person``, ``backpack``,
``handbag``, etc. as distinct object classes.

The mapping is configurable; unknown classes are returned unchanged.
"""

from typing import Dict, Optional


# ── default mapping ─────────────────────────────────────────────────
DEFAULT_MAPPING: Dict[str, str] = {
    # Physical seat geometry
    "seat": "seat",

    # Roboflow custom model seat-state classes
    # These describe what state the SEAT REGION is in, not individual objects.
    "empty_seat": "seat_state_empty",
    "human_occupied": "seat_state_occupied",
    "object_occupied": "seat_state_object",

    # COCO / generic
    "person": "person",
    "item": "item",
}


# ── COCO item classes that should map to "item" ────────────────────
COCO_ITEM_CLASSES = frozenset({
    "backpack", "handbag", "suitcase", "umbrella",
    "bottle", "cell phone", "laptop", "book",
})


def map_class(
    raw_name: str,
    mapping: Optional[Dict[str, str]] = None,
) -> str:
    """Map a raw class name to an internal semantic label.

    Parameters
    ----------
    raw_name : str
        Class name returned by the model.
    mapping : dict or None
        Optional custom mapping; if ``None`` the ``DEFAULT_MAPPING`` is used.

    Returns
    -------
    str
        The mapped semantic label, or the original name (lower-cased)
        if no mapping is known.
    """
    raw_lower = raw_name.strip().lower()
    m = mapping or DEFAULT_MAPPING
    if raw_lower in m:
        return m[raw_lower]
    # COCO common item classes → "item"
    if raw_lower in COCO_ITEM_CLASSES:
        return "item"
    return raw_lower


def is_seat_state_class(semantic_label: str) -> bool:
    """Return ``True`` if the label is a seat-state class from the custom model."""
    return semantic_label.startswith("seat_state_")


def seat_state_to_occupancy(semantic_label: str) -> Optional[str]:
    """Convert a seat-state label to a simple occupancy string.

    Returns ``"vacant"``, ``"occupied"``, ``"object"``, or ``None``.
    """
    return {
        "seat_state_empty": "vacant",
        "seat_state_occupied": "occupied",
        "seat_state_object": "object",
    }.get(semantic_label)


def format_seat_label(seat_id: str) -> str:
    """Format a seat identifier into simplified user-facing Vietnamese text.

    Examples:
        "SEAT_001" -> "Ghế 001"
        "SEAT_1A"  -> "Ghế 1A"
        "1"        -> "Ghế 001"
    """
    sid = str(seat_id).strip()
    clean_id = sid.upper()
    if clean_id.startswith("SEAT_") or clean_id.startswith("SEAT-") or clean_id.startswith("SEAT "):
        sub = clean_id[5:].strip()
        if sub.isdigit():
            return f"Ghế {int(sub):03d}"
        return f"Ghế {sub}"
    elif clean_id.startswith("GHẾ ") or clean_id.startswith("GHE "):
        sub = clean_id.split(None, 1)[1].strip()
        if sub.isdigit():
            return f"Ghế {int(sub):03d}"
        return f"Ghế {sub}"
    elif clean_id.isdigit():
        return f"Ghế {int(clean_id):03d}"
    return f"Ghế {sid}"


def get_user_facing_label(
    class_name: str,
    is_alert: bool = False,
    seat_id: Optional[str] = None,
) -> str:
    """Return simplified user-facing Vietnamese label according to the requirements:
    - Seats: Ghế 001, Ghế 002, etc.
    - People: Người
    - All detected belongings and other relevant objects: Đồ vật
    - Potential forgotten-item alerts: Cảnh báo: Đồ vật có thể bị bỏ quên

    Do not display detailed object names such as phone, wallet, backpack,
    or suitcase in the UI, even if the underlying model predicts those classes.
    """
    if is_alert:
        if seat_id:
            formatted_seat = format_seat_label(seat_id)
            return f"Cảnh báo: Đồ vật có thể bị bỏ quên ({formatted_seat})"
        return "Cảnh báo: Đồ vật có thể bị bỏ quên"

    raw_lower = str(class_name).strip().lower()
    mapped = map_class(raw_lower)

    if mapped == "person" or raw_lower == "person":
        return "Người"
    elif mapped.startswith("seat_state_") or "seat" in raw_lower or mapped == "seat":
        if seat_id:
            return format_seat_label(seat_id)
        return "Ghế"
    else:
        # All belongings, items, backpacks, phones, suitcases, etc.
        return "Đồ vật"

