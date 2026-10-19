"""
src/database/dataset_repository.py
Repository quản lý bảng dataset_images và annotations.
Hỗ trợ upload ảnh, trích xuất frame, gán nhãn bounding box và xuất dữ liệu YOLO.
"""

from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetRepository:
    """Quản lý CRUD cho Dataset và Annotations trong SQLite."""

    def __init__(self, db_connection: sqlite3.Connection):
        self._conn = db_connection

    def add_image(
        self,
        image_path: str,
        filename: str,
        width: int,
        height: int,
        source_type: str = "upload",
        split: str = "train",
        image_id: Optional[str] = None,
    ) -> str:
        """Thêm một ảnh mới vào dataset."""
        img_id = image_id or f"IMG_{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now().isoformat()
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO dataset_images (
                    image_id, image_path, filename, split, width,
                    height, source_type, is_labeled, quality_tag, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    img_id,
                    image_path,
                    filename,
                    split,
                    width,
                    height,
                    source_type,
                    0,
                    "good",
                    now_iso,
                ),
            )
            self._conn.commit()
            return img_id
        except sqlite3.Error as e:
            logger.error(f"Lỗi thêm ảnh vào dataset: {e}")
            return ""

    def get_all_images(
        self, split: Optional[str] = None, labeled_only: Optional[bool] = None
    ) -> List[Dict]:
        """Lấy danh sách ảnh kèm bộ lọc."""
        try:
            cursor = self._conn.cursor()
            query = "SELECT * FROM dataset_images WHERE 1=1"
            params = []
            if split:
                query += " AND split = ?"
                params.append(split)
            if labeled_only is not None:
                query += " AND is_labeled = ?"
                params.append(1 if labeled_only else 0)

            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy danh sách ảnh dataset: {e}")
            return []

    def get_image_by_id(self, image_id: str) -> Optional[Dict]:
        """Lấy thông tin chi tiết của một ảnh."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM dataset_images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy ảnh theo ID: {e}")
            return None

    def delete_image(self, image_id: str) -> bool:
        """Xóa ảnh và các annotation liên quan."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            cursor.execute("DELETE FROM dataset_images WHERE image_id = ?", (image_id,))
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi xóa ảnh dataset: {e}")
            return False

    def save_annotations(self, image_id: str, annotations: List[Dict]) -> bool:
        """Lưu danh sách nhãn bounding box cho một ảnh."""
        try:
            cursor = self._conn.cursor()
            # Xóa các nhãn cũ của ảnh này
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            now_iso = datetime.now().isoformat()

            for ann in annotations:
                ann_id = ann.get("annotation_id") or f"ANN_{uuid.uuid4().hex[:8].upper()}"
                cursor.execute(
                    """
                    INSERT INTO annotations (
                        annotation_id, image_id, class_name,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, is_occluded, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ann_id,
                        image_id,
                        ann["class_name"],
                        ann["bbox_x1"],
                        ann["bbox_y1"],
                        ann["bbox_x2"],
                        ann["bbox_y2"],
                        1 if ann.get("is_occluded") else 0,
                        now_iso,
                    ),
                )

            # Cập nhật cờ is_labeled của ảnh
            is_labeled = 1 if len(annotations) > 0 else 0
            cursor.execute(
                "UPDATE dataset_images SET is_labeled = ? WHERE image_id = ?",
                (is_labeled, image_id),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi lưu annotations cho ảnh {image_id}: {e}")
            return False

    def get_annotations_for_image(self, image_id: str) -> List[Dict]:
        """Lấy tất cả annotation của một ảnh."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM annotations WHERE image_id = ?", (image_id,))
            return [dict(r) for r in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy annotations cho ảnh: {e}")
            return []

    def update_image_split(self, image_id: str, new_split: str) -> bool:
        """Cập nhật tập dữ liệu (train, val, test) cho ảnh."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE dataset_images SET split = ? WHERE image_id = ?",
                (new_split, image_id),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi cập nhật split: {e}")
            return False

    def get_dataset_stats(self) -> Dict:
        """Thống kê tổng quan về dataset: số lượng ảnh, phân bố lớp, phân chia split."""
        stats = {
            "total_images": 0,
            "labeled_images": 0,
            "splits": {"train": 0, "val": 0, "test": 0},
            "class_counts": {},
        }
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total, SUM(is_labeled) AS labeled FROM dataset_images")
            row = cursor.fetchone()
            if row:
                stats["total_images"] = row["total"] or 0
                stats["labeled_images"] = row["labeled"] or 0

            cursor.execute("SELECT split, COUNT(*) AS cnt FROM dataset_images GROUP BY split")
            for r in cursor.fetchall():
                if r["split"] in stats["splits"]:
                    stats["splits"][r["split"]] = r["cnt"]

            cursor.execute("SELECT class_name, COUNT(*) AS cnt FROM annotations GROUP BY class_name")
            for r in cursor.fetchall():
                stats["class_counts"][r["class_name"]] = r["cnt"]

            return stats
        except sqlite3.Error as e:
            logger.error(f"Lỗi lấy thống kê dataset: {e}")
            return stats
