"""
src/services/dataset_service.py
Quản lý tập dữ liệu hình ảnh, gán nhãn annotation và xuất file định dạng YOLO (data.yaml, images, labels).
Hỗ trợ upload ảnh đơn lẻ, upload nhiều ảnh hoặc trích xuất frame từ video giám sát.
"""

import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple
import zipfile
import cv2
import numpy as np

from src.database.dataset_repository import DatasetRepository
from src.utils.logger import get_logger
from config.settings import DATA_DIR, ITEM_CLASSES, PERSON_CLASS

logger = get_logger(__name__)

# Thư mục lưu trữ dataset
DATASET_DIR = DATA_DIR / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATASET_EXPORTS_DIR = DATASET_DIR / "exports"

for d in [DATASET_DIR, DATASET_IMAGES_DIR, DATASET_EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Danh sách lớp chuẩn cho YOLO export
YOLO_CLASSES = [PERSON_CLASS] + [c for c in ITEM_CLASSES if c != PERSON_CLASS]
CLASS_TO_IDX = {c: i for i, c in enumerate(YOLO_CLASSES)}


class DatasetService:
    """Xử lý nghiệp vụ cho trang AI Training & Dataset."""

    def __init__(self, repo: Optional[DatasetRepository] = None):
        self._repo = repo

    def set_repository(self, repo: DatasetRepository):
        self._repo = repo

    def add_image_from_bytes(
        self, file_bytes: bytes, filename: str, split: str = "train", source_type: str = "upload"
    ) -> Optional[str]:
        """Lưu file ảnh upload vào đĩa và thêm vào database."""
        try:
            # Đọc kích thước ảnh bằng OpenCV
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]

            # Lưu ảnh vào DATASET_IMAGES_DIR
            safe_name = f"{Path(filename).stem}_{os.urandom(4).hex()}{Path(filename).suffix}"
            save_path = DATASET_IMAGES_DIR / safe_name
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            if self._repo:
                img_id = self._repo.add_image(
                    image_path=str(save_path),
                    filename=filename,
                    width=w,
                    height=h,
                    source_type=source_type,
                    split=split,
                )
                return img_id
            return safe_name
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh dataset: {e}")
            return None

    def extract_frames_from_video(
        self, video_path: str, interval_seconds: float = 2.0, max_frames: int = 40
    ) -> List[str]:
        """Trích xuất các frame từ video theo khoảng thời gian để tạo dữ liệu gán nhãn."""
        extracted_ids = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = max(1, int(fps * interval_seconds))
            frame_idx = 0
            count = 0

            video_stem = Path(video_path).stem
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_interval == 0:
                    h, w = frame.shape[:2]
                    frame_filename = f"{video_stem}_f{frame_idx}_{os.urandom(3).hex()}.jpg"
                    save_path = DATASET_IMAGES_DIR / frame_filename
                    cv2.imwrite(str(save_path), frame)

                    if self._repo:
                        img_id = self._repo.add_image(
                            image_path=str(save_path),
                            filename=frame_filename,
                            width=w,
                            height=h,
                            source_type="video_extracted",
                            split="train",
                        )
                        if img_id:
                            extracted_ids.append(img_id)
                    count += 1

                frame_idx += 1

            cap.release()
            logger.info(f"Đã trích xuất {len(extracted_ids)} frame từ video: {video_path}")
            return extracted_ids
        except Exception as e:
            logger.error(f"Lỗi trích xuất frame từ video: {e}")
            return []

    def get_images(self, split: Optional[str] = None, labeled_only: Optional[bool] = None) -> List[Dict]:
        """Lấy danh sách ảnh trong dataset."""
        if self._repo:
            return self._repo.get_all_images(split=split, labeled_only=labeled_only)
        return []

    def save_annotations(self, image_id: str, annotations: List[Dict]) -> bool:
        """Lưu danh sách nhãn cho một ảnh."""
        if self._repo:
            return self._repo.save_annotations(image_id, annotations)
        return False

    def get_annotations(self, image_id: str) -> List[Dict]:
        """Lấy nhãn của một ảnh."""
        if self._repo:
            return self._repo.get_annotations_for_image(image_id)
        return []

    def delete_image(self, image_id: str) -> bool:
        """Xóa ảnh khỏi dataset."""
        if self._repo:
            img = self._repo.get_image_by_id(image_id)
            if img and Path(img["image_path"]).exists():
                try:
                    Path(img["image_path"]).unlink()
                except Exception:
                    pass
            return self._repo.delete_image(image_id)
        return False

    def get_stats(self) -> Dict:
        """Lấy thống kê dataset."""
        if self._repo:
            return self._repo.get_dataset_stats()
        return {"total_images": 0, "labeled_images": 0, "splits": {}, "class_counts": {}}

    def export_yolo_dataset(self) -> Tuple[bool, str, str]:
        """
        Xuất toàn bộ dataset ra định dạng chuẩn YOLOv8:
        dataset/
          ├── data.yaml
          ├── images/
          │    ├── train/
          │    ├── val/
          │    └── test/
          └── labels/
               ├── train/
               ├── val/
               └── test/
        Trả về: (thành_công, đường_dẫn_zip, thông_báo)
        """
        if not self._repo:
            return False, "", "Database chưa kết nối"

        images = self._repo.get_all_images()
        if not images:
            return False, "", "Dataset chưa có hình ảnh nào để export"

        export_id = f"yolo_dataset_{os.urandom(4).hex()}"
        export_root = DATASET_EXPORTS_DIR / export_id

        try:
            # Tạo các thư mục con
            for split in ["train", "val", "test"]:
                (export_root / "images" / split).mkdir(parents=True, exist_ok=True)
                (export_root / "labels" / split).mkdir(parents=True, exist_ok=True)

            exported_count = 0
            for img_info in images:
                src_img_path = Path(img_info["image_path"])
                if not src_img_path.exists():
                    continue

                split = img_info.get("split") or "train"
                if split not in ("train", "val", "test"):
                    split = "train"

                # Copy ảnh
                dest_img_path = export_root / "images" / split / src_img_path.name
                shutil.copy2(src_img_path, dest_img_path)

                # Lấy annotations và tạo file .txt
                anns = self._repo.get_annotations_for_image(img_info["image_id"])
                label_txt_path = export_root / "labels" / split / f"{src_img_path.stem}.txt"

                w = img_info.get("width") or 1280
                h = img_info.get("height") or 720

                lines = []
                for ann in anns:
                    cname = ann.get("class_name")
                    class_idx = CLASS_TO_IDX.get(cname, len(YOLO_CLASSES) - 1)

                    # Tọa độ pixel (x1, y1, x2, y2) -> chuẩn hóa YOLO (cx, cy, nw, nh)
                    x1 = float(ann.get("bbox_x1", 0))
                    y1 = float(ann.get("bbox_y1", 0))
                    x2 = float(ann.get("bbox_x2", 0))
                    y2 = float(ann.get("bbox_y2", 0))

                    box_w = max(1.0, x2 - x1)
                    box_h = max(1.0, y2 - y1)
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    nw = box_w / w
                    nh = box_h / h

                    lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                with open(label_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                exported_count += 1

            # Tạo file data.yaml
            yaml_content = f"""# Lost & Found AI - Bus Dataset
path: {export_root.resolve()}
train: images/train
val: images/val
test: images/test

names:
"""
            for idx, cname in enumerate(YOLO_CLASSES):
                yaml_content += f"  {idx}: {cname}\n"

            with open(export_root / "data.yaml", "w", encoding="utf-8") as f:
                f.write(yaml_content)

            # Nén thành file zip
            zip_path = DATASET_EXPORTS_DIR / f"{export_id}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(export_root):
                    for file in files:
                        full_p = Path(root) / file
                        arc_name = full_p.relative_to(export_root)
                        zf.write(full_p, arc_name)

            # Xóa thư mục tạm sau khi nén
            shutil.rmtree(export_root, ignore_errors=True)

            logger.info(f"Đã export {exported_count} ảnh ra {zip_path}")
            return True, str(zip_path), f"Xuất thành công {exported_count} ảnh ra chuẩn YOLOv8!"

        except Exception as e:
            logger.error(f"Lỗi khi export YOLO dataset: {e}")
            return False, "", f"Lỗi export: {str(e)}"
