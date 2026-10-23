"""
src/services/model_service.py
Quản lý các phiên bản Model AI, quy trình huấn luyện thực tế (Real Training Workflow),
kiểm tra tính hợp lệ của dataset, đọc metrics thực tế từ Ultralytics validation,
và quản lý phê duyệt triển khai (Approval Workflow).

TUYỆT ĐỐI KHÔNG TẠO CHỈ SỐ METRICS GIẢ DỰA TRÊN SỐ EPOCH.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import uuid

from src.database.model_repository import ModelRepository
from src.utils.logger import get_logger
from config.settings import MODELS_DIR, YOLO_MODEL_NAME, DATA_DIR, get_current_time

logger = get_logger(__name__)

# Thư mục lưu trữ logs huấn luyện và runs
TRAINING_LOGS_DIR = DATA_DIR / "training_logs"
RUNS_DIR = DATA_DIR / "runs"

for d in [TRAINING_LOGS_DIR, RUNS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ModelService:
    """Quản lý các phiên bản Model và quy trình huấn luyện AI thực sự."""

    def __init__(self, repo: Optional[ModelRepository] = None):
        self._repo = repo
        # Lưu trữ trạng thái các training job đang chạy trong tiến trình riêng: job_id -> dict
        self._active_jobs: Dict[str, Dict] = {}

    def set_repository(self, repo: ModelRepository):
        self._repo = repo

    def get_all_models(self) -> List[Dict]:
        """Lấy tất cả các model."""
        if self._repo:
            return self._repo.get_all_models()
        return []

    def get_active_model(self) -> Optional[Dict]:
        """Lấy model đang hoạt động."""
        if self._repo:
            return self._repo.get_active_model()
        return None

    def activate_model(self, model_id: str) -> Tuple[bool, str]:
        """Kích hoạt một model (chuyển sang active sau khi được phê duyệt)."""
        if not self._repo:
            return False, "Database chưa kết nối"
        ok = self._repo.set_active_model(model_id)
        if ok:
            return True, f"Đã kích hoạt model '{model_id}' thành công!"
        return False, f"Không thể kích hoạt model '{model_id}'"

    def register_model_version(
        self,
        version: str,
        dataset_version: str,
        classes: List[str],
        metrics: Dict,
        confidence_threshold: float = 0.35,
        status: str = "candidate",
        artifact_path: str = "",
        notes: str = "",
    ) -> Tuple[bool, str]:
        """Đăng ký một phiên bản model mới (mặc định là candidate, cần phê duyệt)."""
        if not self._repo:
            return False, "Database chưa kết nối"

        model_id = f"yolov8-{version.lower().replace(' ', '-')}"
        ok = self._repo.add_model_version(
            model_id=model_id,
            version=version,
            dataset_version=dataset_version,
            classes=classes,
            metrics=metrics,
            confidence_threshold=confidence_threshold,
            status=status,
            artifact_path=artifact_path,
            notes=notes,
        )
        if ok:
            return True, f"Đã đăng ký phiên bản model '{model_id}' thành công!"
        return False, "Lỗi khi lưu thông tin model"

    @staticmethod
    def get_training_hardware_info() -> Dict[str, Union[str, bool]]:
        """Kiểm tra phần cứng khả dụng cho huấn luyện (GPU CUDA, Apple Silicon MPS, CPU)."""
        info = {
            "device": "cpu",
            "has_gpu": False,
            "device_name": "CPU",
            "recommendation": "Huấn luyện trên CPU sẽ mất nhiều thời gian. Khuyến nghị chạy trên GPU (NVIDIA CUDA hoặc Apple MPS).",
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["device"] = "cuda"
                info["has_gpu"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                info["recommendation"] = "Hệ thống hỗ trợ GPU CUDA, tối ưu cho huấn luyện nhanh."
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["device"] = "mps"
                info["has_gpu"] = True
                info["device_name"] = "Apple Silicon (MPS GPU)"
                info["recommendation"] = "Hệ thống hỗ trợ Apple Silicon GPU (MPS acceleration)."
        except Exception:
            pass
        return info

    @staticmethod
    def validate_training_dataset(dataset_yaml_path: str) -> Tuple[bool, str, Dict]:
        """
        Kiểm tra dataset có đầy đủ cấu trúc, ảnh và nhãn annotation trước khi huấn luyện:
        - Kiểm tra file data.yaml có tồn tại không.
        - Kiểm tra các thư mục images/train, labels/train có ảnh và file nhãn không.
        - Kiểm tra các file nhãn có chứa bounding box thật hay bị rỗng.
        """
        yaml_file = Path(dataset_yaml_path)
        if not yaml_file.exists() or not yaml_file.is_file():
            return False, f"File cấu hình dataset không tồn tại tại: {dataset_yaml_path}", {}

        try:
            # Đọc nội dung yaml cơ bản
            yaml_content = yaml_file.read_text(encoding="utf-8")
            base_dir = yaml_file.parent

            # Kiểm tra các thư mục con
            train_img_dir = base_dir / "images" / "train"
            val_img_dir = base_dir / "images" / "val"
            train_lbl_dir = base_dir / "labels" / "train"

            train_imgs = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")) if train_img_dir.exists() else []
            val_imgs = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")) if val_img_dir.exists() else []
            train_labels = list(train_lbl_dir.glob("*.txt")) if train_lbl_dir.exists() else []

            # Đếm số nhãn không rỗng
            non_empty_labels = 0
            for lbl in train_labels:
                if lbl.stat().st_size > 0:
                    non_empty_labels += 1

            stats = {
                "train_images": len(train_imgs),
                "val_images": len(val_imgs),
                "total_labels": len(train_labels),
                "non_empty_labels": non_empty_labels,
                "dataset_path": str(base_dir),
            }

            if len(train_imgs) == 0:
                return False, f"Tập train không có hình ảnh nào ({train_img_dir}).", stats

            if non_empty_labels == 0:
                return False, (
                    f"Dataset thiếu annotation: Tìm thấy {len(train_imgs)} ảnh nhưng 0 nhãn bounding box. "
                    "Hãy gán nhãn ít nhất 5-10 ảnh trước khi bắt đầu huấn luyện."
                ), stats

            return True, f"Dataset hợp lệ ({len(train_imgs)} ảnh train, {non_empty_labels} ảnh có nhãn).", stats

        except Exception as e:
            return False, f"Lỗi kiểm tra dataset: {e}", {}

    def generate_training_command(
        self,
        dataset_yaml_path: str,
        base_model: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        device: Optional[str] = None,
        task: str = "detect",
    ) -> str:
        """Sinh lệnh CLI chuẩn Ultralytics để chạy huấn luyện thực tế (detect hoặc segment)."""
        dev = device or self.get_training_hardware_info()["device"]
        task_mode = "segment" if task.lower() in ("segment", "segmentation", "seg") else "detect"
        default_model = "yolov8n-seg.pt" if task_mode == "segment" else "yolov8n.pt"
        model = base_model or default_model
        return (
            f"yolo {task_mode} train "
            f"data='{dataset_yaml_path}' "
            f"model='{model}' "
            f"epochs={epochs} "
            f"batch={batch_size} "
            f"imgsz={imgsz} "
            f"device='{dev}' "
            f"project='{RUNS_DIR / task_mode}' "
            f"name='exp_bus_model'"
        )

    def start_real_training_job(
        self,
        dataset_yaml_path: str,
        new_version: str,
        base_model: Optional[str] = None,
        epochs: int = 30,
        batch_size: int = 16,
        imgsz: int = 640,
        task: str = "detect",
    ) -> Tuple[bool, str, Dict]:
        """
        Khởi tạo tiến trình huấn luyện THỰC SỰ bằng Ultralytics trong tiến trình riêng (background process):
        - Không khóa giao diện Streamlit.
        - Ghi nhật ký đầy đủ ra file log.
        - Kiểm tra tính hợp lệ của dataset trước khi chạy.
        - Hỗ trợ cả tác vụ Object Detection và Instance Segmentation.
        - Trả về job_id để theo dõi tiến độ thời gian thực.
        """
        # 1. Kiểm tra tính hợp lệ của dataset
        valid, msg, stats = self.validate_training_dataset(dataset_yaml_path)
        if not valid:
            logger.warning(f"Từ chối huấn luyện: {msg}")
            return False, msg, {"dataset_stats": stats}

        job_id = f"job_{str(uuid.uuid4())[:8]}"
        log_file_path = TRAINING_LOGS_DIR / f"{job_id}.log"
        task_mode = "segment" if task.lower() in ("segment", "segmentation", "seg") else "detect"
        default_model = "yolov8n-seg.pt" if task_mode == "segment" else "yolov8n.pt"
        chosen_base_model = base_model or default_model

        exp_name = f"train_{task_mode}_{new_version.lower().replace('.', '_')}_{job_id}"
        exp_dir = RUNS_DIR / task_mode / exp_name

        hw = self.get_training_hardware_info()
        device = hw["device"]

        # Lệnh chạy huấn luyện với task rõ ràng
        cmd = [
            sys.executable, "-m", "ultralytics", "train",
            f"task={task_mode}",
            f"data={Path(dataset_yaml_path).resolve()}",
            f"model={chosen_base_model}",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"imgsz={imgsz}",
            f"device={device}",
            f"project={RUNS_DIR / task_mode}",
            f"name={exp_name}",
            "exist_ok=True",
            "verbose=True",
        ]

        logger.info(f"Bắt đầu Training Job {job_id} ({task_mode}): {' '.join(cmd)}")

        try:
            log_f = open(log_file_path, "w", encoding="utf-8")
            log_f.write(f"=== TRAINING JOB {job_id} STARTED AT {get_current_time().isoformat()} ===\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Device: {device} ({hw['device_name']})\n")
            log_f.write(f"Dataset stats: {stats}\n\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(".").resolve()),
            )

            job_info = {
                "job_id": job_id,
                "version": new_version,
                "process": proc,
                "pid": proc.pid,
                "status": "RUNNING",
                "start_time": get_current_time().isoformat(),
                "log_file": str(log_file_path),
                "exp_dir": str(exp_dir),
                "dataset_yaml": str(dataset_yaml_path),
                "epochs": epochs,
                "weights_path": str(exp_dir / "weights" / "best.pt"),
                "error_message": "",
                "metrics": {},
            }
            self._active_jobs[job_id] = job_info

            return True, f"Tiến trình huấn luyện thật đã được khởi chạy trong nền (PID {proc.pid})", {
                "job_id": job_id,
                "log_file": str(log_file_path),
                "device": device,
            }

        except Exception as e:
            logger.error(f"Lỗi khởi chạy huấn luyện: {e}")
            return False, f"Lỗi khởi chạy huấn luyện: {e}", {}

    def get_training_job_status(self, job_id: str) -> Dict:
        """
        Kiểm tra trạng thái tiến trình huấn luyện:
        - Nếu đang chạy: Trả về RUNNING và trích xuất dòng log gần nhất.
        - Nếu đã xong: Đọc metrics THỰC TẾ từ results.csv trong thư mục output của Ultralytics.
        - Nếu lỗi: Trích xuất thông báo lỗi từ log file.
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return {"job_id": job_id, "status": "NOT_FOUND", "message": "Không tìm thấy job"}

        proc: Optional[subprocess.Popen] = job.get("process")
        log_path = Path(job.get("log_file", ""))

        # Đọc 20 dòng log cuối cùng
        recent_logs = ""
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                recent_logs = "\n".join(lines[-25:])
            except Exception:
                recent_logs = "Không thể đọc file log."

        if proc is not None and proc.poll() is None:
            # Vẫn đang chạy
            job["status"] = "RUNNING"
            return {
                "job_id": job_id,
                "status": "RUNNING",
                "pid": job.get("pid"),
                "start_time": job.get("start_time"),
                "recent_logs": recent_logs,
            }

        # Tiến trình đã kết thúc
        return_code = proc.poll() if proc else -1

        if return_code == 0:
            job["status"] = "COMPLETED"
            # Đọc metrics thật từ results.csv
            real_metrics = self._parse_ultralytics_results_csv(Path(job.get("exp_dir", "")))
            job["metrics"] = real_metrics

            weights_p = Path(job.get("weights_path", ""))
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "weights_exist": weights_p.exists(),
                "weights_path": str(weights_p) if weights_p.exists() else "",
                "metrics": real_metrics,
                "recent_logs": recent_logs,
            }
        else:
            job["status"] = "FAILED"
            job["error_message"] = f"Tiến trình kết thúc với mã lỗi {return_code}"
            return {
                "job_id": job_id,
                "status": "FAILED",
                "return_code": return_code,
                "error_message": job["error_message"],
                "recent_logs": recent_logs,
            }

    @staticmethod
    def _parse_ultralytics_results_csv(exp_dir: Path) -> Dict[str, float]:
        """
        Đọc và trích xuất chỉ số metrics THỰC TẾ từ results.csv của Ultralytics.
        Tuyệt đối không sử dụng công thức giả lập.
        """
        results_csv = exp_dir / "results.csv"
        if not results_csv.exists():
            return {}

        try:
            lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) < 2:
                return {}

            headers = [h.strip() for h in lines[0].split(",")]
            last_row = [v.strip() for v in lines[-1].split(",")]

            val_dict = {}
            for h, v in zip(headers, last_row):
                try:
                    val_dict[h] = float(v)
                except ValueError:
                    val_dict[h] = v

            # Trích xuất các trường chuẩn
            metrics = {
                "precision": round(val_dict.get("metrics/precision(B)", 0.0), 4),
                "recall": round(val_dict.get("metrics/recall(B)", 0.0), 4),
                "map50": round(val_dict.get("metrics/mAP50(B)", 0.0), 4),
                "map50_95": round(val_dict.get("metrics/mAP50-95(B)", 0.0), 4),
                "epochs_completed": int(val_dict.get("epoch", 0)),
            }
            return metrics
        except Exception as e:
            logger.error(f"Lỗi đọc results.csv: {e}")
            return {}

    def approve_and_register_completed_job(
        self,
        job_id: str,
        dataset_version: str = "v1.0",
        notes: str = "",
    ) -> Tuple[bool, str]:
        """
        Lưu model đã huấn luyện hoàn tất vào bảng model_versions với trạng thái 'candidate'
        kèm toàn bộ metrics thực tế đã đo đạc.
        """
        status_info = self.get_training_job_status(job_id)
        if status_info.get("status") != "COMPLETED":
            return False, f"Job {job_id} chưa hoàn tất thành công (trạng thái: {status_info.get('status')})"

        metrics = status_info.get("metrics", {})
        if not metrics:
            return False, "Không tìm thấy metrics thực tế từ kết quả huấn luyện."

        job = self._active_jobs[job_id]
        version = job["version"]
        weights_path = status_info.get("weights_path", "")

        # Sao chép weights vào thư mục models/ chính thức
        dest_weights = MODELS_DIR / f"yolov8_{version.lower().replace('.', '_')}.pt"
        try:
            if weights_path and Path(weights_path).exists():
                import shutil
                shutil.copy2(weights_path, dest_weights)
                final_path = str(dest_weights)
            else:
                final_path = weights_path
        except Exception:
            final_path = weights_path

        return self.register_model_version(
            version=version,
            dataset_version=dataset_version,
            classes=["seat", "person", "item"],
            metrics=metrics,
            status="candidate",
            artifact_path=final_path,
            notes=notes or f"Huấn luyện thật hoàn tất từ job {job_id}",
        )

    # Đánh dấu phương thức simulate_training_job là DEPRECATED / VÔ HIỆU HÓA
    def simulate_training_job(self, *args, **kwargs) -> Tuple[bool, Dict, str]:
        """
        ĐÃ VÔ HIỆU HÓA HOÀN TOÀN:
        Hệ thống không còn hỗ trợ tạo metrics mAP/Precision giả lập.
        Hãy sử dụng start_real_training_job() để huấn luyện thực tế.
        """
        err_msg = (
            "VÔ HIỆU HÓA: simulate_training_job() đã bị loại bỏ theo yêu cầu computer vision thực thụ. "
            "Hệ thống không sinh metrics giả lập. Hãy sử dụng start_real_training_job()."
        )
        logger.warning(err_msg)
        return False, {}, err_msg
