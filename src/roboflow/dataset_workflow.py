"""
src/roboflow/dataset_workflow.py
Workflow for Roboflow dataset management, annotation validation,
version generation, training kickoff, evaluation retrieval, and model activation.

Uses the official Roboflow REST API endpoints:
  - Project summary & class stats: GET /{workspace}/{project}
  - Image upload: POST /{workspace}/{project}/upload
  - Version generation: POST /{workspace}/{project}/generate
  - Version status & details: GET /{workspace}/{project}/{version}
  - Model training kickoff: POST /{workspace}/{project}/{version}/train
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoboflowDatasetWorkflow:
    """Manages the end-to-end dataset & training lifecycle on Roboflow."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project_id: Optional[str] = None,
        base_api_url: str = "https://api.roboflow.com",
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.workspace = workspace or os.getenv("ROBOFLOW_WORKSPACE", "le-uc-s-workspace")
        self.project_id = project_id or os.getenv("ROBOFLOW_PROJECT_ID", "bus-seat-and-belongings-monitori")
        self.base_api_url = base_api_url.rstrip("/")
        self.timeout = int(os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "15"))

    def _require_auth(self):
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not configured.")
        if not self.workspace or not self.project_id:
            raise ValueError("ROBOFLOW_WORKSPACE or ROBOFLOW_PROJECT_ID is not configured.")

    # ────────────────────────────────────────────────────────────────
    # 1. Project Summary & Class Stats
    # ────────────────────────────────────────────────────────────────
    def get_project_summary(self) -> Dict[str, Any]:
        """Fetch project details, splits, class distribution, and version list."""
        self._require_auth()
        url = f"{self.base_api_url}/{self.workspace}/{self.project_id}"
        resp = requests.get(url, params={"api_key": self.api_key}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        proj = data.get("project", {})

        versions = []
        for v in data.get("versions", []):
            vid = v.get("id") or str(v.get("version"))
            v_num = v.get("version")
            if v_num is None and "/" in str(vid):
                try:
                    v_num = int(str(vid).split("/")[-1])
                except Exception:
                    v_num = 1
            versions.append({
                "id": vid,
                "version": v_num,
                "name": v.get("name"),
                "images": v.get("images", 0),
                "splits": v.get("splits", {}),
                "generating": v.get("generating", False),
                "progress": v.get("progress", 1),
            })

        return {
            "workspace": self.workspace,
            "project_id": self.project_id,
            "name": proj.get("name", self.project_id),
            "type": proj.get("type", "object-detection"),
            "images_total": proj.get("images", 0),
            "unannotated_total": proj.get("unannotated", 0),
            "classes": proj.get("classes", {}),
            "splits": proj.get("splits", {}),
            "versions": versions,
        }

    # ────────────────────────────────────────────────────────────────
    # 2. Annotation Validation
    # ────────────────────────────────────────────────────────────────
    def validate_annotations(self) -> Dict[str, Any]:
        """
        Validate current dataset annotation state:
        - Checks for unannotated images that need labeling
        - Checks class balance across project classes
        - Checks whether minimum splits are met for version creation
        """
        summary = self.get_project_summary()
        total_images = summary["images_total"]
        unannotated = summary["unannotated_total"]
        annotated = max(0, total_images - unannotated)
        classes = summary["classes"]
        splits = summary["splits"]

        issues: List[str] = []
        recommendations: List[str] = []

        if unannotated > 0:
            issues.append(f"{unannotated} / {total_images} images are unannotated.")
            recommendations.append("Label remaining images in Roboflow annotator before generating a new version.")

        if annotated < 10:
            issues.append("Less than 10 annotated images available. Training accuracy will be limited.")
            recommendations.append("Add and annotate more representative frames from the bus camera.")

        # Check for class presence
        expected_classes = {"empty_seat", "human_occupied", "object_occupied"}
        found_expected = expected_classes.intersection(set(classes.keys()))
        if not found_expected:
            recommendations.append(
                f"Project classes {list(classes.keys())} do not match standard seat state classes "
                f"({list(expected_classes)}). Ensure class_mapper.py aligns with actual labels."
            )

        is_ready_for_version = annotated >= 10 and (unannotated / total_images < 0.5 if total_images > 0 else False)

        return {
            "is_valid": len(issues) == 0,
            "ready_for_version": is_ready_for_version,
            "total_images": total_images,
            "annotated_images": annotated,
            "unannotated_images": unannotated,
            "classes": classes,
            "splits": splits,
            "issues": issues,
            "recommendations": recommendations,
        }

    # ────────────────────────────────────────────────────────────────
    # 3. Image Upload (Single & Batch)
    # ────────────────────────────────────────────────────────────────
    def upload_image(
        self,
        image_path: Union[str, Path],
        split: str = "train",
        tag: Optional[str] = None,
        batch_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local image file to the Roboflow project."""
        self._require_auth()
        p = Path(image_path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")

        url = f"{self.base_api_url}/{self.workspace}/{self.project_id}/upload"
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "split": split,
            "name": p.name,
        }
        if tag:
            params["tag"] = tag
        if batch_name:
            params["batch"] = batch_name

        with open(p, "rb") as f:
            files = {"file": (p.name, f, "image/jpeg")}
            resp = requests.post(url, params=params, files=files, timeout=self.timeout)

        resp.raise_for_status()
        return resp.json()

    def upload_batch(
        self,
        image_paths: List[Union[str, Path]],
        split: str = "train",
        tag: Optional[str] = None,
        batch_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload multiple images with progress tracking."""
        successful = []
        failed = []

        for p in image_paths:
            try:
                res = self.upload_image(p, split=split, tag=tag, batch_name=batch_name)
                successful.append({"path": str(p), "result": res})
            except Exception as e:
                logger.error(f"Failed to upload {p}: {e}")
                failed.append({"path": str(p), "error": str(e)})

        return {
            "total": len(image_paths),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful": successful,
            "failed": failed,
        }

    # ────────────────────────────────────────────────────────────────
    # 4. Dataset Version Generation
    # ────────────────────────────────────────────────────────────────
    def generate_version(
        self,
        version_name: Optional[str] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request generation of a new frozen dataset version on Roboflow.
        Default preprocessing: auto-orient, resize to 512x512 (matching existing project version 1).
        """
        self._require_auth()
        url = f"{self.base_api_url}/{self.workspace}/{self.project_id}/generate"

        payload: Dict[str, Any] = {
            "preprocessing": preprocessing or {
                "auto-orient": True,
                "resize": {"width": 512, "height": 512, "format": "Stretch to"},
            },
            "augmentation": augmentation or {},
        }
        if version_name:
            payload["name"] = version_name

        resp = requests.post(
            url,
            params={"api_key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ────────────────────────────────────────────────────────────────
    # 5. Version Status & Evaluation Metrics
    # ────────────────────────────────────────────────────────────────
    def get_version_details(self, version_number: int) -> Dict[str, Any]:
        """Fetch details for a specific dataset version."""
        self._require_auth()
        url = f"{self.base_api_url}/{self.workspace}/{self.project_id}/{version_number}"
        resp = requests.get(url, params={"api_key": self.api_key}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("version", data)

    def get_model_evaluation(self, version_number: int) -> Dict[str, Any]:
        """
        Retrieve performance metrics for a trained model on a specific version.
        Extracts mAP50, precision, recall from the version's training record.
        """
        v_info = self.get_version_details(version_number)
        trainings = v_info.get("trainings", [])

        if not trainings:
            return {
                "version": version_number,
                "status": "not_trained",
                "metrics": {},
                "model_id": None,
                "message": f"Version {version_number} has not been trained yet.",
            }

        latest_training = trainings[-1]
        model_type = latest_training.get("modelType", "unknown")
        status = latest_training.get("status", "unknown")
        models = latest_training.get("models", [])
        model_id = None
        metrics: Dict[str, Any] = {}

        if models:
            m = models[0]
            model_id = m.get("modelId")
            metrics = m.get("metrics", {})
        elif latest_training.get("modelIds"):
            model_id = latest_training["modelIds"][0]

        return {
            "version": version_number,
            "training_id": latest_training.get("trainingId"),
            "status": status,
            "model_type": model_type,
            "model_id": model_id,
            "metrics": {
                "map50": metrics.get("map50", metrics.get("map", 0.0)),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
            },
            "raw_training": latest_training,
        }

    # ────────────────────────────────────────────────────────────────
    # 6. Kick off Model Training
    # ────────────────────────────────────────────────────────────────
    def start_training(self, version_number: int) -> Dict[str, Any]:
        """
        Start cloud training on Roboflow for the specified version.
        Note: requires available training credits on the workspace plan.
        """
        self._require_auth()
        url = f"{self.base_api_url}/{self.workspace}/{self.project_id}/{version_number}/train"
        resp = requests.post(url, params={"api_key": self.api_key}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ────────────────────────────────────────────────────────────────
    # 7. Model Activation (Register to Local DB & Settings)
    # ────────────────────────────────────────────────────────────────
    def activate_model(
        self,
        version_number: int,
        model_id: Optional[str] = None,
        model_service: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        Activate a trained Roboflow model version:
        1. Fetch evaluation metrics from Roboflow.
        2. Register candidate/active model in local ModelService database.
        3. Switch active model so subsequent inferences use it.
        """
        eval_info = self.get_model_evaluation(version_number)
        m_id = model_id or eval_info.get("model_id") or f"{self.project_id}/{version_number}"
        metrics = eval_info.get("metrics", {})

        if model_service is not None:
            ok, msg = model_service.register_model_version(
                version=f"roboflow-v{version_number}",
                dataset_version=f"roboflow-{self.project_id}-v{version_number}",
                classes=list(self.get_project_summary().get("classes", {}).keys()),
                metrics=metrics,
                status="active",
                notes=f"Roboflow Serverless Workflow Model: {m_id} (Type: {eval_info.get('model_type', 'rfdetr')})",
            )
            if ok:
                model_service.activate_model(f"yolov8-roboflow-v{version_number}")
                return True, f"Model Roboflow v{version_number} ({m_id}) đã được kích hoạt thành công trong hệ thống!"
            return ok, msg

        return True, f"Model Roboflow v{version_number} ({m_id}) thông tin đã được kiểm chứng!"
