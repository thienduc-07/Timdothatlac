"""
src/roboflow/roboflow_client.py
Client for submitting images to a Roboflow Serverless Workflow.

Uses the ``inference-sdk`` package (``InferenceHTTPClient``) for proper
request/response handling.  Falls back to raw ``requests`` if the SDK
is unavailable.

Configuration is read from environment variables (loaded via dotenv):
  * ROBOFLOW_API_KEY       – API key for authentication.
  * ROBOFLOW_WORKFLOW_URL  – Base inference URL (default: serverless).
  * ROBOFLOW_WORKSPACE     – Workspace URL slug.
  * ROBOFLOW_WORKFLOW_ID   – Workflow URL slug (not document ID).
"""

import os
import json
import time
import base64
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoboflowWorkflowClient:
    """Client for submitting images to a Roboflow Serverless Workflow.

    Preferred path: ``inference-sdk`` ``InferenceHTTPClient.run_workflow``.
    Fallback path: raw ``requests.post`` to the REST endpoint.
    """

    _semaphore = threading.Semaphore(int(os.getenv("ROBOFLOW_MAX_CONCURRENT", "2")))
    _last_submit: Dict[str, float] = {}
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.api_key: str = os.getenv("ROBOFLOW_API_KEY", "")
        self.api_url: str = os.getenv(
            "ROBOFLOW_WORKFLOW_URL", "https://serverless.roboflow.com"
        )
        self.workspace: str = os.getenv("ROBOFLOW_WORKSPACE", "le-uc-s-workspace")
        self.project_id: str = os.getenv("ROBOFLOW_PROJECT_ID", "bus-seat-and-belongings-monitori")
        self.workflow_id: str = os.getenv("ROBOFLOW_WORKFLOW_ID", "bus-seat-and-belongings-monitori")
        self.model_version: str = os.getenv("ROBOFLOW_MODEL_VERSION", "1")
        self.timeout: int = int(os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "10"))
        self.min_interval: float = float(
            os.getenv("ROBOFLOW_SUBMIT_INTERVAL_SECONDS", "1.0")
        )

        if not self.api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY is not set. "
                "Add it to .env or export it as an environment variable."
            )

        self.last_inference_info: Dict[str, Any] = {
            "requested_version": self.model_version,
            "requested_model_id": f"{self.project_id}/{self.model_version}",
            "status": "INITIALIZED",
            "actual_model_id": None,
            "error": None,
        }
        self.last_switch_status: Dict[str, Any] = {
            "requested_version": self.model_version,
            "target_model_id": f"{self.project_id}/{self.model_version}",
            "switched_at": time.time(),
            "status": "DEFAULT",
        }

        # Try to create the SDK client; set flag for fallback path.
        self._sdk_client = None
        try:
            from inference_sdk import InferenceHTTPClient

            self._sdk_client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            )
            logger.info(
                "RoboflowWorkflowClient initialised with inference-sdk "
                f"(api_url={self.api_url}, workspace={self.workspace}, "
                f"workflow_id={self.workflow_id}, version={self.model_version})"
            )
        except ImportError:
            logger.warning(
                "inference-sdk not installed. Falling back to raw HTTP requests."
            )

    def _mask_key(self, text: str) -> str:
        """Sanitize text to ensure raw API keys are never exposed in logs or UI."""
        if not self.api_key or not text:
            return text
        masked = (
            f"{self.api_key[:4]}****{self.api_key[-4:]}"
            if len(self.api_key) >= 8
            else "****"
        )
        return text.replace(self.api_key, masked)

    def _sanitize_error(self, err_msg: str) -> str:
        """Sanitize exception message to protect credentials."""
        return self._mask_key(str(err_msg))

    @property
    def configured(self) -> bool:
        """Return True if the client has valid credentials and target workflow/project configured."""
        return bool(self.api_key and (self.workflow_id or self.project_id))

    @property
    def is_configured(self) -> bool:
        """Alias for configured."""
        return self.configured

    # ── rate-limit ──────────────────────────────────────────────────

    def _can_submit(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last_submit.get(key, 0)
            if now - last >= self.min_interval:
                self._last_submit[key] = now
                return True
        return False

    # ── main entry-point ────────────────────────────────────────────

    def submit_image(
        self, image_path: Path, use_workflow: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Submit a single image for inference with the active model version.

        If use_workflow is True and model_version == '1', invokes the serverless workflow.
        Otherwise (default and when version != 1), queries the exact versioned hosted model endpoint,
        guaranteeing that switching version genuinely calls the chosen version without silent fallback.

        Returns the parsed response dict or None on failure.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            logger.error(f"Image file not found: {image_path}")
            return None

        key = str(image_path)
        if not self._can_submit(key):
            return None

        if not self._semaphore.acquire(blocking=False):
            return None

        try:
            # If workflow is explicitly requested AND version is "1", run workflow
            if use_workflow is True and str(self.model_version).strip() == "1":
                return self._submit_workflow(image_path)
            # Otherwise, use version-targeted inference (supports version 1, 2, 3...)
            return self._submit_versioned(image_path)
        finally:
            self._semaphore.release()

    def _submit_versioned(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Submit to the specific model version endpoint on Roboflow."""
        target_model_id = f"{self.project_id}/{self.model_version}"
        self.last_inference_info = {
            "requested_version": str(self.model_version),
            "requested_model_id": target_model_id,
            "status": "PENDING",
            "actual_model_id": None,
            "error": None,
        }

        try:
            if self._sdk_client:
                # inference-sdk direct infer targets exact model_id (e.g. project/2)
                res = self._sdk_client.infer(str(image_path), model_id=target_model_id)
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]
                self.last_inference_info["status"] = "SUCCESS"
                self.last_inference_info["actual_model_id"] = target_model_id
                self.last_inference_info["error"] = None
                return res
            else:
                return self._submit_http_versioned(image_path)
        except Exception as exc:
            clean_err = self._sanitize_error(str(exc))
            self.last_inference_info["status"] = "FAILED"
            self.last_inference_info["actual_model_id"] = None
            self.last_inference_info["error"] = clean_err
            logger.error(
                f"Roboflow inference failed for model '{target_model_id}' "
                f"(requested version: {self.model_version}): {clean_err}"
            )
            # NEVER silently fallback to another version; report genuine error
            return None

    def _submit_http_versioned(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Raw HTTP fallback to version-specific endpoint: https://detect.roboflow.com/{project_id}/{version}."""
        import requests

        target_model_id = f"{self.project_id}/{self.model_version}"
        url = f"https://detect.roboflow.com/{self.project_id}/{self.model_version}?api_key={self.api_key}"

        try:
            with open(image_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"file": f},
                    timeout=self.timeout,
                )
            if resp.status_code != 200:
                clean_err = self._sanitize_error(f"Status {resp.status_code}: {resp.text}")
                self.last_inference_info["status"] = "FAILED"
                self.last_inference_info["actual_model_id"] = None
                self.last_inference_info["error"] = clean_err
                logger.error(
                    f"Roboflow HTTP request to model '{target_model_id}' failed: {clean_err}"
                )
                return None

            data = resp.json()
            self.last_inference_info["status"] = "SUCCESS"
            self.last_inference_info["actual_model_id"] = target_model_id
            self.last_inference_info["error"] = None
            return data
        except Exception as exc:
            clean_err = self._sanitize_error(str(exc))
            self.last_inference_info["status"] = "FAILED"
            self.last_inference_info["actual_model_id"] = None
            self.last_inference_info["error"] = clean_err
            logger.error(
                f"Roboflow HTTP request to model '{target_model_id}' error: {clean_err}"
            )
            return None

    def _submit_workflow(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Submit via Roboflow Serverless Workflow (Version 1 base workflow)."""
        self.last_inference_info = {
            "requested_version": str(self.model_version),
            "requested_model_id": f"{self.workspace}/{self.workflow_id}",
            "status": "PENDING",
            "actual_model_id": None,
            "error": None,
        }
        try:
            if self._sdk_client and self.workspace and self.workflow_id:
                result = self._sdk_client.run_workflow(
                    workspace_name=self.workspace,
                    workflow_id=self.workflow_id,
                    images={"image": str(image_path)},
                )
                if result and isinstance(result, list) and len(result) > 0:
                    result = result[0]
                self.last_inference_info["status"] = "SUCCESS"
                self.last_inference_info["actual_model_id"] = f"{self.project_id}/1"
                return result
            else:
                return self._submit_http_workflow(image_path)
        except Exception as exc:
            clean_err = self._sanitize_error(str(exc))
            self.last_inference_info["status"] = "FAILED"
            self.last_inference_info["actual_model_id"] = None
            self.last_inference_info["error"] = clean_err
            logger.error(f"Roboflow workflow request failed: {clean_err}")
            return None

    def _submit_http_workflow(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Fallback: raw POST to workflow with base64-encoded image."""
        import requests

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        url = (
            f"{self.api_url}/{self.workspace}/workflows/{self.workflow_id}"
            if self.workspace and self.workflow_id
            else self.api_url
        )

        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {"type": "base64", "value": img_b64},
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        outputs = data.get("outputs", [])
        if outputs and isinstance(outputs, list):
            self.last_inference_info["status"] = "SUCCESS"
            self.last_inference_info["actual_model_id"] = f"{self.project_id}/1"
            return outputs[0]
        self.last_inference_info["status"] = "SUCCESS"
        self.last_inference_info["actual_model_id"] = f"{self.project_id}/1"
        return data

    # ── batch ───────────────────────────────────────────────────────

    def batch_submit(
        self, image_paths: List[Path]
    ) -> Dict[Path, Optional[Dict[str, Any]]]:
        """Submit multiple images concurrently."""
        results: Dict[Path, Optional[Dict[str, Any]]] = {}
        threads: List[threading.Thread] = []

        def worker(p: Path) -> None:
            results[p] = self.submit_image(p)

        for p in image_paths:
            t = threading.Thread(target=worker, args=(p,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return results

    # ── parsing helpers ─────────────────────────────────────────────

    @staticmethod
    def parse_workflow_output(
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalise a workflow response into a dict with a ``predictions`` key.

        Roboflow Workflow output keys depend on the workflow spec.  Common
        shapes:
        - ``{"predictions": [...]}``         — detection output block
        - ``{"model_predictions": {...}}``   — object detection model output
        - ``{"output": {...}}``              — generic output

        The returned dict always has a ``predictions`` key containing a list
        of dicts, each with at least ``class``, ``confidence``, and ``bbox``
        (as x1, y1, x2, y2).
        """
        if payload is None:
            return {"predictions": []}

        # Direct predictions list
        if "predictions" in payload:
            preds = payload["predictions"]
            if isinstance(preds, dict) and "predictions" in preds:
                preds = preds["predictions"]
            if isinstance(preds, list):
                return {"predictions": _normalise_pred_list(preds)}
            return {"predictions": []}

        # Nested under model_predictions (common workflow block output key)
        if "model_predictions" in payload:
            mp = payload["model_predictions"]
            if isinstance(mp, dict) and "predictions" in mp:
                return {"predictions": _normalise_pred_list(mp["predictions"])}

        # Last resort: look for any list value that looks like predictions
        for _key, val in payload.items():
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], dict) and ("class" in val[0] or "class_name" in val[0]):
                    return {"predictions": _normalise_pred_list(val)}

        return {"predictions": [], "raw": payload}

    def get_workflow_info(self) -> Dict[str, Any]:
        """Return connection info (no raw secrets exposed) for display in the UI."""
        masked_key = (
            f"{self.api_key[:4]}****{self.api_key[-4:]}"
            if len(self.api_key) >= 8
            else "****"
        )
        return {
            "api_url": self.api_url,
            "workspace": self.workspace,
            "project_id": self.project_id,
            "workflow_id": self.workflow_id,
            "model_version": self.model_version,
            "target_model_id": f"{self.project_id}/{self.model_version}",
            "masked_api_key": masked_key,
            "api_key_masked": masked_key,
            "sdk_available": self._sdk_client is not None,
            "configured": bool(self.api_key and self.workspace and (self.workflow_id or self.project_id)),
            "last_inference_info": self.last_inference_info,
            "last_switch_status": self.last_switch_status,
        }

    def list_available_versions(self) -> List[Dict[str, Any]]:
        """Fetch available dataset and model versions from Roboflow REST API.

        Never throws an exception; returns an empty list on failure.
        """
        if not self.api_key or not self.workspace or not self.project_id:
            return []

        try:
            import requests
            url = f"https://api.roboflow.com/{self.workspace}/{self.project_id}?api_key={self.api_key}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                raw_versions = data.get("versions", [])
                out = []
                for v in raw_versions:
                    vid = v.get("id", "")
                    ver_num = vid.split("/")[-1] if "/" in vid else str(vid)
                    out.append({
                        "id": vid,
                        "version": ver_num,
                        "name": v.get("name", f"Version {ver_num}"),
                        "images": v.get("images", 0),
                        "created": v.get("created", ""),
                        "generating": v.get("generating", False),
                        "progress": v.get("progress", 1.0),
                    })
                return out
            else:
                logger.warning(f"Roboflow API returned status {resp.status_code} when listing versions")
        except Exception as e:
            logger.warning(f"Could not fetch Roboflow versions: {e}")
        return []

    def get_project_metadata(self) -> Dict[str, Any]:
        """Fetch project metadata (type, classes, splits, image count) without exposing secrets."""
        if not self.api_key or not self.workspace or not self.project_id:
            return {"configured": False}

        try:
            import requests
            url = f"https://api.roboflow.com/{self.workspace}/{self.project_id}?api_key={self.api_key}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                proj = data.get("project", {})
                classes = data.get("classes", proj.get("classes", {}))
                return {
                    "configured": True,
                    "workspace": self.workspace,
                    "project_id": self.project_id,
                    "name": proj.get("name", self.project_id),
                    "type": proj.get("type", "object-detection"),
                    "total_images": proj.get("images", 0),
                    "unannotated": proj.get("unannotated", 0),
                    "classes": classes,
                    "versions_count": len(data.get("versions", [])),
                    "active_version": self.model_version,
                }
        except Exception as e:
            logger.warning(f"Could not fetch Roboflow project metadata: {e}")
        return {"configured": False, "error": "Unable to connect"}

    def set_active_version(self, version: str) -> bool:
        """Switch the active model version in memory and environment without editing code.

        Invalidates previous submission rate-limit cache and updates switch metadata.
        """
        ver_str = str(version).strip()
        if not ver_str:
            return False
        self.model_version = ver_str
        os.environ["ROBOFLOW_MODEL_VERSION"] = ver_str

        # Invalidate cache so old version results are never mistaken for new version
        with self._lock:
            self._last_submit.clear()

        target_model_id = f"{self.project_id}/{ver_str}"
        self.last_switch_status = {
            "requested_version": ver_str,
            "target_model_id": target_model_id,
            "switched_at": time.time(),
            "status": "ACTIVE",
        }
        self.last_inference_info = {
            "requested_version": ver_str,
            "requested_model_id": target_model_id,
            "status": "VERSION_SWITCHED",
            "actual_model_id": None,
            "error": None,
        }
        logger.info(f"Switched active Roboflow model version to: {ver_str} (Target model ID: {target_model_id})")
        return True


# ── private helpers ──────────────────────────────────────────────

def _normalise_pred_list(preds: list) -> list:
    """Ensure each prediction dict has ``class``, ``confidence``, ``bbox`` (xyxy), and ``polygon``."""
    out = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        cls = p.get("class") or p.get("class_name") or ""
        conf = float(p.get("confidence", 0))

        # bbox can come as {x, y, width, height} or as xyxy
        if "x" in p and "y" in p and "width" in p and "height" in p:
            cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
            bbox = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
        elif "bbox" in p:
            raw = p["bbox"]
            if isinstance(raw, dict):
                cx, cy = raw.get("x", 0), raw.get("y", 0)
                w, h = raw.get("width", 0), raw.get("height", 0)
                bbox = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
            elif isinstance(raw, (list, tuple)) and len(raw) == 4:
                bbox = [int(v) for v in raw]
            else:
                bbox = [0, 0, 0, 0]
        else:
            bbox = [0, 0, 0, 0]

        # Extract polygon if segmentation mask/points exist in any standard format
        polygon = None
        raw_pts = None
        for k in ("points", "polygon", "polygons", "segmentation", "mask", "masks"):
            if k in p and p[k]:
                raw_pts = p[k]
                break

        if raw_pts is not None and isinstance(raw_pts, list):
            # Handle nested list [[x1,y1], [x2,y2], ...]
            if len(raw_pts) > 0 and isinstance(raw_pts[0], list) and len(raw_pts[0]) > 0 and isinstance(raw_pts[0][0], (list, dict)):
                raw_pts = raw_pts[0]
            valid_pts = []
            for pt in raw_pts:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    valid_pts.append((float(pt["x"]), float(pt["y"])))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    valid_pts.append((float(pt[0]), float(pt[1])))
            if len(valid_pts) >= 3:
                polygon = valid_pts

        out.append({"class": cls, "confidence": conf, "bbox": bbox, "polygon": polygon})
    return out



# Example usage (only runs when executed as a script).
if __name__ == "__main__":
    client = RoboflowWorkflowClient()
    sample = Path("/tmp/sample.jpg")
    if sample.exists():
        out = client.submit_image(sample)
        print(json.dumps(out, indent=2))
