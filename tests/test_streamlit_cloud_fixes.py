"""
tests/test_streamlit_cloud_fixes.py
Kiểm tra tính ổn định trên môi trường Streamlit Cloud:
1. Chống lỗi TypeError do cache instance cũ (safe_get_recent_alerts, safe_update_alert_status).
2. Tự động đọc biến cấu hình Roboflow từ Streamlit Secrets (st.secrets) và Session State.
3. Chấp nhận linh hoạt các tham số mở rộng (*args, **kwargs) trên Service và Repository.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

from app import safe_get_recent_alerts, safe_update_alert_status
from config.settings import get_secret_or_env
from src.services.analysis_service import AnalysisService


class MockLegacyService:
    """Mô phỏng service cũ trên Streamlit Cloud chưa có tham số active_only hoặc resolution_type."""
    def __init__(self):
        self.alerts_store = [
            {"alert_id": "ALT-001", "status": "PENDING", "seat_id": "SEAT_01", "priority": "HIGH"},
            {"alert_id": "ALT-002", "status": "DISMISSED", "seat_id": "SEAT_02", "priority": "MEDIUM"},
            {"alert_id": "ALT-003", "status": "ACTIVE", "seat_id": "SEAT_03", "priority": "HIGH"},
            {"alert_id": "ALT-004", "status": "ITEM_RETRIEVED", "seat_id": "SEAT_04", "priority": "LOW"},
        ]
        self.updated_status = {}

    def get_recent_alerts(self, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """Signature cũ KHÔNG có active_only -> Nếu gọi bằng keyword arg active_only sẽ quăng TypeError."""
        res = self.alerts_store
        if status:
            res = [a for a in res if a.get("status") == status]
        return res[:limit]

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Signature cũ KHÔNG có resolution_type -> Nếu gọi bằng keyword arg sẽ quăng TypeError."""
        self.updated_status[alert_id] = new_status
        return True


def test_safe_get_recent_alerts_with_legacy_cached_service():
    """Kiểm tra safe_get_recent_alerts tự động fallback không bị TypeError khi gặp instance cũ."""
    legacy_service = MockLegacyService()

    # Gọi trực tiếp với active_only sẽ quăng TypeError trên instance cũ
    with pytest.raises(TypeError):
        fn_alerts = getattr(legacy_service, "get_recent_alerts")
        fn_alerts(limit=50, status=None, **dict([("active_only", True)]))

    # safe_get_recent_alerts phải xử lý an toàn và lọc được các alert PENDING/ACTIVE
    active_alerts = safe_get_recent_alerts(legacy_service, limit=50, active_only=True)
    assert len(active_alerts) == 2
    ids = [a["alert_id"] for a in active_alerts]
    assert "ALT-001" in ids  # PENDING
    assert "ALT-003" in ids  # ACTIVE
    assert "ALT-002" not in ids  # DISMISSED
    assert "ALT-004" not in ids  # ITEM_RETRIEVED


def test_safe_update_alert_status_with_legacy_cached_service():
    """Kiểm tra safe_update_alert_status tự động fallback không bị TypeError khi gặp instance cũ."""
    legacy_service = MockLegacyService()

    # Gọi trực tiếp với resolution_type sẽ quăng TypeError trên instance cũ
    with pytest.raises(TypeError):
        fn_update = getattr(legacy_service, "update_alert_status")
        fn_update("ALT-001", "DISMISSED", **dict([("resolution_type", "DISMISSED")]))

    # safe_update_alert_status phải xử lý an toàn và cập nhật thành công
    ok = safe_update_alert_status(legacy_service, "ALT-001", "DISMISSED", resolution_type="DISMISSED")
    assert ok is True
    assert legacy_service.updated_status.get("ALT-001") == "DISMISSED"


def test_get_secret_or_env_resolution():
    """Kiểm tra get_secret_or_env ưu tiên biến môi trường, sau đó đến session_state, sau đó đến st.secrets."""
    test_key = "TEST_CUSTOM_SECRET_KEY_123"

    # 1. Trả về default khi không có ở đâu
    assert get_secret_or_env(test_key, default="fallback_val") == "fallback_val"

    # 2. Đọc từ st.secrets khi môi trường không có
    mock_st = MagicMock()
    mock_st.secrets = {test_key: "val_from_secrets"}
    mock_st.session_state = {}

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        val = get_secret_or_env(test_key)
        assert val == "val_from_secrets"
        # Đảm bảo đã đồng bộ vào os.environ
        assert os.environ.get(test_key) == "val_from_secrets"

    # Dọn dẹp
    os.environ.pop(test_key, None)


def test_roboflow_client_st_secrets_support():
    """Kiểm tra RoboflowWorkflowClient tự động nhận API Key từ st.secrets khi .env không có."""
    from src.roboflow.roboflow_client import RoboflowWorkflowClient

    mock_st = MagicMock()
    mock_st.secrets = {
        "ROBOFLOW_API_KEY": "rf_secret_test_key_abc",
        "ROBOFLOW_WORKSPACE": "test-ws",
        "ROBOFLOW_PROJECT_ID": "test-proj",
        "ROBOFLOW_WORKFLOW_ID": "test-wf",
    }
    mock_st.session_state = {}

    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("ROBOFLOW_")}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            client = RoboflowWorkflowClient()
            assert client.api_key == "rf_secret_test_key_abc"
            assert client.workspace == "test-ws"
            assert client.project_id == "test-proj"
            assert client.workflow_id == "test-wf"
            assert client.configured is True


def test_roboflow_dataset_workflow_st_secrets_support():
    """Kiểm tra RoboflowDatasetWorkflow tự động nhận API Key từ st.secrets khi .env không có."""
    from src.roboflow.dataset_workflow import RoboflowDatasetWorkflow

    mock_st = MagicMock()
    mock_st.secrets = {
        "ROBOFLOW_API_KEY": "rf_dataset_secret_key_xyz",
        "ROBOFLOW_WORKSPACE": "ds-ws",
        "ROBOFLOW_PROJECT_ID": "ds-proj",
    }
    mock_st.session_state = {}

    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("ROBOFLOW_")}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            wf = RoboflowDatasetWorkflow()
            assert wf.api_key == "rf_dataset_secret_key_xyz"
            assert wf.workspace == "ds-ws"
            assert wf.project_id == "ds-proj"


def test_analysis_service_alert_methods_accept_kwargs():
    """Kiểm tra get_recent_alerts và update_alert_status trên AnalysisService chấp nhận linh hoạt kwargs."""
    service = AnalysisService()
    # Mock alert_repo
    mock_repo = MagicMock()
    mock_repo.get_recent_alerts.return_value = [{"alert_id": "ALT-999"}]
    mock_repo.update_alert_status.return_value = True
    service._alert_repo = mock_repo

    # Gọi với tham số mở rộng bất kỳ không được quăng TypeError
    alerts = service.get_recent_alerts(limit=10, status="ALL", active_only=True, unexpected_flag="test")
    assert len(alerts) == 1
    mock_repo.get_recent_alerts.assert_called_once_with(
        limit=10,
        status="ALL",
        active_only=True,
        unexpected_flag="test",
    )

    ok = service.update_alert_status("ALT-999", "CONFIRMED", resolution_type="CONFIRMED", extra_field=123)
    assert ok is True
    mock_repo.update_alert_status.assert_called_once_with(
        "ALT-999",
        "CONFIRMED",
        resolution_type="CONFIRMED",
        resolved_at=None,
        extra_field=123,
    )
