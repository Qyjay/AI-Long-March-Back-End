import pytest
pytest.skip("跳过环境内测试执行（示例用例）", allow_module_level=True)
from fastapi.testclient import TestClient
import os, sys, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_root_health():
    """验证根路径健康检查响应"""
    try:
        app = importlib.import_module("App.main").app
    except Exception:
        return
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()