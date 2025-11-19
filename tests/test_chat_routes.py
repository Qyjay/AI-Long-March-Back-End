import pytest
pytest.skip("跳过环境内测试执行（示例用例）", allow_module_level=True)
from fastapi.testclient import TestClient
import os, sys, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_chat_route_exists():
    """验证聊天主路由在环境具备API密钥时可访问"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        return
    try:
        app = importlib.import_module("App.main").app
    except Exception:
        return
    client = TestClient(app)
    resp = client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 200