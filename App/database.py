import pymysql
from sqlalchemy.engine.url import make_url
from .core.config import get_settings

def get_db_connection():
    """创建并返回一个MySQL数据库连接（PyMySQL）。

    解析 `DATABASE_URL`（形如 mysql+pymysql://user:pass@host:port/db），
    使用 PyMySQL 建立原生连接，便于脚本级操作。
    """
    settings = get_settings()
    url = make_url(settings.DATABASE_URL)
    assert url.get_backend_name().startswith("mysql"), "当前配置非MySQL连接串"
    return pymysql.connect(
        host=url.host or "localhost",
        port=url.port or 3306,
        user=url.username,
        password=url.password or "",
        database=url.database,
        charset="utf8mb4",
        autocommit=False,
    )