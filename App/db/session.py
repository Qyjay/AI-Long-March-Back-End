from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker, Session
from ..core.config import get_settings

def ensure_database_exists(database_url: str) -> None:
    """确保MySQL数据库存在，不存在则创建。

    解析连接串，先连接到服务器级别（不指定数据库），执行
    `CREATE DATABASE IF NOT EXISTS`，字符集使用utf8mb4。
    """
    url = make_url(database_url)
    if url.get_backend_name().startswith("mysql"):
        server_url = url.set(database="mysql")
        engine = create_engine(server_url)
        db_name = url.database
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        engine.dispose()

def get_engine():
    """创建并返回 SQLAlchemy 引擎（MySQL自动建库）"""
    settings = get_settings()
    ensure_database_exists(settings.DATABASE_URL)
    return create_engine(settings.DATABASE_URL)

def get_session() -> Session:
    """创建并返回一个新的数据库会话"""
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()