from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker
from .core.config import get_settings

# 导入模型
from .models import Base, Node, Route, UserProgress

def ensure_mysql_database(database_url: str) -> None:
    """确保MySQL数据库存在，不存在则创建。"""
    url = make_url(database_url)
    if url.get_backend_name().startswith("mysql"):
        server_url = url.set(database="mysql")
        engine = create_engine(server_url)
        db_name = url.database
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        engine.dispose()

if __name__ == "__main__":
    """初始化数据库，创建所有表"""
    settings = get_settings()
    ensure_mysql_database(settings.DATABASE_URL)
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成！")
