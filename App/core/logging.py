from loguru import logger
from .config import get_settings

def setup_logging() -> None:
    """初始化日志配置，按环境变量设置日志级别与格式"""
    settings = get_settings()
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.LOG_LEVEL.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )

def get_logger():
    """获取全局日志记录器"""
    return logger