from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """项目配置类，负责加载环境变量与默认值"""

    # 基础服务配置
    APP_NAME: str = Field(default="LongMarch Backend")
    LOG_LEVEL: str = Field(default="INFO")

    # CORS 配置（逗号分隔的域名列表）
    CORS_ORIGINS: str = Field(default="*")

    # 数据库配置（切换为 MySQL，默认本地 root/root）
    DATABASE_URL: str = Field(default="mysql+pymysql://root:root@localhost:3306/app")

    # LLM 与 Embedding 相关配置（全部来自 .env）
    LLM_MODEL: str = Field(default="qwen-plus")
    DASHSCOPE_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_BASE_URL: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    EMBEDDING_MODEL: str = Field(default="text-embedding-v4")

    # RAG 与索引目录
    RAG_DOCX_PATH: str = Field(default=r"App\重走长征路.docx")
    RAG_JSON_PATH: str = Field(default=r"data.json")
    FAISS_DIR: str = Field(default=r"./faiss_index")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """获取配置单例，避免重复读取环境"""
    # 简单的模块级缓存即可满足需求
    global _settings
    try:
        return _settings
    except NameError:
        _settings = Settings()
        return _settings