from typing import List, Optional, Dict, Any
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api.routers.chat import router as chat_router
from .core.config import get_settings
from .db.session import get_session
from .models import Node as NodeModel, Route as RouteModel, UserProgress as UserProgressModel
from .core.logging import setup_logging, get_logger

app = FastAPI()

# 初始化配置与日志，并设置 CORS
settings = get_settings()
setup_logging()
logger = get_logger()
allow_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册（聊天模块）
app.include_router(chat_router)

class Node(BaseModel):
    """节点数据模型"""
    id: str
    name_zh: str
    name_en: str
    lng: float
    lat: float
    time: str
    time_en: str
    summary_zh: str
    summary_en: str
    description_zh: str
    description_en: str
    poem_zh: str
    poem_en: str
    image: str
    gallery: List[str]
    assets: dict
    historical_significance: str
    casualties: Optional[str] = None
    weather: Optional[str] = None
    terrain: Optional[str] = None
    participants: Optional[str] = None
    enemy_forces: Optional[str] = None
    battle_duration: Optional[str] = None
    meeting_duration: Optional[str] = None
    key_decisions: Optional[str] = None
    interactive_decision: Optional[Dict[str, Any]] = None

class Route(BaseModel):
    """路线数据模型"""
    id: str
    name: str
    description: str
    distance_km: float
    duration_days: int
    coordinates: List[List[float]]

class GeoJSONFeature(BaseModel):
    """GeoJSON Feature（用于路线返回）"""
    type: str
    properties: Dict[str, Any]
    geometry: Dict[str, Any]

class FeatureCollection(BaseModel):
    """GeoJSON FeatureCollection（用于路线返回）"""
    type: str
    features: List[GeoJSONFeature]

class UserProgress(BaseModel):
    """用户进度模型"""
    user_id: str
    unlocked_nodes: List[str]
    achievements: List[str]

class History(BaseModel):
    """历史内容聚合模型"""
    summary: str
    total_nodes: int
    key_topics: List[str]

class Achievement(BaseModel):
    """成就模型"""
    id: str
    name: str
    description: str = ""

class Scene(BaseModel):
    """场景模型（简化版：每个节点派生一个场景）"""
    id: str
    node_id: str
    title: str
    description: str

class DemoScript(BaseModel):
    """演示脚本模型（占位实现）"""
    title: str
    sections: List[str]

@app.get("/nodes", response_model=List[Node])
async def get_nodes():
    """查询所有节点（SQLAlchemy）"""
    try:
        session = get_session()
        records = session.query(NodeModel).all()
        result = []
        for r in records:
            result.append(Node(
                id=r.id,
                name_zh=r.name_zh,
                name_en=r.name_en,
                lng=r.lng,
                lat=r.lat,
                time=r.time,
                time_en=r.time_en,
                summary_zh=r.summary_zh,
                summary_en=r.summary_en,
                description_zh=r.description_zh,
                description_en=r.description_en,
                poem_zh=r.poem_zh,
                poem_en=r.poem_en,
                image=r.image,
                gallery=r.gallery or [],
                assets=r.assets or {},
                interactive_decision=r.interactive_decision or None,
                historical_significance=r.historical_significance,
                casualties=r.casualties,
                weather=r.weather,
                terrain=r.terrain,
                participants=r.participants,
                enemy_forces=r.enemy_forces,
                battle_duration=r.battle_duration,
                meeting_duration=r.meeting_duration,
                key_decisions=r.key_decisions,
            ))
        session.close()
        return result
    except Exception as e:
        logger.error(f"查询节点失败: {e}")
        raise HTTPException(status_code=500, detail="查询节点失败")

@app.get("/nodes/{node_id}", response_model=Node)
async def get_node(node_id: str):
    """按ID查询节点（SQLAlchemy）"""
    try:
        session = get_session()
        r = session.query(NodeModel).filter(NodeModel.id == node_id).first()
        session.close()
        if not r:
            raise HTTPException(status_code=404, detail="节点不存在")
        return Node(
            id=r.id,
            name_zh=r.name_zh,
            name_en=r.name_en,
            lng=r.lng,
            lat=r.lat,
            time=r.time,
            time_en=r.time_en,
            summary_zh=r.summary_zh,
            summary_en=r.summary_en,
            description_zh=r.description_zh,
            description_en=r.description_en,
            poem_zh=r.poem_zh,
            poem_en=r.poem_en,
            image=r.image,
            gallery=r.gallery or [],
            assets=r.assets or {},
            interactive_decision=r.interactive_decision or None,
            historical_significance=r.historical_significance,
            casualties=r.casualties,
            weather=r.weather,
            terrain=r.terrain,
            participants=r.participants,
            enemy_forces=r.enemy_forces,
            battle_duration=r.battle_duration,
            meeting_duration=r.meeting_duration,
            key_decisions=r.key_decisions,
        )
    except Exception as e:
        logger.error(f"查询节点失败: {e}")
        raise HTTPException(status_code=500, detail="查询节点失败")

@app.get("/route", response_model=FeatureCollection)
async def get_route():
    """查询路线（GeoJSON FeatureCollection，SQLAlchemy）"""
    try:
        session = get_session()
        records = session.query(RouteModel).all()
        session.close()
        if not records:
            raise HTTPException(status_code=404, detail="未配置路线")
        features: List[GeoJSONFeature] = []
        for r in records:
            features.append(GeoJSONFeature(
                type="Feature",
                properties={
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "distance_km": r.distance_km,
                    "duration_days": r.duration_days,
                },
                geometry={
                    "type": "LineString",
                    "coordinates": r.coordinates or [],
                },
            ))
        return FeatureCollection(type="FeatureCollection", features=features)
    except Exception as e:
        logger.error(f"查询路线失败: {e}")
        raise HTTPException(status_code=500, detail="查询路线失败")

@app.post("/progress")
async def save_progress(progress: UserProgress):
    """保存用户进度（SQLAlchemy）"""
    try:
        session = get_session()
        record = UserProgressModel(
            user_id=progress.user_id,
            unlocked_nodes=progress.unlocked_nodes,
            achievements=progress.achievements,
        )
        session.add(record)
        session.commit()
        session.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"保存进度失败: {e}")
        raise HTTPException(status_code=500, detail="保存进度失败")

@app.get("/history", response_model=History)
async def get_history():
    """获取历史内容聚合
    - 汇总节点数量与若干关键主题（由节点的历史意义字段派生）
    """
    try:
        session = get_session()
        records = session.query(NodeModel).all()
        session.close()
        total = len(records)
        topics: List[str] = []
        for r in records:
            if getattr(r, "historical_significance", None):
                topics.append(str(r.historical_significance))
            if len(topics) >= 5:
                break
        summary = f"包含 {total} 个历史节点，涵盖长征相关关键主题。"
        return History(summary=summary, total_nodes=total, key_topics=topics)
    except Exception as e:
        logger.error(f"聚合历史内容失败: {e}")
        raise HTTPException(status_code=500, detail="聚合历史内容失败")

@app.get("/achievements", response_model=List[Achievement])
async def get_achievements():
    """获取成就列表
    - 聚合用户进度中的成就字段并去重
    """
    try:
        session = get_session()
        records = session.query(UserProgressModel).all()
        session.close()
        uniq = {}
        for r in records:
            for a in (r.achievements or []):
                if a not in uniq:
                    uniq[a] = Achievement(id=a, name=a, description="")
        return list(uniq.values())
    except Exception as e:
        logger.error(f"查询成就失败: {e}")
        raise HTTPException(status_code=500, detail="查询成就失败")

@app.get("/scenes", response_model=List[Scene])
async def get_scenes(nodeId: Optional[str] = None):
    """获取场景列表
    - 简化：每个节点派生一个场景；支持按节点筛选
    """
    try:
        session = get_session()
        records = session.query(NodeModel).all()
        session.close()
        result: List[Scene] = []
        for r in records:
            if nodeId and r.id != nodeId:
                continue
            result.append(Scene(
                id=r.id,
                node_id=r.id,
                title=f"{r.name_zh} 场景",
                description=r.summary_zh or r.description_zh or ""
            ))
        return result
    except Exception as e:
        logger.error(f"查询场景失败: {e}")
        raise HTTPException(status_code=500, detail="查询场景失败")

@app.get("/scenes/{node_id}", response_model=Scene)
async def get_scene_by_node(node_id: str):
    """按节点ID获取场景（简化：节点即场景）"""
    try:
        session = get_session()
        r = session.query(NodeModel).filter(NodeModel.id == node_id).first()
        session.close()
        if not r:
            raise HTTPException(status_code=404, detail="节点不存在")
        return Scene(
            id=r.id,
            node_id=r.id,
            title=f"{r.name_zh} 场景",
            description=r.summary_zh or r.description_zh or ""
        )
    except Exception as e:
        logger.error(f"查询场景失败: {e}")
        raise HTTPException(status_code=500, detail="查询场景失败")

@app.get("/demo-script", response_model=DemoScript)
async def get_demo_script():
    """获取演示脚本（占位实现）"""
    try:
        session = get_session()
        records = session.query(NodeModel).all()
        session.close()
        names = [r.name_zh for r in records][:5]
        sections = [f"第{i+1}站：{n}" for i, n in enumerate(names)] or ["长征线路导览", "节点讲解示例"]
        return DemoScript(title="重走长征路演示脚本", sections=sections)
    except Exception as e:
        logger.error(f"获取演示脚本失败: {e}")
        raise HTTPException(status_code=500, detail="获取演示脚本失败")

@app.get("/")
async def root():
    """健康检查与欢迎信息"""
    return {"message": "FastAPI流式聊天服务器已启动"}

@app.get("/health")
async def health():
    """健康检查端点（供前端探测服务状态）"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
# 静态资源挂载（提供 /static/assets/* 图像、音频等）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
