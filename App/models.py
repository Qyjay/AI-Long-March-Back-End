from sqlalchemy import Column, String, Integer, Float, JSON, Text
from sqlalchemy.ext.declarative import declarative_base

# 声明基类
Base = declarative_base()

class Node(Base):
    __tablename__ = 'nodes'

    id = Column(String(255), primary_key=True)
    name_zh = Column(String(255))
    name_en = Column(String(255))
    lng = Column(Float)
    lat = Column(Float)
    time = Column(String(255))
    time_en = Column(String(255))
    summary_zh = Column(Text)
    summary_en = Column(Text)
    description_zh = Column(Text)
    description_en = Column(Text)
    poem_zh = Column(Text)
    poem_en = Column(Text)
    image = Column(String(255))
    gallery = Column(JSON)
    assets = Column(JSON)
    interactive_decision = Column(JSON)
    historical_significance = Column(Text)
    casualties = Column(String(255))
    weather = Column(String(255))
    terrain = Column(String(255))
    participants = Column(String(255))
    enemy_forces = Column(String(255))
    battle_duration = Column(String(255))
    meeting_duration = Column(String(255))
    key_decisions = Column(String(255))

class Route(Base):
    __tablename__ = 'routes'

    id = Column(String(255), primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    distance_km = Column(Float)
    duration_days = Column(Integer)
    coordinates = Column(JSON)

class UserProgress(Base):
    __tablename__ = 'user_progress'

    user_id = Column(String(255), primary_key=True)
    unlocked_nodes = Column(JSON)
    achievements = Column(JSON)