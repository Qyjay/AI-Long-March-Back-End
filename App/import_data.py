import os
import json
from .database import get_db_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'front-end', 'src', 'data')
STATIC_DATA_DIR = os.path.join(BASE_DIR, 'static', 'assets', 'jsons')

def import_nodes():
    """导入节点数据到 MySQL（含表结构创建）。

    数据来源优先：back-end/static/assets/jsons/nodes.mock.json；
    若不存在则回退到 front-end/src/data/nodes.mock.json。
    """
    candidate_static = os.path.join(STATIC_DATA_DIR, 'nodes.mock.json')
    candidate_front = os.path.join(FRONT_DATA_DIR, 'nodes.mock.json')
    src_path = candidate_static if os.path.exists(candidate_static) else candidate_front
    with open(src_path, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id VARCHAR(255) PRIMARY KEY,
            name_zh VARCHAR(255),
            name_en VARCHAR(255),
            lng DOUBLE,
            lat DOUBLE,
            time VARCHAR(255),
            time_en VARCHAR(255),
            summary_zh TEXT,
            summary_en TEXT,
            description_zh TEXT,
            description_en TEXT,
            poem_zh TEXT,
            poem_en TEXT,
            image VARCHAR(255),
            gallery JSON,
            assets JSON,
            interactive_decision JSON,
            historical_significance TEXT,
            casualties VARCHAR(255),
            weather VARCHAR(255),
            terrain VARCHAR(255),
            participants VARCHAR(255),
            enemy_forces VARCHAR(255),
            battle_duration VARCHAR(255),
            meeting_duration VARCHAR(255),
            key_decisions VARCHAR(255)
        )
        """
    )
    try:
        cursor.execute("ALTER TABLE nodes MODIFY summary_zh TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY summary_en TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY description_zh TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY description_en TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY poem_zh TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY poem_en TEXT")
        cursor.execute("ALTER TABLE nodes MODIFY historical_significance TEXT")
        cursor.execute("ALTER TABLE nodes ADD COLUMN interactive_decision JSON")
    except Exception:
        pass

    for node in nodes:
        image_path = node.get("image") or ""
        if image_path and not image_path.startswith("/static/"):
            image_path = f"/static/assets/{image_path}"
        gallery = [g.replace("/assets/", "/static/assets/") for g in (node.get("gallery") or [])]
        assets_raw = node.get("assets") or {}
        assets = {k: (v.replace("/assets/", "/static/assets/") if isinstance(v, str) else v) for k, v in assets_raw.items()}
        cursor.execute(
            """
            INSERT INTO nodes (
                id, name_zh, name_en, lng, lat, time, time_en, summary_zh, summary_en,
                description_zh, description_en, poem_zh, poem_en, image, gallery, assets, interactive_decision,
                historical_significance, casualties, weather, terrain, participants,
                enemy_forces, battle_duration, meeting_duration, key_decisions
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE 
                image = VALUES(image),
                gallery = VALUES(gallery),
                assets = VALUES(assets),
                interactive_decision = VALUES(interactive_decision)
            """,
            (
                node.get("id"),
                node.get("name_zh"),
                node.get("name_en"),
                node.get("lng"),
                node.get("lat"),
                node.get("time"),
                node.get("time_en"),
                node.get("summary_zh"),
                node.get("summary_en"),
                node.get("description_zh"),
                node.get("description_en"),
                node.get("poem_zh"),
                node.get("poem_en"),
                image_path,
                json.dumps(gallery, ensure_ascii=False),
                json.dumps(assets, ensure_ascii=False),
                json.dumps(node.get("interactive_decision"), ensure_ascii=False),
                node.get("historical_significance"),
                node.get("casualties"),
                node.get("weather"),
                node.get("terrain"),
                node.get("participants"),
                node.get("enemy_forces"),
                node.get("battle_duration"),
                node.get("meeting_duration"),
                node.get("key_decisions"),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

def import_routes():
    """导入路线数据到 MySQL（GeoJSON FeatureCollection）。

    数据来源优先：back-end/static/assets/jsons/route.mock.json；
    若不存在则回退到 front-end/src/data/route.mock.json。
    """
    candidate_static = os.path.join(STATIC_DATA_DIR, 'route.mock.json')
    candidate_front = os.path.join(FRONT_DATA_DIR, 'route.mock.json')
    src_path = candidate_static if os.path.exists(candidate_static) else candidate_front
    with open(src_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS routes (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            description VARCHAR(255),
            distance_km DOUBLE,
            duration_days INT,
            coordinates JSON
        )
        """
    )

    features = data.get("features", [])
    for f in features:
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [])
        cursor.execute(
            """
            INSERT INTO routes (id, name, description, distance_km, duration_days, coordinates)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE id = id
            """,
            (
                props.get("id"),
                props.get("name"),
                props.get("description"),
                props.get("distance_km"),
                props.get("duration_days"),
                json.dumps(coords, ensure_ascii=False),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

def import_scenes():
    """导入场景数据到 MySQL。

    数据来源：front-end/src/data/scenes.mock.json
    """
    src_path = os.path.join(FRONT_DATA_DIR, 'scenes.mock.json')
    with open(src_path, "r", encoding="utf-8") as file:
        scenes = json.load(file)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scenes (
            id VARCHAR(255) PRIMARY KEY,
            node_id VARCHAR(255),
            title VARCHAR(255),
            type VARCHAR(255),
            background VARCHAR(255),
            music VARCHAR(255),
            characters JSON,
            dialogues JSON,
            choices JSON
        )
        """
    )

    for s in scenes:
        bg = (s.get("background") or "").replace("/assets/", "/static/assets/")
        music = (s.get("music") or "").replace("/assets/", "/static/assets/")
        characters = s.get("characters", [])
        for ch in characters:
            if isinstance(ch.get("avatar"), str):
                ch["avatar"] = ch["avatar"].replace("/assets/", "/static/assets/")
        cursor.execute(
            """
            INSERT INTO scenes (id, node_id, title, type, background, music, characters, dialogues, choices)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE id = id
            """,
            (
                s.get("id"),
                s.get("nodeId") or s.get("node_id"),
                s.get("title"),
                s.get("type"),
                bg,
                music,
                json.dumps(characters, ensure_ascii=False),
                json.dumps(s.get("dialogues", []), ensure_ascii=False),
                json.dumps(s.get("choices", []), ensure_ascii=False),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

def import_achievements():
    """导入成就数据到 MySQL。

    数据来源：front-end/src/data/achievements.mock.json
    """
    src_path = os.path.join(FRONT_DATA_DIR, 'achievements.mock.json')
    with open(src_path, "r", encoding="utf-8") as file:
        achs = json.load(file)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS achievements (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            description VARCHAR(1024),
            icon VARCHAR(255),
            category VARCHAR(255),
            points INT,
            rarity VARCHAR(255),
            unlock_condition JSON
        )
        """
    )

    for a in achs:
        icon = (a.get("icon") or "").replace("/assets/", "/static/assets/")
        cursor.execute(
            """
            INSERT INTO achievements (id, name, description, icon, category, points, rarity, unlock_condition)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE id = id
            """,
            (
                a.get("id"),
                a.get("name"),
                a.get("description"),
                icon,
                a.get("category"),
                a.get("points"),
                a.get("rarity"),
                json.dumps(a.get("unlockCondition", {}), ensure_ascii=False),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

def import_history():
    """导入历史综合内容到 MySQL。

    数据来源：front-end/src/data/history.mock.json
    """
    src_path = os.path.join(FRONT_DATA_DIR, 'history.mock.json')
    with open(src_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history_content (
            id VARCHAR(255) PRIMARY KEY,
            people JSON,
            stories JSON,
            tourism JSON
        )
        """
    )

    tourism = data.get("tourism", [])
    for t in tourism:
        if isinstance(t.get("image"), str):
            t["image"] = t["image"].replace("/assets/", "/static/assets/")

    cursor.execute(
        """
        INSERT INTO history_content (id, people, stories, tourism)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE id = id
        """,
        (
            'default',
            json.dumps(data.get("people", []), ensure_ascii=False),
            json.dumps(data.get("stories", []), ensure_ascii=False),
            json.dumps(tourism, ensure_ascii=False),
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

def import_historical_media():
    """导入历史媒体图集到 MySQL。

    数据来源：front-end/src/data/historical-media.mock.json
    """
    src_path = os.path.join(FRONT_DATA_DIR, 'historical-media.mock.json')
    with open(src_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    groups = data.get("historicalMedia", {})

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_media (
            id VARCHAR(255) PRIMARY KEY,
            node_key VARCHAR(255),
            items JSON
        )
        """
    )

    for key, items in groups.items():
        normalized = []
        for it in items:
            it = dict(it)
            for field in ("thumbnail", "url", "poster"):
                if isinstance(it.get(field), str):
                    it[field] = it[field].replace("/assets/", "/static/assets/")
            normalized.append(it)
        cursor.execute(
            """
            INSERT INTO historical_media (id, node_key, items)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE id = id
            """,
            (
                key,
                key,
                json.dumps(normalized, ensure_ascii=False),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # 默认仅执行基础数据导入：节点与路线（静态目录优先）
    import_nodes()
    import_routes()
