# API 文档

## 1. 节点相关接口

### 获取所有节点信息
- **端点**: `GET /nodes`
- **描述**: 获取所有节点的详细信息。
- **响应模型**: `List[Node]`
- **示例请求**:
  ```bash
  curl -X GET "http://localhost:8000/nodes"
  ```
- **示例响应**:
  ```json
  [
    {
      "id": "node1",
      "name_zh": "节点1",
      "name_en": "Node 1",
      "lng": 116.404,
      "lat": 39.915,
      "time": "2023-01-01",
      "time_en": "2023-01-01",
      "summary_zh": "节点1的摘要",
      "summary_en": "Summary of Node 1",
      "description_zh": "节点1的详细描述",
      "description_en": "Detailed description of Node 1",
      "poem_zh": "节点1的诗句",
      "poem_en": "Poem of Node 1",
      "image": "node1.jpg",
      "gallery": ["image1.jpg", "image2.jpg"],
      "assets": {"key": "value"},
      "historical_significance": "历史意义",
      "casualties": "伤亡情况",
      "weather": "天气情况",
      "terrain": "地形描述",
      "participants": "参与者",
      "enemy_forces": "敌方力量",
      "battle_duration": "战斗持续时间",
      "meeting_duration": "会议持续时间",
      "key_decisions": "关键决策"
    }
  ]
  ```

### 获取特定节点信息
- **端点**: `GET /nodes/{node_id}`
- **描述**: 根据节点 ID 获取特定节点的详细信息。
- **参数**:
  - `node_id` (路径参数): 节点 ID。
- **响应模型**: `Node`
- **示例请求**:
  ```bash
  curl -X GET "http://localhost:8000/nodes/node1"
  ```
- **示例响应**:
  ```json
  {
    "id": "node1",
    "name_zh": "节点1",
    "name_en": "Node 1",
    "lng": 116.404,
    "lat": 39.915,
    "time": "2023-01-01",
    "time_en": "2023-01-01",
    "summary_zh": "节点1的摘要",
    "summary_en": "Summary of Node 1",
    "description_zh": "节点1的详细描述",
    "description_en": "Detailed description of Node 1",
    "poem_zh": "节点1的诗句",
    "poem_en": "Poem of Node 1",
    "image": "node1.jpg",
    "gallery": ["image1.jpg", "image2.jpg"],
    "assets": {"key": "value"},
    "historical_significance": "历史意义",
    "casualties": "伤亡情况",
    "weather": "天气情况",
    "terrain": "地形描述",
    "participants": "参与者",
    "enemy_forces": "敌方力量",
    "battle_duration": "战斗持续时间",
    "meeting_duration": "会议持续时间",
    "key_decisions": "关键决策"
  }
  ```

## 2. 路线相关接口

### 获取路线信息
- **端点**: `GET /route`
- **描述**: 获取当前路线的详细信息。
- **响应模型**: `Route`
- **示例请求**:
  ```bash
  curl -X GET "http://localhost:8000/route"
  ```
- **示例响应**:
  ```json
  {
    "id": "route1",
    "name": "长征路线",
    "description": "长征路线的详细描述",
    "distance_km": 1000.5,
    "duration_days": 30,
    "coordinates": [[116.404, 39.915], [116.405, 39.916]]
  }
  ```

## 3. 用户进度相关接口

### 保存用户进度
- **端点**: `POST /progress`
- **描述**: 保存用户的进度信息。
- **请求体模型**: `UserProgress`
- **示例请求**:
  ```bash
  curl -X POST "http://localhost:8000/progress" -H "Content-Type: application/json" -d '{
    "user_id": "user1",
    "unlocked_nodes": ["node1", "node2"],
    "achievements": ["achievement1", "achievement2"]
  }'
  ```
- **示例响应**:
  ```json
  {
    "status": "success"
  }
  ```

## 4. 聊天相关接口

### 剧情决策解析（流式输出）
- **端点**: `POST /chat/explanation`
- **描述**: 提供剧情决策的解析，支持流式输出。
- **请求体参数**:
  - `route`: 路线名称。
  - `choice`: 用户的选择。
  - `real_history_choice`: 历史真实决策。
  - `history_background`: 历史背景。
  - `tactical_logic`: 战术逻辑。
- **示例请求**:
  ```bash
  curl -X POST "http://localhost:8000/chat/explanation" -H "Content-Type: application/json" -d '{
    "route": "长征路线",
    "choice": "正面突破",
    "real_history_choice": "分兵佯攻",
    "history_background": "历史背景描述",
    "tactical_logic": "战术逻辑描述"
  }'
  ```
- **响应**: 流式输出，内容为讲解员风格的解析。

### 诗词赏析（流式输出）
- **端点**: `POST /chat/poem`
- **描述**: 提供诗词赏析，支持流式输出。
- **请求体参数**:
  - `route`: 路线名称。
  - `poem`: 诗词内容。
  - `creation_background`: 创作背景。
  - `poem_analysis`: 诗词解析。
  - `spirit`: 精神内涵。
  - `meanings`: 当代意义。
- **示例请求**:
  ```bash
  curl -X POST "http://localhost:8000/chat/poem" -H "Content-Type: application/json" -d '{
    "route": "长征路线",
    "poem": "七律·长征",
    "creation_background": "创作背景描述",
    "poem_analysis": "诗词解析描述",
    "spirit": "精神内涵描述",
    "meanings": "当代意义描述"
  }'
  ```
- **响应**: 流式输出，内容为诗词赏析。

## 5. RAG 系统功能

### 检索相关文档
- **方法**: `search_documents(query: str, k: int = 3)`
- **描述**: 根据查询检索相关文档。
- **参数**:
  - `query`: 查询字符串。
  - `k`: 返回的文档数量（默认为 3）。
- **返回值**: 包含文档内容和元数据的列表。

### 构建提示词
- **方法**: `build_explanation_prompt`, `build_poem_prompt`, `build_qa_prompt`
- **描述**: 根据输入参数构建不同风格的提示词，用于生成响应。

## 6. 流式处理器

### 处理流式输出
- **类**: `StreamProcessor`
- **描述**: 处理流式响应，生成流式数据块。