import os
import json
import asyncio
import tempfile
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from typing import List, Any, Dict
from asyncio import Queue
from App.core.config import get_settings
from typing import Optional

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_client() -> OpenAI:
    """懒加载初始化并返回 OpenAI 兼容客户端（DashScope）"""
    settings = get_settings()
    api_key = settings.DASHSCOPE_API_KEY
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请在环境变量或 .env 中设置")
    global _client
    try:
        return _client
    except NameError:
        _client = OpenAI(api_key=api_key, base_url=settings.OPENAI_BASE_URL)
        return _client

client: Optional[OpenAI] = None

class StreamRetriever(BaseRetriever):
    """支持流式输出的检索器"""
    
    def __init__(self, vectorstore: FAISS, k: int = 3):
        # 不直接设置属性，而是通过父类初始化
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """使用新的方法名避免弃用警告"""
        return self._vectorstore.similarity_search(query, k=self._k)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步版本"""
        return self._get_relevant_documents(query)

class RAGSystem:
    def __init__(self, json_file_path: str, persist_directory: str = "./faiss_index"):
        self.json_file_path = json_file_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化RAG系统"""
        try:
            # 初始化Ollama嵌入模型
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
            print("Ollama嵌入模型初始化成功")
            
            # 尝试加载现有的向量数据库
            if os.path.exists(self.persist_directory):
                print("加载现有的FAISS索引...")
                self.vectorstore = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("FAISS索引加载成功")
            else:
                print("创建新的FAISS索引...")
                self._create_vectorstore()
            
            # 初始化检索器
            self.retriever = StreamRetriever(self.vectorstore, k=3)
            
        except Exception as e:
            print(f"初始化RAG系统失败: {e}")
            # 如果加载失败，重新创建
            self._create_vectorstore()
            self.retriever = StreamRetriever(self.vectorstore, k=3)
    
    def _create_vectorstore(self):
        """创建向量数据库"""
        try:
            # 加载JSON文档
            print(f"加载JSON文件: {self.json_file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.json_file_path):
                print(f"JSON文件不存在: {self.json_file_path}")
                # 创建示例数据文件
                self._create_sample_data()
            
            # 使用JSONLoader加载文档
            def metadata_func(record: dict, metadata: dict) -> dict:
                metadata["source"] = self.json_file_path
                if "category" in record:
                    metadata["category"] = record["category"]
                if "title" in record:
                    metadata["title"] = record["title"]
                return metadata
            
            loader = JSONLoader(
                file_path=self.json_file_path,
                jq_schema='.[]',
                text_content=False,
                metadata_func=metadata_func
            )
            
            documents = loader.load()
            print(f"原始文档数量: {len(documents)}")
            
            if documents:
                # 分割文档
                splits = self.text_splitter.split_documents(documents)
                print(f"分割后文档数量: {len(splits)}")
                
                # 创建向量数据库（使用Ollama嵌入模型）
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                
                # 保存向量数据库
                self.vectorstore.save_local(self.persist_directory)
                print(f"FAISS索引创建成功，保存到: {self.persist_directory}")
            else:
                # 创建包含默认文档的向量数据库
                print("没有加载到文档，创建默认文档...")
                self._create_default_vectorstore()
                
        except Exception as e:
            print(f"创建向量数据库失败: {e}")
            # 创建包含默认文档的向量数据库作为备用
            self._create_default_vectorstore()
    
    def _create_sample_data(self):
        """创建示例数据文件"""
        sample_data = [
            {
                "content": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。",
                "category": "定义",
                "title": "人工智能定义"
            },
            {
                "content": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                "category": "技术",
                "title": "机器学习"
            },
            {
                "content": "深度学习使用具有多个层的神经网络来学习数据的层次表示。",
                "category": "技术", 
                "title": "深度学习"
            },
            {
                "content": "自然语言处理使计算机能够理解、解释和生成人类语言。",
                "category": "应用",
                "title": "自然语言处理"
            },
            {
                "content": "计算机视觉使机器能够解释和理解视觉世界。",
                "category": "应用",
                "title": "计算机视觉"
            }
        ]
        
        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"已创建示例数据文件: {self.json_file_path}")
    
    def _create_default_vectorstore(self):
        """创建默认的向量数据库"""
        default_docs = [Document(
            page_content="这是一个默认的文档，请添加您的JSON数据文件。",
            metadata={"source": "default", "category": "system"}
        )]
        self.vectorstore = FAISS.from_documents(default_docs, self.embeddings)
        self.vectorstore.save_local(self.persist_directory)
        print("已创建默认向量数据库")
    
    def search_documents(self, query: str, k: int = 3) -> List[Dict]:
        """检索相关文档"""
        try:
            if self.retriever is None or self.vectorstore is None:
                return []
            
            # 使用新的 invoke 方法替代弃用的 get_relevant_documents
            docs = self.retriever.invoke(query)
            results = []
            
            for i, doc in enumerate(docs):
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': 1.0 - (i * 0.1)  # 简单的分数估算
                })
            
            return results
            
        except Exception as e:
            print(f"文档检索失败: {e}")
            return []
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """构建包含检索上下文的提示词"""
        
        if context_docs:
            context_text = "\n\n".join([
                f"【相关文档 {i+1}】\n{doc['content']}" 
                for i, doc in enumerate(context_docs)
            ])
        else:
            context_text = "暂无相关文档"
        
        prompt_template = """基于以下检索到的相关信息，请回答用户的问题。如果信息不足以完全回答问题，请基于你的知识进行补充，

检索到的相关信息：
{context}

用户问题：{question}

请根据以上信息提供准确、有用的回答："""
        
        return prompt_template.format(
            context=context_text,
            question=query
        )

def _init_rag() -> RAGSystem:
    """初始化并返回 RAG 系统实例"""
    settings = get_settings()
    return RAGSystem(settings.RAG_JSON_PATH, settings.FAISS_DIR)

rag_system = _init_rag()

class StreamProcessor:
    """处理流式输出的类"""
    
    def __init__(self):
        self.content_queue = Queue()
    
    async def process_stream(self, completion, context_docs):
        """处理流式响应"""
        try:
            # 发送开始信号和检索上下文
            await self.content_queue.put({
                "type": "start",
                "context_docs": context_docs
            })
            
            full_content = ""
            usage_info = None
            
            # 逐块处理流式响应
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    
                    # 发送内容块
                    await self.content_queue.put({
                        "type": "content",
                        "content": content,
                        "full_content": full_content
                    })
                    
                    # 添加小延迟，让前端能看清流式效果
                    await asyncio.sleep(0.01)
                    
                elif chunk.usage:
                    usage_info = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
            
            # 发送结束信号
            await self.content_queue.put({
                "type": "complete",
                "full_content": full_content,
                "usage": usage_info,
                "context_docs": context_docs
            })
            
        except Exception as e:
            await self.content_queue.put({
                "type": "error",
                "message": str(e)
            })
    
    async def generate_responses(self):
        """生成响应数据"""
        while True:
            try:
                data = await self.content_queue.get()
                if data is None:  # 结束信号
                    break
                
                yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield "data: " + json.dumps(error_data, ensure_ascii=False) + "\n\n"

@app.post("/chat")
async def chat_stream(request: Request):
    """流式聊天接口 - 带LangChain风格RAG功能"""
    
    # 解析请求体
    try:
        body = await request.json()
        user_message = body.get('message', '请介绍一下自己')
        print(f"收到用户消息: {user_message}")
    except:
        user_message = "请介绍一下自己"
    
    # RAG检索相关文档
    print("正在进行向量检索...")
    context_docs = rag_system.search_documents(user_message, k=3)
    print(f"检索到 {len(context_docs)} 个相关文档")
    
    # 构建包含上下文的提示词
    system_prompt = rag_system.build_prompt(user_message, context_docs)
    
    # 创建流处理器
    stream_processor = StreamProcessor()
    
    async def generate():
        """流式生成器"""
        try:
            # 发起流式请求
            completion = get_client().chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # 启动流处理任务
            processing_task = asyncio.create_task(
                stream_processor.process_stream(completion, context_docs)
            )
            
            # 生成响应
            async for response in stream_processor.generate_responses():
                yield response
            
            # 发送结束信号到队列
            await stream_processor.content_queue.put(None)
            
            # 等待处理任务完成
            await processing_task
            
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield "data: " + json.dumps(error_data, ensure_ascii=False) + "\n\n"
            print(f"流式生成错误: {e}")
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
@app.get("/")
async def root():
    return {"message": "基于Ollama嵌入模型的RAG流式聊天服务器已启动"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)