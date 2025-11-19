from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from ...ChatModel import StreamProcessor, get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from ...services.rag.dispatcher import get_rag_system
from ...core.logging import get_logger
from ...core.config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger()

@router.post("/")
async def chat_stream(request: Request):
    """流式聊天接口 - 带RAG检索"""
    try:
        body = await request.json()
        user_message = body.get('message', '请介绍一下自己')
        logger.info(f"收到用户消息: {user_message}")
    except Exception:
        user_message = "请介绍一下自己"

    logger.info("正在进行向量检索...")
    rag_system = get_rag_system()
    context_docs = rag_system.search_documents(user_message, k=3)
    logger.info(f"检索到 {len(context_docs)} 个相关文档")

    system_prompt = rag_system.build_qa_prompt(user_message, context_docs)
    stream_processor = StreamProcessor()

    async def generate():
        """SSE流生成器，统一事件格式"""
        try:
            llm = get_llm()
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            completion = llm.stream(messages)

            processing_task = asyncio.create_task(
                stream_processor.process_stream(completion, context_docs)
            )

            async for response in stream_processor.generate_responses():
                yield response

            await stream_processor.content_queue.put(None)
            await processing_task

        except Exception as e:
            # LLM不可用时，生成降级SSE输出
            logger.error(f"流式生成错误，启用降级输出: {e}")
            fallback = "基于检索到的资料，以下是简要讲解：\n"
            for i, d in enumerate(context_docs[:2]):
                src = d.get('metadata', {}).get('source', '未知')
                content = (d.get('content', '') or '')[:300]
                fallback += f"【素材{i+1} - 来源：{src}】\n{content}\n"
            await stream_processor.process_fallback(context_docs, fallback)
            async for response in stream_processor.generate_responses():
                yield response
            await stream_processor.content_queue.put(None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/explanation")
async def chat_explanation_stream(request: Request):
    """讲解员风格聊天接口 - 流式输出（统一事件格式）"""
    try:
        body = await request.json()
        route = body.get('route', '')
        choice = body.get('choice', '')
        real_history_choice = body.get('real_history_choice', '')
        history_background = body.get('history_background', '')
        tactical_logic = body.get('tactical_logic', '')
        logger.info(f"收到讲解请求: {route} - {choice}")
    except Exception as e:
        return {"error": f"请求参数解析失败: {str(e)}"}

    logger.info("正在进行向量检索...")
    query = f"{route} {choice} {real_history_choice}"
    rag_system = get_rag_system()
    context_docs = rag_system.search_documents(query, k=3)
    logger.info(f"检索到 {len(context_docs)} 个相关文档")

    system_prompt = rag_system.build_explanation_prompt(
        route, choice, real_history_choice, history_background, tactical_logic
    )
    stream_processor = StreamProcessor()

    async def generate():
        """SSE流生成器，统一事件格式"""
        try:
            llm = get_llm()
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"请分析在{route}节点选择{choice}的情况")]
            completion = llm.stream(messages)

            processing_task = asyncio.create_task(
                stream_processor.process_stream(completion, context_docs)
            )

            async for response in stream_processor.generate_responses():
                yield response

            await stream_processor.content_queue.put(None)
            await processing_task

        except Exception as e:
            logger.error(f"流式生成错误，启用降级输出: {e}")
            fallback = f"节点 {route} 的选择 {choice} 解析（降级输出）：\n"
            for i, d in enumerate(context_docs[:2]):
                src = d.get('metadata', {}).get('source', '未知')
                content = (d.get('content', '') or '')[:300]
                fallback += f"【素材{i+1} - 来源：{src}】\n{content}\n"
            await stream_processor.process_fallback(context_docs, fallback)
            async for response in stream_processor.generate_responses():
                yield response
            await stream_processor.content_queue.put(None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/poem")
async def chat_poem_stream(request: Request):
    """诗词赏析风格聊天接口 - 流式输出（统一事件格式）"""
    try:
        body = await request.json()
        route = body.get('route', '')
        poem = body.get('poem', '')
        creation_background = body.get('creation_background', '')
        poem_analysis = body.get('poem_analysis', '')
        spirit = body.get('spirit', '')
        meanings = body.get('meanings', '')
        logger.info(f"收到诗词赏析请求: {route} - {poem}")
    except Exception as e:
        return {"error": f"请求参数解析失败: {str(e)}"}

    logger.info("正在进行向量检索...")
    query = f"{route} {poem} 长征诗词"
    rag_system = get_rag_system()
    context_docs = rag_system.search_documents(query, k=3)
    logger.info(f"检索到 {len(context_docs)} 个相关文档")

    system_prompt = rag_system.build_poem_prompt(
        route, poem, creation_background, poem_analysis, spirit, meanings
    )
    stream_processor = StreamProcessor()

    async def generate():
        """SSE流生成器，统一事件格式"""
        try:
            llm = get_llm()
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"请赏析{route}节点的诗词{poem}")]
            completion = llm.stream(messages)

            processing_task = asyncio.create_task(
                stream_processor.process_stream(completion, context_docs)
            )

            async for response in stream_processor.generate_responses():
                yield response

            await stream_processor.content_queue.put(None)
            await processing_task

        except Exception as e:
            logger.error(f"流式生成错误，启用降级输出: {e}")
            fallback = f"诗词赏析（节点 {route}，作品 {poem}）降级输出：\n"
            for i, d in enumerate(context_docs[:2]):
                src = d.get('metadata', {}).get('source', '未知')
                content = (d.get('content', '') or '')[:300]
                fallback += f"【素材{i+1} - 来源：{src}】\n{content}\n"
            await stream_processor.process_fallback(context_docs, fallback)
            async for response in stream_processor.generate_responses():
                yield response
            await stream_processor.content_queue.put(None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/qa")
async def chat_qa_stream(request: Request):
    """问答风格聊天接口 - 流式输出（统一事件格式）"""
    try:
        body = await request.json()
        question = body.get('question', '请介绍一下长征')
        logger.info(f"收到问答请求: {question}")
    except Exception:
        question = "请介绍一下长征"

    logger.info("正在进行向量检索...")
    rag_system = get_rag_system()
    context_docs = rag_system.search_documents(question, k=3)
    logger.info(f"检索到 {len(context_docs)} 个相关文档")

    system_prompt = rag_system.build_qa_prompt(question, context_docs)
    stream_processor = StreamProcessor()

    async def generate():
        """SSE流生成器，统一事件格式"""
        try:
            llm = get_llm()
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
            completion = llm.stream(messages)

            processing_task = asyncio.create_task(
                stream_processor.process_stream(completion, context_docs)
            )

            async for response in stream_processor.generate_responses():
                yield response

            await stream_processor.content_queue.put(None)
            await processing_task

        except Exception as e:
            error_data = {"type": "error", "content": "", "meta": {"message": str(e)}}
            yield "data: " + json.dumps(error_data, ensure_ascii=False) + "\n\n"
            logger.error(f"流式生成错误: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )