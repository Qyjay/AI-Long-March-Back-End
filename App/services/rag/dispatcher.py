def get_rag_system():
    """返回默认的 RAG 系统实例（DashScope API 实现）"""
    from ...ChatModel import rag_system
    return rag_system