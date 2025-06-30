"""
检索执行器模块
负责从知识库检索相关信息
"""
from typing import List, Dict, Any

from executors.base_executor import ExecutorBase
from core.data_structures import Task, ExecutorError
from core.context_manager import ContextManager
from knowledge.knowledge_base import ChromaKnowledgeBase


class RetrievalExecutor(ExecutorBase):
    """检索执行器：从向量知识库检索相关文档"""
    
    def __init__(self, kb: ChromaKnowledgeBase):
        super().__init__()
        self.kb = kb

    async def execute(self, task: Task, context: ContextManager) -> List[Dict[str, Any]]:
        """执行检索任务"""
        logic_input = task.logic_input
        query_to_retrieve = logic_input.get("query")
        filter_param = logic_input.get("filter")
        
        if not query_to_retrieve or not isinstance(query_to_retrieve, str):
            raise ExecutorError("RetrievalExecutor: 'query' (string) is required.")
        
        resolved_query = self._resolve_references(query_to_retrieve, context)
        
        actual_filter = None
        if filter_param and isinstance(filter_param, dict) and filter_param:
            actual_filter = filter_param
        elif filter_param:
            print(f"  [RetrievalExec Warn] Invalid filter for task '{task.id}': {filter_param}. Ignoring.")
        
        task.thought = (task.thought or "") + f"KB检索查询: '{resolved_query}', 过滤器: {actual_filter}.".strip()
        
        retrieved_docs_with_meta = self.kb.retrieve(resolved_query, top_k=3, filter_dict=actual_filter)
        print('='*20)
        print(retrieved_docs_with_meta)
        print('='*20)
        if not retrieved_docs_with_meta:
            task.thought += "\n未检索到任何匹配文档."
            return []
        
        task.thought += f"\n检索到 {len(retrieved_docs_with_meta)} 个文档对象."
        return retrieved_docs_with_meta

    def get_schema(self) -> Dict[str, Any]:
        """获取检索执行器的模式信息"""
        return {
            "name": "RetrievalExecutor",
            "description": "从向量知识库中检索与查询相关的文本片段。可指定元数据过滤器。",
            "logic_input_schema": {
                "query": "string (检索查询语句, 可引用 {{task_id.result}})",
                "filter": "dict (可选, ChromaDB元数据过滤器)"
            }
        }
