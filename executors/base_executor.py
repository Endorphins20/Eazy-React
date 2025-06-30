"""
执行器基类模块
定义所有执行器的基础接口
"""
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.data_structures import Task, ExecutorError
from core.context_manager import ContextManager
from model.client import OpenAIChatLLM


class ExecutorBase(ABC):
    """执行器基类"""
    
    def __init__(self, llm_client: Optional[OpenAIChatLLM] = None):
        self.llm_client = llm_client

    @abstractmethod
    async def execute(self, task: Task, context: ContextManager) -> Any:
        """执行任务的抽象方法"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """获取执行器模式信息的抽象方法"""
        pass

    def _resolve_references(self, data_template: Any, context: ContextManager) -> Any:
        """解析模板中的引用"""
        if isinstance(data_template, str):
            def replace_match(match):
                ref_full = match.group(1).strip()
                task_id_ref, attr_ref = ref_full.split('.', 1) if '.' in ref_full else (ref_full, "result")
                ref_task = context.get_task(task_id_ref)
                
                if ref_task and ref_task.status == "completed":
                    target_obj = ref_task.result
                    if attr_ref == "result":
                        if isinstance(target_obj, dict) and "answer_summary" in target_obj:
                            return str(target_obj["answer_summary"])
                        if isinstance(target_obj, list):
                            return "\n".join([f"- {item}" for item in map(str, target_obj)])
                        return str(target_obj)
                    elif isinstance(target_obj, dict) and attr_ref in target_obj:
                        return str(target_obj[attr_ref])
                    else:
                        print(f"  [Exec Warn] Unsupported attr '{attr_ref}' or not found for {{ {ref_full} }}.")
                else:
                    print(f"  [Exec Warn] Cannot resolve ref {{ {ref_full} }}. "
                         f"Task '{task_id_ref}' status: {ref_task.status if ref_task else 'not_found'}.")
                
                return f"{{引用错误: {ref_full}}}"

            return re.sub(r"\{\{([\w_\-\d\.]+)\}\}", replace_match, data_template)
        elif isinstance(data_template, dict):
            return {k: self._resolve_references(v, context) for k, v in data_template.items()}
        elif isinstance(data_template, list):
            return [self._resolve_references(item, context) for item in data_template]
        
        return data_template
