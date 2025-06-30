"""
数据结构和类型定义模块
定义系统中使用的主要数据结构
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict


# 类型别名
LogicInput = Dict[str, Any]


@dataclass
class Task:
    """任务数据结构"""
    id: str
    executor_name: str
    task_description: str
    logic_input: LogicInput
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None
    thought: Optional[str] = None


class DeduceExecutorOutput(TypedDict, total=False):
    """推理执行器输出结构"""
    answer_summary: str
    is_sufficient: bool
    new_questions_or_entities: List[str]
    raw_llm_response: str


class ExecutorError(Exception):
    """执行器错误异常"""
    pass
