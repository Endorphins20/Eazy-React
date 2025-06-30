"""
完成执行器模块
用于标记任务完成
"""
from typing import Dict, Any

from executors.base_executor import ExecutorBase
from core.data_structures import Task
from core.context_manager import ContextManager


class FinishExecutor(ExecutorBase):
    """完成执行器：标记流程结束"""
    
    async def execute(self, task: Task, context: ContextManager) -> str:
        """执行完成任务"""
        task.thought = "收到Finish指令，流程结束。"
        print(f"  [FinishExecutor] Task {task.id} executed.")
        return "已完成所有必要步骤。"

    def get_schema(self) -> Dict[str, Any]:
        """获取完成执行器的模式信息"""
        return {
            "name": "FinishAction",
            "description": "当问题已解决或无法继续时调用此动作结束规划。",
            "logic_input_schema": {
                "reason": "string (可选，结束原因)"
            }
        }
