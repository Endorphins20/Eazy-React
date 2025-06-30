"""
上下文管理器模块
负责管理任务上下文和执行历史
"""
from typing import Dict, List, Optional, Any
from core.data_structures import Task


class ContextManager:
    """上下文管理器：管理任务执行状态和历史"""
    
    def __init__(self, user_query: str):
        self.user_query = user_query
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []

    def add_task_from_planner(self, task_data: Dict, base_id_prefix: str, iter_num: int, step_in_iter: int) -> Task:
        """从规划器数据创建并添加任务"""
        llm_id = task_data.get("id")
        task_id = (llm_id if llm_id and isinstance(llm_id, str) and llm_id.strip() 
                  else f"{base_id_prefix}_iter{iter_num}_step{step_in_iter}")
        
        orig_id = task_id
        ctr = 0
        while task_id in self.tasks:
            ctr += 1
            task_id = f"{orig_id}_v{ctr}"
        
        if ctr > 0:
            print(f"  [CM Warn] Task ID '{orig_id}' conflict. Renamed '{task_id}'.")
        
        task = Task(
            id=task_id,
            executor_name=task_data["executor_name"],
            task_description=task_data["task_description"],
            logic_input=task_data["logic_input"],
            dependencies=task_data.get("dependencies", [])
        )
        
        self.tasks[task.id] = task
        self.execution_order.append(task.id)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取指定任务"""
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: str, 
                          result: Optional[Any] = None, thought: Optional[str] = None):
        """更新任务状态"""
        task = self.get_task(task_id)
        if task:
            task.status = status
            task.result = result if result is not None else task.result
            task.thought = ((task.thought or "") + f"\n{thought}".strip() 
                           if thought and (task.thought or "") else thought or task.thought)
        else:
            print(f"  [CM Warn] Task ID {task_id} not found for status update.")

    def get_task_history_for_prompt(self, truncate_len: int = 150) -> str:
        """获取用于提示词的任务历史"""
        hist = []
        
        for tid in self.execution_order:
            t = self.get_task(tid)
            if t and t.status in ["completed", "failed"]:
                res_disp_parts = []
                
                if t.status == "completed":
                    if isinstance(t.result, dict) and t.executor_name == "DeduceExecutor":
                        d_out = t.result
                        res_disp_parts.append(
                            f"    推理总结: {str(d_out.get('answer_summary', 'N/A'))[:truncate_len]}"
                            f"{'...' if len(str(d_out.get('answer_summary', 'N/A'))) > truncate_len else ''}"
                        )
                        res_disp_parts.append(f"    信息是否充分: {d_out.get('is_sufficient', True)}")
                        new_q = d_out.get('new_questions_or_entities', [])
                        if new_q:
                            res_disp_parts.append(
                                f"    建议进一步查询: {', '.join(new_q)[:truncate_len]}"
                                f"{'...' if len(', '.join(new_q)) > truncate_len else ''}"
                            )
                    elif isinstance(t.result, list) and t.executor_name == "RetrievalExecutor":
                        retrieved_ids = [
                            str(item.get("metadata", {}).get("id", f"unnamed_chunk_{i}"))
                            for i, item in enumerate(t.result)
                        ]
                        res_disp_parts.append(
                            f"    检索到 {len(t.result)} 片段. IDs: {', '.join(retrieved_ids)[:truncate_len - 20]}..."
                        )
                    else:
                        res_s = str(t.result)
                        res_disp_parts.append(
                            f"    结果: {res_s[:truncate_len]}{'...' if len(res_s) > truncate_len else ''}"
                        )
                else:
                    res_disp_parts.append(f"    执行失败: {str(t.result)[:truncate_len]}...")
                
                res_final_disp = "\n".join(res_disp_parts)
                th_s = str(t.thought or "N/A")
                th_s = th_s[:truncate_len] + "..." if len(th_s) > truncate_len else th_s
                
                hist.append(
                    f"  - Task ID: {t.id}\n    Desc: {t.task_description}\n"
                    f"    Exec: {t.executor_name}\n    Status: {t.status}\n"
                    f"{res_final_disp}\n    Thought: {th_s}"
                )
        
        return "\n\n".join(hist) if hist else "尚未执行任何历史任务。"

    def get_summary_for_generator(self) -> str:
        """获取用于生成器的总结"""
        summary_parts = []
        
        for i, task_id in enumerate(self.execution_order):
            task = self.get_task(task_id)
            if task:
                result_str = "N/A"
                thought_str = str(task.thought or '未记录思考过程')
                
                if task.status == 'completed':
                    if isinstance(task.result, dict) and task.executor_name == "DeduceExecutor":
                        result_str = (
                            f"推理总结: {task.result.get('answer_summary', 'N/A')}, "
                            f"信息是否充分: {task.result.get('is_sufficient')}"
                        )
                        if task.result.get('new_questions_or_entities'):
                            result_str += f", 建议探究: {task.result['new_questions_or_entities']}"
                    elif isinstance(task.result, list) and task.executor_name == "RetrievalExecutor":
                        result_str = f"检索到 {len(task.result)} 个相关片段。"
                    elif isinstance(task.result, str):
                        result_str = task.result
                    else:
                        result_str = f"复杂类型结果 (摘要: {str(task.result)[:100]}...)"
                elif task.status == 'failed':
                    result_str = f"执行失败: {str(task.result)[:150]}..."
                elif task.status == 'skipped':
                    result_str = "因依赖失败或条件不满足而跳过。"
                else:
                    result_str = f"当前状态: {task.status}"
                
                result_str_summary = result_str[:250] + "..." if len(result_str) > 250 else result_str
                thought_str_summary = thought_str[:200] + "..." if len(thought_str) > 200 else thought_str
                
                summary_parts.append(
                    f"步骤 {i + 1} (ID: {task.id}):\n"
                    f"  目标: {task.task_description}\n"
                    f"  执行工具: {task.executor_name}\n"
                    f"  执行思考: {thought_str_summary}\n"
                    f"  产出/状态: {result_str_summary}"
                )
        
        if not summary_parts:
            return "未能执行任何步骤，或没有可总结的产出。"
        
        return "\n\n".join(summary_parts)

    def collect_retrieved_references_for_generator(self) -> List[Dict]:
        """收集用于生成器的检索引用"""
        refs = []
        
        for tid in self.execution_order:
            t = self.get_task(tid)
            if (t and t.executor_name == "RetrievalExecutor" and 
                t.status == "completed" and isinstance(t.result, list)):
                for item in t.result:
                    if isinstance(item, dict) and "content" in item:
                        refs.append({
                            "content": item["content"],
                            "document_name": item.get("metadata", {}).get("source_name", f"源_{t.id}"),
                            "id": item.get("metadata", {}).get("id", f"ref_{t.id}_{len(refs)}")
                        })
        
        return refs
