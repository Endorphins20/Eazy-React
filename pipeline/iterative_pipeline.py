"""
迭代流水线模块
协调整个任务执行流程
"""
import re
from typing import Dict, List

from planning.planner import Planner
from generation.answer_generator import AnswerGenerator
from executors.base_executor import ExecutorBase
from core.context_manager import ContextManager
from core.data_structures import ExecutorError


class IterativePipeline:
    """迭代式任务执行流水线"""
    
    def __init__(self, planner: Planner, 
                 executors: Dict[str, ExecutorBase], 
                 generator: AnswerGenerator,
                 max_iterations: int = 5):
        self.planner = planner
        self.executors = executors
        self.generator = generator
        self.max_iterations = max_iterations

    async def run(self, user_query: str) -> str:
        """运行完整的问答流程"""
        print(f"\n🚀 IterativePipeline for query: \"{user_query}\"")
        
        ctx = ContextManager(user_query)
        schemas = [ex.get_schema() for ex in self.executors.values()]
        final_ans = "处理中遇到问题，未能得出最终答案。"
        planner_overall_thought = ""
        current_plan_status = "requires_more_steps"
        final_iter_num_for_log = 0
        
        # 迭代执行
        for i_iter in range(self.max_iterations):
            final_iter_num_for_log = i_iter + 1
            print(f"\n--- Iteration {final_iter_num_for_log} / {self.max_iterations} ---")
            
            # 规划阶段
            print(f"📝 Planning phase (Iteration {final_iter_num_for_log})...")
            try:
                status, thought, steps_data = await self.planner.plan_next_steps(user_query, ctx, schemas)
                planner_overall_thought += f"\nIter {final_iter_num_for_log} Planner Thought: {thought}"
                current_plan_status = status
                
                print(f"  [Planner Out] Status: {current_plan_status}, Thought: {thought}")
                if steps_data:
                    print(f"  [Planner Out] Next Steps Planned ({len(steps_data)}): "
                         f"{[s.get('task_description', 'N/A') for s in steps_data]}")
                else:
                    print("  [Planner Out] No new steps planned.")
                
                # 检查规划状态
                if current_plan_status == "finished":
                    print("  [Pipe] Planner: finished.")
                    break
                if current_plan_status == "cannot_proceed":
                    print("  [Pipe] Planner: cannot_proceed.")
                    final_ans = f"无法继续：{thought}"
                    break
                if not steps_data:
                    if i_iter > 0:
                        print("  [Pipe] No new steps & not finished explicitly. Assuming completion.")
                        break
                    else:
                        print("  [Pipe Err] Planner: no initial steps.")
                        return f"无法制定初步计划。Planner Thought: {thought}"
                        
            except Exception as e:
                print(f"  [Pipe Err] Planning error: {e}")
                import traceback
                traceback.print_exc()
                final_ans = f"规划阶段意外错误：{e}"
                break
            
            # 执行阶段
            print(f"\n⚙️ Execution phase (Iteration {final_iter_num_for_log})...")
            await self._execute_task_dag_segment(steps_data, ctx, final_iter_num_for_log)
            
            # 检查是否有完成动作被执行
            finish_executed = any(
                t.executor_name == "FinishAction" and t.status == "completed"
                for t in ctx.tasks.values()
                if t.id in [s.get("id", "") for s in steps_data]
            )
            if finish_executed:
                print(f"  [Pipe] FinishAction completed. Ending iterations.")
                current_plan_status = "finished"
                break
        
        # 生成阶段
        print(f"\n💬 Generation phase after {final_iter_num_for_log} iteration(s)...")
        try:
            if current_plan_status == "cannot_proceed" and "无法继续" in final_ans:
                pass  # 保持错误信息
            else:
                final_ans = await self.generator.generate_final_answer(user_query, ctx)
        except Exception as e:
            print(f"  [Pipe Err] Generation error: {e}")
            import traceback
            traceback.print_exc()
            final_ans = f"生成答案意外错误：{e}"
        
        print(f"\n💡 Final Answer: {final_ans}")
        return final_ans

    async def _execute_task_dag_segment(self, tasks_data: List[Dict], 
                                       ctx: ContextManager, iter_num: int) -> bool:
        """执行任务DAG片段"""
        if not tasks_data:
            return True
        
        prefix = re.sub(r'[^\w\s-]', '', ctx.user_query[:10]).strip().replace(' ', '_') or "q"
        added_ids = [
            ctx.add_task_from_planner(td, prefix, iter_num, i).id 
            for i, td in enumerate(tasks_data)
        ]
        
        cache = set()
        success = True
        
        for tid in added_ids:
            task = ctx.get_task(tid)
            if task and task.status == "pending":
                await self._execute_task_with_dependencies(tid, ctx, cache)
                if ctx.get_task(tid).status != "completed":
                    success = False
        
        return success

    async def _execute_task_with_dependencies(self, task_id: str, 
                                            ctx: ContextManager, cache: set):
        """递归执行任务及其依赖"""
        task = ctx.get_task(task_id)
        if not task:
            print(f"  [Pipe Err] Task {task_id} def not found for exec.")
            return
        
        if task.status != "pending":
            return
        
        cache.add(task_id)
        
        # 先执行依赖任务
        for dep_id in task.dependencies:
            dep_task = ctx.get_task(dep_id)
            if not dep_task:
                emsg = f"T {task.id} undef dep ID: {dep_id}. Failed."
                ctx.update_task_status(task.id, "failed", result=emsg, thought=emsg)
                print(f"  [Pipe Err] {emsg}")
                return
            
            if dep_task.status == "pending":
                await self._execute_task_with_dependencies(dep_id, ctx, cache)
        
        # 检查依赖是否满足
        deps_met = True
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_task = ctx.get_task(dep_id)
                if not dep_task or dep_task.status != "completed":
                    deps_met = False
                    ts = ((task.thought or "") + 
                         f"Dep {dep_id} not met (status: {dep_task.status if dep_task else 'N/A'}). Skip.")
                    ctx.update_task_status(task.id, "skipped", thought=ts)
                    print(f"  [Pipe] Task {task.id} skip, dep {dep_id} not met.")
                    break
            
            if not deps_met:
                return
        
        if task.status != "pending":
            return
        
        # 执行当前任务
        executor = self.executors.get(task.executor_name)
        if not executor:
            emsg = f"Exec '{task.executor_name}' not found for task '{task.task_description}'."
            ctx.update_task_status(task.id, "failed", result=emsg, thought=emsg)
            print(f"  [Pipe Err] {emsg}")
            return
        
        ctx.update_task_status(task.id, "running", thought=f"Start: {task.task_description}")
        print(f"\n▶️ Iter Exec Task: {task.id} - \"{task.task_description}\" ({task.executor_name})")
        
        try:
            result = await executor.execute(task, ctx)
            ctx.update_task_status(task.id, "completed", result=result, thought=task.thought)
        except ExecutorError as e:
            emsg = f"ExecErr T {task.id}: {e}"
            ft = (task.thought or "") + f"\nExecErr: {e}"
            ctx.update_task_status(task.id, "failed", result=emsg, thought=ft)
            print(f"🛑 {emsg}")
        except Exception as e:
            emsg = f"UnexpectedErr T {task.id}: {e}"
            import traceback
            tb = traceback.format_exc()
            print(f"🛑 {emsg}\n{tb}")
            ft = (task.thought or "") + f"\nUnexpectedErr: {e}"
            ctx.update_task_status(task.id, "failed", result=emsg, thought=ft)
