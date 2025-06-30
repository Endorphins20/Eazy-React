"""
规划器模块
负责任务规划和决策
"""
import json
from typing import List, Dict, Tuple

from llm.client import OpenAIChatLLM
from core.context_manager import ContextManager
from prompts.templates import PlannerPrompt


class Planner:
    """任务规划器：基于当前状态规划下一步行动"""
    
    def __init__(self, llm_client: OpenAIChatLLM, prompt: PlannerPrompt):
        self.llm_client = llm_client
        self.prompt_template = prompt

    async def plan_next_steps(self, user_query: str, 
                             context: ContextManager, 
                             available_executors: List[Dict]) -> Tuple[str, str, List[Dict]]:
        """规划下一步任务"""
        # 构建执行器描述
        exec_desc_parts = [
            f"  - 名称: \"{s['name']}\"\n"
            f"    描述: \"{s['description']}\"\n"
            f"    输入参数模式 (logic_input_schema): {json.dumps(s.get('logic_input_schema', 'N/A'), ensure_ascii=False)}"
            for s in available_executors
        ]
        exec_desc = "\n".join(exec_desc_parts)
        
        # 获取历史信息
        history_str = context.get_task_history_for_prompt()
        
        # 构建用户提示词
        user_prompt_str = self.prompt_template.format(
            user_query=user_query,
            available_executors_description=exec_desc,
            task_history=history_str
        )
        
        # 调用LLM进行规划
        response_json = await self.llm_client.generate_structured_json(
            user_prompt_str,
            system_prompt_str=PlannerPrompt.SYSTEM_PROMPT,
            temperature=0.01  # 极低温度确保一致性
        )
        
        # 解析响应
        plan_status = response_json.get("plan_status", "error")
        final_thought = response_json.get("final_thought", "LLM未能提供规划思考。")
        next_steps_data = response_json.get("next_steps", [])
        
        if not isinstance(next_steps_data, list):
            print(f"  [Planner Err] LLM 'next_steps' not list. Got: {next_steps_data}. Assuming none.")
            next_steps_data = []
        
        # 验证步骤数据
        valid_steps = [
            td for td in next_steps_data 
            if isinstance(td, dict) and all(
                k in td for k in ["id", "executor_name", "task_description", "logic_input"]
            )
        ]
        
        if len(valid_steps) != len(next_steps_data):
            print(f"  [Planner Warn] Some planned steps were invalid and skipped by planner output validation.")
        
        return plan_status, final_thought, valid_steps
