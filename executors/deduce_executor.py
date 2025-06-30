"""
推理执行器模块
负责基于上下文进行推理和判断
"""
import json
from typing import Dict, Any, Optional

from executors.base_executor import ExecutorBase
from core.data_structures import Task, ExecutorError, DeduceExecutorOutput
from core.context_manager import ContextManager
from model.client import OpenAIChatLLM
from prompts.templates import BasePrompt, DeducePrompt


class DeduceExecutor(ExecutorBase):
    """推理执行器：基于上下文进行推理、总结、判断或抽取"""
    
    def __init__(self, llm_client: OpenAIChatLLM, 
                 prompt_template: DeducePrompt,
                 specialized_prompts: Optional[Dict[str, BasePrompt]] = None):
        super().__init__(llm_client)
        self.default_prompt_template = prompt_template
        self.specialized_prompts = specialized_prompts or {}

    async def execute(self, task: Task, context: ContextManager) -> DeduceExecutorOutput:
        """执行推理任务"""
        logic_input = task.logic_input
        goal = logic_input.get("reasoning_goal")
        raw_ctx_data = logic_input.get("context_data")
        op_type = logic_input.get("operation_type")
        
        if not goal or not isinstance(goal, str) or raw_ctx_data is None:
            raise ExecutorError("DeduceExecutor: 'reasoning_goal'(str) & 'context_data' required.")
        
        res_ctx_data = self._resolve_references(raw_ctx_data, context)
        ctx_data_str = (json.dumps(res_ctx_data, ensure_ascii=False, indent=2) 
                       if isinstance(res_ctx_data, (list, dict)) 
                       else str(res_ctx_data))
        
        # 选择合适的提示词模板
        prompt_to_use = (self.specialized_prompts.get(op_type, self.default_prompt_template) 
                        if op_type else self.default_prompt_template)
        sys_prompt = getattr(prompt_to_use, "SYSTEM_PROMPT", DeducePrompt.SYSTEM_PROMPT)
        
        if op_type and prompt_to_use != self.default_prompt_template:
            print(f"  [DeduceExecutor] Using specialized prompt for op_type: {op_type}")
        
        prompt_str = prompt_to_use.format(reasoning_goal=goal, context_data=ctx_data_str)
        task.thought = (task.thought or "") + f"演绎目标({op_type or 'default'}): {goal}. 上下文(摘要): {ctx_data_str[:100]}...".strip()
        
        # 调用LLM进行推理
        resp_json = await self.llm_client.generate_structured_json(
            prompt_str, system_prompt_str=sys_prompt, temperature=0.0
        )
        
        # 解析响应
        summary = resp_json.get("answer_summary", "未能从LLM响应中解析出答案总结。")
        is_suff = bool(resp_json.get("is_sufficient", False))
        new_qs_raw = resp_json.get("new_questions_or_entities", [])
        
        new_qs = []
        if isinstance(new_qs_raw, list):
            new_qs = [str(item).strip() for item in new_qs_raw 
                     if isinstance(item, str) and str(item).strip()]
        elif isinstance(new_qs_raw, str) and str(new_qs_raw).strip():
            new_qs = [str(new_qs_raw).strip()]
        
        task.thought += f"\nLLM演绎响应(结构化): sufficient={is_suff}, new_qs={new_qs}, summary_preview='{summary[:50]}...'"
        
        output: DeduceExecutorOutput = {
            "answer_summary": summary,
            "is_sufficient": is_suff,
            "new_questions_or_entities": new_qs,
            "raw_llm_response": json.dumps(resp_json, ensure_ascii=False)
        }
        
        return output

    def get_schema(self) -> Dict[str, Any]:
        """获取推理执行器的模式信息"""
        return {
            "name": "DeduceExecutor",
            "description": "基于上下文进行推理、总结、判断或抽取。会判断信息是否充分并给出下一步查询建议。",
            "logic_input_schema": {
                "reasoning_goal": "string (具体推理目标)",
                "context_data": "any (推理所需上下文,可引用 {{task_id.result}})",
                "operation_type": "string (可选, 如 summarize, extract_info)"
            }
        }
