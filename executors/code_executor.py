"""
代码执行器模块
负责生成并执行Python代码
"""
import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, Any

from executors.base_executor import ExecutorBase
from core.data_structures import Task, ExecutorError
from core.context_manager import ContextManager
from llm.client import OpenAIChatLLM
from prompts.templates import CodeExecutionPrompt


class CodeExecutor(ExecutorBase):
    """代码执行器：生成并执行Python代码"""
    
    def __init__(self, llm_client: OpenAIChatLLM, prompt: CodeExecutionPrompt):
        super().__init__(llm_client)
        self.prompt_template = prompt

    async def execute(self, task: Task, context: ContextManager) -> str:
        """执行代码生成和运行任务"""
        logic_input = task.logic_input
        code_gen_prompt = logic_input.get("code_generation_prompt")
        rel_data = logic_input.get("relevant_data", "")
        
        if not code_gen_prompt or not isinstance(code_gen_prompt, str):
            raise ExecutorError("CodeExecutor: 'code_generation_prompt' required.")
        
        res_code_prompt = self._resolve_references(code_gen_prompt, context)
        res_rel_data = self._resolve_references(rel_data, context)
        
        rel_data_prompt = (json.dumps(res_rel_data, ensure_ascii=False, indent=2) 
                          if isinstance(res_rel_data, (list, dict)) 
                          else str(res_rel_data) if res_rel_data else "")
        
        llm_prompt = self.prompt_template.format(
            code_generation_prompt=res_code_prompt,
            relevant_data=rel_data_prompt
        )
        
        task.thought = (task.thought or "") + f"代码生成目标: {res_code_prompt}. ".strip()
        
        # 生成代码
        code_block = await self.llm_client.generate(
            llm_prompt, 
            system_prompt_str=CodeExecutionPrompt.SYSTEM_PROMPT
        )
        
        # 清理代码块
        code = code_block.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        if not code:
            task.thought += "\nLLM未能生成代码."
            raise ExecutorError("CodeExecutor: LLM no code.")
        
        task.thought += f"\n生成的代码:\n---\n{code}\n---"
        
        # 执行代码
        try:
            with open("t.py", "w", encoding="utf-8") as f:
                f.write(code)
            
            p = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "t.py"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            
            if p.returncode != 0:
                task.thought += f"\n代码错误码{p.returncode}. stderr:\n{p.stderr or '无'}"
                raise ExecutorError(f"Code exec err {p.returncode}:\n{p.stderr or '无'}")
            
            out = p.stdout.strip()
            task.thought += f"\n代码输出: {out}"
            return out
            
        except subprocess.TimeoutExpired:
            task.thought += "\n代码超时."
            raise ExecutorError("Code timeout.")
        except Exception as e:
            task.thought += f"\n代码本地执行错误: {e}"
            raise ExecutorError(f"Code local exec err: {e}")
        finally:
            if os.path.exists("t.py"):
                os.remove("t.py")

    def get_schema(self) -> Dict[str, Any]:
        """获取代码执行器的模式信息"""
        return {
            "name": "CodeExecutor",
            "description": "生成并执行Python代码。代码应print()结果。输入可引用 {{task_id.result}}。",
            "logic_input_schema": {
                "code_generation_prompt": "string (代码生成指令)",
                "relevant_data": "any (可选, 代码所需数据)"
            }
        }
