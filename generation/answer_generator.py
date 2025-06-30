"""
答案生成器模块
负责基于执行历史生成最终答案
"""
from model.client import OpenAIChatLLM
from core.context_manager import ContextManager
from prompts.templates import UserProvidedReferGeneratorPrompt


class AnswerGenerator:
    """答案生成器：基于任务执行历史生成最终答案"""
    
    def __init__(self, llm_client: OpenAIChatLLM, prompt: UserProvidedReferGeneratorPrompt):
        self.llm_client = llm_client
        self.prompt_template = prompt

    async def generate_final_answer(self, user_query: str, context: ContextManager) -> str:
        """生成最终答案"""
        # 获取任务执行总结
        summary = context.get_summary_for_generator()
        
        # 收集检索到的引用
        refs = context.collect_retrieved_references_for_generator()
        
        # 构建提示词
        prompt_str = self.prompt_template.format(
            summary_of_executed_steps=summary,
            user_query=user_query,
            retrieved_references=refs
        )
        
        # 生成答案
        return await self.llm_client.generate(prompt_str, temperature=0.2)  # 稍高温度增加创造性
