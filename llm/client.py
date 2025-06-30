"""
LLM客户端模块
封装与大语言模型的交互逻辑
"""
import asyncio
import json
import os
from typing import List, Dict, Optional


class OpenAIChatLLM:
    """OpenAI兼容的聊天LLM客户端"""
    
    def __init__(self, model_name: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library not found. `pip install openai`.")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "deepseek-r1-distill-qwen-32b")
        
        if not self.api_key:
            raise ValueError("API key not found.")
        
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(f"[OpenAIChatLLM] 客户端就绪: 模型 {self.model_name}, URL {self.base_url or 'OpenAI default'}")

    async def _make_api_call(self, messages: List[Dict[str, str]], 
                           expect_json: bool = False, 
                           temperature: float = 0.01,
                           **kwargs) -> str:
        """执行API调用"""
        try:
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }
            
            if expect_json:
                if "dashscope" in (self.base_url or "").lower() and self.model_name.startswith("qwen"):
                    completion_params["extra_body"] = {"result_format": "message"}
                else:
                    completion_params["response_format"] = {"type": "json_object"}
            
            response = await asyncio.to_thread(self.client.chat.completions.create, **completion_params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"  [OpenAIChatLLM Error] API调用失败: {e}")
            raise RuntimeError(f"LLM API call failed: {e}")

    async def generate(self, prompt_str: str, 
                      system_prompt_str: Optional[str] = None,
                      temperature: float = 0.1, **kwargs) -> str:
        """生成文本响应"""
        messages = []
        if system_prompt_str:
            messages.append({"role": "system", "content": system_prompt_str})
        messages.append({"role": "user", "content": prompt_str})
        
        return await self._make_api_call(messages, expect_json=False, temperature=temperature, **kwargs)

    async def generate_structured_json(self, prompt_str: str,
                                     system_prompt_str: Optional[str] = None,
                                     temperature: float = 0.01, **kwargs) -> Dict:
        """生成结构化JSON响应"""
        messages = []
        if system_prompt_str:
            messages.append({"role": "system", "content": system_prompt_str})
        
        user_content = f"{prompt_str}\n\n请确保您的回复是一个合法的、单独的JSON对象，不包含任何其他解释性文本或markdown标记。"
        messages.append({"role": "user", "content": user_content})
        
        response_str = await self._make_api_call(messages, expect_json=True, temperature=temperature, **kwargs)
        
        # 清理响应
        cleaned_response_str = response_str.strip()
        if cleaned_response_str.startswith("```json"):
            cleaned_response_str = cleaned_response_str[7:]
        if cleaned_response_str.endswith("```"):
            cleaned_response_str = cleaned_response_str[:-3]
        cleaned_response_str = cleaned_response_str.strip()
        
        try:
            return json.loads(cleaned_response_str)
        except json.JSONDecodeError as e:
            error_msg = f"LLM did not return valid JSON. Error: {e}. Raw: '{response_str}'"
            print(f"  [OpenAIChatLLM Error] {error_msg}")
            raise ValueError(error_msg)
