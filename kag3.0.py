import asyncio
import json
import os
import re
import subprocess  # 用于安全执行代码
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import shutil  # 用于删除目录

# --- OpenAI/LLM Client Configuration ---
# (与上一轮相同，用户需设置环境变量)

# --- Langchain and Chroma Imports (来自您的代码) ---
from langchain_core.documents import Document
from langchain_chroma import Chroma
# Embedding function - 使用您提供的 DashScope (QwenEmbeddingFunction)
# from langchain_community.embeddings import DashScopeEmbeddings # 直接用下面的包装类
from tongyiembedding import QwenEmbeddingFunction  # 来自您的代码

# --- ChromaKnowledgeBase Class (直接从您的代码中复制过来) ---
CHROMA_PERSIST_DIRECTORY = "chroma_db_kag_concrete"  # 改个名字以防冲突
CHROMA_COLLECTION_NAME = "kag_concrete_documents"


class ChromaKnowledgeBase:
	def __init__(self,
				 embedding_function: Callable,
				 initial_documents: Optional[List[Document]] = None,
				 persist_directory: str = CHROMA_PERSIST_DIRECTORY,
				 collection_name: str = CHROMA_COLLECTION_NAME,
				 force_rebuild: bool = False):

		print(f"  [ChromaKB] 初始化知识库，持久化目录: {persist_directory}, 集合: {collection_name}")
		self.embedding_function = embedding_function
		self.persist_directory = persist_directory
		self.collection_name = collection_name
		self.vectorstore: Optional[Chroma] = None

		if force_rebuild and os.path.exists(persist_directory):
			print(f"  [ChromaKB] force_rebuild=True, 正在删除已存在的持久化目录: {persist_directory}")
			try:
				shutil.rmtree(persist_directory)
			except OSError as e:
				print(f"  [ChromaKB Error] 删除目录失败: {e}. 可能需要手动删除。")

		if os.path.exists(persist_directory) and not force_rebuild:
			print(f"  [ChromaKB] 正在从 '{persist_directory}' 加载已存在的 Chroma 向量库...")
			try:
				self.vectorstore = Chroma(
					persist_directory=self.persist_directory,
					embedding_function=self.embedding_function,
					collection_name=self.collection_name
				)
				print(f"  [ChromaKB] 成功加载向量库。集合 '{self.collection_name}'.")
			except Exception as e:
				print(f"  [ChromaKB Error] 从 '{persist_directory}' 加载失败: {e}")
				print(f"  [ChromaKB] 将尝试基于提供的 initial_documents (如果存在) 创建新的向量库。")
				self.vectorstore = None

		if self.vectorstore is None:
			if initial_documents:
				print(f"  [ChromaKB] 正在为 {len(initial_documents)} 个文档构建新的 Chroma 向量库...")
				self.vectorstore = Chroma.from_documents(
					documents=initial_documents,
					embedding=self.embedding_function,
					persist_directory=self.persist_directory,
					collection_name=self.collection_name
				)
				print(f"  [ChromaKB] 新向量库构建并持久化完成。")
			else:
				print(f"  [ChromaKB] 没有提供初始文档，且未加载现有向量库。知识库将为空。")
				self.vectorstore = Chroma(  # 创建一个空的，但可持久化的
					persist_directory=self.persist_directory,
					embedding_function=self.embedding_function,
					collection_name=self.collection_name
				)
				print(f"  [ChromaKB] 空的持久化 Chroma 集合 '{self.collection_name}' 已准备就绪。")

	def add_documents(self, documents: List[Document]):
		if not self.vectorstore:
			print("  [ChromaKB Error] Vectorstore 未初始化，无法添加文档。")
			# 考虑在这里创建，如果 initial_documents 为空时没有创建
			if documents:
				print(f"  [ChromaKB] Vectorstore 为空, 尝试从当前文档创建...")
				self.vectorstore = Chroma.from_documents(
					documents=documents,
					embedding=self.embedding_function,
					persist_directory=self.persist_directory,
					collection_name=self.collection_name
				)
				print(f"  [ChromaKB] 基于新文档创建并持久化完成。")
				return  # 已经添加了
			else:  # 没有文档可添加，也没有vectorstore
				return

		if documents:
			print(f"  [ChromaKB] 正在向集合 '{self.collection_name}' 添加 {len(documents)} 个新文档...")
			self.vectorstore.add_documents(documents)
			print(f"  [ChromaKB] 文档添加完成。")

	def retrieve(self, query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
		# 返回 List[Dict[str, Any]]，每个字典包含 "content", "metadata", "score"
		if not self.vectorstore:
			print("  [ChromaKB Error] Vectorstore 未初始化，无法检索。")
			return []
		try:
			# 检查集合是否真的存在并且可以计数
			if self.vectorstore._collection is None or self.vectorstore._collection.count() == 0:
				print("  [ChromaKB] 知识库集合为空或未正确加载，无法检索。")
				return []
		except Exception as e:  # 捕获可能的 Chroma client 或 collection 访问错误
			print(f"  [ChromaKB Warning] 无法获取集合计数或集合无效，将尝试检索: {e}")
		# 即使无法计数，也尝试检索，让similarity_search自己报错（如果底层有问题）

		print(f"  [ChromaKB] 正在为查询 '{query}' 检索最相关的 {top_k} 个文档 (过滤器: {filter_dict})...")
		try:
			results_with_scores = self.vectorstore.similarity_search_with_score(
				query, k=top_k, filter=filter_dict
			)
		except Exception as e:
			print(f"  [ChromaKB Error] Chroma similarity search failed: {e}")
			return []

		processed_results = []
		if results_with_scores:
			for doc, score in results_with_scores:
				processed_results.append({
					"id": doc.metadata.get("id", None),
					"content": doc.page_content,
					"metadata": doc.metadata,
					"score": float(score)
				})
		print(f"  [ChromaKB] 检索到 {len(processed_results)} 个文档。")
		return processed_results


# --- LLM客户端 (OpenAIChatLLM from previous response) ---
class OpenAIChatLLM:
	def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
		try:
			import openai
		except ImportError:
			raise ImportError("OpenAI library not found. Please install it with `pip install openai`.")

		self.api_key = 'sk-af4423da370c478abaf68b056f547c6e'
		self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
		self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "qwen-plus")

		if not self.api_key:
			raise ValueError(
				"API key not found. Please set OPENAI_API_KEY environment variable or pass it to the constructor.")

		self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
		print(f"[OpenAIChatLLM] 客户端初始化成功，将使用模型: {self.model_name} at {self.base_url or 'OpenAI default'}")

	async def _make_api_call(self, messages: List[Dict[str, str]], expect_json: bool = False, **kwargs) -> str:
		try:
			completion_params = {
				"model": self.model_name,
				"messages": messages,
				"temperature": 0.1,
				**kwargs
			}
			if expect_json:
				if "dashscope" in (self.base_url or "").lower() and self.model_name.startswith("qwen"):
					completion_params["extra_body"] = {"result_format": "message"}
					print("  [OpenAIChatLLM] DashScope Qwen模型，设置 result_format: message 以期待JSON")
				else:
					completion_params["response_format"] = {"type": "json_object"}
					print("  [OpenAIChatLLM] 设置 response_format: json_object")

			# print(f"  [OpenAIChatLLM] Calling model with params: model={completion_params['model']}, temp={completion_params['temperature']}")
			# print(f"  [OpenAIChatLLM] Messages (brief): Role {messages[-1]['role']}, Content: {messages[-1]['content'][:100]}...")

			response = await asyncio.to_thread(self.client.chat.completions.create, **completion_params)
			content = response.choices[0].message.content
			if not content:
				print("  [OpenAIChatLLM Error] 未能从API响应中获取内容.")
				return ""
			return content.strip()
		except Exception as e:
			print(f"  [OpenAIChatLLM Error] API调用失败: {e}")
			import traceback
			traceback.print_exc()
			raise RuntimeError(f"LLM API call failed: {e}")

	async def generate(self, prompt_str: str, system_prompt_str: Optional[str] = None, **kwargs) -> str:
		messages = []
		if system_prompt_str:
			messages.append({"role": "system", "content": system_prompt_str})
		messages.append({"role": "user", "content": prompt_str})
		return await self._make_api_call(messages, expect_json=False, **kwargs)

	async def generate_structured_json(self, prompt_str: str, system_prompt_str: Optional[str] = None,
									   **kwargs) -> Dict:
		messages = []
		if system_prompt_str:
			messages.append({"role": "system", "content": system_prompt_str})
		user_content = f"{prompt_str}\n\n请确保您的回复是一个合法的、单独的JSON对象，不包含任何其他解释性文本或markdown标记。"
		messages.append({"role": "user", "content": user_content})

		response_str = await self._make_api_call(messages, expect_json=True, **kwargs)

		cleaned_response_str = response_str.strip()
		if cleaned_response_str.startswith("```json"):
			cleaned_response_str = cleaned_response_str[7:]
		if cleaned_response_str.endswith("```"):
			cleaned_response_str = cleaned_response_str[:-3]
		cleaned_response_str = cleaned_response_str.strip()

		try:
			return json.loads(cleaned_response_str)
		except json.JSONDecodeError as e:
			error_msg = f"LLM did not return valid JSON. Error: {e}. Raw response: '{response_str}'"
			print(f"  [OpenAIChatLLM Error] {error_msg}")
			raise ValueError(error_msg)


# --- Prompts (from previous response, ensure PlannerPrompt is updated) ---
class BasePrompt:
	def __init__(self, template: str, variables: List[str]):
		self.template_str = template
		self.variables = variables

	def format(self, **kwargs) -> str:
		formatted_prompt = self.template_str
		for var_name in self.variables:
			if var_name not in kwargs:
				raise ValueError(f"Missing variable: {var_name} for prompt template.")
			formatted_prompt = formatted_prompt.replace(f"${{{var_name}}}", str(kwargs[var_name]))
		return formatted_prompt


class PlannerPrompt(BasePrompt):
	SYSTEM_PROMPT = """
    你是一个高度智能的AI任务规划助手。
    你的目标是根据用户提出的复杂问题，将其分解为一系列逻辑清晰、可执行的子任务。
    你必须严格按照以下JSON格式输出任务计划。每个任务都是一个JSON对象，整个计划是一个JSON列表。

    任务对象结构:
    {
      "id": "string (任务的唯一标识符, 例如 task_0, task_userquery_1)",
      "executor_name": "string (执行该任务的工具名称, 从提供的可用工具列表中选择)",
      "task_description": "string (对这个子任务的简短中文描述)",
      "logic_input": {
        // 这个对象的具体字段取决于所选的 executor_name
        // 请参考下面可用工具描述中每个工具的 'logic_input_schema'
      },
      "dependencies": ["string"] (一个列表，包含当前任务所依赖的其他任务的ID。初始任务此列表为空)
    }

    处理逻辑:
    1.  仔细分析用户问题和已经执行的历史任务（如果有）。
    2.  将用户问题分解为若干个原子步骤，每个步骤都应明确对应一个可用工具。
    3.  为每个步骤选择最合适的工具，并根据工具的`logic_input_schema`准备其输入参数。
    4.  **关键**：如果一个任务的输入（`logic_input`中的字段）需要依赖之前任务的输出，请使用占位符 `{{task_id.result}}` 来表示。例如，如果 `task_1` 的某个输入需要 `task_0` 的结果，则该输入值应为 `{{task_0.result}}`。这里的 `task_id` 必须是先前步骤中定义的 `id`。
    5.  确保最终输出是一个符合上述结构的JSON列表。不要包含任何额外的解释、注释或markdown标记。只输出JSON。
    """
	USER_TEMPLATE = """
    可用工具如下:
    ---BEGIN EXECUTOR DESCRIPTIONS---
    ${available_executors_description}
    ---END EXECUTOR DESCRIPTIONS---

    历史任务及结果 (如果存在):
    ---BEGIN TASK HISTORY---
    ${task_history}
    ---END TASK HISTORY---

    当前用户问题: "${user_query}"

    请根据上述信息，为解决当前用户问题制定任务计划。
    输出任务计划 (一个JSON对象列表):
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["user_query", "available_executors_description", "task_history"])


class DeducePrompt(BasePrompt):
	SYSTEM_PROMPT = "你是一个严谨的AI推理助手。"
	USER_TEMPLATE = """
    请根据以下提供的上下文信息，回答或者完成指定的推理目标。
    请严格依据所提供的上下文作答，不要使用任何外部知识或进行不合理的假设。
    如果信息不足以回答，请明确指出“信息不足”。

    推理目标:
    ${reasoning_goal}

    上下文信息:
    ${context_data}

    你的回答:
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["reasoning_goal", "context_data"])


class CodeExecutionPrompt(BasePrompt):
	SYSTEM_PROMPT = "你是一个Python代码生成和执行助手。"
	USER_TEMPLATE = """
    请根据以下指令和相关数据，生成一段Python代码来解决问题。
    代码必须通过 `print()` 输出其最终计算结果。不要包含任何解释或注释，只输出纯代码。

    指令:
    ${code_generation_prompt}

    相关数据 (如果提供):
    ${relevant_data}

    生成的Python代码 (请确保它只包含代码本身，并用print()输出结果):
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["code_generation_prompt", "relevant_data"])


class GenerationPrompt(BasePrompt):
	SYSTEM_PROMPT = "你是一个优秀的AI沟通助手。"
	USER_TEMPLATE = """
    你的任务是根据以下解决用户问题的完整思考过程和各步骤的最终结果，生成一个流畅、简洁、用户友好的最终答案。

    用户的原始问题是: "${user_query}"

    解决过程和各步骤结果:
    ---BEGIN REASONING STEPS---
    ${reasoning_steps_summary}
    ---END REASONING STEPS---

    请整合以上信息，给出最终答案。如果某些步骤未能成功或信息不足，也请在答案中恰当说明。
    请直接给出答案，不要添加如“最终答案是：”这样的前缀。
    最终答案:
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["user_query", "reasoning_steps_summary"])


# --- DataStructures (from previous response) ---
LogicInput = Dict[str, Any]


@dataclass
class Task:
	id: str
	executor_name: str
	task_description: str
	logic_input: LogicInput
	dependencies: List[str] = field(default_factory=list)
	status: str = "pending"
	result: Optional[Any] = None
	thought: Optional[str] = None


# --- ContextManager (from previous response) ---
class ContextManager:
	def __init__(self, user_query: str):
		self.user_query = user_query
		self.tasks: Dict[str, Task] = {}
		self.execution_order: List[str] = []

	def add_task_from_planner(self, task_data: Dict, base_id_prefix: str, existing_task_count: int) -> Task:
		task_id = task_data.get("id")
		if not task_id or not isinstance(task_id, str):
			task_id = f"{base_id_prefix}_task_{existing_task_count}"
			print(
				f"  [ContextManager] Warning: Planner did not provide a valid string ID for a task. Generated ID: {task_id}")

		original_task_id = task_id
		counter = 0  # Start counter at 0 for the first potential conflict
		while task_id in self.tasks:
			counter += 1
			task_id = f"{original_task_id}_v{counter}"
		if counter > 0:
			print(
				f"  [ContextManager] Warning: Task ID '{original_task_id}' from planner already exists or was generated multiple times. Renamed to '{task_id}'.")

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
		return self.tasks.get(task_id)

	def update_task_status(self, task_id: str, status: str, result: Optional[Any] = None,
						   thought: Optional[str] = None):
		task = self.get_task(task_id)
		if task:
			task.status = status
			if result is not None:
				task.result = result
			current_thought = task.thought or ""
			if thought:  # Append new thought to existing, if any
				task.thought = f"{current_thought}\n{thought}".strip() if current_thought else thought
		else:
			print(f"  [ContextManager] Warning: Task ID {task_id} not found for status update.")

	def get_task_history_for_prompt(self) -> str:
		history = []
		for task_id in self.execution_order:
			task = self.tasks.get(task_id)
			if task and task.status in ["completed", "failed"]:
				result_str = str(task.result)
				if len(result_str) > 200:
					result_str = result_str[:200] + "..."
				history.append(
					f"  - Task ID: {task.id}\n"
					f"    Description: {task.task_description}\n"
					f"    Executor: {task.executor_name}\n"
					f"    Status: {task.status}\n"
					f"    Result: {result_str}\n"
					f"    Thought: {task.thought or 'N/A'}"
				)
		return "\n".join(history) if history else "无已执行的历史任务。"

	def get_summary_for_generator(self) -> str:
		summary = []
		for i, task_id in enumerate(self.execution_order):
			task = self.tasks.get(task_id)
			if task:
				result_str = str(task.result)
				if len(result_str) > 300:
					result_str = result_str[:300] + "..."
				summary.append(
					f"步骤 {i + 1} (Task ID: {task.id}):\n"
					f"  描述: {task.task_description}\n"
					f"  工具: {task.executor_name}\n"
					f"  思考过程: {task.thought or '未记录'}\n"
					f"  状态: {task.status}\n"
					f"  结果: {result_str if task.status == 'completed' else '未完成或失败'}"
				)
		return "\n\n".join(summary)


# --- Executors (from previous response, RetrievalExecutor modified) ---
class ExecutorError(Exception): pass


class ExecutorBase(ABC):
	def __init__(self, llm_client: Optional[OpenAIChatLLM] = None):
		self.llm_client = llm_client

	@abstractmethod
	async def execute(self, task: Task, context: ContextManager) -> Any:
		pass

	@abstractmethod
	def get_schema(self) -> Dict[str, Any]:
		pass

	def _resolve_references(self, data_template: Any, context: ContextManager) -> Any:
		if isinstance(data_template, str):
			def replace_match(match):
				ref_full = match.group(1).strip()
				task_id_ref, attr_ref = ref_full.split('.', 1) if '.' in ref_full else (
				ref_full, "result")  # Default to result

				ref_task = context.get_task(task_id_ref)
				if ref_task and ref_task.status == "completed":
					if attr_ref == "result":
						if isinstance(ref_task.result, list):
							# 将列表结果格式化为多行字符串，供LLM消费
							return "\n".join([f"- {item}" for item in map(str, ref_task.result)])
						return str(ref_task.result)
					else:
						print(f"  [Executor Warning] 不支持引用任务属性 '{attr_ref}' in {{ {ref_full} }}.")
						return match.group(0)
				else:
					status_msg = f"(Task '{task_id_ref}' not found or not completed: status={ref_task.status if ref_task else 'not_found'})"
					print(f"  [Executor Warning] 无法解析引用 {{ {ref_full} }}. {status_msg}")
					return f"{{引用错误: {ref_full} {status_msg}}}"

			return re.sub(r"\{\{([\w_\-\d\.]+)\}\}", replace_match, data_template)
		elif isinstance(data_template, dict):
			return {k: self._resolve_references(v, context) for k, v in data_template.items()}
		elif isinstance(data_template, list):
			return [self._resolve_references(item, context) for item in data_template]
		return data_template


# (请将此修改后的 RetrievalExecutor 替换掉之前代码中的版本)

# --- Executors (部分修改) ---
# class ExecutorError(Exception): pass # 假设已定义
# class ExecutorBase(ABC): ... # 假设已定义, 包含 _resolve_references
# from .knowledge_base import ChromaKnowledgeBase # 确保正确导入

class RetrievalExecutor(ExecutorBase):
	"""使用 ChromaKnowledgeBase 进行检索的执行器"""

	def __init__(self, kb: ChromaKnowledgeBase):  # 改为接收ChromaKnowledgeBase
		super().__init__()  # RetrievalExecutor 通常不需要 LLM
		self.kb = kb

	async def execute(self, task: Task, context: ContextManager) -> List[str]:  # 返回字符串列表
		logic_input = task.logic_input
		query_to_retrieve = logic_input.get("query")
		filter_param_from_planner = logic_input.get("filter")  # 可能为 None, {}, 或有效过滤器

		if not query_to_retrieve or not isinstance(query_to_retrieve, str):
			raise ExecutorError("RetrievalExecutor: 'query' (string) is required in logic_input.")

		resolved_query = self._resolve_references(query_to_retrieve, context)

		# 【错误修正核心代码】
		# 确保传递给 ChromaDB 的过滤器是有效的，或者为 None
		actual_filter_for_chroma: Optional[Dict[str, Any]] = None
		if filter_param_from_planner and isinstance(filter_param_from_planner, dict):
			# 只有当 filter_param_from_planner 是一个非空字典时，才将其用作过滤器
			if filter_param_from_planner:  # 检查字典是否为空
				actual_filter_for_chroma = filter_param_from_planner
			else:  # 如果 filter_param_from_planner 是一个空字典 {}
				print(
					f"  [RetrievalExecutor] 警告: Planner 为任务 '{task.id}' 的 'filter' 参数提供了一个空字典。将视作无过滤器处理。")
				actual_filter_for_chroma = None
		elif filter_param_from_planner is not None:
			# 如果 filter_param 不是字典也不是 None (例如，可能是错误类型的字符串等)，则警告并视为无过滤器
			print(
				f"  [RetrievalExecutor] 警告: Planner 为任务 '{task.id}' 的 'filter' 参数提供了无效类型: {type(filter_param_from_planner)}。将视作无过滤器处理。")
			actual_filter_for_chroma = None
		# 如果 filter_param_from_planner 本身就是 None，则 actual_filter_for_chroma 保持为 None

		current_thought = task.thought or ""
		task.thought = f"{current_thought}知识库检索查询: '{resolved_query}', 应用的过滤器: {actual_filter_for_chroma}.".strip()

		# ChromaKnowledgeBase.retrieve 返回 List[Dict[str, Any]]
		# 每个字典: {"id": ..., "content": ..., "metadata": ..., "score": ...}
		retrieved_docs_with_meta = self.kb.retrieve(
			resolved_query,
			top_k=3,
			filter_dict=actual_filter_for_chroma  # 使用修正后的过滤器
		)

		# 提取 page_content 列表供下游使用
		retrieved_contents = [doc["content"] for doc in retrieved_docs_with_meta if
							  doc and "content" in doc]  # 增加对doc是否为None的检查

		if not retrieved_contents:
			task.thought += "\n未检索到任何匹配文档。"
			return ["抱歉，未在知识库中找到与您查询直接相关的信息。"]  # 保持返回列表形式

		task.thought += f"\n检索到 {len(retrieved_contents)} 个文档片段。"
		return retrieved_contents

	def get_schema(self) -> Dict[str, Any]:  # (与之前相同)
		return {
			"name": "RetrievalExecutor",
			"description": "从配置的向量知识库中检索与查询相关的文本片段。可以指定元数据过滤器。",
			"logic_input_schema": {
				"query": "string (要在知识库中检索的查询语句，可引用先前步骤 {{task_id.result}})",
				"filter": "dict (可选, 用于ChromaDB的元数据过滤器, e.g., {\"year\": 2020})"
			}
		}

class DeduceExecutor(ExecutorBase):
	def __init__(self, llm_client: OpenAIChatLLM, prompt: DeducePrompt):
		super().__init__(llm_client)
		self.prompt_template = prompt

	async def execute(self, task: Task, context: ContextManager) -> str:
		logic_input = task.logic_input
		reasoning_goal = logic_input.get("reasoning_goal")
		raw_context_data = logic_input.get("context_data")

		if not reasoning_goal or not isinstance(reasoning_goal, str) or \
				raw_context_data is None:  # context_data can be various types after resolution
			raise ExecutorError("DeduceExecutor: 'reasoning_goal' (string) and 'context_data' are required.")

		resolved_context_data = self._resolve_references(raw_context_data, context)

		context_data_str = ""
		if isinstance(resolved_context_data, list):
			context_data_str = "\n".join([f"- {item}" for item in resolved_context_data])
		elif isinstance(resolved_context_data, dict):
			context_data_str = json.dumps(resolved_context_data, ensure_ascii=False, indent=2)
		else:
			context_data_str = str(resolved_context_data)

		prompt_str = self.prompt_template.format(reasoning_goal=reasoning_goal, context_data=context_data_str)
		task.thought = f"演绎目标: {reasoning_goal}. 使用的上下文 (摘要): {context_data_str[:200]}..."

		response = await self.llm_client.generate(prompt_str, system_prompt_str=DeducePrompt.SYSTEM_PROMPT)
		return response

	def get_schema(self) -> Dict[str, Any]:
		return {
			"name": "DeduceExecutor",
			"description": "基于提供的上下文信息进行推理、总结、判断或抽取。上下文可引用先前步骤的结果。",
			"logic_input_schema": {
				"reasoning_goal": "string (本次推理的具体目标或子问题)",
				"context_data": "any (进行推理所必需的所有背景知识或数据，可包含 {{task_id.result}})",
				"operation_type": "string (可选，如 summarize, extract_info, judge)"
			}
		}


class CodeExecutor(ExecutorBase):  # (与上一轮回复中的版本相同)
	def __init__(self, llm_client: OpenAIChatLLM, prompt: CodeExecutionPrompt):
		super().__init__(llm_client)
		self.prompt_template = prompt

	async def execute(self, task: Task, context: ContextManager) -> str:
		logic_input = task.logic_input
		code_generation_prompt_str = logic_input.get("code_generation_prompt")
		relevant_data = logic_input.get("relevant_data", "")
		if not code_generation_prompt_str or not isinstance(code_generation_prompt_str, str):
			raise ExecutorError("CodeExecutor: 'code_generation_prompt' (string) is required.")
		resolved_code_gen_prompt = self._resolve_references(code_generation_prompt_str, context)
		resolved_relevant_data = self._resolve_references(relevant_data, context)
		relevant_data_for_prompt = ""
		if isinstance(resolved_relevant_data, (list, dict)):
			relevant_data_for_prompt = json.dumps(resolved_relevant_data, ensure_ascii=False, indent=2)
		elif resolved_relevant_data:
			relevant_data_for_prompt = str(resolved_relevant_data)
		llm_prompt_str = self.prompt_template.format(
			code_generation_prompt=resolved_code_gen_prompt,
			relevant_data=relevant_data_for_prompt
		)
		task.thought = f"代码生成目标: {resolved_code_gen_prompt}. "
		generated_code_with_markers = await self.llm_client.generate(llm_prompt_str,
																	 system_prompt_str=CodeExecutionPrompt.SYSTEM_PROMPT)
		generated_code = generated_code_with_markers.strip()
		if generated_code.startswith("```python"): generated_code = generated_code[9:]
		if generated_code.startswith("```"): generated_code = generated_code[3:]
		if generated_code.endswith("```"): generated_code = generated_code[:-3]
		generated_code = generated_code.strip()
		if not generated_code:
			task.thought += "\nLLM未能生成任何可执行代码。"
			raise ExecutorError("CodeExecutor: LLM did not generate any Python code.")
		task.thought += f"\n生成的代码:\n---\n{generated_code}\n---"
		try:
			with open("temp_code_to_execute.py", "w", encoding="utf-8") as f:
				f.write(generated_code)
			process = await asyncio.to_thread(
				subprocess.run, [sys.executable, "temp_code_to_execute.py"],
				capture_output=True, text=True, timeout=10, check=False
			)
			if process.returncode != 0:
				error_output = process.stderr or "Unknown execution error"
				task.thought += f"\n代码执行返回非零退出码 {process.returncode}. Stderr:\n{error_output}"
				raise ExecutorError(
					f"Generated code execution failed with exit code {process.returncode}:\n{error_output}")
			output_value = process.stdout.strip()
			task.thought += f"\n代码执行标准输出: {output_value}"
			return output_value
		except subprocess.TimeoutExpired:
			task.thought += "\n代码执行超时。"
			raise ExecutorError("Generated code execution timed out.")
		except Exception as e:
			task.thought += f"\n执行生成的代码时发生本地错误: {str(e)}"
			raise ExecutorError(f"Error during local setup/execution of generated code: {e}")
		finally:
			if os.path.exists("temp_code_to_execute.py"): os.remove("temp_code_to_execute.py")

	def get_schema(self) -> Dict[str, Any]:
		return {
			"name": "CodeExecutor",
			"description": "生成并执行Python代码来解决计算或数据处理问题。代码应通过print()输出结果。输入可以引用先前步骤的结果。",
			"logic_input_schema": {
				"code_generation_prompt": "string (生成代码的目标和指令，可包含 {{task_id.result}})",
				"relevant_data": "any (可选, 代码执行可能需要的数据，可以是字符串、列表、字典或引用 {{task_id.result}})"
			}
		}


# --- Planner (from previous response) ---
class Planner:
	def __init__(self, llm_client: OpenAIChatLLM, prompt: PlannerPrompt):
		self.llm_client = llm_client
		self.prompt_template = prompt

	async def create_plan(self, user_query: str, context: ContextManager, available_executors: List[Dict]) -> List[
		Dict]:
		executors_description_parts = []
		for ex_schema in available_executors:
			input_schema_str = ex_schema.get('logic_input_schema', '未定义')
			if isinstance(input_schema_str, dict):
				input_schema_str = json.dumps(input_schema_str, ensure_ascii=False)
			executors_description_parts.append(
				f"  - 名称: \"{ex_schema['name']}\"\n"
				f"    描述: \"{ex_schema['description']}\"\n"
				f"    输入参数模式 (logic_input_schema): {input_schema_str}"
			)
		executors_description = "\n".join(executors_description_parts)
		task_history_str = context.get_task_history_for_prompt()
		user_prompt_str = self.prompt_template.format(
			user_query=user_query,
			available_executors_description=executors_description,
			task_history=task_history_str
		)
		print(f"  [Planner] Sending planning prompt to LLM (length: {len(user_prompt_str)} chars)...")
		planned_task_data_list = await self.llm_client.generate_structured_json(
			user_prompt_str, system_prompt_str=PlannerPrompt.SYSTEM_PROMPT
		)
		if not isinstance(planned_task_data_list, list):
			print(f"  [Planner Error] LLM did not return a list for the plan. Response: {planned_task_data_list}")
			if isinstance(planned_task_data_list, dict):
				for key in ["tasks", "plan", "steps", "actions"]:  # 常见包装键
					if isinstance(planned_task_data_list.get(key), list):
						planned_task_data_list = planned_task_data_list[key]
						print(f"  [Planner] Extracted task list from key '{key}'.")
						break
				else:
					raise ValueError("Planner LLM response, even after checking common keys, is not a list.")
			else:
				raise ValueError("Planner LLM response is not a list or a dict containing a list.")
		validated_tasks_data = []
		for i, task_data in enumerate(planned_task_data_list):
			if not isinstance(task_data, dict):
				print(f"  [Planner Warning] Plan item {i} is not a dictionary: {task_data}. Skipping.")
				continue
			if not all(k in task_data for k in ["executor_name", "task_description", "logic_input"]):
				print(
					f"  [Planner Warning] Plan item {i} missing required fields (executor_name, task_description, logic_input): {task_data}. Skipping.")
				continue
			if "id" not in task_data or not isinstance(task_data["id"], str) or not task_data["id"].strip():
				auto_id = f"{context.user_query[:10].replace(' ', '_').replace('?', '')}_plan_step_{i}"  # 使用 context.tasks 计数不准确，因为context此时可能为空
				print(
					f"  [Planner Warning] Plan item {i} missing or invalid ID. Assigning '{auto_id}'. Original: {task_data.get('id')}")
				task_data["id"] = auto_id
			validated_tasks_data.append(task_data)
		return validated_tasks_data


# --- AnswerGenerator (from previous response) ---
class AnswerGenerator:
	def __init__(self, llm_client: OpenAIChatLLM, prompt: GenerationPrompt):
		self.llm_client = llm_client
		self.prompt_template = prompt

	async def generate_final_answer(self, user_query: str, context: ContextManager) -> str:
		reasoning_summary = context.get_summary_for_generator()
		user_prompt_str = self.prompt_template.format(
			user_query=user_query,
			reasoning_steps_summary=reasoning_summary
		)
		print(f"  [Generator] Sending generation prompt to LLM (summary length: {len(reasoning_summary)} chars)...")
		final_answer = await self.llm_client.generate(user_prompt_str, system_prompt_str=GenerationPrompt.SYSTEM_PROMPT)
		return final_answer


# --- Pipeline (from previous response, minor logging/error improvements) ---
class IterativePipeline:  # (名字可以保留，但行为是静态DAG执行)
	def __init__(self,
				 planner: Planner,
				 executors: Dict[str, ExecutorBase],
				 generator: AnswerGenerator,
				 max_iterations: int = 1):
		self.planner = planner
		self.executors = executors
		self.generator = generator
		self.max_iterations = max_iterations

	async def _execute_task_with_dependencies(self, task_id: str, context: ContextManager, executed_tasks_cache: set):
		if task_id in executed_tasks_cache:
			task_obj = context.get_task(task_id)
			# if task_obj and task_obj.status != "pending": # 如果不是pending，说明已经被处理或正在处理
			# print(f"  [Pipeline] Task {task_id} already processed or in queue (status: {task_obj.status if task_obj else 'N/A'}). Skipping.")
			return

		task_to_run = context.get_task(task_id)
		if not task_to_run:
			print(f"  [Pipeline Error] Task {task_id} definition not found in context. Cannot execute.")
			return  # Should not happen if plan is consistent

		# 先将任务ID加入缓存，表示开始处理（包括依赖检查）
		executed_tasks_cache.add(task_id)

		# 1. 解决依赖
		print(f"  [Pipeline] Checking dependencies for task {task_id}: {task_to_run.dependencies}")
		for dep_id in task_to_run.dependencies:
			if dep_id not in executed_tasks_cache:  # 只有当依赖项本身还未被“处理过”时才递归
				dep_task_obj = context.get_task(dep_id)
				if not dep_task_obj:  # 依赖的任务ID在计划中就不存在
					error_msg = f"Task {task_to_run.id} has an undefined dependency ID: {dep_id} not found in plan. Marking task as failed."
					context.update_task_status(task_to_run.id, "failed", result=error_msg, thought=error_msg)
					print(f"  [Pipeline Error] {error_msg}")
					return
				await self._execute_task_with_dependencies(dep_id, context, executed_tasks_cache)

		# 2. 检查依赖是否都成功完成
		dependencies_met = True
		if task_to_run.dependencies:  # Only check if there are dependencies
			for dep_id in task_to_run.dependencies:
				dep_task = context.get_task(dep_id)
				if not dep_task or dep_task.status != "completed":
					dependencies_met = False
					current_thought = task_to_run.thought or ""
					task_thought = f"{current_thought}依赖任务 {dep_id} 未成功完成 (状态: {dep_task.status if dep_task else '不存在或未定义'})，跳过执行本任务。"
					context.update_task_status(task_to_run.id, "skipped", thought=task_thought)
					print(
						f"  [Pipeline] Task {task_to_run.id} ('{task_to_run.task_description}') skipped, dependency {dep_id} not met (status: {dep_task.status if dep_task else 'N/A'}).")
					break
			if not dependencies_met:
				return
		else:
			print(f"  [Pipeline] Task {task_id} has no dependencies.")

		# 3. 执行当前任务 (确保它真的是pending状态)
		if task_to_run.status != "pending":
			print(
				f"  [Pipeline] Task {task_id} is not pending (status: {task_to_run.status}). Skipping execution phase for this task.")
			return

		executor = self.executors.get(task_to_run.executor_name)
		if not executor:
			error_msg = f"Executor '{task_to_run.executor_name}' not found for task '{task_to_run.task_description}'."
			context.update_task_status(task_to_run.id, "failed", result=error_msg, thought=error_msg)
			print(f"  [Pipeline Error] {error_msg}")
			return

		print(
			f"\n▶️ Executing Task: {task_to_run.id} - \"{task_to_run.task_description}\" (using {task_to_run.executor_name})")
		context.update_task_status(task_to_run.id, "running", thought=f"开始执行: {task_to_run.task_description}")
		try:
			result = await executor.execute(task_to_run, context)
			context.update_task_status(task_to_run.id, "completed", result=result, thought=task_to_run.thought)
			print(
				f"✅ Task {task_to_run.id} Result : {str(result)}")
		except ExecutorError as e:
			error_str = f"执行器错误 for task {task_to_run.id}: {e}"
			final_thought = (task_to_run.thought or "") + f"\n执行器错误: {e}"
			context.update_task_status(task_to_run.id, "failed", result=error_str, thought=final_thought)
			print(f"🛑 {error_str}")
		except Exception as e:
			error_str = f"执行任务时发生意外错误 {task_to_run.id}: {e}"
			import traceback
			tb_str = traceback.format_exc()
			print(f"🛑 {error_str}\n{tb_str}")
			final_thought = (task_to_run.thought or "") + f"\n意外系统错误: {e}"
			context.update_task_status(task_to_run.id, "failed", result=error_str, thought=final_thought)

	async def run(self, user_query: str) -> str:
		print(f"\n🚀 Pipeline starting for query: \"{user_query}\"")
		context = ContextManager(user_query)
		available_executors_schemas = [ex.get_schema() for ex in self.executors.values()]

		final_answer = "抱歉，处理您的问题时遇到了一些麻烦。"  # Default error

		# 1. 规划
		print(f"\n📝 Planning phase...")
		try:
			planned_tasks_data_list = await self.planner.create_plan(user_query, context, available_executors_schemas)
			if not planned_tasks_data_list:
				print("  [Pipeline] Planner did not generate any tasks. Cannot proceed.")
				return "抱歉，我无法为您的问题制定有效的执行计划。"

			# 清空旧任务，并将新计划加入上下文
			context.tasks.clear()
			context.execution_order.clear()
			print("\n📋 Plan Received from LLM (before adding to context):")
			for i, task_data_item in enumerate(planned_tasks_data_list):
				print(f"  Raw Task Data {i}: {task_data_item}")
				# 使用 context.user_query 生成一个基础ID前缀，确保其对文件名友好
				base_id_prefix_for_task = re.sub(r'[^\w\s-]', '', context.user_query[:15]).strip().replace(' ',
																										   '_') or "query"
				context.add_task_from_planner(task_data_item, base_id_prefix_for_task, i)


		except ValueError as ve:
			print(f"  [Pipeline Error] Planning phase failed due to invalid LLM response: {ve}")
			return f"抱歉，我在规划如何解决您的问题时遇到了错误（LLM返回格式问题）：{ve}"
		except RuntimeError as rte:
			print(f"  [Pipeline Error] Planning phase LLM call failed: {rte}")
			return f"抱歉，连接到语言模型进行规划时失败：{rte}"
		except Exception as e:
			print(f"  [Pipeline Error] Unexpected error during planning: {e}")
			import traceback;
			traceback.print_exc()
			return f"抱歉，规划阶段出现意外错误：{e}"

		print("\n📋 Tasks in Context after planning:")
		for task_id_in_order in context.execution_order:
			task_obj = context.get_task(task_id_in_order)
			if task_obj:
				print(
					f"  - ID: {task_obj.id}, Executor: {task_obj.executor_name}, Desc: \"{task_obj.task_description}\", Deps: {task_obj.dependencies}")
			else:
				print(f"  - Error: Task ID {task_id_in_order} found in execution_order but not in tasks dict.")

		# 2. 执行计划 (处理DAG)
		print("\n⚙️ Execution phase...")
		executed_tasks_this_run = set()

		# 按任务在上下文中的顺序（即LLM规划的顺序）尝试执行
		# _execute_task_with_dependencies 会递归处理实际依赖顺序
		for task_id_to_process in context.execution_order:
			# 只有当任务是pending的时候才启动它，因为依赖执行可能已经处理了它
			task_obj_to_process = context.get_task(task_id_to_process)
			if task_obj_to_process and task_obj_to_process.status == "pending":
				await self._execute_task_with_dependencies(task_id_to_process, context, executed_tasks_this_run)

		failed_or_skipped_tasks = [t for t in context.tasks.values() if t.status in ["failed", "skipped"]]
		if failed_or_skipped_tasks:
			print("\n⚠️ Some tasks failed or were skipped during execution:")
			for ft in failed_or_skipped_tasks:
				print(f"  - Task ID: {ft.id}, Status: {ft.status}, Reason/Result: {str(ft.result)[:200]}")
		else:
			print("  All planned tasks appear to have completed successfully or were appropriately handled.")

		# 3. 生成最终答案
		print("\n💬 Generation phase...")
		try:
			final_answer = await self.generator.generate_final_answer(user_query, context)
		except RuntimeError as rte:
			print(f"  [Pipeline Error] Generation phase LLM call failed: {rte}")
			final_answer = f"抱歉，生成最终答案时连接语言模型失败：{rte}"
		except Exception as e:
			print(f"  [Pipeline Error] Unexpected error during generation: {e}")
			import traceback;
			traceback.print_exc()
			final_answer = f"抱歉，生成最终答案阶段出现意外错误：{e}"

		print(f"\n💡 Final Answer: {final_answer}")
		return final_answer


# --- 主程序入口 ---
async def run_main_logic_with_user_data():
	# 1. 初始化LLM客户端
	api_key = 'sk-af4423da370c478abaf68b056f547c6e'
	base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")

	if not api_key:
		print("错误：未找到 OPENAI_API_KEY 环境变量。请设置后再运行。")
		return

	llm_client = OpenAIChatLLM(model_name=model_name, api_key=api_key, base_url=base_url)

	# 2. 准备用户数据并初始化Chroma知识库
	# 使用您提供的DashScope Embedding Key
	# 请确保您已安装 langchain-community 和 tongyiembedding
	# pip install langchain-community langchain-chroma tongyiembedding
	try:
		# 使用您代码中提供的 QwenEmbeddingFunction
		# 这里假设您的 DASHSCOPE_API_KEY 环境变量也适用于 QwenEmbeddingFunction
		# 如果QwenEmbeddingFunction需要不同的key，请相应调整

		from tongyiembedding import QwenEmbeddingFunction

		embedding_function = QwenEmbeddingFunction(api_key='sk-af4423da370c478abaf68b056f547c6e')

		print(f"  [Embedding] 使用 QwenEmbeddingFunction 初始化 embedding_function。")
	except Exception as e:
		print(f"  [Embedding Error] 初始化 QwenEmbeddingFunction 失败: {e}")
		print("  请确保 'tongyiembedding' 库已安装且API Key有效。")
		return

	# 将您提供的字典列表转换为Langchain Document对象
	initial_user_docs_as_dicts = [
		{"page_content": "【一枝黄花】性状:本品长30～100cm",
		 "metadata": {"id": "doc_yzyh", "source_name": "一枝黄花说明"}},
		{"page_content": "【正柴胡饮颗粒】检查: 应符合颗粒剂项下有关的各项规定（通则0104)。",
		 "metadata": {"id": "doc_zchyk", "source_name": "正柴胡饮颗粒说明"}},
		{"page_content": "0104颗粒剂除另有规定外，颗粒剂应进行以下相应检查。【粒度】",
		 "metadata": {"id": "doc_tongze0104_part1", "source_name": "药典通则0104"}},
		{"page_content": "通则0104继续：【干燥失重】...",
		 "metadata": {"id": "doc_tongze0104_part2", "source_name": "药典通则0104"}}
	]
	initial_langchain_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in
							  initial_user_docs_as_dicts]
	initial_langchain_docs = [
		Document(
			page_content="【一枝黄花】性状:本品长30～100cm。根茎短粗，簇生淡黄色细根。茎圆柱形，直径0.2～0.5cm；表面黄绿色、灰棕色或暗紫红色，有棱线，上部被毛；质脆，易折断，断面纤维性，有髓。单叶互生，多皱缩、破碎，完整叶片展平后呈卵形或披针形，长1～9cm，宽0.3～1.5cm；先端稍尖或钝，全缘或有不规则的疏锯齿，基部下延成柄。头状花序直径约0.7cm，排成总状，偶有黄色舌状花残留，多皱缩扭曲，苞片3层，卵状披针形。瘦果细小，冠毛黄白色。气微香，味微苦辛。",
			metadata={"year": 1993, "rating": 7.7, "genre": "science fiction", "id": "movie_1"},
			# Added an 'id' for clarity
		),
		# 加入之前知识库的文档，转换为Document格式
		Document(page_content="【正柴胡饮颗粒】检查: 应符合颗粒剂项下有关的各项规定（通则0104)。",
				 metadata={"topic": "programming", "language": "Python", "id": "tech_py"}),
		Document(page_content="""0104颗粒剂除另有规定外，颗粒剂应进行以下相应检查。
								【粒度】除另有规定外，照粒度和粒度分布测定法（通则0982第二法 双筛分法）测定，不能通过一号筛与能通过五号筛的总和不得超过15％。
								【水分】中药颗粒剂照水分测定法（通则0832）测定，除另有规定外，水分不得超过8.0％。
								【干燥失重】除另有规定外，化学药品和生物制品颗粒剂照干燥失重测定法（通则0831）测定，于105℃干燥（含糖颗粒应在80℃减压干燥）至恒重，减失重量不得超过2.0%。
								【溶化性】除另有规定外，颗粒剂照下述方法检查，溶化性应符合规定。含中药原粉的颗粒剂不进行溶化性检查。
								可溶颗粒检查法 取供试品10g（中药单剂量包装取1袋），加热水200ml，搅拌5分钟，立即观察，可溶颗粒应全部溶化或轻微浑浊。
								泡腾颗粒检查法 取供试品3袋，将内容物分别转移至盛有200ml水的烧杯中，水温为15～25℃，应迅速产生气体而呈泡腾状，5分钟内颗粒均应完全分散或溶解在水中。
								颗粒剂按上述方法检查，均不得有异物，中药颗粒还不得有焦屑。
								混悬颗粒以及已规定检查溶出度或释放度的颗粒剂可不进行溶化性检查。
								【装量差异】单剂量包装的颗粒剂按下述方法检查，应符合规定。
								检查法 取供试品10袋（瓶），除去包装，分别精密称定每袋（瓶）内容物的重量，求出每袋（瓶）内容物的装量与平均装量。每袋（瓶）装量与平均装量相比较［凡无含量测定的颗粒剂或有标示装量的颗粒剂，每袋（瓶）装量应与标示装量比较］，超出装量差异限度的颗粒剂不得多于2袋（瓶），并不得有1袋（瓶）超出装量差异限度1倍。
								<table border="1" ><tr>
								<td colspan="1" rowspan="1">平均装量或标示装量</td>
								<td colspan="1" rowspan="1">装量差异限度</td>
								</tr><tr>
								<td colspan="1" rowspan="1">1.0g及1.0g以下</td>
								<td colspan="1" rowspan="1">±10%</td>
								</tr><tr>
								<td colspan="1" rowspan="1">1.0g以上至1.5g </td>
								<td colspan="1" rowspan="1">±8%</td>
								</tr><tr>
								<td colspan="1" rowspan="1">1.5g以上至6.0g </td>
								<td colspan="1" rowspan="1">±7%</td>
								</tr><tr>
								<td colspan="1" rowspan="1">6.0g以上</td>
								<td colspan="1" rowspan="1">±5%</td>
								</tr></table>
								凡规定检查含量均匀度的颗粒剂，一般不再进行装量差异检查。
								【装量】多剂量包装的颗粒剂，照最低装量检查法（通则0942）检查，应符合规定。
								【微生物限度】以动物、植物、矿物质来源的非单体成分制成的颗粒剂，生物制品颗粒剂，照非无菌产品微生物限度检查：微生物计数法（通则1105）和控制菌检查法（通则1106）及非无菌药品微生物限度标准（通则1107）检查，应符合规定。规定检查杂菌的生物制品颗粒剂，可不进行微生物限度检查。0104颗粒剂（更昔洛韦）""",
				 metadata={"topic": "AI", "type": "LLM", "id": "tech_llm"}),
		Document(page_content="""1107非无菌药品微生物限度标准 3．非无菌化学药品制剂、生物制品制剂、不含药材原粉的中药制剂的微生物限度标准见表1。
									表1 非无菌化学药品制剂、生物制品制剂、不含药材原粉的中药制剂的微生物限度标准
									<table border="1">

										<tr>

											<th>给药途径</th>

											<th>制剂类型</th>

											<th>需氧菌总数（cfu／g、cfu／ml或cfu／10c㎡）</th>

											<th>霉菌和酵母菌总数（cfu／g、cfu／ml或cfu／10c㎡）</th>

											<th>控制菌</th>

										</tr>

										<tr>

											<td rowspan="2">口服给药</td>

											<td>固体制剂</td>

											<td>10³</td>

											<td>10²</td>

											<td>不得检出大肠埃希菌（1g或1ml）；含脏器提取物的制剂还不得检出沙门菌（10g或10ml）</td>

										</tr>

										<tr>

											<td>液体及半固体制剂</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出大肠埃希菌（1g或1ml）；含脏器提取物的制剂还不得检出沙门菌（10g或10ml）</td>

										</tr>

										<tr>

											<td rowspan="3">口腔黏膜给药制剂<br>齿龈给药制剂<br>鼻用制剂</td>

											<td>-</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出大肠埃希菌、金黄色葡萄球菌、铜绿假单胞菌（1g、1ml或10c㎡）</td>

										</tr>

										<tr>

											<td>耳用制剂</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌（1g、1ml或10c㎡）</td>

										</tr>

										<tr>

											<td>皮肤给药制剂</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌（1g、1ml或10c㎡）</td>

										</tr>

										<tr>

											<td>呼吸道吸入给药制剂</td>

											<td>-</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出大肠埃希菌、金黄色葡萄球菌、铜绿假单胞菌、耐胆盐革兰阴性菌（1g或1ml）</td>

										</tr>

										<tr>

											<td>阴道、尿道给药制剂</td>

											<td>-</td>

											<td>10²</td>

											<td>10¹</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌、白色念珠菌（1g、1ml或10c㎡）；中药制剂还不得检出梭菌（1g、1ml或10c㎡）</td>

										</tr>

										<tr>

											<td rowspan="2">直肠给药</td>

											<td>固体及半固体制剂</td>

											<td>10³</td>

											<td>10²</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌（1g 或1ml）</td>

										</tr>

										<tr>

											<td>液体制剂</td>

											<td>10²</td>

											<td>10²</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌（1g 或1ml）</td>

										</tr>

										<tr>

											<td>其他局部给药制剂</td>

											<td>-</td>

											<td>10²</td>

											<td>10²</td>

											<td>不得检出金黄色葡萄球菌、铜绿假单胞菌（1g、1ml或10c㎡）</td>

										</tr>

									</table>
									注：①化学药品制剂和生物制品制剂若含有未经提取的动植物来源的成分及矿物质，还不得检出沙门菌（10g或10ml）。""",
				 metadata={"topic": "technology", "product": "iPhone", "id": "tech_iphone"}),
		Document(page_content="北京是中华人民共和国的首都，也是中国的政治、文化、科技创新和国际交往中心。",
				 metadata={"topic": "geography", "city": "Beijing", "id": "geo_beijing"}),
		Document(page_content="FAISS (Facebook AI Similarity Search) 是一个用于高效相似性搜索和密集向量聚类的库。",
				 metadata={"topic": "technology", "library": "FAISS", "id": "tech_faiss"})
	]
	from md2Document import read_md_file, create_document_from_md
	initial_langchain_docs = []
	folder_path = r'D:\Master\llm\database\kag\测试数据集'  # Windows路径
	# 获取文件夹中所有文件
	files = os.listdir(folder_path)
	# 筛选出所有 .md 文件
	md_files = [file for file in files if file.endswith('.md')]
	# 创建Document对象列表

	for file in md_files:
		file_path = os.path.join(folder_path, file)  # 拼接完整的文件路径
		document = create_document_from_md(file_path)  # 创建Document对象
		initial_langchain_docs.append(document)

	try:
		# force_rebuild=True 确保每次运行时都重新构建索引，便于测试
		# 在生产中，通常会设为 False 以加载现有索引
		chroma_kb = ChromaKnowledgeBase(
			initial_documents=initial_langchain_docs,
			embedding_function=embedding_function,  # Langchain-compatible embedding function
			persist_directory=CHROMA_PERSIST_DIRECTORY,
			collection_name=CHROMA_COLLECTION_NAME,
			force_rebuild=True
		)
	except Exception as e:
		print(f"创建或加载Chroma知识库失败: {e}")
		import traceback;
		traceback.print_exc()
		return

	# 3. 初始化 Prompts
	planner_prompt = PlannerPrompt()
	deduce_prompt = DeducePrompt()
	code_exec_prompt = CodeExecutionPrompt()
	generation_prompt = GenerationPrompt()

	# 4. 初始化 Executors
	retrieval_executor = RetrievalExecutor(kb=chroma_kb)  # 使用ChromaKB
	deduce_executor = DeduceExecutor(llm_client, deduce_prompt)
	code_executor = CodeExecutor(llm_client, code_exec_prompt)

	executors_map = {
		"RetrievalExecutor": retrieval_executor,
		"DeduceExecutor": deduce_executor,
		"CodeExecutor": code_executor,
	}

	# 5. 初始化 Planner
	planner = Planner(llm_client, planner_prompt)

	# 6. 初始化 Generator
	generator = AnswerGenerator(llm_client, generation_prompt)

	# 7. 初始化 Pipeline
	pipeline = IterativePipeline(planner, executors_map, generator, max_iterations=1)

	# 8. 运行用户指定的查询
	user_query_to_run = "正柴胡饮颗粒的检查内容有哪些方面？请详细说明。"
	# user_query_to_run = "一只huanghua的性状"
	print(f"\n🚀 Running user query: \"{user_query_to_run}\"")
	final_answer = await pipeline.run(user_query_to_run)
	print(f"\n🏁 Final Answer for user query: \n{final_answer}")


if __name__ == "__main__":
	# 确保在运行此脚本前设置 OPENAI_API_KEY, OPENAI_BASE_URL (如果需要), 和 LLM_MODEL_NAME 环境变量
	# 例如:
	# export OPENAI_API_KEY="sk-yourdashscopekeyoropenaikey"
	# export OPENAI_BASE_URL="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)" # (如果用DashScope)
	# export LLM_MODEL_NAME="qwen-plus"
	# export DASHSCOPE_API_KEY_FOR_EMBEDDING="your_dashscope_key_if_different" (如果embedding key不同)

	# 或者在代码中直接修改 OpenAIChatLLM 和 QwenEmbeddingFunction 的API Key (不推荐用于生产)

	print("开始执行主逻辑...")
	print(f"当前工作目录: {os.getcwd()}")


	asyncio.run(run_main_logic_with_user_data())