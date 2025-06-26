import asyncio
import json
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable, TypedDict
from dataclasses import dataclass, field
import shutil

# --- Langchain and Chroma Imports ---
from langchain_core.documents import Document
from langchain_chroma import Chroma
from tongyiembedding import QwenEmbeddingFunction

# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = "chroma_db_kag_recursive"
CHROMA_COLLECTION_NAME = "kag_recursive_documents"


# --- ChromaKnowledgeBase (基本不变) ---
class ChromaKnowledgeBase:
	def __init__(self, embedding_function: Callable, initial_documents: Optional[List[Document]] = None,
				 persist_directory: str = CHROMA_PERSIST_DIRECTORY, collection_name: str = CHROMA_COLLECTION_NAME,
				 force_rebuild: bool = False):
		print(f"  [ChromaKB] 初始化知识库: {persist_directory}, 集合: {collection_name}")
		self.embedding_function = embedding_function
		self.persist_directory = persist_directory
		self.collection_name = collection_name
		self.vectorstore: Optional[Chroma] = None
		if force_rebuild and os.path.exists(persist_directory):
			print(f"  [ChromaKB] force_rebuild=True, 删除目录: {persist_directory}")
			try:
				shutil.rmtree(persist_directory)
			except OSError as e:
				print(f"  [ChromaKB Error] 删除目录失败: {e}.")
		if os.path.exists(persist_directory) and not force_rebuild:
			print(f"  [ChromaKB] 从 '{persist_directory}' 加载已存在向量库...")
			try:
				self.vectorstore = Chroma(persist_directory=self.persist_directory,
										  embedding_function=self.embedding_function,
										  collection_name=self.collection_name)
				print(f"  [ChromaKB] 成功加载向量库 '{self.collection_name}'.")
			except Exception as e:
				print(f"  [ChromaKB Error] 从 '{persist_directory}' 加载失败: {e}. 将尝试新建。")
				self.vectorstore = None
		if self.vectorstore is None:
			if initial_documents:
				print(f"  [ChromaKB] 为 {len(initial_documents)} 个文档构建新向量库...")
				self.vectorstore = Chroma.from_documents(documents=initial_documents,
														 embedding=self.embedding_function,
														 persist_directory=self.persist_directory,
														 collection_name=self.collection_name)
				print(f"  [ChromaKB] 新向量库构建并持久化完成。")
			else:
				print(f"  [ChromaKB] 无初始文档，创建空的持久化集合。")
				self.vectorstore = Chroma(persist_directory=self.persist_directory,
										  embedding_function=self.embedding_function,
										  collection_name=self.collection_name)
				print(f"  [ChromaKB] 空的持久化 Chroma 集合 '{self.collection_name}' 已准备就绪。")

	def add_documents(self, documents: List[Document]):  # (与之前实现相同)
		if not self.vectorstore:
			if documents:
				print(f"  [ChromaKB] Vectorstore 为空, 尝试从当前 {len(documents)} 个文档创建...")
				self.vectorstore = Chroma.from_documents(documents=documents, embedding=self.embedding_function,
														 persist_directory=self.persist_directory,
														 collection_name=self.collection_name)
				print(f"  [ChromaKB] 基于新文档创建并持久化完成。");
				return
			else:
				print("  [ChromaKB Error] Vectorstore 未初始化且无文档可添加."); return
		if documents:
			print(f"  [ChromaKB] 向集合 '{self.collection_name}' 添加 {len(documents)} 个新文档...")
			self.vectorstore.add_documents(documents);
			print(f"  [ChromaKB] 文档添加完成。")

	def retrieve(self, query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[
		Dict[str, Any]]:  # (与之前实现相同)
		if not self.vectorstore: print("  [ChromaKB Error] Vectorstore 未初始化."); return []
		try:
			if self.vectorstore._collection is None or self.vectorstore._collection.count() == 0:
				print("  [ChromaKB] 知识库集合为空或未正确加载。");
				return []
		except Exception as e:
			print(f"  [ChromaKB Warning] 无法获取集合计数: {e}")
		print(f"  [ChromaKB] 检索查询 '{query}', top_k={top_k}, 过滤器: {filter_dict}...")
		try:
			results_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k, filter=filter_dict)
		except Exception as e:
			print(f"  [ChromaKB Error] Chroma similarity search failed: {e}"); return []
		processed_results = [{"id": doc.metadata.get("id", f"retrieved_{i}"), "content": doc.page_content,
							  "metadata": doc.metadata, "score": float(score)}
							 for i, (doc, score) in enumerate(results_with_scores)]
		print(f"  [ChromaKB] 检索到 {len(processed_results)} 个文档。")
		return processed_results


# --- LLM Client (OpenAIChatLLM - 与之前相同) ---
class OpenAIChatLLM:
	def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
		try:
			import openai
		except ImportError:
			raise ImportError("OpenAI library not found. `pip install openai`.")
		self.api_key = api_key or os.getenv("OPENAI_API_KEY")
		self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
		self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "qwen-plus")
		if not self.api_key: raise ValueError("API key not found.")
		self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
		print(f"[OpenAIChatLLM] 客户端就绪: 模型 {self.model_name}, URL {self.base_url or 'OpenAI default'}")

	async def _make_api_call(self, messages: List[Dict[str, str]], expect_json: bool = False, temperature: float = 0.1,
							 **kwargs) -> str:
		try:
			completion_params = {"model": self.model_name, "messages": messages, "temperature": temperature, **kwargs}
			if expect_json:
				if "dashscope" in (self.base_url or "").lower() and self.model_name.startswith("qwen"):
					completion_params["extra_body"] = {"result_format": "message"}
				else:
					completion_params["response_format"] = {"type": "json_object"}
			response = await asyncio.to_thread(self.client.chat.completions.create, **completion_params)
			content = response.choices[0].message.content
			return content.strip() if content else ""
		except Exception as e:
			print(f"  [OpenAIChatLLM Error] API调用失败: {e}"); raise RuntimeError(f"LLM API call failed: {e}")

	async def generate(self, prompt_str: str, system_prompt_str: Optional[str] = None, temperature: float = 0.1,
					   **kwargs) -> str:
		messages = []
		if system_prompt_str: messages.append({"role": "system", "content": system_prompt_str})
		messages.append({"role": "user", "content": prompt_str})
		return await self._make_api_call(messages, expect_json=False, temperature=temperature, **kwargs)

	async def generate_structured_json(self, prompt_str: str, system_prompt_str: Optional[str] = None,
									   temperature: float = 0.1, **kwargs) -> Dict:
		messages = []
		if system_prompt_str: messages.append({"role": "system", "content": system_prompt_str})
		user_content = f"{prompt_str}\n\n请确保您的回复是一个合法的、单独的JSON对象，不包含任何其他解释性文本或markdown标记。"
		messages.append({"role": "user", "content": user_content})
		response_str = await self._make_api_call(messages, expect_json=True, temperature=temperature, **kwargs)
		cleaned_response_str = response_str.strip()
		if cleaned_response_str.startswith("```json"): cleaned_response_str = cleaned_response_str[7:]
		if cleaned_response_str.endswith("```"): cleaned_response_str = cleaned_response_str[:-3]
		cleaned_response_str = cleaned_response_str.strip()
		try:
			return json.loads(cleaned_response_str)
		except json.JSONDecodeError as e:
			error_msg = f"LLM did not return valid JSON. Error: {e}. Raw: '{response_str}'"
			print(f"  [OpenAIChatLLM Error] {error_msg}");
			raise ValueError(error_msg)


# --- Prompts ---
class BasePrompt:
	def __init__(self, template: str, variables: List[str]):
		self.template_str = template;
		self.variables = variables

	def format(self, **kwargs) -> str:
		formatted_prompt = self.template_str
		for var_name in self.variables:
			if var_name not in kwargs: raise ValueError(f"Missing variable: {var_name}")
			formatted_prompt = formatted_prompt.replace(f"${{{var_name}}}", str(kwargs[var_name]))
		return formatted_prompt


class PlannerPrompt(BasePrompt):  # REFINED: Enhanced for iterative, reflective planning
	SYSTEM_PROMPT = """
    你是一位高度智能的AI任务规划和反思专家。
    你的核心目标是：根据用户提出的复杂问题 `${user_query}` 和已经执行的历史任务 `${task_history}`，分析当前状态，并决定下一步行动。
    这个行动可能是：
    a) 规划新的、具体的子任务来获取缺失信息或进行必要推理。
    b) 判断问题已经完全解决。
    c) 判断问题因信息不足或工具限制而无法进一步解决。

    你必须严格按照以下JSON格式输出你的决策和计划：

    **输出格式:**
    ```json
    {
      "plan_status": "finished" | "cannot_proceed" | "requires_more_steps",
      "final_thought": "string (对当前整体进展的清晰思考和判断，解释你的plan_status)",
      "next_steps": [ // 仅当 plan_status == "requires_more_steps" 时非空
        {
          "id": "string (新任务的唯一ID, 例如 task_iter2_step0, 确保与历史ID不冲突)",
          "executor_name": "string (从可用工具列表中选择，例如 'RetrievalExecutor', 'DeduceExecutor', 'FinishAction')",
          "task_description": "string (对此新任务目标的清晰、具体中文描述)",
          "logic_input": {
            // 字段依赖于 executor_name, 参考工具描述中的 'logic_input_schema'
            // 特别注意：为 DeduceExecutor 的 'context_data' 字段提供高度相关且简洁的上下文。
            // 可以直接从 `${user_query}` 或 `${task_history}` 中某个特定任务的 `answer_summary` 或 `retrieved_content` 中提取。
            // 如果需要之前任务的完整结果，使用 "{{task_id.result}}" 占位符。
          },
          "dependencies": ["string"] // 依赖的历史任务ID列表
        }
        // ... 通常一次只规划1-2个最关键的后续步骤 ...
      ]
    }
    ```

    **核心指令与思考链 (Chain-of-Thought for Reflective Planning):**
    1.  **回顾目标 (Recall Goal)**: 清晰理解用户原始问题 `${user_query}` 的最终目标是什么。
    2.  **审视历史 (Analyze History - `${task_history}`中每个任务的 `Result Details`)**:
        * 哪些子问题已经被回答了？答案是什么 (`answer_summary`)？
        * 信息是否充分 (`is_sufficient`)？
        * 是否识别出了新的查询点 (`new_questions_or_entities`)？
        * `RetrievalExecutor` 检索到了哪些关键信息？
        * 之前的 `Thought` 中有哪些未解决的线索？
    3.  **差距评估 (Gap Assessment)**: 对比当前已知信息和用户原始问题的最终目标，还缺少哪些核心信息片段或逻辑步骤？
    4.  **决策制定 (Decision Making & Justification -> `final_thought` and `plan_status`)**:
        * **已解决?** 如果所有必要信息都已在历史中确认（例如，通过 `is_sufficient: true` 的Deduce步骤），并且能够完整回答用户问题，则 `plan_status: "finished"`。`final_thought` 应总结是如何解决的。
        * **无法解决?** 如果关键信息通过历史中的检索和演绎步骤都未能获取（例如，多次相关的 `RetrievalExecutor` 失败或返回空，或者 `DeduceExecutor` 持续报告 `is_sufficient: false` 且没有新的有效查询点），并且你判断没有其他可用工具能解决，则 `plan_status: "cannot_proceed"`。`final_thought` 应解释原因。
        * **需要更多步骤?** 如果上述两者都不是，则 `plan_status: "requires_more_steps"`。`final_thought` 应明确指出当前已完成什么、还缺少什么，以及下一步计划（`next_steps`）的目标是什么。
    5.  **行动规划 (Action Formulation for `next_steps` - 如果 `requires_more_steps`)**:
        * **针对性**: 规划1-2个最直接解决当前核心差距的步骤。
        * **工具选择**:
            * 如果历史提示 `new_questions_or_entities` 或你的分析表明需要查找新信息，优先规划 `RetrievalExecutor`。其 `logic_input.query` 应尽可能具体，可以基于这些 `new_questions_or_entities` 或对历史结果的分析来构造。
            * 如果已有一些信息片段，需要整合、分析、总结、判断或抽取，规划 `DeduceExecutor`。其 `logic_input.reasoning_goal` 要明确，`logic_input.context_data` 应精确提供必要的上下文（例如，引用刚检索到的 `{{retrieval_task_id.result}}`，或历史中某个 `DeduceExecutor` 的 `answer_summary`）。
            * 如果判断所有子问题都已解决，信息已完整，规划一个 `FinishAction` 任务，`logic_input` 中可以简单说明理由。
        * **ID 和依赖**: 为新任务分配唯一的 `id`。正确设置 `dependencies`，指向提供输入的历史任务ID。
    6.  **严格JSON输出**: 你的全部输出必须是合法的、单一的JSON对象。
    """
	USER_TEMPLATE = """
    --- 可用工具 ---
    ${available_executors_description}
    --- 可用工具结束 ---

    --- 历史任务及结果 (最近的步骤在最后) ---
    ${task_history}
    --- 历史任务及结果结束 ---

    --- 用户原始问题 ---
    "${user_query}"
    --- 用户原始问题结束 ---

    请根据以上信息，反思并规划。输出JSON对象:
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["user_query", "available_executors_description", "task_history"])


class DeducePrompt(BasePrompt):  # REFINED: To guide LLM for structured output
	SYSTEM_PROMPT = """
    你是一位严谨的AI推理专家。你的任务是根据提供的“上下文信息”，精确地回答或完成“推理目标”。
    请严格按照以下JSON格式输出你的结论和评估：
    ```json
    {
      "answer_summary": "string (对推理目标的直接、简洁的回答或总结)",
      "is_sufficient": boolean (true 如果你认为提供的上下文信息足以完全回答推理目标，否则 false),
      "new_questions_or_entities": [
        "string" // 如果 is_sufficient 为 false，列出需要进一步调查或检索的具体问题、术语或实体名称。如果信息充分则为空列表[]。
      ]
    }
    ```
    - 如果上下文信息不包含回答推理目标所需的内容，请在 `answer_summary` 中明确指出信息不足，并将 `is_sufficient` 设为 `false`。
    - `new_questions_or_entities` 对于引导后续步骤至关重要，请尽可能具体。例如，如果上下文中提到“通则XXXX”，但不包含其细节，那么“通则XXXX的详细内容”就是一个好的`new_questions_or_entities`。
    - 不要添加任何额外的解释或markdown标记，只输出JSON对象。
    """
	USER_TEMPLATE = """
    推理目标:
    ${reasoning_goal}

    上下文信息:
    ${context_data}

    请输出JSON格式的推理结果:
    """

	def __init__(self):
		super().__init__(self.USER_TEMPLATE, ["reasoning_goal", "context_data"])


class CodeExecutionPrompt(BasePrompt):  # (与之前相同)
	SYSTEM_PROMPT = "你是一个Python代码生成和执行助手。"
	USER_TEMPLATE = "请根据以下指令和相关数据，生成一段Python代码来解决问题。\n代码必须通过 `print()` 输出其最终计算结果。不要包含任何解释或注释，只输出纯代码。\n\n指令:\n${code_generation_prompt}\n\n相关数据 (如果提供):\n${relevant_data}\n\n生成的Python代码 (请确保它只包含代码本身，并用print()输出结果):"

	def __init__(self): super().__init__(self.USER_TEMPLATE, ["code_generation_prompt", "relevant_data"])


class UserProvidedReferGeneratorPrompt(BasePrompt):  # (与之前相同, 确保get_now可用)
	def __init__(self, language: str = "zh"):
		try:
			from kag.common.utils import get_now
		except ImportError:
			def get_now(language='zh'):
				return "当前日期"

			print("[UserProvidedReferGeneratorPrompt] Warning: kag.common.utils.get_now not found.")
		self.template_zh = (
					f"你是一个信息分析专家，今天是{get_now(language='zh')}。" + "基于给定的引用信息回答问题。\n输出答案，如果答案中存在引用信息，则需要reference的id字段，如果不是检索结果，则不需要标记引用\n输出时，不需要重复输出参考文献\n引用要求，使用类似<reference id=\"chunk:1_2\"></reference>表示\n如果根据引用信息无法回答，则使用模型内的知识回答，但是必须通过合适的方式提示用户，是基于检索内容还是引用文档\n示例1：\n任务过程上下文：\n根据常识岳父是妻子的爸爸，所以需要首先找到张三的妻子，然后找到妻子的爸爸\n给定的引用信息：'\nreference：\n[\n{\n    \"content\": \"张三 妻子 王五\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_1\"\n},\n{\n    \"content\": \"王五 父亲 王四\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_2\"\n}\n]'\n问题：'张三的岳父是谁？'\n\n张三的妻子是王五<reference id=\"chunk:1_1\"></reference>，而王五的父亲是王四<reference id=\"chunk:1_2\"></reference>，所以张三的岳父是王四\n\n\n输出语调要求通顺，不要有机械感，输出的语言要和问题的语言保持一致\n任务过程上下文信息：'${summary_of_executed_steps}'\n给定的引用信息：'${formatted_references}'\n问题：'${user_query}'")
		self.template_en = self.template_zh  # 简化，实际应有英文版
		self.template_zh = (
		f"你是一个信息分析专家，今天是{get_now(language='zh')}。"
		"你的任务是基于【任务过程上下文信息】中各步骤的【事实性产出】（尤其是成功的推理结论和检索到的信息）以及【给定的引用信息】，来全面、详细地回答用户的问题：'${user_query}'。"
		"请确保你的回答直接、准确，并能体现出多方面的信息综合。不要重复或过多阐述任务规划的思考过程，而是聚焦于实际获得的结果。"
		# ... 后续关于引用格式、信息不足处理等指令保持不变 ...
		"\n任务过程上下文信息：'${summary_of_executed_steps}'"
		"\n给定的引用信息：'${formatted_references}'\n问题：'${user_query}'"
		)
		current_template = self.template_zh if language == "zh" else self.template_en
		super().__init__(current_template, ["summary_of_executed_steps", "user_query", "formatted_references"])

	def format(self, summary_of_executed_steps: str, user_query: str, retrieved_references: List[Dict]) -> str:
		ref_list_for_prompt = []
		for i, ref_item in enumerate(retrieved_references):
			ref_list_for_prompt.append({"content": ref_item.get("content", ""),
										"document_name": ref_item.get("metadata", {}).get("source_name",
																						  f"检索文档{i + 1}"),
										"id": ref_item.get("metadata", {}).get("id", f"retrieved_chunk_{i}")})
		formatted_references_str = json.dumps(ref_list_for_prompt, ensure_ascii=False, indent=2)
		return super().format(summary_of_executed_steps=summary_of_executed_steps, user_query=user_query,
							  formatted_references=formatted_references_str)


# --- DataStructures & ContextManager (ContextManager slightly adapted) ---
LogicInput = Dict[str, Any]


@dataclass
class Task:
	id: str;
	executor_name: str;
	task_description: str;
	logic_input: LogicInput
	dependencies: List[str] = field(default_factory=list);
	status: str = "pending"
	result: Optional[Any] = None;
	thought: Optional[str] = None  # result can now be a Dict for DeduceExecutor


class ContextManager:
	def __init__(self, user_query: str):
		self.user_query = user_query;
		self.tasks: Dict[str, Task] = {}
		self.execution_order: List[str] = []  # 保持任务添加/规划顺序

	def add_task_from_planner(self, task_data: Dict, base_id_prefix: str, current_iteration: int,
							  step_in_iter: int) -> Task:
		llm_provided_id = task_data.get("id")
		if llm_provided_id and isinstance(llm_provided_id, str) and llm_provided_id.strip():
			task_id = llm_provided_id
		else:
			task_id = f"{base_id_prefix}_iter{current_iteration}_step{step_in_iter}"
		original_task_id = task_id;
		counter = 0
		while task_id in self.tasks: counter += 1; task_id = f"{original_task_id}_v{counter}"
		if counter > 0: print(f"  [CM Warn] Task ID '{original_task_id}' conflict. Renamed to '{task_id}'.")
		task = Task(id=task_id, executor_name=task_data["executor_name"],
					task_description=task_data["task_description"],
					logic_input=task_data["logic_input"], dependencies=task_data.get("dependencies", []))
		self.tasks[task.id] = task;
		self.execution_order.append(task.id);
		return task

	def get_task(self, task_id: str) -> Optional[Task]:
		return self.tasks.get(task_id)

	def update_task_status(self, task_id: str, status: str, result: Optional[Any] = None,
						   thought: Optional[str] = None):
		task = self.get_task(task_id)
		if task:
			task.status = status
			if result is not None: task.result = result
			current_thought = task.thought or "";
			task.thought = f"{current_thought}\n{thought}".strip() if current_thought and thought else thought or current_thought
		else:
			print(f"  [CM Warn] Task ID {task_id} not found for status update.")

	def get_task_history_for_prompt(self) -> str:  # REFINED: To show structured DeduceExecutor result
		history = []
		for task_id in self.execution_order:
			task = self.tasks.get(task_id)
			if task and task.status in ["completed", "failed"]:
				result_display_parts = []
				if task.status == "completed":
					if isinstance(task.result, dict) and task.executor_name == "DeduceExecutor":  # Structured result
						deduce_out = task.result
						result_display_parts.append(
							f"    推理总结: {str(deduce_out.get('answer_summary', 'N/A'))[:100]}{'...' if len(str(deduce_out.get('answer_summary', 'N/A'))) > 100 else ''}")
						result_display_parts.append(f"    信息是否充分: {deduce_out.get('is_sufficient', True)}")
						new_q = deduce_out.get('new_questions_or_entities', [])
						if new_q: result_display_parts.append(
							f"    建议进一步查询: {', '.join(new_q)[:100]}{'...' if len(', '.join(new_q)) > 100 else ''}")
					elif isinstance(task.result, list) and task.executor_name == "RetrievalExecutor":
						result_display_parts.append(
							f"    检索到 {len(task.result)} 个片段。内容(摘要): {str(task.result[0].get('content') if task.result else '空')[:80]}...")
					else:  # Other executors or simple string result
						res_str = str(task.result);
						result_display_parts.append(f"    结果: {res_str[:150]}{'...' if len(res_str) > 150 else ''}")
				else:  # Failed task
					result_display_parts.append(f"    执行失败: {str(task.result)[:100]}...")

				result_final_display = "\n".join(result_display_parts)
				thought_str = str(task.thought or "N/A");
				thought_str = thought_str[:100] + "..." if len(thought_str) > 100 else thought_str

				history.append(
					f"  - Task ID: {task.id}\n    Desc: {task.task_description}\n    Exec: {task.executor_name}\n    Status: {task.status}\n{result_final_display}\n    Thought: {thought_str}")
		return "\n\n".join(history) if history else "尚未执行任何历史任务。"

	# In class ContextManager:
	def get_summary_for_generator(self) -> str:
		summary_parts = []
		print("  [ContextManager] Generating summary for AnswerGenerator...")
		for i, task_id in enumerate(self.execution_order):  # 按任务规划/添加顺序
			task = self.get_task(task_id)
			if task:
				result_str = "N/A"
				thought_str = str(task.thought or '未记录思考过程')

				if task.status == 'completed':
					if isinstance(task.result, dict) and task.executor_name == "DeduceExecutor":
						# 这是 DeduceExecutor 的结构化输出
						answer_summary = task.result.get('answer_summary', '未能提取总结')
						is_sufficient = task.result.get('is_sufficient', False)
						new_qs = task.result.get('new_questions_or_entities', [])
						result_str = f"推理结论: \"{answer_summary}\" (信息是否充分: {is_sufficient})"
						if new_qs:
							result_str += f" 建议进一步探究: {', '.join(new_qs)}"
					elif isinstance(task.result, list) and task.executor_name == "RetrievalExecutor":
						# task.result 是 List[Dict[str,Any]]
						num_retrieved = len(task.result)
						if num_retrieved > 0:
							# 显示检索到的文档的ID或名称，而不是完整内容，避免摘要过长
							# 完整内容会通过 collect_retrieved_references_for_generator 单独提供给 ReferGeneratorPrompt
							doc_ids_or_names = [
								item.get("metadata", {}).get("id", f"检索项{idx + 1}")
								for idx, item in enumerate(task.result)
							]
							result_str = f"检索到 {num_retrieved} 个相关文档/片段 (ID/名称: {', '.join(doc_ids_or_names)})。"
						else:
							result_str = "未检索到相关文档。"
					elif isinstance(task.result, str):  # 例如 CodeExecutor 或旧版 DeduceExecutor
						result_str = task.result
					else:  # 其他复杂类型
						result_str = f"复杂类型结果 (摘要: {str(task.result)[:100]}...)"
				elif task.status == 'failed':
					result_str = f"执行失败: {str(task.result)[:150]}..."
				elif task.status == 'skipped':
					result_str = "因依赖失败或条件不满足而跳过。"
				else:  # pending, running
					result_str = f"当前状态: {task.status}"

				# 截断，以防Prompt过长
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

	def collect_retrieved_references_for_generator(self) -> List[Dict]:  # (与之前相同)
		references = []
		for task_id in self.execution_order:
			task = self.get_task(task_id)
			if task and task.executor_name == "RetrievalExecutor" and task.status == "completed" and isinstance(
					task.result, list):
				for retrieved_item in task.result:
					if isinstance(retrieved_item, dict) and "content" in retrieved_item:
						references.append({"content": retrieved_item["content"],
										   "document_name": retrieved_item.get("metadata", {}).get("source_name",
																								   f"来源文档_{task.id}"),
										   "id": retrieved_item.get("metadata", {}).get("id",
																						f"ref_{task.id}_{len(references)}")})
		return references


# --- Executors (DeduceExecutor modified, FinishExecutor added) ---
class ExecutorError(Exception): pass


class ExecutorBase(ABC):  # (与之前相同)
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
				ref_full = match.group(1).strip();
				task_id_ref, attr_ref = ref_full.split('.', 1) if '.' in ref_full else (ref_full, "result")
				ref_task = context.get_task(task_id_ref)
				if ref_task and ref_task.status == "completed":
					if attr_ref == "result":
						# Special handling for structured DeduceExecutor result if referenced directly
						if isinstance(ref_task.result, dict) and "answer_summary" in ref_task.result:
							return str(ref_task.result["answer_summary"])  # Use the summary part
						if isinstance(ref_task.result, list): return "\n".join(
							[f"- {item}" for item in map(str, ref_task.result)])
						return str(ref_task.result)
					else:
						print(f"  [Exec Warn] Unsupported attr '{attr_ref}' in {{ {ref_full} }}.")
				else:
					print(
						f"  [Exec Warn] Cannot resolve ref {{ {ref_full} }}. Task '{task_id_ref}' status: {ref_task.status if ref_task else 'not_found'}.")
				return f"{{引用错误: {ref_full}}}"

			return re.sub(r"\{\{([\w_\-\d\.]+)\}\}", replace_match, data_template)
		elif isinstance(data_template, dict):
			return {k: self._resolve_references(v, context) for k, v in data_template.items()}
		elif isinstance(data_template, list):
			return [self._resolve_references(item, context) for item in data_template]
		return data_template


class RetrievalExecutor(ExecutorBase):  # (与之前相同)
	def __init__(self, kb: ChromaKnowledgeBase):
		super().__init__(); self.kb = kb

	async def execute(self, task: Task, context: ContextManager) -> List[Dict[str, Any]]:
		logic_input = task.logic_input;
		query_to_retrieve = logic_input.get("query");
		filter_param = logic_input.get("filter")
		if not query_to_retrieve or not isinstance(query_to_retrieve, str): raise ExecutorError(
			"RetrievalExecutor: 'query' (string) is required.")
		resolved_query = self._resolve_references(query_to_retrieve, context);
		actual_filter = None
		if filter_param and isinstance(filter_param, dict) and filter_param:
			actual_filter = filter_param
		elif filter_param:
			print(f"  [RetrievalExec Warn] Invalid filter for task '{task.id}': {filter_param}. Ignoring.")
		current_thought = task.thought or "";
		task.thought = f"{current_thought}KB检索查询: '{resolved_query}', 过滤器: {actual_filter}.".strip()
		retrieved_docs_with_meta = self.kb.retrieve(resolved_query, top_k=3, filter_dict=actual_filter)
		if not retrieved_docs_with_meta: task.thought += "\n未检索到任何匹配文档."; return []
		task.thought += f"\n检索到 {len(retrieved_docs_with_meta)} 个文档对象.";
		return retrieved_docs_with_meta

	def get_schema(self) -> Dict[str, Any]:
		return {"name": "RetrievalExecutor",
				"description": "从向量知识库中检索与查询相关的文本片段。可指定元数据过滤器。",
				"logic_input_schema": {"query": "string (检索查询语句, 可引用 {{task_id.result}})",
									   "filter": "dict (可选, ChromaDB元数据过滤器)"}}


# REFINED: DeduceExecutor to return structured output
class DeduceExecutorOutput(TypedDict, total=False):
	answer_summary: str
	is_sufficient: bool
	new_questions_or_entities: List[str]
	raw_llm_response: str


class DeduceExecutor(ExecutorBase):
	def __init__(self, llm_client: OpenAIChatLLM, prompt_template: DeducePrompt,
				 specialized_prompts: Optional[Dict[str, BasePrompt]] = None):  # Renamed prompt to prompt_template
		super().__init__(llm_client)
		self.default_prompt_template = prompt_template  # Use the renamed variable
		self.specialized_prompts = specialized_prompts if specialized_prompts else {}

	async def execute(self, task: Task, context: ContextManager) -> DeduceExecutorOutput:
		logic_input = task.logic_input;
		reasoning_goal = logic_input.get("reasoning_goal")
		raw_context_data = logic_input.get("context_data");
		operation_type = logic_input.get("operation_type")

		if not reasoning_goal or not isinstance(reasoning_goal, str) or raw_context_data is None:
			raise ExecutorError("DeduceExecutor: 'reasoning_goal' (string) and 'context_data' are required.")

		resolved_context_data = self._resolve_references(raw_context_data, context)
		context_data_str = json.dumps(resolved_context_data, ensure_ascii=False, indent=2) if isinstance(
			resolved_context_data, (list, dict)) else str(resolved_context_data)

		prompt_to_use = self.default_prompt_template
		system_prompt_to_use = DeducePrompt.SYSTEM_PROMPT  # Default system prompt
		if operation_type and operation_type in self.specialized_prompts:
			prompt_to_use = self.specialized_prompts[operation_type]
			system_prompt_to_use = getattr(prompt_to_use, "SYSTEM_PROMPT",
										   system_prompt_to_use)  # Specialized system prompt if available
			print(f"  [DeduceExecutor] Using specialized prompt for op_type: {operation_type}")

		prompt_str = prompt_to_use.format(reasoning_goal=reasoning_goal, context_data=context_data_str)
		current_thought = task.thought or ""
		task.thought = f"{current_thought}演绎目标({operation_type or 'default'}): {reasoning_goal}. 上下文(摘要): {context_data_str[:100]}...".strip()

		# DeduceExecutor's LLM now needs to output JSON
		response_json = await self.llm_client.generate_structured_json(prompt_str,
																	   system_prompt_str=system_prompt_to_use,
																	   temperature=0.0)  # Low temp for structured

		# Validate and structure the output
		answer_summary = response_json.get("answer_summary", "未能从LLM响应中解析出答案。")
		is_sufficient = response_json.get("is_sufficient", False)  # Default to False if not specified
		new_qs = response_json.get("new_questions_or_entities", [])
		if not isinstance(new_qs, list): new_qs = [str(new_qs)] if new_qs else []  # Ensure it's a list of strings

		task.thought += f"\nLLM演绎响应(结构化): sufficient={is_sufficient}, new_qs={new_qs}, summary={answer_summary[:50]}..."

		return {
			"answer_summary": answer_summary,
			"is_sufficient": is_sufficient,
			"new_questions_or_entities": new_qs,
			"raw_llm_response": json.dumps(response_json)  # Store the full JSON response string
		}

	def get_schema(self) -> Dict[str, Any]:  # (与之前相同)
		return {"name": "DeduceExecutor",
				"description": "基于提供的上下文信息进行推理、总结、判断或抽取。上下文可引用先前步骤的结果。会判断信息是否充分并给出下一步查询建议。",
				"logic_input_schema": {"reasoning_goal": "string (具体推理目标)",
									   "context_data": "any (推理所需上下文，可引用 {{task_id.result}})",
									   "operation_type": "string (可选, 如 summarize, extract_info, judge, refine_query)"}}


class CodeExecutor(ExecutorBase):  # (与之前相同)
	def __init__(self, llm_client: OpenAIChatLLM, prompt: CodeExecutionPrompt):
		super().__init__(llm_client); self.prompt_template = prompt

	async def execute(self, task: Task, context: ContextManager) -> str:
		logic_input = task.logic_input;
		code_gen_prompt = logic_input.get("code_generation_prompt");
		rel_data = logic_input.get("relevant_data", "")
		if not code_gen_prompt or not isinstance(code_gen_prompt, str): raise ExecutorError(
			"CodeExecutor: 'code_generation_prompt' required.")
		res_code_prompt = self._resolve_references(code_gen_prompt, context);
		res_rel_data = self._resolve_references(rel_data, context)
		rel_data_prompt = json.dumps(res_rel_data, ensure_ascii=False, indent=2) if isinstance(res_rel_data,
																							   (list, dict)) else str(
			res_rel_data) if res_rel_data else ""
		llm_prompt = self.prompt_template.format(code_generation_prompt=res_code_prompt, relevant_data=rel_data_prompt)
		task.thought = (task.thought or "") + f"代码生成目标: {res_code_prompt}. ".strip()
		code_block = await self.llm_client.generate(llm_prompt, system_prompt_str=CodeExecutionPrompt.SYSTEM_PROMPT)
		code = code_block.strip();
		if code.startswith("```python"): code = code[9:]
		if code.startswith("```"): code = code[3:]
		if code.endswith("```"): code = code[:-3]
		code = code.strip()
		if not code: task.thought += "\nLLM未能生成代码."; raise ExecutorError("CodeExecutor: LLM no code.")
		task.thought += f"\n生成的代码:\n---\n{code}\n---"
		try:
			with open("t.py", "w", encoding="utf-8") as f:
				f.write(code)
			p = await asyncio.to_thread(subprocess.run, [sys.executable, "t.py"], capture_output=True, text=True,
										timeout=10, check=False)
			if p.returncode != 0: task.thought += f"\n代码错误码{p.returncode}. stderr:\n{p.stderr or '无'}"; raise ExecutorError(
				f"Code exec err {p.returncode}:\n{p.stderr or '无'}")
			out = p.stdout.strip();
			task.thought += f"\n代码输出: {out}";
			return out
		except subprocess.TimeoutExpired:
			task.thought += "\n代码超时."; raise ExecutorError("Code timeout.")
		except Exception as e:
			task.thought += f"\n代码本地执行错误: {e}"; raise ExecutorError(f"Code local exec err: {e}")
		finally:
			if os.path.exists("t.py"): os.remove("t.py")

	def get_schema(self) -> Dict[str, Any]:
		return {"name": "CodeExecutor",
				"description": "生成并执行Python代码。代码应print()结果。输入可引用 {{task_id.result}}。",
				"logic_input_schema": {"code_generation_prompt": "string (代码生成指令)",
									   "relevant_data": "any (可选, 代码所需数据)"}}


class FinishExecutor(ExecutorBase):  # (与之前相同)
	async def execute(self, task: Task,
					  context: ContextManager) -> str: task.thought = "收到Finish指令，流程结束。"; print(
		f"  [FinishExecutor] Task {task.id} executed."); return "已完成所有必要步骤。"

	def get_schema(self) -> Dict[str, Any]: return {"name": "FinishAction",
													"description": "当问题已解决或无法继续时调用此动作结束规划。",
													"logic_input_schema": {"reason": "string (可选，结束原因)"}}


# --- Planner (与之前相同，使用新的PlannerPrompt) ---
class Planner:
	def __init__(self, llm_client: OpenAIChatLLM, prompt: PlannerPrompt):
		self.llm_client = llm_client; self.prompt_template = prompt

	async def plan_next_steps(self, user_query: str, context: ContextManager, available_executors: List[Dict]) -> Tuple[
		str, str, List[Dict]]:
		exec_desc_parts = [
			f"  - 名称: \"{s['name']}\"\n    描述: \"{s['description']}\"\n    输入参数模式 (logic_input_schema): {json.dumps(s.get('logic_input_schema', 'N/A'), ensure_ascii=False)}"
			for s in available_executors]
		exec_desc = "\n".join(exec_desc_parts)
		history_str = context.get_task_history_for_prompt()
		user_prompt_str = self.prompt_template.format(user_query=user_query, available_executors_description=exec_desc,
													  task_history=history_str)
		# print(f"  [Planner] Sending planning prompt (history length: {len(history_str)} chars)...")
		response_json = await self.llm_client.generate_structured_json(user_prompt_str,
																	   system_prompt_str=PlannerPrompt.SYSTEM_PROMPT,
																	   temperature=0.0)  # Planner needs to be deterministic
		plan_status = response_json.get("plan_status", "error");
		final_thought = response_json.get("final_thought", "LLM未能提供规划思考。");
		next_steps_data = response_json.get("next_steps", [])
		if not isinstance(next_steps_data, list): print(
			f"  [Planner Err] LLM 'next_steps' not list. Got: {next_steps_data}. Assuming none."); next_steps_data = []
		valid_steps = [td for td in next_steps_data if isinstance(td, dict) and all(
			k in td for k in ["id", "executor_name", "task_description", "logic_input"])]
		if len(valid_steps) != len(next_steps_data): print(f"  [Planner Warn] Some planned steps were invalid.")
		return plan_status, final_thought, valid_steps


# --- AnswerGenerator (与之前相同，使用UserProvidedReferGeneratorPrompt) ---
class AnswerGenerator:
	def __init__(self, llm_client: OpenAIChatLLM,
				 prompt: UserProvidedReferGeneratorPrompt): self.llm_client = llm_client; self.prompt_template = prompt

	async def generate_final_answer(self, user_query: str, context: ContextManager) -> str:
		summary = context.get_summary_for_generator();
		refs = context.collect_retrieved_references_for_generator()
		prompt_str = self.prompt_template.format(summary_of_executed_steps=summary, user_query=user_query,
												 retrieved_references=refs)
		# print(f"  [Generator] Sending generation prompt (summary length: {len(summary)}, refs: {len(refs)})...")
		return await self.llm_client.generate(
			prompt_str)  # System prompt might be part of UserProvidedReferGeneratorPrompt's template


# --- Pipeline (与之前相同，使用新的迭代逻辑) ---
class IterativePipeline:
	def __init__(self, planner: Planner, executors: Dict[str, ExecutorBase], generator: AnswerGenerator,
				 max_iterations: int = 5):
		self.planner = planner;
		self.executors = executors;
		self.generator = generator;
		self.max_iterations = max_iterations

	async def _execute_task_dag_segment(self, tasks_data: List[Dict], ctx: ContextManager, iter_num: int) -> bool:
		if not tasks_data: return True
		prefix = re.sub(r'[^\w\s-]', '', ctx.user_query[:10]).strip().replace(' ', '_') or "q"
		added_ids = [ctx.add_task_from_planner(td, prefix, iter_num, i).id for i, td in enumerate(tasks_data)]
		# print(f"  [Pipeline Iter {iter_num}] Added tasks: {added_ids}")
		# for tid in added_ids: task = ctx.get_task(tid); print(f"    - ID:{task.id}, Exec:{task.executor_name}, Desc:\"{task.task_description}\", Deps:{task.dependencies}")
		cache = set();
		success = True
		for tid in added_ids:
			task = ctx.get_task(tid)
			if task and task.status == "pending":
				await self._execute_task_with_dependencies(tid, ctx, cache)
				if ctx.get_task(tid).status != "completed": success = False
		return success

	async def _execute_task_with_dependencies(self, task_id: str, ctx: ContextManager, cache: set):
		task = ctx.get_task(task_id)
		if not task: print(f"  [Pipe Err] Task {task_id} def not found."); return
		if task.status != "pending": return
		cache.add(task_id)
		for dep_id in task.dependencies:
			dep_task = ctx.get_task(dep_id)
			if not dep_task: emsg = f"T {task.id} undef dep ID: {dep_id}. Failed."; ctx.update_task_status(task.id,
																										   "failed",
																										   result=emsg,
																										   thought=emsg); print(
				f"  [Pipe Err] {emsg}"); return
			if dep_task.status == "pending": await self._execute_task_with_dependencies(dep_id, ctx, cache)
		deps_met = True
		if task.dependencies:
			for dep_id in task.dependencies:
				dep_task = ctx.get_task(dep_id)
				if not dep_task or dep_task.status != "completed":
					deps_met = False;
					ts = (
									 task.thought or "") + f"Dep {dep_id} not met (status: {dep_task.status if dep_task else 'N/A'}). Skip."
					ctx.update_task_status(task.id, "skipped", thought=ts);
					print(f"  [Pipe] Task {task.id} skip, dep {dep_id} not met.");
					break
			if not deps_met: return
		if task.status != "pending": return
		executor = self.executors.get(task.executor_name)
		if not executor: emsg = f"Exec '{task.executor_name}' not found for task '{task.task_description}'."; ctx.update_task_status(
			task.id, "failed", result=emsg, thought=emsg); print(f"  [Pipe Err] {emsg}"); return
		print(f"\n▶️ Iter Exec Task: {task.id} - \"{task.task_description}\" ({task.executor_name})")
		ctx.update_task_status(task.id, "running", thought=f"Start: {task.task_description}")
		try:
			result = await executor.execute(task, ctx)
			ctx.update_task_status(task.id, "completed", result=result, thought=task.thought)
		# print(f"✅ Task {task.id} Result (short): {str(result)[:100]}{'...' if result and len(str(result)) > 100 else ''}")
		except ExecutorError as e:
			emsg = f"ExecErr T {task.id}: {e}"; ft = (task.thought or "") + f"\nExecErr: {e}"; ctx.update_task_status(
				task.id, "failed", result=emsg, thought=ft); print(f"🛑 {emsg}")
		except Exception as e:
			emsg = f"UnexpectedErr T {task.id}: {e}"; import traceback; tb = traceback.format_exc(); print(
				f"🛑 {emsg}\n{tb}"); ft = (task.thought or "") + f"\nUnexpectedErr: {e}"; ctx.update_task_status(task.id,
																												"failed",
																												result=emsg,
																												thought=ft)

	# In class IterativePipeline:
	async def run(self, user_query: str) -> str:
		print(f"\n🚀 IterativePipeline starting for query: \"{user_query}\"")
		context = ContextManager(user_query)
		available_executors_schemas = [ex.get_schema() for ex in self.executors.values()]

		current_plan_status = "requires_more_steps"  # Initial status
		i_iter = 0  # Initialize iteration counter

		for i_iter in range(self.max_iterations):
			current_iteration_num = i_iter + 1
			print(f"\n--- Iteration {current_iteration_num} / {self.max_iterations} ---")

			print(f"📝 Planning phase (Iteration {current_iteration_num})...")
			try:
				plan_status, planner_thought, next_steps_data = await self.planner.plan_next_steps(
					user_query, context, available_executors_schemas
				)
				print(f"  [Planner Output] Status: {plan_status}, Thought: {planner_thought}")
				if next_steps_data:
					print(
						f"  [Planner Output] Next Steps Planned ({len(next_steps_data)}): {[s.get('task_description', 'N/A') for s in next_steps_data]}")
				else:
					print("  [Planner Output] No new steps planned for this iteration.")

				current_plan_status = plan_status  # Update overall plan status

			# Log planner's thought, but don't let it become the final answer directly
			# context.update_task_status(f"planner_thought_iter_{current_iteration_num}", "info", thought=planner_thought) # Optional logging

			# ... (error handling for planning as before) ...
			except Exception as e:
				# ... (error handling) ...
				break  # Break from loop on planning error

			if current_plan_status == "finished":
				print("  [Pipeline] Planner determined the process is finished. Proceeding to generation.")
				break
			if current_plan_status == "cannot_proceed":
				print("  [Pipeline] Planner determined it cannot proceed.")
				# final_answer will be set by generator based on current context and this thought
				# We can pass this thought to the generator via context if needed, or generator just uses task history
				# For now, let generator synthesize based on available task results.
				break

			if not next_steps_data:
				print(
					"  [Pipeline] No new steps planned by Planner, and not explicitly finished. Will proceed to generation with current context.")
				break

			print(f"\n⚙️ Execution phase (Iteration {current_iteration_num})...")
			segment_successful = await self._execute_task_dag_segment(next_steps_data, context, current_iteration_num)

			if not segment_successful:
				print(
					f"  [Pipeline Warning] Iteration {current_iteration_num} encountered errors. Planner will try to adapt.")

			finish_task_executed = any(
				task.executor_name == "FinishAction" and task.status == "completed"
				for task in context.tasks.values() if task.id in [data["id"] for data in next_steps_data]
			)
			if finish_task_executed:
				print(f"  [Pipeline] FinishAction task completed. Ending iterations.")
				current_plan_status = "finished"  # Ensure status reflects finish
				break

		# --- Generation Phase ---
		# This phase is ALWAYS called after the loop finishes or breaks,
		# unless a critical unrecoverable error occurred earlier.
		print(f"\n💬 Generation phase after {i_iter + 1} iteration(s) (or fewer if loop broke early)...")
		final_answer = "未能生成最终答案。"  # Default if generator fails
		try:
			# If planner said "cannot_proceed", its thought is valuable context for the generator
			if current_plan_status == "cannot_proceed" and planner_thought:
				# We can prepend this to the summary or pass it specially
				# For now, let the generator work with the task history; planner_thought is mainly for control flow.
				print(f"  [Generator] Note: Planner indicated 'cannot_proceed'. Reason: {planner_thought}")

			final_answer = await self.generator.generate_final_answer(user_query, context)
		except RuntimeError as rte:
			print(
				f"  [Pipeline Error] Generation LLM call failed: {rte}"); final_answer = f"生成答案时连接LLM失败：{rte}"
		except Exception as e:
			print(f"  [Pipeline Error] Unexpected error during generation: {e}"); import \
				traceback; traceback.print_exc(); final_answer = f"生成答案阶段意外错误：{e}"

		print(f"\n💡 Final Answer: {final_answer}")
		return final_answer


# _execute_task_dag_segment 和 _execute_task_with_dependencies 保持不变
# ... (rest of the IterativePipeline class as before) ...

# --- 主程序入口 ---
async def run_main_logic_with_user_data_recursive_optimized():
	# --- LLM and Embedding Setup ---
	# (Same as previous, ensure env vars are set: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL_NAME, DASHSCOPE_API_KEY_FOR_EMBEDDING)
	api_key = 'sk-af4423da370c478abaf68b056f547c6e'
	base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
	if "YOUR_API_KEY" in api_key: print("错误：请设置有效的 OPENAI_API_KEY。"); return
	llm_client = OpenAIChatLLM(model_name=model_name, api_key=api_key, base_url=base_url)
	try:
		from tongyiembedding import QwenEmbeddingFunction

		embedding_function = QwenEmbeddingFunction(api_key='sk-af4423da370c478abaf68b056f547c6e')
	except Exception as e:
		print(f"  [Embedding Error] QwenEmbeddingFunction 初始化失败: {e}"); return

	# --- Knowledge Base Data ---
	initial_user_docs_as_dicts = [
		{"page_content": "【正柴胡饮颗粒】检查: 应符合颗粒剂项下有关的各项规定（通则0104)。这是主要检查依据。",
		 "metadata": {"id": "doc_zchyk_check", "source_name": "正柴胡饮颗粒说明书-检查章节"}},
		{
			"page_content": "药典通则0104 - 颗粒剂检查要点：【粒度】要求不能通过一号筛与能通过五号筛的总和不得超过15％。【水分】中药颗粒剂水分不得超过8.0％。【溶化性】可溶颗粒5分钟内全部溶化或呈轻微浑浊；泡腾颗粒5分钟内完全分散或溶解。均不得有异物，中药颗粒还不得有焦屑。【装量差异】单剂量包装与平均装量比较，差异需在限度内，例如1g以下为±10%。【微生物限度】需符合非无菌产品微生物限度标准。",
			"metadata": {"id": "doc_tongze0104_summary_v2", "source_name": "药典通则0104核心摘要"}},
		{
			"page_content": "通则0104详细说明之【水分测定】：采用甲苯法或减压干燥法。对于含糖或易熔化辅料的颗粒，宜在较低温度（如60-80℃）减压干燥至恒重。",
			"metadata": {"id": "doc_tongze0104_water", "source_name": "药典通则0104-水分测定细节"}},
		{
			"page_content": "通则0104详细说明之【粒度分布】：使用标准药筛进行筛分，记录各筛上物及筛下物的重量百分比。对于难溶性药物，需注意其在特定介质中的分散性。",
			"metadata": {"id": "doc_tongze0104_size", "source_name": "药典通则0104-粒度分布细节"}}
	]
	initial_langchain_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in
							  initial_user_docs_as_dicts]
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
		chroma_kb = ChromaKnowledgeBase(initial_documents=initial_langchain_docs, embedding_function=embedding_function,
										force_rebuild=False,
										persist_directory = 'chroma_db_kag_recursive_1')
	except Exception as e:
		print(f"创建Chroma知识库失败: {e}"); import traceback; traceback.print_exc(); return

	# --- Initialize Components ---
	planner_prompt = PlannerPrompt()  # Uses the new reflective prompt
	deduce_prompt_template = DeducePrompt()  # The one that asks for structured output
	code_exec_prompt = CodeExecutionPrompt()
	refer_generator_prompt = UserProvidedReferGeneratorPrompt(language="zh")

	retrieval_executor = RetrievalExecutor(kb=chroma_kb)
	deduce_executor = DeduceExecutor(llm_client, deduce_prompt_template)  # Default deduce prompt
	code_executor = CodeExecutor(llm_client, code_exec_prompt)
	finish_executor = FinishExecutor()

	executors_map = {
		"RetrievalExecutor": retrieval_executor, "DeduceExecutor": deduce_executor,
		"CodeExecutor": code_executor, "FinishAction": finish_executor
	}

	planner = Planner(llm_client, planner_prompt)
	generator = AnswerGenerator(llm_client, refer_generator_prompt)
	pipeline = IterativePipeline(planner, executors_map, generator, max_iterations=4)  # Max 4 iterations

	# --- Run Query ---
	# user_query_to_run = "正柴胡饮颗粒的检查内容有哪些方面？请详细说明。"
	user_query_to_run = "正柴胡饮颗粒的主要检查标准是什么？"

	print(f"\n🚀 Running RECURSIVE-LIKE optimized query: \"{user_query_to_run}\"")
	final_answer = await pipeline.run(user_query_to_run)
	print(f"\n🏁🏁🏁🏁🏁 RECURSIVE-LIKE FINAL ANSWER (for query: '{user_query_to_run}') 🏁🏁🏁🏁🏁\n{final_answer}")


if __name__ == "__main__":
	print("开始执行“类递归”优化版主逻辑...")
	# ... (环境变量和依赖提示) ...
	# Forcing API Key for this test run in case environment variables are not set by user.
	# USER SHOULD REPLACE THIS WITH THEIR ACTUAL KEY OR ENV VARS.
	# THIS IS NOT SAFE FOR PRODUCTION.

	asyncio.run(run_main_logic_with_user_data_recursive_optimized())
