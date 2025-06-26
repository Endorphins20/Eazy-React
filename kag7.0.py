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


# --- ChromaKnowledgeBase (与您之前运行成功的版本一致) ---
class ChromaKnowledgeBase:  # ... (代码与您上一轮成功运行的版本相同，此处省略以减少篇幅) ...
	def __init__(self, embedding_function: Callable, initial_documents: Optional[List[Document]] = None,
				 persist_directory: str = CHROMA_PERSIST_DIRECTORY, collection_name: str = CHROMA_COLLECTION_NAME,
				 force_rebuild: bool = False):
		print(f"  [ChromaKB] 初始化知识库: {persist_directory}, 集合: {collection_name}")
		self.embedding_function = embedding_function;
		self.persist_directory = persist_directory
		self.collection_name = collection_name;
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

	def add_documents(self, documents: List[Document]):
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

	def retrieve(self, query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
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
class OpenAIChatLLM:  # (与之前版本相同)
	def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
		try:
			import openai
		except ImportError:
			raise ImportError("OpenAI library not found. `pip install openai`.")
		self.api_key = api_key or os.getenv("OPENAI_API_KEY")
		self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
		self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "deepseek-r1-distill-qwen-32b")
		if not self.api_key: raise ValueError("API key not found.")
		self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
		print(f"[OpenAIChatLLM] 客户端就绪: 模型 {self.model_name}, URL {self.base_url or 'OpenAI default'}")

	async def _make_api_call(self, messages: List[Dict[str, str]], expect_json: bool = False, temperature: float = 0.01,
							 **kwargs) -> str:  # 更低的temperature
		try:
			completion_params = {"model": self.model_name, "messages": messages, "temperature": temperature, **kwargs}
			if expect_json:
				if "dashscope" in (self.base_url or "").lower() and self.model_name.startswith("qwen"):
					completion_params["extra_body"] = {"result_format": "message"}
				else:
					completion_params["response_format"] = {"type": "json_object"}
			response = await asyncio.to_thread(self.client.chat.completions.create, **completion_params)
			content = response.choices[0].message.content;
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
									   temperature: float = 0.01, **kwargs) -> Dict:  # 更低的temperature
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
			error_msg = f"LLM did not return valid JSON. Error: {e}. Raw: '{response_str}'";
			print(f"  [OpenAIChatLLM Error] {error_msg}");
			raise ValueError(error_msg)


# --- Prompts ---
class BasePrompt:  # (与之前相同)
	def __init__(self, template: str, variables: List[str]):
		self.template_str = template; self.variables = variables

	def format(self, **kwargs) -> str:
		formatted_prompt = self.template_str
		for var_name in self.variables:
			if var_name not in kwargs: raise ValueError(f"Missing variable: {var_name}")
			formatted_prompt = formatted_prompt.replace(f"${{{var_name}}}", str(kwargs[var_name]))
		return formatted_prompt


class PlannerPrompt(BasePrompt):  # REFINED: For deeper recursive-like planning
	SYSTEM_PROMPT = """
    你是一位高度智能且具备卓越反思能力的AI任务规划专家。
    你的核心目标是：根据用户提出的复杂问题 `${user_query}` 和已经执行的历史任务 `${task_history}`，分析当前状态，并决定下一步【最优行动】。
    你必须严格按照以下JSON格式输出你的决策和计划：

    **输出JSON格式:**
    ```json
    {
      "plan_status": "finished" | "cannot_proceed" | "requires_more_steps",
      "final_thought": "string (对当前整体进展的清晰思考和判断，解释你的plan_status。例如，如果requires_more_steps，清晰说明还需要什么具体信息以及下一步的核心目标)",
      "next_steps": [ // 仅当 plan_status == "requires_more_steps" 时非空
        {
          "id": "string (新任务的唯一ID, 例如 task_iter2_step0, 确保与历史ID不冲突)",
          "executor_name": "string (从可用工具列表中选择)",
          "task_description": "string (对此新任务目标的清晰、具体中文描述)",
          "logic_input": { /* 取决于 executor_name */ },
          "dependencies": ["string"] // 依赖的历史任务ID列表
        }
        // ... 通常一次只规划1个，最多2个高度相关的后续步骤 ...
      ]
    }
    ```

    **核心指令与思考链 (Chain-of-Thought for Reflective Planning):**
    1.  **回顾目标 (Recall Goal)**: 清晰、完整地理解用户原始问题 `${user_query}` 的每一个细节和最终期望。
    2.  **审视历史 (Analyze History - `${task_history}`中每个任务的 `Result Details`)**:
        * **信息提取**: 哪些子问题已被回答？答案 (`answer_summary`) 的【具体内容】是什么？
        * **【关键评估点 - 严格检查】**: 最近的 `DeduceExecutor` 步骤的结果中：
            * `is_sufficient` 是否为 `false`？
            * 如果为 `false`，它在其 `new_questions_or_entities` 列表中明确列出了哪些【具体的、尚未被充分探究的标准编号、文件名、实体名或子问题】？这些是解决问题的【核心待办线索】！
        * `RetrievalExecutor` 检索到了哪些信息？这些信息是否【真的】已被后续的 `DeduceExecutor` 步骤【彻底地、针对性地】分析过（特别是当用户问题指向这些检索内容的细节时）？
    3.  **差距评估 (Gap Assessment)**: 对比当前所有已知信息（来自历史任务的成功结果）和用户原始问题的最终目标（特别是那些要求“详细说明”、“具体要求”的部分），目前还缺少哪些【具体的细节】或【明确被引用的标准/文件（如通则XXXX）的详细内容】？
    4.  **决策制定 (`final_thought` 和 `plan_status`)**:
        * **已解决 (finished)?** 【严格条件】当且仅当：所有用户原始问题中明确要求的方面都得到了解答，所有在推理过程中被识别为需要进一步探究的 `new_questions_or_entities`（尤其是标准/通则的细节）都已经被成功检索其详细内容、并被后续的 `DeduceExecutor` 步骤分析确认信息充分（`is_sufficient: true`）后，才可判断为 `finished`。此时，规划一个 `FinishAction` 任务。`final_thought` 应总结是如何一步步解决的。
        * **无法解决 (cannot_proceed)?** 若关键信息缺失，且历史中的 `new_questions_or_entities` 已尝试检索但无果（例如，多次相关的 `RetrievalExecutor` 对特定标准编号的查询返回空或不相关内容），或工具无法获取，则 `plan_status: "cannot_proceed"`。`final_thought` 应解释原因。
        * **需要更多步骤 (requires_more_steps)?** 否则，即信息尚不完整，或有新的线索需要追查，则 `plan_status: "requires_more_steps"`。
    5.  **行动规划 (`next_steps` - 如果 `requires_more_steps`)**:
        * **【最高优先级：递归深挖引用和新线索】**: 
            * 如果历史中有 `DeduceExecutor` 的结果在其 `new_questions_or_entities` 列表中明确指出了需要查询【特定标准、文件编号或实体】的详细内容（例如，“查询通则1107的详细内容”，“获取《中国药典》通则1107中关于微生物限度的具体标准是什么？”），并且这些条目【对应的详细内容尚未通过后续的 `RetrievalExecutor` 成功获取并被充分分析过】：
                * **立即规划**一个 `RetrievalExecutor` 任务。其 `logic_input.query` 应【直接针对这些 `new_questions_or_entities` 中的一项进行精确查询】。例如，如果 `new_questions_or_entities` 中有“通则1107中关于微生物限度的具体标准是什么？”，则检索查询就应该是这个或非常相似的，以确保能命中包含该通则细节的文档。
                * 这是当前迭代最优先要处理的任务。通常，处理完一个这样的关键缺失信息点后，就应该结束本轮规划，等待下一轮迭代基于新获取的信息进行再规划和整合。
        * **整合信息**: 如果上一步是检索（特别是针对 `new_questions_or_entities` 的检索），并且检索到了有价值的新信息，下一步通常是规划 `DeduceExecutor` 任务。其 `logic_input.reasoning_goal` 设为“整合新检索到的关于[新实体/标准]的信息，并结合先前关于[相关主题]的结论，以更全面地回答[某个子问题或原始问题中与此新信息相关的方面]”。`logic_input.context_data` 应精确引用【所有相关的】检索结果和先前步骤的关键结论。
        * **初始规划/其他探索**: 若无明确的 `new_questions_or_entities` 指引，则根据对用户问题的理解进行初步的 `RetrievalExecutor` 或 `DeduceExecutor` 规划。
        * **聚焦**: 仍然建议一次只规划1个核心步骤（可能是一个检索 + 一个后续的演绎，或者只是一个关键的补充检索）。
    6.  **ID 和依赖**: 为新任务分配唯一的 `id`。正确设置 `dependencies`。
    7.  **严格JSON输出**。
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

	def __init__(self): super().__init__(self.USER_TEMPLATE,
										 ["user_query", "available_executors_description", "task_history"])


class DeducePrompt(BasePrompt):  # REFINED: Very strong guidance on new_questions for referenced standards
	SYSTEM_PROMPT = """
    你是一位极其严谨和细致的AI推理专家。你的任务是根据提供的“上下文信息”，精确地回答或完成“推理目标”。
    你必须对上下文中出现的所有【标准、通则、法规、文件编号、或被明确引用的专有名称】（例如“通则0104”、“中华人民共和国药典通则1107”、“GB/T标准X”）进行严格的审视。

    请严格按照以下JSON格式输出你的结论和评估：
    ```json
    {
      "answer_summary": "string (对推理目标的直接、简洁的回答或总结。如果信息不足，明确说明当前已知什么，并【清晰指出具体还缺少哪些信息或哪些被引用的标准/文件细节】才能完整回答推理目标。例如：'根据上下文，正柴胡饮颗粒的微生物限度应参照通则1107，但当前未提供通则1107的具体限值。')",
      "is_sufficient": boolean (true 如果你认为提供的上下文信息【已包含回答推理目标所需的所有必要细节，特别是所有被明确引用或提及的标准/文件的具体内容，无需任何补充查询】，否则 false),
      "new_questions_or_entities": [
        "string" // 【至关重要指令 - 必须严格执行】:
                  // 1. 如果 is_sufficient 为 false，这里【必须】列出为了获得【缺失的关键细节】需要查询的具体问题。
                  // 2. 【特别注意】：如果在上下文中发现任何【标准、通则、法规、文件编号或专有名称】被明确引用（例如“详见通则1107”、“依据GB/T标准X执行”、“按照XYZ操作手册进行”），但该标准/文件/名称的【具体内容、定义或详细条款并未在当前提供给你的上下文中清晰、完整地列出】，那么你【必须】将一个用于获取该标准/文件/名称【详细内容】的明确查询（例如：“查询《中国药典》通则1107关于微生物限度的详细规定”、“获取GB/T标准X的全文”、“查找XYZ操作手册中关于XX部分的具体步骤”）作为条目加入此列表。并且，在这种情况下，除非该引用对于当前推理目标而言完全不重要，否则 `is_sufficient` 通常应为 `false`。
                  // 确保列表中的每一项都是一个明确的、可用于下一步检索的查询目标。如果信息完全充分且所有引用都有细节支撑，则为空列表[]。
      ]
    }
    ```
    - 不要添加任何额外的解释或markdown标记，只输出JSON对象。
    """
	USER_TEMPLATE = "推理目标:\n${reasoning_goal}\n\n上下文信息:\n${context_data}\n\n请输出JSON格式的推理结果:"

	def __init__(self): super().__init__(self.USER_TEMPLATE, ["reasoning_goal", "context_data"])


class CodeExecutionPrompt(BasePrompt):  # (与之前相同)
	SYSTEM_PROMPT = "你是一个Python代码生成和执行助手。"
	USER_TEMPLATE = "请根据以下指令和相关数据，生成一段Python代码来解决问题。\n代码必须通过 `print()` 输出其最终计算结果。不要包含任何解释或注释，只输出纯代码。\n\n指令:\n${code_generation_prompt}\n\n相关数据 (如果提供):\n${relevant_data}\n\n生成的Python代码 (请确保它只包含代码本身，并用print()输出结果):"

	def __init__(self): super().__init__(self.USER_TEMPLATE, ["code_generation_prompt", "relevant_data"])


class UserProvidedReferGeneratorPrompt(BasePrompt):  # (与之前相同)
	def __init__(self, language: str = "zh"):
		try:
			from kag.common.utils import get_now
		except ImportError:
			_get_now_imported = False

			def get_now(language='zh'):
				return "当前日期"

			if not hasattr(UserProvidedReferGeneratorPrompt, '_get_now_imported_warning_shown'):
				print("[UserProvidedReferGeneratorPrompt] Warning: kag.common.utils.get_now not found.")
				UserProvidedReferGeneratorPrompt._get_now_imported_warning_shown = True
		self.template_zh = (
					f"你是一个信息分析专家，今天是{get_now(language='zh')}。" + "基于给定的引用信息回答问题。\n输出答案，如果答案中存在引用信息，则需要reference的id字段，如果不是检索结果，则不需要标记引用\n输出时，不需要重复输出参考文献\n引用要求，使用类似<reference id=\"chunk:1_2\"></reference>表示\n如果根据引用信息无法回答，则使用模型内的知识回答，但是必须通过合适的方式提示用户，是基于检索内容还是引用文档\n示例1：\n任务过程上下文：\n根据常识岳父是妻子的爸爸，所以需要首先找到张三的妻子，然后找到妻子的爸爸\n给定的引用信息：'\nreference：\n[\n{\n    \"content\": \"张三 妻子 王五\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_1\"\n},\n{\n    \"content\": \"王五 父亲 王四\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_2\"\n}\n]'\n问题：'张三的岳父是谁？'\n\n张三的妻子是王五<reference id=\"chunk:1_1\"></reference>，而王五的父亲是王四<reference id=\"chunk:1_2\"></reference>，所以张三的岳父是王四\n\n\n输出语调要求通顺，不要有机械感，输出的语言要和问题的语言保持一致\n任务过程上下文信息：'${summary_of_executed_steps}'\n给定的引用信息：'${formatted_references}'\n问题：'${user_query}'")
		self.template_en = self.template_zh
		current_template = self.template_zh if language == "zh" else self.template_en
		super().__init__(current_template, ["summary_of_executed_steps", "user_query", "formatted_references"])

	def format(self, summary_of_executed_steps: str, user_query: str, retrieved_references: List[Dict]) -> str:
		ref_list_for_prompt = []
		for i, ref_item in enumerate(retrieved_references): ref_list_for_prompt.append(
			{"content": ref_item.get("content", ""),
			 "document_name": ref_item.get("metadata", {}).get("source_name", f"检索文档{i + 1}"),
			 "id": ref_item.get("metadata", {}).get("id", f"retrieved_chunk_{i}")})
		formatted_references_str = json.dumps(ref_list_for_prompt, ensure_ascii=False, indent=2)
		return super().format(summary_of_executed_steps=summary_of_executed_steps, user_query=user_query,
							  formatted_references=formatted_references_str)


# --- DataStructures & ContextManager (与之前相同) ---
LogicInput = Dict[str, Any]


@dataclass
class Task: id: str; executor_name: str; task_description: str; logic_input: LogicInput; dependencies: List[
	str] = field(default_factory=list); status: str = "pending"; result: Optional[Any] = None; thought: Optional[
	str] = None


class ContextManager:  # (与之前版本相同)
	def __init__(self, user_query: str):
		self.user_query = user_query
		self.tasks: Dict[str, Task] = {}
		self.execution_order: List[str] = []

	def add_task_from_planner(self, task_data: Dict, base_id_prefix: str, iter_num: int, step_in_iter: int) -> Task:
		llm_id = task_data.get("id");
		task_id = llm_id if llm_id and isinstance(llm_id,
												  str) and llm_id.strip() else f"{base_id_prefix}_iter{iter_num}_step{step_in_iter}"
		orig_id = task_id;
		ctr = 0
		while task_id in self.tasks: ctr += 1; task_id = f"{orig_id}_v{ctr}"
		if ctr > 0: print(f"  [CM Warn] Task ID '{orig_id}' conflict. Renamed '{task_id}'.")
		task = Task(id=task_id, executor_name=task_data["executor_name"],
					task_description=task_data["task_description"], logic_input=task_data["logic_input"],
					dependencies=task_data.get("dependencies", []));
		self.tasks[task.id] = task;
		self.execution_order.append(task.id);
		return task

	def get_task(self, task_id: str) -> Optional[Task]:
		return self.tasks.get(task_id)

	def update_task_status(self, task_id: str, status: str, result: Optional[Any] = None,
						   thought: Optional[str] = None):
		task = self.get_task(task_id)
		if task:
			task.status = status; task.result = result if result is not None else task.result; task.thought = (
																														  task.thought or "") + f"\n{thought}".strip() if thought and (
						task.thought or "") else thought or task.thought
		else:
			print(f"  [CM Warn] Task ID {task_id} not found for status update.")

	def get_task_history_for_prompt(self) -> str:  # (与之前版本相同)
		hist = [];
		truncate_len = 150  # Increased truncation slightly for more context
		for tid in self.execution_order:
			t = self.get_task(tid)
			if t and t.status in ["completed", "failed"]:
				res_disp_parts = [];
				if t.status == "completed":
					if isinstance(t.result, dict) and t.executor_name == "DeduceExecutor":
						d_out = t.result
						res_disp_parts.append(
							f"    推理总结: {str(d_out.get('answer_summary', 'N/A'))[:truncate_len]}{'...' if len(str(d_out.get('answer_summary', 'N/A'))) > truncate_len else ''}")
						res_disp_parts.append(
							f"    信息是否充分: {d_out.get('is_sufficient', True)}")  # Default to True if missing, though Deduce should always provide it
						new_q = d_out.get('new_questions_or_entities', [])
						if new_q: res_disp_parts.append(
							f"    建议进一步查询: {', '.join(new_q)[:truncate_len]}{'...' if len(', '.join(new_q)) > truncate_len else ''}")
					elif isinstance(t.result, list) and t.executor_name == "RetrievalExecutor":
						retrieved_ids = [str(item.get("metadata", {}).get("id", f"unnamed_chunk_{i}")) for i, item in
										 enumerate(t.result)]
						res_disp_parts.append(
							f"    检索到 {len(t.result)} 片段. IDs: {', '.join(retrieved_ids)[:truncate_len - 20]}...")
					else:
						res_s = str(t.result);
						res_disp_parts.append(
							f"    结果: {res_s[:truncate_len]}{'...' if len(res_s) > truncate_len else ''}")
				else:
					res_disp_parts.append(f"    执行失败: {str(t.result)[:truncate_len]}...")
				res_final_disp = "\n".join(res_disp_parts);
				th_s = str(t.thought or "N/A");
				th_s = th_s[:truncate_len] + "..." if len(th_s) > truncate_len else th_s
				hist.append(
					f"  - Task ID: {t.id}\n    Desc: {t.task_description}\n    Exec: {t.executor_name}\n    Status: {t.status}\n{res_final_disp}\n    Thought: {th_s}")
		return "\n\n".join(hist) if hist else "尚未执行任何历史任务。"

	def get_summary_for_generator(self) -> str:  # (与之前相同)
		summary_parts = []
		for i, task_id in enumerate(self.execution_order):
			task = self.get_task(task_id)
			if task:
				result_str = "N/A";
				thought_str = str(task.thought or '未记录思考过程')
				if task.status == 'completed':
					if isinstance(task.result, dict) and task.executor_name == "DeduceExecutor":
						result_str = f"推理总结: {task.result.get('answer_summary', 'N/A')}, 信息是否充分: {task.result.get('is_sufficient')}" + (
							f", 建议探究: {task.result['new_questions_or_entities']}" if task.result.get(
								'new_questions_or_entities') else "")
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
					f"步骤 {i + 1} (ID: {task.id}):\n  目标: {task.task_description}\n  执行工具: {task.executor_name}\n  执行思考: {thought_str_summary}\n  产出/状态: {result_str_summary}")
		if not summary_parts: return "未能执行任何步骤，或没有可总结的产出。"
		return "\n\n".join(summary_parts)

	def collect_retrieved_references_for_generator(self) -> List[Dict]:  # (与之前相同)
		refs = [];
		for tid in self.execution_order:
			t = self.get_task(tid)
			if t and t.executor_name == "RetrievalExecutor" and t.status == "completed" and isinstance(t.result, list):
				for item in t.result:
					if isinstance(item, dict) and "content" in item: refs.append({"content": item["content"],
																				  "document_name": item.get("metadata",
																											{}).get(
																					  "source_name", f"源_{t.id}"),
																				  "id": item.get("metadata", {}).get(
																					  "id", f"ref_{t.id}_{len(refs)}")})
		return refs


# --- Executors (与之前相同，DeduceExecutor 返回新结构体) ---
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

	def _resolve_references(self, data_template: Any,
							context: ContextManager) -> Any:  # (与之前相同，已包含对DeduceExecutor结构化结果的answer_summary的引用)
		if isinstance(data_template, str):
			def replace_match(match):
				ref_full = match.group(1).strip();
				task_id_ref, attr_ref = ref_full.split('.', 1) if '.' in ref_full else (ref_full, "result")
				ref_task = context.get_task(task_id_ref)
				if ref_task and ref_task.status == "completed":
					target_obj = ref_task.result
					if attr_ref == "result":
						if isinstance(target_obj, dict) and "answer_summary" in target_obj: return str(
							target_obj["answer_summary"])
						if isinstance(target_obj, list): return "\n".join(
							[f"- {item}" for item in map(str, target_obj)])
						return str(target_obj)
					elif isinstance(target_obj, dict) and attr_ref in target_obj:
						return str(target_obj[attr_ref])
					else:
						print(f"  [Exec Warn] Unsupported attr '{attr_ref}' or not found for {{ {ref_full} }}.")
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
		task.thought = (task.thought or "") + f"KB检索查询: '{resolved_query}', 过滤器: {actual_filter}.".strip()
		retrieved_docs_with_meta = self.kb.retrieve(resolved_query, top_k=3, filter_dict=actual_filter)
		if not retrieved_docs_with_meta: task.thought += "\n未检索到任何匹配文档."; return []
		task.thought += f"\n检索到 {len(retrieved_docs_with_meta)} 个文档对象.";
		return retrieved_docs_with_meta

	def get_schema(self) -> Dict[str, Any]:
		return {"name": "RetrievalExecutor",
				"description": "从向量知识库中检索与查询相关的文本片段。可指定元数据过滤器。",
				"logic_input_schema": {"query": "string (检索查询语句, 可引用 {{task_id.result}})",
									   "filter": "dict (可选, ChromaDB元数据过滤器)"}}


class DeduceExecutorOutput(TypedDict, total=False):  # (与之前相同)
	answer_summary: str;
	is_sufficient: bool;
	new_questions_or_entities: List[str];
	raw_llm_response: str


class DeduceExecutor(ExecutorBase):  # (与之前相同，使用新的DeducePrompt)
	def __init__(self, llm_client: OpenAIChatLLM, prompt_template: DeducePrompt,
				 specialized_prompts: Optional[Dict[str, BasePrompt]] = None):
		super().__init__(llm_client);
		self.default_prompt_template = prompt_template;
		self.specialized_prompts = specialized_prompts or {}

	async def execute(self, task: Task, context: ContextManager) -> DeduceExecutorOutput:
		logic_input = task.logic_input;
		goal = logic_input.get("reasoning_goal");
		raw_ctx_data = logic_input.get("context_data");
		op_type = logic_input.get("operation_type")
		if not goal or not isinstance(goal, str) or raw_ctx_data is None: raise ExecutorError(
			"DeduceExecutor: 'reasoning_goal'(str) & 'context_data' required.")
		res_ctx_data = self._resolve_references(raw_ctx_data, context)
		ctx_data_str = json.dumps(res_ctx_data, ensure_ascii=False, indent=2) if isinstance(res_ctx_data,
																							(list, dict)) else str(
			res_ctx_data)
		prompt_to_use = self.specialized_prompts.get(op_type,
													 self.default_prompt_template) if op_type else self.default_prompt_template
		sys_prompt = getattr(prompt_to_use, "SYSTEM_PROMPT", DeducePrompt.SYSTEM_PROMPT)
		if op_type and prompt_to_use != self.default_prompt_template: print(
			f"  [DeduceExecutor] Using specialized prompt for op_type: {op_type}")
		prompt_str = prompt_to_use.format(reasoning_goal=goal, context_data=ctx_data_str)
		task.thought = (
								   task.thought or "") + f"演绎目标({op_type or 'default'}): {goal}. 上下文(摘要): {ctx_data_str[:100]}...".strip()
		resp_json = await self.llm_client.generate_structured_json(prompt_str, system_prompt_str=sys_prompt,
																   temperature=0.0)
		summary = resp_json.get("answer_summary", "未能从LLM响应中解析出答案总结。");
		is_suff = bool(resp_json.get("is_sufficient", False))
		new_qs_raw = resp_json.get("new_questions_or_entities", []);
		new_qs = [str(item).strip() for item in new_qs_raw if
				  isinstance(item, str) and str(item).strip()] if isinstance(new_qs_raw, list) else [
			str(new_qs_raw).strip()] if isinstance(new_qs_raw, str) and str(new_qs_raw).strip() else []
		task.thought += f"\nLLM演绎响应(结构化): sufficient={is_suff}, new_qs={new_qs}, summary_preview='{summary[:50]}...'"
		output: DeduceExecutorOutput = {"answer_summary": summary, "is_sufficient": is_suff,
										"new_questions_or_entities": new_qs,
										"raw_llm_response": json.dumps(resp_json, ensure_ascii=False)};
		return output

	def get_schema(self) -> Dict[str, Any]:
		return {"name": "DeduceExecutor",
				"description": "基于上下文进行推理、总结、判断或抽取。会判断信息是否充分并给出下一步查询建议。",
				"logic_input_schema": {"reasoning_goal": "string (具体推理目标)",
									   "context_data": "any (推理所需上下文,可引用 {{task_id.result}})",
									   "operation_type": "string (可选, 如 summarize, extract_info)"}}


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


# --- Planner (与之前相同) ---
class Planner:  # (与之前相同)
	def __init__(self, llm_client: OpenAIChatLLM, prompt: PlannerPrompt):
		self.llm_client = llm_client; self.prompt_template = prompt

	async def plan_next_steps(self, user_query: str, context: ContextManager, available_executors: List[Dict]) -> Tuple[
		str, str, List[Dict]]:
		exec_desc_parts = [
			f"  - 名称: \"{s['name']}\"\n    描述: \"{s['description']}\"\n    输入参数模式 (logic_input_schema): {json.dumps(s.get('logic_input_schema', 'N/A'), ensure_ascii=False)}"
			for s in available_executors]
		exec_desc = "\n".join(exec_desc_parts);
		history_str = context.get_task_history_for_prompt()
		user_prompt_str = self.prompt_template.format(user_query=user_query, available_executors_description=exec_desc,
													  task_history=history_str)
		response_json = await self.llm_client.generate_structured_json(user_prompt_str,
																	   system_prompt_str=PlannerPrompt.SYSTEM_PROMPT,
																	   temperature=0.01)  # Very low temp for planner
		plan_status = response_json.get("plan_status", "error");
		final_thought = response_json.get("final_thought", "LLM未能提供规划思考。");
		next_steps_data = response_json.get("next_steps", [])
		if not isinstance(next_steps_data, list): print(
			f"  [Planner Err] LLM 'next_steps' not list. Got: {next_steps_data}. Assuming none."); next_steps_data = []
		valid_steps = [td for td in next_steps_data if isinstance(td, dict) and all(
			k in td for k in ["id", "executor_name", "task_description", "logic_input"])]
		if len(valid_steps) != len(next_steps_data): print(
			f"  [Planner Warn] Some planned steps were invalid and skipped by planner output validation.")
		return plan_status, final_thought, valid_steps


# --- AnswerGenerator (与之前相同) ---
class AnswerGenerator:  # (与之前相同)
	def __init__(self, llm_client: OpenAIChatLLM,
				 prompt: UserProvidedReferGeneratorPrompt): self.llm_client = llm_client; self.prompt_template = prompt

	async def generate_final_answer(self, user_query: str, context: ContextManager) -> str:
		summary = context.get_summary_for_generator();
		refs = context.collect_retrieved_references_for_generator()
		prompt_str = self.prompt_template.format(summary_of_executed_steps=summary, user_query=user_query,
												 retrieved_references=refs)
		return await self.llm_client.generate(prompt_str, temperature=0.2)  # Slightly higher temp for generation


# --- Pipeline (与之前相同，使用迭代逻辑) ---
class IterativePipeline:  # (与之前相同)
	def __init__(self, planner: Planner, executors: Dict[str, ExecutorBase], generator: AnswerGenerator,
				 max_iterations: int = 5):  # Default to 5 iterations
		self.planner = planner;
		self.executors = executors;
		self.generator = generator;
		self.max_iterations = max_iterations

	async def _execute_task_dag_segment(self, tasks_data: List[Dict], ctx: ContextManager,
										iter_num: int) -> bool:  # (与之前相同)
		if not tasks_data: return True
		prefix = re.sub(r'[^\w\s-]', '', ctx.user_query[:10]).strip().replace(' ', '_') or "q"
		added_ids = [ctx.add_task_from_planner(td, prefix, iter_num, i).id for i, td in enumerate(tasks_data)]
		cache = set();
		success = True
		for tid in added_ids:
			task = ctx.get_task(tid)
			if task and task.status == "pending":
				await self._execute_task_with_dependencies(tid, ctx, cache)
				if ctx.get_task(tid).status != "completed": success = False
		return success

	async def _execute_task_with_dependencies(self, task_id: str, ctx: ContextManager, cache: set):  # (与之前相同)
		task = ctx.get_task(task_id)
		if not task: print(f"  [Pipe Err] Task {task_id} def not found for exec."); return
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
		ctx.update_task_status(task.id, "running", thought=f"Start: {task.task_description}")
		print(f"\n▶️ Iter Exec Task: {task.id} - \"{task.task_description}\" ({task.executor_name})")
		try:
			result = await executor.execute(task, ctx)
			ctx.update_task_status(task.id, "completed", result=result, thought=task.thought)
		except ExecutorError as e:
			emsg = f"ExecErr T {task.id}: {e}"; ft = (task.thought or "") + f"\nExecErr: {e}"; ctx.update_task_status(
				task.id, "failed", result=emsg, thought=ft); print(f"🛑 {emsg}")
		except Exception as e:
			emsg = f"UnexpectedErr T {task.id}: {e}"; import traceback; tb = traceback.format_exc(); print(
				f"🛑 {emsg}\n{tb}"); ft = (task.thought or "") + f"\nUnexpectedErr: {e}"; ctx.update_task_status(task.id,
																												"failed",
																												result=emsg,
																												thought=ft)

	async def run(self, user_query: str) -> str:  # (与之前相同)
		print(f"\n🚀 IterativePipeline for query: \"{user_query}\"")
		ctx = ContextManager(user_query);
		schemas = [ex.get_schema() for ex in self.executors.values()]
		final_ans = "处理中遇到问题，未能得出最终答案。";
		planner_overall_thought = "";
		current_plan_status = "requires_more_steps"
		final_iter_num_for_log = 0
		for i_iter in range(self.max_iterations):
			final_iter_num_for_log = i_iter + 1
			print(f"\n--- Iteration {final_iter_num_for_log} / {self.max_iterations} ---")
			print(f"📝 Planning phase (Iteration {final_iter_num_for_log})...")
			try:
				status, thought, steps_data = await self.planner.plan_next_steps(user_query, ctx, schemas)
				planner_overall_thought += f"\nIter {final_iter_num_for_log} Planner Thought: {thought}"
				current_plan_status = status
				print(f"  [Planner Out] Status: {current_plan_status}, Thought: {thought}")
				if steps_data:
					print(
						f"  [Planner Out] Next Steps Planned ({len(steps_data)}): {[s.get('task_description', 'N/A') for s in steps_data]}")
				else:
					print("  [Planner Out] No new steps planned.")
				if current_plan_status == "finished": print("  [Pipe] Planner: finished."); break
				if current_plan_status == "cannot_proceed": print(
					"  [Pipe] Planner: cannot_proceed."); final_ans = f"无法继续：{thought}"; break
				if not steps_data:
					if i_iter > 0:
						print("  [Pipe] No new steps & not finished explicitly. Assuming completion."); break
					else:
						print(
							"  [Pipe Err] Planner: no initial steps."); return f"无法制定初步计划。Planner Thought: {thought}"
			except Exception as e:
				print(f"  [Pipe Err] Planning error: {e}"); import \
					traceback; traceback.print_exc(); final_ans = f"规划阶段意外错误：{e}"; break
			print(f"\n⚙️ Execution phase (Iteration {final_iter_num_for_log})...")
			await self._execute_task_dag_segment(steps_data, ctx, final_iter_num_for_log)
			finish_executed = any(
				t.executor_name == "FinishAction" and t.status == "completed" for t in ctx.tasks.values() if
				t.id in [s.get("id", "") for s in steps_data])  # Check ID properly
			if finish_executed: print(
				f"  [Pipe] FinishAction completed. Ending iterations."); current_plan_status = "finished"; break
		print(f"\n💬 Generation phase after {final_iter_num_for_log} iteration(s)...")
		try:
			if current_plan_status == "cannot_proceed" and "无法继续" in final_ans:
				pass
			else:
				final_ans = await self.generator.generate_final_answer(user_query, ctx)
		except Exception as e:
			print(f"  [Pipe Err] Generation error: {e}"); import \
				traceback; traceback.print_exc(); final_ans = f"生成答案意外错误：{e}"
		print(f"\n💡 Final Answer: {final_ans}");
		return final_ans


# --- 主程序入口 ---
async def run_main_logic_with_user_data_deep_recursive_v2():  # Renamed main function
	# --- LLM and Embedding Setup ---
	api_key = 'sk-af4423da370c478abaf68b056f547c6e'
	base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
	embedding_api_key = 'sk-af4423da370c478abaf68b056f547c6e'

	if not api_key or not embedding_api_key:
		print("错误：请确保 OPENAI_API_KEY 和 DASHSCOPE_API_KEY_FOR_EMBEDDING (或其回退 OPENAI_API_KEY) 已设置。")
		return
	llm_client = OpenAIChatLLM(model_name=model_name, api_key=api_key, base_url=base_url)
	try:
		embedding_function = QwenEmbeddingFunction(api_key=embedding_api_key)
	except Exception as e:
		print(f"  [Embedding Error] QwenEmbeddingFunction 初始化失败: {e}"); return

	# --- REFINED Knowledge Base Data for Multi-Hop Retrieval Test ---
	initial_user_docs_as_dicts = [
		{
			"page_content": "【正柴胡饮颗粒】药品说明书（摘要）\n【检查】应符合颗粒剂项下有关的各项规定（详见《中国药典》通则0104）。其余按品种标准执行。",
			"metadata": {"id": "zchy_spec_main_v3", "source_name": "正柴胡饮颗粒说明书摘要"}},
		{
			"page_content": "《中国药典》通则0104 - 颗粒剂（概述）\n本通则为颗粒剂的通用质量控制要求。具体检查项目包括：【性状】、【鉴别】、【检查】（如粒度、水分、溶化性、装量差异、微生物限度等）、【含量测定】等。\n关于【微生物限度】，颗粒剂应符合现行版《中国药典》通则1105（非无菌产品微生物限度检查：微生物计数法）、通则1106（非无菌产品微生物限度检查：控制菌检查法）和通则1107（非无菌药品微生物限度标准）的相关规定。本通则0104不详述这些微生物限度标准的具体限值。",
			"metadata": {"id": "tg0104_overview_v3", "source_name": "药典通则0104概述"}},
		{
			"page_content": "《中国药典》通则1107 - 非无菌药品微生物限度标准（节选）\n本标准规定了各类非无菌药品所需控制的微生物限度。\n对于口服固体制剂（如颗粒剂）：\n1. 需氧菌总数：每1g（或1ml）不得过1000 cfu。\n2. 霉菌和酵母菌总数：每1g（或1ml）不得过100 cfu。\n3. 控制菌：每1g（或1ml）不得检出大肠埃希菌；对于含动物脏器、组织或血液成分的制剂，每10g（或10ml）不得检出沙门菌。",
			"metadata": {"id": "tg1107_details_v3", "source_name": "药典通则1107-微生物限度细节"}},
		{
			"page_content": "《中国药典》通则0104 - 颗粒剂（粒度与水分细节）\n【粒度】（通则0982第二法）不能通过一号筛（2.00mm）与能通过五号筛（0.250mm）的药粉总和不得超过总重量的15％。\n【水分】（通则0832第一法）中药颗粒剂不得过8.0％。",
			"metadata": {"id": "tg0104_sizewater_v3", "source_name": "药典通则0104-粒度与水分细节"}}
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
	# md_folder_path = r'D:\Master\llm\database\kag\测试数据集'
	# if os.path.exists(md_folder_path): # ... (您的 md 加载逻辑) ...
	# else: print(f"  [Warning] Markdown文件夹 {md_folder_path} 不存在。")

	try:
		chroma_kb = ChromaKnowledgeBase(initial_documents=initial_langchain_docs, embedding_function=embedding_function,
										force_rebuild=False,
										persist_directory = 'chroma_db_kag_recursive_2')
	except Exception as e:
		print(f"创建Chroma知识库失败: {e}"); import traceback; traceback.print_exc(); return

	# --- Initialize Components (using refined prompts) ---
	planner_prompt = PlannerPrompt()
	deduce_prompt_template = DeducePrompt()
	code_exec_prompt = CodeExecutionPrompt()
	refer_generator_prompt = UserProvidedReferGeneratorPrompt(language="zh")

	retrieval_executor = RetrievalExecutor(kb=chroma_kb)
	deduce_executor = DeduceExecutor(llm_client, deduce_prompt_template)
	code_executor = CodeExecutor(llm_client, code_exec_prompt)
	finish_executor = FinishExecutor()

	executors_map = {
		"RetrievalExecutor": retrieval_executor, "DeduceExecutor": deduce_executor,
		"CodeExecutor": code_executor, "FinishAction": finish_executor
	}
	planner = Planner(llm_client, planner_prompt)
	generator = AnswerGenerator(llm_client, refer_generator_prompt)
	pipeline = IterativePipeline(planner, executors_map, generator, max_iterations=8)  # 增加迭代次数以允许更深的递归

	# --- Run Query Designed to Force Multi-Hop Retrieval ---
	user_query_to_run = "你是谁"

	print(f"\n🚀 Running DEEP RECURSIVE V2 query: \"{user_query_to_run}\"")
	final_answer = await pipeline.run(user_query_to_run)
	print(f"\n🏁🏁🏁🏁🏁 DEEP RECURSIVE V2 FINAL ANSWER (for query: '{user_query_to_run}') 🏁🏁🏁🏁🏁\n{final_answer}")


if __name__ == "__main__":
	print("开始执行“深度类递归V2”优化版主逻辑...")
	# ... (环境变量和依赖提示 - 与之前相同) ...

	asyncio.run(run_main_logic_with_user_data_deep_recursive_v2())
