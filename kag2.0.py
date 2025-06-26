import json
import os
import openai
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional, Callable  # Added Callable

# Langchain and Chroma imports
from langchain_core.documents import Document
from langchain_chroma import Chroma
# Embedding function placeholder - CHOOSE ONE or provide your own
# Option 1: DashScope Embeddings (Recommended for consistency if you have key & it works)
from langchain_community.embeddings import DashScopeEmbeddings

# Option 2: HuggingFace Sentence Transformers (Local, free)
# from langchain_community.embeddings import HuggingFaceEmbeddings


# --- 配置区 ---
DASHSCOPE_API_KEY = 'sk-af4423da370c478abaf68b056f547c6e'
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")

# ChromaDB persistence settings
CHROMA_PERSIST_DIRECTORY = "chroma_db_kag"
CHROMA_COLLECTION_NAME = "kag_documents"

# Embedding model for Chroma - choose one or provide your own `embedding_function`
# If using DashScopeEmbeddings:
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v2"  # Check DashScope docs for latest/best model
# If using HuggingFaceEmbeddings:
# HF_EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'


# 初始化 DashScope Client (for LLM, not embeddings if using HF)
if not DASHSCOPE_API_KEY:
	print("错误：请设置 DASHSCOPE_API_KEY 环境变量。")
	exit(1)
try:
	client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
	print(f"[DashScope] LLM 客户端初始化成功，将使用模型: {LLM_MODEL_NAME} at {DASHSCOPE_BASE_URL}")
except Exception as e:
	print(f"[DashScope] LLM 客户端初始化失败: {e}");
	exit(1)


# --- ChromaKnowledgeBase Class ---
class ChromaKnowledgeBase:
	def __init__(self,
				 embedding_function: Callable,  # Expecting a Langchain-compatible embedding function
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
			import shutil
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
				# Check if the collection actually has documents
				# Chroma's count() method might require the client to be initialized or collection to be loaded.
				# A simple check: try a dummy get.
				# test_get = self.vectorstore.get(limit=1) # This can be slow if collection is large
				# count = self.vectorstore._collection.count() # Accessing private member, risky
				# For now, assume loading is successful if no error. A more robust check might be needed.
				# A simple way to check might be to see if a search returns anything or if a count is > 0
				# However, an empty loaded collection is also possible.
				print(f"  [ChromaKB] 成功加载向量库。集合 '{self.collection_name}'.")
			# You might want to add a more specific check here if an empty collection is an issue
			# For example, self.vectorstore.get(limit=1)['ids'] would be empty for a new collection.
			# Or, if initial_documents are provided and we loaded, ensure they are there (more complex)

			except Exception as e:  # Broad exception for now, Chroma can raise various things
				print(f"  [ChromaKB Error] 从 '{persist_directory}' 加载失败: {e}")
				print(f"  [ChromaKB] 将尝试基于提供的 initial_documents (如果存在) 创建新的向量库。")
				self.vectorstore = None  # Ensure it's None so it gets rebuilt if initial_documents are present

		if self.vectorstore is None:  # If not loaded or loading failed, or force_rebuild was true (and dir deleted)
			if initial_documents:
				print(f"  [ChromaKB] 正在为 {len(initial_documents)} 个文档构建新的 Chroma 向量库...")
				self.vectorstore = Chroma.from_documents(
					documents=initial_documents,
					embedding=self.embedding_function,
					persist_directory=self.persist_directory,
					collection_name=self.collection_name
					# ids=[f"doc_{i}" for i in range(len(initial_documents))] # Optional: provide custom IDs
				)
				# Chroma.from_documents handles persistence if persist_directory is provided
				print(f"  [ChromaKB] 新向量库构建并持久化完成。")
			else:
				print(f"  [ChromaKB] 没有提供初始文档，且未加载现有向量库。知识库将为空（或使用空的持久化集合）。")
				# Initialize an empty Chroma store that persists, so it can be added to later
				self.vectorstore = Chroma(
					persist_directory=self.persist_directory,
					embedding_function=self.embedding_function,
					collection_name=self.collection_name
				)
				print(f"  [ChromaKB] 空的持久化 Chroma 集合 '{self.collection_name}' 已准备就绪。")

	def add_documents(self, documents: List[Document]):
		if not self.vectorstore:
			print("  [ChromaKB Error] Vectorstore 未初始化，无法添加文档。")
			# Potentially initialize it here if it makes sense for the workflow
			# self.vectorstore = Chroma.from_documents(...)
			return
		if documents:
			print(f"  [ChromaKB] 正在向集合 '{self.collection_name}' 添加 {len(documents)} 个新文档...")
			# ids_to_add = [f"doc_{self.vectorstore._collection.count() + i}" for i in range(len(documents))] # Requires count
			self.vectorstore.add_documents(documents)  # Chroma will generate IDs if not provided
			# Note: Chroma automatically persists changes if persist_directory was set at init.
			print(f"  [ChromaKB] 文档添加完成。")

	def retrieve(self, query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
		if not self.vectorstore:
			print("  [ChromaKB Error] Vectorstore 未初始化，无法检索。")
			return []

		# A quick check if the collection is empty.
		# This is a workaround as Chroma's count isn't always straightforward to get without client.
		# test_get_for_empty_check = self.vectorstore.get(limit=1)
		# if not test_get_for_empty_check or not test_get_for_empty_check.get('ids'):
		#     print("  [ChromaKB] 知识库集合为空，无法检索。")
		#     return []
		# A more direct count if collection object is accessible and stable:
		try:
			if self.vectorstore._collection.count() == 0:
				print("  [ChromaKB] 知识库集合为空，无法检索。")
				return []
		except Exception as e:
			print(f"  [ChromaKB Warning] 无法获取集合计数，将尝试检索: {e}")

		print(f"  [ChromaKB] 正在为查询 '{query}' 检索最相关的 {top_k} 个文档 (过滤器: {filter_dict})...")
		try:
			# similarity_search_with_score returns (Document, score) tuples.
			# Score is distance by default (lower is better).
			# For cosine similarity, some Chroma versions/setups return 0 for perfect match, 1 for dissimilar.
			# For L2 distance, it's the squared L2 norm.
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
					"id": doc.metadata.get("id", None),  # Try to get an ID from metadata if you set one
					"content": doc.page_content,
					"metadata": doc.metadata,
					"score": float(score)  # Score is often distance, so lower is better.
					# If it's cosine similarity (where higher is better), this might need adjustment.
					# Langchain Chroma typically returns distance (e.g. L2 norm or cosine distance)
				})
		print(f"  [ChromaKB] 检索到 {len(processed_results)} 个文档。")
		return processed_results




# --- LogicalStep, LLM Call, Operators, Engine (largely same as before) ---
# (Copied from previous correct version, ensure these are complete in your actual file)
class LogicalStep:
	def __init__(self, operator_type: str, params: dict, step_id: Optional[str] = None):
		self.step_id = step_id
		self.operator_type = operator_type
		if not isinstance(params, dict):
			print(
				f"  [LogicalStep CRITICAL Warning] 'params' for operator_type '{operator_type}' (step_id: {step_id}) was initialized with non-dict type: {type(params)}. Forcing to empty dict. Value: {params}")
			self.params = {}
		else:
			self.params = params
		self.result: Any = None
		self.status: str = "pending"

	def __repr__(self):
		return f"Step(id={self.step_id}, op='{self.operator_type}', params={self.params}, status='{self.status}', result='{self.result}')"

	@classmethod
	def from_dict(cls, data: Dict[str, Any], original_question_id: Optional[str], step_counter: int) -> 'LogicalStep':
		step_id_base = original_question_id if original_question_id else "unknown_q"
		step_id = f"{step_id_base}-s{step_counter}"
		operator_type = data.get("operator_type", "unknown_operator")
		raw_params = data.get("params")
		parsed_params = {}
		if isinstance(raw_params, dict):
			parsed_params = raw_params
		elif isinstance(raw_params, str):
			print(
				f"  [LogicalStep Warning] 'params' field for step_id '{step_id}', operator '{operator_type}' was a STRING: '{raw_params}'. Attempting to interpret.")
			if operator_type == 'retrieve':
				parsed_params = {"query": raw_params}
			elif operator_type == 'reason':
				parsed_params = {"question_or_goal": raw_params, "context_source_step_ids": []}
			else:
				print(
					f"  [LogicalStep Warning] Cannot safely interpret string params for unknown operator type '{operator_type}'. Defaulting to empty params."); parsed_params = {}
		elif raw_params is None:
			print(
				f"  [LogicalStep Info] 'params' field missing for step_id '{step_id}', operator '{operator_type}'. Defaulting to empty dict."); parsed_params = {}
		else:
			print(
				f"  [LogicalStep Warning] 'params' field for step_id '{step_id}', operator '{operator_type}' was unexpected type: {type(raw_params)}. Defaulting to empty dict. Value: {raw_params}"); parsed_params = {}
		return cls(step_id=step_id, operator_type=operator_type, params=parsed_params)


def call_dashscope_llm(
		system_prompt: str, user_prompt: str, model_name: str = LLM_MODEL_NAME, expect_json_output: bool = False
) -> Optional[Any]:
	messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
	print(f"  [DashScope] 正在调用模型 '{model_name}' (期望JSON: {expect_json_output})...")
	try:
		completion_params = {"model": model_name, "messages": messages, "temperature": 0.1}
		extra_body_params = {}
		if expect_json_output:
			completion_params["response_format"] = {"type": "json_object"}
			extra_body_params["enable_thinking"] = False
		if extra_body_params: completion_params["extra_body"] = extra_body_params
		response = client.chat.completions.create(**completion_params)
		content = response.choices[0].message.content
		if not content: print("  [DashScope Error] 未能从API响应中获取内容."); return None
		if expect_json_output:
			try:
				return json.loads(content)
			except json.JSONDecodeError as e:
				print(f"  [DashScope Error] 期望JSON输出，但解析失败: {e}");
				print(f"  [DashScope Raw Response] 收到的内容: {content}")
				cleaned_content = content.strip()
				if cleaned_content.startswith("```json"):
					cleaned_content = cleaned_content[7:];
					cleaned_content = cleaned_content[:-3] if cleaned_content.endswith("```") else cleaned_content
					try:
						return json.loads(cleaned_content.strip())
					except json.JSONDecodeError as e2:
						print(f"  [DashScope Error] 清理后再次解析JSON失败: {e2}"); return None
				return None
		else:
			return content
	except openai.APIError as e:
		print(f"  [DashScope API Error] API调用失败: {e.status_code if hasattr(e, 'status_code') else 'Unknown'}")
		if hasattr(e, 'message'): print(f"  [DashScope API Error] Message: {e.message}")
		if hasattr(e, 'body') and e.body: print(f"  [DashScope API Error] Body: {e.body}")
		if hasattr(e, 'response') and e.response:
			try:
				print(f"  [DashScope API Error] Raw Response Content: {e.response.text}")
			except:
				pass
		return None
	except Exception as e:
		print(f"  [DashScope Error] 调用LLM时发生未知错误: {e}"); import traceback; traceback.print_exc(); return None


class PlanningOperator:
	def decompose(self, natural_question: str, original_question_id: str) -> List[LogicalStep]:
		system_prompt = """
        你是一个智能助手，负责将复杂的用户问题分解为一系列可执行的逻辑步骤。
        每个步骤是一个JSON对象，应包含 "operator_type" (字符串) 和 "params" (一个JSON对象/字典)。
        可用的 operator_type 包括: "retrieve", "reason"。
        对于 "retrieve" 操作符, "params" 对象应包含一个 "query" (字符串) 键。例如: {"query": "Python 创造者"}
        对于 "reason" 操作符, "params" 对象应包含一个 "question_or_goal" (字符串) 键，以及一个可选的 "context_source_step_ids" (字符串列表) 键，这些ID应该是相对于你当前生成的步骤列表的序号（例如 ["s1"], ["s1", "s2"]）。
        请以JSON列表的形式返回这些步骤。确保输出是合法的JSON数组。
        示例:
        [
            {"operator_type": "retrieve", "params": {"query": "Python 创造者"}},
            {"operator_type": "retrieve", "params": {"query": "Python 发布时间"}},
            {"operator_type": "reason", "params": {"question_or_goal": "总结Python的创造者和发布时间", "context_source_step_ids": ["s1", "s2"]}}
        ]
        """
		user_prompt = f"请分解以下问题: \"{natural_question}\"\n请确保 context_source_step_ids 中的ID是相对于你生成的步骤列表的序号（例如 s1, s2 等）。"
		response_data = call_dashscope_llm(system_prompt, user_prompt, expect_json_output=True)
		print(f"  [Planning DEBUG] LLM 分解阶段原始输出: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
		steps = []
		if response_data and isinstance(response_data, list):
			for i, step_data_dict in enumerate(response_data):
				step_counter = i + 1
				if isinstance(step_data_dict, dict):
					steps.append(LogicalStep.from_dict(step_data_dict, original_question_id, step_counter))
				else:
					print(f"  [Planning Warning] 分解步骤返回了非字典类型的数据项: {step_data_dict}")
		elif response_data:
			print(f"  [Planning Warning] 分解步骤期望列表但收到: {type(response_data)}")
		if not steps:
			print(f"  [Planning Warning] 未能从LLM获取有效分解计划，或计划为空。将使用默认检索。")
			steps.append(LogicalStep.from_dict({"operator_type": "retrieve", "params": {"query": natural_question}},
											   original_question_id, 1))
		print(f"  [PlanningOperator] 分解完成，生成 {len(steps)} 个步骤。")
		return steps


class RetrievalOperator:
	def __init__(self, kb: ChromaKnowledgeBase): self.kb = kb  # 使用 ChromaKnowledgeBase

	def retrieve(self, query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
		return self.kb.retrieve(query, top_k, filter_dict=filter_dict)


class ReasoningOperator:
	def reason(self, context_snippets: List[str], question_or_goal: str) -> str:
		system_prompt = """
        你是一个智能助手，根据提供的上下文信息来回答问题或达成目标。
        请简洁地回答。如果上下文信息不足，请明确指出。你的回答应该是纯文本。
        """
		context_str = "\n\n".join([f"上下文片段 {i + 1}:\n{snippet}" for i, snippet in enumerate(context_snippets)])
		if not context_str: context_str = "没有提供上下文信息。"
		user_prompt = f"问题/目标: \"{question_or_goal}\"\n\n可用的上下文信息如下:\n{context_str}\n\n请根据以上信息回答问题/目标。"
		response_text = call_dashscope_llm(system_prompt, user_prompt, expect_json_output=False)
		if response_text and isinstance(response_text, str):
			print(f"  [ReasoningOperator] 推理完成。"); return response_text
		else:
			print(f"  [Reasoning Warning] 推理步骤未能从LLM获取有效文本响应."); return "抱歉，推理时遇到问题或未获取LLM响应。"


class KAGLikeEngine:
	def __init__(self, po: PlanningOperator, ro: RetrievalOperator, rso: ReasoningOperator):
		self.planning_operator, self.retrieval_operator, self.reasoning_operator = po, ro, rso
		self.task_counter, self.global_step_id_counter = 0, 0

	def _get_next_global_step_id(self, q_id: str, prefix: str = "gs") -> str:
		self.global_step_id_counter += 1;
		return f"{q_id}-{prefix}{self.global_step_id_counter}"

	def _reflect_and_plan_next(self, oq: str, wms: str, mi: int, ci: int, cqi: str) -> Tuple[
		List[LogicalStep], bool, Optional[str]]:
		system_prompt = f"""
        你是一个智能任务协调器。评估当前进展并决定下一步。当前是迭代 {ci}/{mi}。
        任务:
        1. 判断原始问题是否已完全解决。
        2. 若解决，在 "final_answer_if_solved" 字段提供答案。
        3. 若未解决且有迭代空间，决定是否需新步骤。新步骤对象必须含 "operator_type" (字符串) 和 "params" (JSON对象/字典)。"context_source_step_ids" (字符串列表) 应引用工作内存摘要中已存在的全局步骤ID。
        返回严格的JSON对象，含 "is_solved" (布尔), "final_answer_if_solved" (字符串/null), "next_steps" (JSON对象列表/空列表)。
        """
		user_prompt = f"原始问题: \"{oq}\"\n\n工作内存摘要:\n{wms}\n\n请反思并计划下一步。"
		response_data = call_dashscope_llm(system_prompt, user_prompt, expect_json_output=True)
		print(f"  [Reflect DEBUG] LLM 反思阶段原始输出: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
		new_steps, is_solved, final_answer = [], False, None
		if response_data and isinstance(response_data, dict):
			is_solved = bool(response_data.get("is_solved", False))
			final_answer = response_data.get("final_answer_if_solved")
			raw_next_steps = response_data.get("next_steps", [])
			if isinstance(raw_next_steps, list):
				for i, step_data_dict in enumerate(raw_next_steps):
					if isinstance(step_data_dict, dict):
						new_step_id = self._get_next_global_step_id(cqi, prefix=f"r{ci}-s")
						op_type = step_data_dict.get("operator_type", "unknown_operator")
						raw_params = step_data_dict.get("params")
						params_for_new_step = {}
						if isinstance(raw_params, dict):
							params_for_new_step = raw_params
						elif isinstance(raw_params, str):
							print(
								f"  [Reflect Warning] 新步骤的 'params' 是字符串: '{raw_params}'. 为op '{op_type}'尝试解释.")
							if op_type == 'retrieve':
								params_for_new_step = {"query": raw_params}
							elif op_type == 'reason':
								params_for_new_step = {"question_or_goal": raw_params, "context_source_step_ids": []}
							else:
								params_for_new_step = {}
						elif raw_params is None:
							params_for_new_step = {}
						else:
							print(
								f"  [Reflect Warning] 新步骤的 'params' 类型未知: {type(raw_params)}. 用空字典."); params_for_new_step = {}
						new_steps.append(
							LogicalStep(operator_type=op_type, params=params_for_new_step, step_id=new_step_id))
			print(f"  [Reflect] 反思完成。is_solved={is_solved}, new_steps_count={len(new_steps)}")
		else:
			print(f"  [Reflect Warning] 反思步骤未能从LLM获取有效JSON决策。假设未解决且无新步骤。")
		return new_steps, is_solved, final_answer

	def execute_step(self, step: LogicalStep, working_memory: Dict[str, LogicalStep]) -> LogicalStep:
		print(f"\n[Engine] 正在执行步骤: {step.step_id} ({step.operator_type})")
		step.status = "in_progress"
		try:
			if not isinstance(step.params, dict):
				print(
					f"  [Engine CRITICAL Error] 步骤 {step.step_id} 的 params 不是字典: {step.params} (类型: {type(step.params)}). 跳过执行。")
				step.status = "failed";
				return step
			if step.operator_type == "retrieve":
				query = step.params.get("query", "")
				# Retrieve from Chroma might also take a filter if needed from params
				filter_param = step.params.get("filter")
				if not query:
					print(f"  [Engine Warning] 步骤 {step.step_id} (retrieve) 查询为空."); step.result = []
				else:
					step.result = self.retrieval_operator.retrieve(query, filter_dict=filter_param)
			elif step.operator_type == "reason":
				context_snippets = []
				source_step_ids_raw = step.params.get("context_source_step_ids", [])
				source_step_ids_processed: List[str] = []
				if isinstance(source_step_ids_raw, str):
					source_step_ids_processed = [source_step_ids_raw]
				elif isinstance(source_step_ids_raw, int):
					relative_id = f"s{source_step_ids_raw}";
					print(
						f"  [Engine Warning] ... context_source_step_ids 是整数: {source_step_ids_raw}. 解释为 '{relative_id}'.");
					source_step_ids_processed = [relative_id]
				elif isinstance(source_step_ids_raw, list):
					for item in source_step_ids_raw:
						if isinstance(item, str):
							source_step_ids_processed.append(item)
						elif isinstance(item, int):
							relative_id = f"s{item}"; print(
								f"  [Engine Warning] ... context_source_step_ids 列表内含整数: {item}. 解释为 '{relative_id}'."); source_step_ids_processed.append(
								relative_id)
						else:
							print(f"  [Engine Warning] ... context_source_step_ids 列表内含非预期类型: {item}. 已忽略.")
				else:
					print(
						f"  [Engine Warning] ... context_source_step_ids 类型非预期: {type(source_step_ids_raw)}. 用空列表."); source_step_ids_processed = []
				current_q_prefix = step.step_id.split("-s")[0].split("-gs")[0].split("-r")[0]
				for prev_step_id_ref in source_step_ids_processed:
					actual_prev_step_id = prev_step_id_ref
					if prev_step_id_ref.startswith("s") and not prev_step_id_ref.startswith(
							current_q_prefix) and prev_step_id_ref[1:].isdigit():
						actual_prev_step_id = f"{current_q_prefix}-{prev_step_id_ref}"
					if actual_prev_step_id in working_memory and working_memory[actual_prev_step_id].result:
						prev_res, op_type = working_memory[actual_prev_step_id].result, working_memory[
							actual_prev_step_id].operator_type
						if op_type == "retrieve" and isinstance(prev_res,
																list):  # Chroma retrieve returns list of dicts
							for item_dict in prev_res:  # each item_dict is like {"content": ..., "metadata": ..., "score": ...}
								if isinstance(item_dict, dict) and "content" in item_dict:
									context_snippets.append(
										item_dict["content"])  # Pass only content to reasoner for now
						elif isinstance(prev_res, str):
							context_snippets.append(prev_res)  # From a previous reason step
					else:
						print(
							f"  [Engine Warning] 步骤 {step.step_id} (reason) 无法找到上下文源ID: '{prev_step_id_ref}' (尝试匹配: '{actual_prev_step_id}')")
				question_or_goal = step.params.get("question_or_goal", "请根据上下文总结。")
				if not context_snippets: print(f"  [Engine Warning] 步骤 {step.step_id} (reason) 无上下文。")
				step.result = self.reasoning_operator.reason(context_snippets, question_or_goal)
			else:
				print(f"  [Engine Error] 未知操作类型: {step.operator_type}"); step.status = "failed"; return step
			step.status = "completed";
			print(f"[Engine] 步骤 {step.step_id} 完成.")
		except Exception as e:
			print(f"  [Engine Error] 步骤 {step.step_id} 执行失败: {e}"); import \
				traceback; traceback.print_exc(); step.status = "failed"
		return step

	def solve(self, natural_question: str, max_iterations: int = 3) -> str:
		self.task_counter += 1;
		current_question_id = f"q{self.task_counter}";
		self.global_step_id_counter = 0
		print(f"\n--- [Engine] 开始解决问题 ({current_question_id}): '{natural_question}' ---")
		working_memory: Dict[str, LogicalStep] = {}
		current_plan: List[LogicalStep] = self.planning_operator.decompose(natural_question, current_question_id)
		if not current_plan: print("[Engine] 初始计划为空。"); return "无法制定计划。"
		final_answer, is_solved, current_iteration = "未能找到明确答案.", False, 0
		executed_step_ids = set()
		while not is_solved and current_iteration < max_iterations:
			current_iteration += 1;
			print(f"\n[Engine] ---迭代 {current_iteration}---")
			if not current_plan: print("[Engine] 当前计划为空，进入反思。")
			active_plan_this_iter = [s for s in current_plan if
									 s.step_id not in executed_step_ids and s.status == "pending"]
			for step_to_execute in active_plan_this_iter:
				executed_step = self.execute_step(step_to_execute, working_memory)
				working_memory[executed_step.step_id] = executed_step
				executed_step_ids.add(executed_step.step_id)
			wm_summary_list = [
				f"  Step ID: {s.step_id}, Type: {s.operator_type}, Status: {s.status}, Result: {str(s.result)[:100].replace(chr(10), ' ') + '...' if s.result and len(str(s.result)) > 100 else str(s.result).replace(chr(10), ' ')}"
				for sid, s in sorted(working_memory.items())]
			wms_for_llm = "已执行步骤概览:\n" + "\n".join(wm_summary_list) if wm_summary_list else "目前还没有已执行的步骤。"
			new_steps, reflection_says_solved, llm_final_answer = self._reflect_and_plan_next(
				natural_question, wms_for_llm, max_iterations, current_iteration, current_question_id)
			is_solved = reflection_says_solved
			if is_solved:
				if llm_final_answer and llm_final_answer.strip().lower() not in ["", "null"]:
					final_answer = llm_final_answer
				else:
					sorted_reason_steps = sorted(
						[s for s in working_memory.values() if s.operator_type == "reason" and s.status == "completed"],
						key=lambda x: x.step_id, reverse=True)
					if sorted_reason_steps:
						final_answer = str(sorted_reason_steps[0].result)
					else:
						final_answer = "问题已解决(据LLM反思)，但未提取到具体文本答案。"
				break
			current_plan = new_steps
			if not current_plan:
				print("[Engine] 反思后无新计划，且问题未解决。尝试最后总结。")
				all_ctx_for_final_reason = []
				for s_id, s_obj in working_memory.items():
					if s_obj.status == "completed" and s_obj.result:
						if s_obj.operator_type == "retrieve" and isinstance(s_obj.result, list):
							for item in s_obj.result:  # item is a dict like {"content": ..., "metadata": ..., "score": ...}
								if isinstance(item, dict) and "content" in item:
									all_ctx_for_final_reason.append(item["content"])
						elif s_obj.operator_type == "reason" and isinstance(s_obj.result, str):
							all_ctx_for_final_reason.append(s_obj.result)
				if all_ctx_for_final_reason:
					final_answer = self.reasoning_operator.reason(all_ctx_for_final_reason, natural_question)
				else:
					final_answer = "未能收集到足够信息进行最终总结。"
				break
		if not is_solved and current_iteration >= max_iterations:
			print("[Engine] 达到最大迭代次数。")
			all_ctx_for_timeout_reason = []
			for s_id, s_obj in working_memory.items():
				if s_obj.status == "completed" and s_obj.result:
					if s_obj.operator_type == "retrieve" and isinstance(s_obj.result, list):
						for item in s_obj.result:
							if isinstance(item, dict) and "content" in item:
								all_ctx_for_timeout_reason.append(item["content"])
					elif s_obj.operator_type == "reason" and isinstance(s_obj.result, str):
						all_ctx_for_timeout_reason.append(s_obj.result)
			if all_ctx_for_timeout_reason:
				final_answer = self.reasoning_operator.reason(all_ctx_for_timeout_reason,
															  f"请根据以下信息总结关于 '{natural_question}' 的答案。")
			else:
				final_answer = "处理超时，未能找到明确答案。"
		print(f"\n--- [Engine] 问题解决结束 ({current_question_id}) ---");
		return final_answer


# --- 主程序入口 ---
if __name__ == "__main__":
	# 1. 定义您的 Embedding Function
	# 您需要确保 embedding_function 是一个可调用的对象，它接受一个字符串列表并返回一个嵌入列表。
	# 例如，使用 DashScope Embeddings:
	from tongyiembedding import QwenEmbeddingFunction

	embedding_function = QwenEmbeddingFunction(api_key='sk-af4423da370c478abaf68b056f547c6e')
	# 2. 准备 Langchain Document 对象
	# 这是用户提供的示例数据
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

	# 初始化 ChromaKnowledgeBase
	try:
		chroma_kb = ChromaKnowledgeBase(
			initial_documents=initial_langchain_docs,
			embedding_function=embedding_function,
			persist_directory=CHROMA_PERSIST_DIRECTORY,
			collection_name=CHROMA_COLLECTION_NAME,
			force_rebuild=True  # <--- 设置为 True 来强制重建索引
		)
	# 成功重建后，下次运行时可以将 force_rebuild 改回 False

	except Exception as e:
		print(f"创建或加载Chroma知识库失败: {e}")
		print(
			"请确保您已安装 langchain-chroma, langchain-core, 和 embedding function 所需的库 (如 sentence-transformers 或 langchain-community for DashScopeEmbeddings)。")
		exit(1)

	planning_op = PlanningOperator()
	retrieval_op = RetrievalOperator(kb=chroma_kb)
	reasoning_op = ReasoningOperator()
	engine = KAGLikeEngine(planning_op, retrieval_op, reasoning_op)

	questions = [
		"正柴胡饮颗粒的检查内容.",  # Should hit Chroma
	]
	for i, q in enumerate(questions):
		print(f"\n==================== 处理问题 {i + 1}: {q} ====================")
		# 演示如何使用元数据过滤器 (可选)
		filter_for_query = None
		if "2019" in q and "wholesome" in q:  # 这是一个非常简单的触发器
			filter_for_query = {"year": 2019}
			print(f"  [Engine] 将为此查询应用过滤器: {filter_for_query}")

		# 修改 RetrievalOperator.retrieve 以接受过滤器，或在引擎中修改如何调用它
		# 为简单起见，我已修改 RetrievalOperator.retrieve 来接受 filter_dict
		# 但在 KAGLikeEngine.execute_step 中，retrieve 步骤的 params 也需要能传递 filter
		# 这需要在 PlanningOperator 的LLM prompt中指导它生成带filter的retrieve步骤
		# 目前 execute_step 中的 retrieve 会查找 params["filter"]
		# 为了简单演示，我们将直接修改一个查询的检索步骤参数（通常这应由LLM规划）

		if filter_for_query:
			# 这是手动修改计划的简化方式，理想情况下LLM会生成带过滤器的检索步骤
			# 对于这个演示，我们假设第一个分解步骤是针对这个问题的检索
			# engine.solve 将调用 planning_op.decompose, 如果我们想在分解后注入过滤器，会比较复杂
			# 更简单的方式是，如果 KAGLikeEngine.solve 可以接受一个初始的 plan
			# 或者，在 RetrievalOperator.retrieve 中添加一个参数，并在 execute_step 中传递
			# 我已经在 execute_step 中添加了 filter_param = step.params.get("filter")
			# 所以，如果LLM的分解步骤能生成 "filter": {"year": 2019} 就会生效。
			# 由于LLM现在不会这样做，所以这个特定过滤器的演示不会直接通过当前规划流程生效。
			# 除非我们在 PlanningOperator 的 prompt 中明确指导它使用元数据。
			pass

		answer = engine.solve(q, max_iterations=3)
		print(f"\n>>> 最终答案 Q{i + 1} ('{q}'): {answer}")
		print("=======================================================\n")