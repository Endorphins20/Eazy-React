import json
import os
import openai
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional

# --- 配置区 ---
DASHSCOPE_API_KEY = 'sk-af4423da370c478abaf68b056f547c6e'
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")

if not DASHSCOPE_API_KEY:
	print("错误：请设置 DASHSCOPE_API_KEY 环境变量。")
	print("例如: export DASHSCOPE_API_KEY=\"sk-yourkey\"")
	exit(1)

try:
	client = OpenAI(
		api_key=DASHSCOPE_API_KEY,
		base_url=DASHSCOPE_BASE_URL,
	)
	print(f"[DashScope] 客户端初始化成功，将使用模型: {LLM_MODEL_NAME} at {DASHSCOPE_BASE_URL}")
except Exception as e:
	print(f"[DashScope] 客户端初始化失败: {e}")
	exit(1)


# --- 类定义 ---
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
					f"  [LogicalStep Warning] Cannot safely interpret string params for unknown operator type '{operator_type}'. Defaulting to empty params.")
				parsed_params = {}
		elif raw_params is None:
			print(
				f"  [LogicalStep Info] 'params' field missing for step_id '{step_id}', operator '{operator_type}'. Defaulting to empty dict.")
			parsed_params = {}
		else:
			print(
				f"  [LogicalStep Warning] 'params' field for step_id '{step_id}', operator '{operator_type}' was unexpected type: {type(raw_params)}. Defaulting to empty dict. Value: {raw_params}")
			parsed_params = {}

		return cls(
			step_id=step_id,
			operator_type=operator_type,
			params=parsed_params
		)


class SimulatedKnowledgeBase:
	def __init__(self):
		self.documents = {
			"doc1": "Python是一种广泛使用的高级编程语言，由Guido van Rossum首次发布于1991年。",
			"doc2": "大型语言模型（LLM）是深度学习模型，能够理解和生成人类语言文本。",
			"doc3": "苹果公司设计了iPhone，它是一款智能手机。iPhone首次发布于2007年。",
			"doc4": "北京是中国的首都，拥有悠久的历史和丰富的文化遗产。"
		}

	def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
		print(f"  [KB] 检索与 '{query}' 相关的文档...")
		results = []
		query_terms = set(query.lower().split())
		for doc_id, content in self.documents.items():
			doc_terms = set(content.lower().split())
			common_terms = query_terms.intersection(doc_terms)
			if common_terms:
				score = len(common_terms)
				if "python" in query.lower() and "python" in content.lower(): score += 3
				if (
						"guido" in query.lower() or "rossum" in query.lower() or "创造者" in query.lower()) and "guido van rossum" in content.lower(): score += 5
				if (
						"发布" in query.lower() or "时间" in query.lower() or "1991" in query.lower()) and "1991年" in content.lower(): score += 5
				if (
						"iphone" in query.lower() or "苹果" in query.lower() or "apple" in query.lower()) and "iphone" in content.lower(): score += 3
				if (
						"首都" in query.lower() or "北京" in query.lower()) and "北京" in content.lower() and "首都" in content.lower(): score += 5
				results.append({"id": doc_id, "content": content, "score": score})
		if not results:
			print(f"  [KB] 未检索到与 '{query}' 相关的文档。")
			return []
		results.sort(key=lambda x: x["score"], reverse=True)
		print(f"  [KB] 检索到 {len(results)} 个相关文档 (最高分 {results[0]['score']})，返回前 {top_k} 个。")
		return results[:top_k]


# --- DashScope LLM 调用辅助函数 ---
def call_dashscope_llm(
		system_prompt: str,
		user_prompt: str,
		model_name: str = LLM_MODEL_NAME,
		expect_json_output: bool = False
) -> Optional[Any]:
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]
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


# --- Operator 类 ---
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
        如果问题很简单，可能只需要一个 "retrieve" 步骤，或者一个直接的 "reason" 步骤。
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
			print(f"  [Planning Warning] 未能从LLM获取有效的分解计划，或计划为空。将使用默认检索。")
			steps.append(LogicalStep.from_dict(
				{"operator_type": "retrieve", "params": {"query": natural_question}},
				original_question_id, 1
			))

		print(f"  [PlanningOperator] 分解完成，生成 {len(steps)} 个步骤。")
		return steps


class RetrievalOperator:
	def __init__(self, kb: SimulatedKnowledgeBase): self.kb = kb

	def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]: return self.kb.retrieve(query, top_k)


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


# --- KAGLikeEngine 类 ---
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
				if not query:
					print(f"  [Engine Warning] 步骤 {step.step_id} (retrieve) 查询为空."); step.result = []
				else:
					step.result = self.retrieval_operator.retrieve(query)
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
						if op_type == "retrieve" and isinstance(prev_res, list):
							for item_dict in prev_res:
								if isinstance(item_dict, dict) and "content" in item_dict: context_snippets.append(
									item_dict["content"])
						elif isinstance(prev_res, str):
							context_snippets.append(prev_res)
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
				# Corrected line below
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
				all_ctx = [item["content"] for s in working_memory.values() if
						   s.status == "completed" and s.operator_type == "retrieve" and isinstance(s.result, list) for
						   item in s.result if isinstance(item, dict) and "content" in item] + \
						  [str(s.result) for s in working_memory.values() if
						   s.status == "completed" and s.operator_type == "reason" and isinstance(s.result, str)]
				if all_ctx:
					final_answer = self.reasoning_operator.reason(all_ctx, natural_question)
				else:
					final_answer = "未能收集到足够信息进行最终总结。"
				break

		if not is_solved and current_iteration >= max_iterations:
			print("[Engine] 达到最大迭代次数。")
			all_ctx = [item["content"] for s in working_memory.values() if
					   s.status == "completed" and s.operator_type == "retrieve" and isinstance(s.result, list) for item
					   in s.result if isinstance(item, dict) and "content" in item] + \
					  [str(s.result) for s in working_memory.values() if
					   s.status == "completed" and s.operator_type == "reason" and isinstance(s.result, str)]
			if all_ctx:
				final_answer = self.reasoning_operator.reason(all_ctx,
															  f"请根据以下信息总结关于 '{natural_question}' 的答案。")
			else:
				final_answer = "处理超时，未能找到明确答案。"
		print(f"\n--- [Engine] 问题解决结束 ({current_question_id}) ---");
		return final_answer


# --- 主程序入口 ---
if __name__ == "__main__":
	sim_kb = SimulatedKnowledgeBase()
	planning_op = PlanningOperator()
	retrieval_op = RetrievalOperator(kb=sim_kb)
	reasoning_op = ReasoningOperator()
	engine = KAGLikeEngine(planning_op, retrieval_op, reasoning_op)
	questions = [
		"Python的创造者是谁？它何时发布？",
		"iPhone是什么，它是由谁设计的？",
		"中国的首都是哪里？",
		"请告诉我关于大型语言模型的基本信息以及Python的发布年份。"
	]
	for i, q in enumerate(questions):
		print(f"\n==================== 处理问题 {i + 1}: {q} ====================")
		answer = engine.solve(q, max_iterations=3)
		print(f"\n>>> 最终答案 Q{i + 1} ('{q}'): {answer}")
		print("=======================================================\n")