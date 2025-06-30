"""
提示词模板模块
包含所有系统使用的提示词模板
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json


class BasePrompt:
    """基础提示词类"""
    def __init__(self, template: str, variables: List[str]):
        self.template_str = template
        self.variables = variables

    def format(self, **kwargs) -> str:
        formatted_prompt = self.template_str
        for var_name in self.variables:
            if var_name not in kwargs:
                raise ValueError(f"Missing variable: {var_name}")
            formatted_prompt = formatted_prompt.replace(f"${{{var_name}}}", str(kwargs[var_name]))
        return formatted_prompt


class PlannerPrompt(BasePrompt):
    """规划器提示词模板"""
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
    3.  **差距评估 (Gap Assessment)**: 对比当前所有已知信息（来自历史任务的成功结果）和用户原始问题的最终目标（特别是那些要求"详细说明"、"具体要求"的部分），目前还缺少哪些【具体的细节】或【明确被引用的标准/文件（如通则XXXX）的详细内容】？
    4.  **决策制定 (`final_thought` 和 `plan_status`)**:
        * **已解决 (finished)?** 【严格条件】当且仅当：所有用户原始问题中明确要求的方面都得到了解答，所有在推理过程中被识别为需要进一步探究的 `new_questions_or_entities`（尤其是标准/通则的细节）都已经被成功检索其详细内容、并被后续的 `DeduceExecutor` 步骤分析确认信息充分（`is_sufficient: true`）后，才可判断为 `finished`。此时，规划一个 `FinishAction` 任务。`final_thought` 应总结是如何一步步解决的。
        * **无法解决 (cannot_proceed)?** 若关键信息缺失，且历史中的 `new_questions_or_entities` 已尝试检索但无果（例如，多次相关的 `RetrievalExecutor` 对特定标准编号的查询返回空或不相关内容），或工具无法获取，则 `plan_status: "cannot_proceed"`。`final_thought` 应解释原因。
        * **需要更多步骤 (requires_more_steps)?** 否则，即信息尚不完整，或有新的线索需要追查，则 `plan_status: "requires_more_steps"`。
    5.  **行动规划 (`next_steps` - 如果 `requires_more_steps`)**:
        * **【最高优先级：递归深挖引用和新线索】**: 
            * 如果历史中有 `DeduceExecutor` 的结果在其 `new_questions_or_entities` 列表中明确指出了需要查询【特定标准、文件编号或实体】的详细内容（例如，"查询通则1107的详细内容"，"获取《中国药典》通则1107中关于微生物限度的具体标准是什么？"），并且这些条目【对应的详细内容尚未通过后续的 `RetrievalExecutor` 成功获取并被充分分析过】：
                * **立即规划**一个 `RetrievalExecutor` 任务。其 `logic_input.query` 应【直接针对这些 `new_questions_or_entities` 中的一项进行精确查询】。例如，如果 `new_questions_or_entities` 中有"通则1107中关于微生物限度的具体标准是什么？"，则检索查询就应该是这个或非常相似的，以确保能命中包含该通则细节的文档。
                * 这是当前迭代最优先要处理的任务。通常，处理完一个这样的关键缺失信息点后，就应该结束本轮规划，等待下一轮迭代基于新获取的信息进行再规划和整合。
        * **整合信息**: 如果上一步是检索（特别是针对 `new_questions_or_entities` 的检索），并且检索到了有价值的新信息，下一步通常是规划 `DeduceExecutor` 任务。其 `logic_input.reasoning_goal` 设为"整合新检索到的关于[新实体/标准]的信息，并结合先前关于[相关主题]的结论，以更全面地回答[某个子问题或原始问题中与此新信息相关的方面]"。`logic_input.context_data` 应精确引用【所有相关的】检索结果和先前步骤的关键结论。
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

    def __init__(self):
        super().__init__(self.USER_TEMPLATE, ["user_query", "available_executors_description", "task_history"])


class DeducePrompt(BasePrompt):
    """推理器提示词模板"""
    SYSTEM_PROMPT = """
    你是一位极其严谨和细致的AI推理专家。你的任务是根据提供的"上下文信息"，精确地回答或完成"推理目标"。
    你必须对上下文中出现的所有【标准、通则、法规、文件编号、或被明确引用的专有名称】（例如"通则0104"、"中华人民共和国药典通则1107"、"GB/T标准X"）进行严格的审视。

    请严格按照以下JSON格式输出你的结论和评估：
    ```json
    {
      "answer_summary": "string (对推理目标的直接、简洁的回答或总结。如果信息不足，明确说明当前已知什么，并【清晰指出具体还缺少哪些信息或哪些被引用的标准/文件细节】才能完整回答推理目标。例如：'根据上下文，正柴胡饮颗粒的微生物限度应参照通则1107，但当前未提供通则1107的具体限值。')",
      "is_sufficient": boolean (true 如果你认为提供的上下文信息【已包含回答推理目标所需的所有必要细节，特别是所有被明确引用或提及的标准/文件的具体内容，无需任何补充查询】，否则 false),
      "new_questions_or_entities": [
        "string" // 【至关重要指令 - 必须严格执行】:
                  // 1. 如果 is_sufficient 为 false，这里【必须】列出为了获得【缺失的关键细节】需要查询的具体问题。
                  // 2. 【特别注意】：如果在上下文中发现任何【标准、通则、法规、文件编号或专有名称】被明确引用（例如"详见通则1107"、"依据GB/T标准X执行"、"按照XYZ操作手册进行"），但该标准/文件/名称的【具体内容、定义或详细条款并未在当前提供给你的上下文中清晰、完整地列出】，那么你【必须】将一个用于获取该标准/文件/名称【详细内容】的明确查询（例如："查询《中国药典》通则1107关于微生物限度的详细规定"、"获取GB/T标准X的全文"、"查找XYZ操作手册中关于XX部分的具体步骤"）作为条目加入此列表。并且，在这种情况下，除非该引用对于当前推理目标而言完全不重要，否则 `is_sufficient` 通常应为 `false`。
                  // 确保列表中的每一项都是一个明确的、可用于下一步检索的查询目标。如果信息完全充分且所有引用都有细节支撑，则为空列表[]。
      ]
    }
    ```
    - 不要添加任何额外的解释或markdown标记，只输出JSON对象。
    """
    
    USER_TEMPLATE = "推理目标:\n${reasoning_goal}\n\n上下文信息:\n${context_data}\n\n请输出JSON格式的推理结果:"

    def __init__(self):
        super().__init__(self.USER_TEMPLATE, ["reasoning_goal", "context_data"])


class CodeExecutionPrompt(BasePrompt):
    """代码执行提示词模板"""
    SYSTEM_PROMPT = "你是一个Python代码生成和执行助手。"
    
    USER_TEMPLATE = """请根据以下指令和相关数据，生成一段Python代码来解决问题。
代码必须通过 `print()` 输出其最终计算结果。不要包含任何解释或注释，只输出纯代码。

指令:
${code_generation_prompt}

相关数据 (如果提供):
${relevant_data}

生成的Python代码 (请确保它只包含代码本身，并用print()输出结果):"""

    def __init__(self):
        super().__init__(self.USER_TEMPLATE, ["code_generation_prompt", "relevant_data"])


class UserProvidedReferGeneratorPrompt(BasePrompt):
    """用户提供的引用生成器提示词模板"""
    
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
            f"你是一个信息分析专家，今天是{get_now(language='zh')}。" + 
            "基于给定的引用信息回答问题。\n输出答案，如果答案中存在引用信息，则需要reference的id字段，如果不是检索结果，则不需要标记引用\n输出时，不需要重复输出参考文献\n引用要求，使用类似<reference id=\"chunk:1_2\"></reference>表示\n如果根据引用信息无法回答，则使用模型内的知识回答，但是必须通过合适的方式提示用户，是基于检索内容还是引用文档\n示例1：\n任务过程上下文：\n根据常识岳父是妻子的爸爸，所以需要首先找到张三的妻子，然后找到妻子的爸爸\n给定的引用信息：'\nreference：\n[\n{\n    \"content\": \"张三 妻子 王五\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_1\"\n},\n{\n    \"content\": \"王五 父亲 王四\",\n    \"document_name\": \"张三介绍\",\n    \"id\": \"chunk:1_2\"\n}\n]'\n问题：'张三的岳父是谁？'\n\n张三的妻子是王五<reference id=\"chunk:1_1\"></reference>，而王五的父亲是王四<reference id=\"chunk:1_2\"></reference>，所以张三的岳父是王四\n\n\n输出语调要求通顺，不要有机械感，输出的语言要和问题的语言保持一致\n任务过程上下文信息：'${summary_of_executed_steps}'\n给定的引用信息：'${formatted_references}'\n问题：'${user_query}'"
        )
        self.template_en = self.template_zh
        current_template = self.template_zh if language == "zh" else self.template_en
        super().__init__(current_template, ["summary_of_executed_steps", "user_query", "formatted_references"])

    def format(self, summary_of_executed_steps: str, user_query: str, retrieved_references: List[Dict]) -> str:
        ref_list_for_prompt = []
        for i, ref_item in enumerate(retrieved_references):
            ref_list_for_prompt.append({
                "content": ref_item.get("content", ""),
                "document_name": ref_item.get("metadata", {}).get("source_name", f"检索文档{i + 1}"),
                "id": ref_item.get("metadata", {}).get("id", f"retrieved_chunk_{i}")
            })
        formatted_references_str = json.dumps(ref_list_for_prompt, ensure_ascii=False, indent=2)
        return super().format(
            summary_of_executed_steps=summary_of_executed_steps,
            user_query=user_query,
            formatted_references=formatted_references_str
        )
