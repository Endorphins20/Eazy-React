"""
主程序入口模块
包含系统的主要运行逻辑
"""
import asyncio
import os
from typing import List
from langchain_core.documents import Document

from config.settings import (
    API_KEY, EMBEDDING_API_KEY, MODEL_NAME, BASE_URL,
    DOCUMENT_FOLDER_PATH, MAX_ITERATIONS,CHROMA_PERSIST_DIRECTORY,EMBEDDING_BASE_URL,EMBEDDING_MODEL_NAME
)
from model.client import OpenAIChatLLM
from knowledge.knowledge_base import ChromaKnowledgeBase
from prompts.templates import (
    PlannerPrompt, DeducePrompt, CodeExecutionPrompt, UserProvidedReferGeneratorPrompt
)
from executors import (
    RetrievalExecutor, DeduceExecutor, CodeExecutor, FinishExecutor
)
from planning.planner import Planner
from generation.answer_generator import AnswerGenerator
from pipeline.iterative_pipeline import IterativePipeline
from utils.document_loader import load_documents_from_folder, create_default_test_documents

try:
    from model.embedding import QwenEmbeddingFunction
except ImportError:
    print("[Warning] tongyiembedding module not found. Embedding may not work.")
    QwenEmbeddingFunction = None


class KAGSystem:
    """KAG系统主类"""
    
    def __init__(self):
        self.persist_directory = CHROMA_PERSIST_DIRECTORY
        self.llm_client = None
        self.kb = None
        self.pipeline = None
    
    async def initialize(self, force_rebuild_kb: bool = False):
        """初始化系统"""
        print("🔧 初始化KAG系统...")
        
        # 初始化LLM客户端
        api_key = API_KEY
        embedding_api_key = EMBEDDING_API_KEY

        model_name = MODEL_NAME
        embedding_model_name = EMBEDDING_MODEL_NAME
        
        base_url = BASE_URL
        embedding_base_url = EMBEDDING_BASE_URL
        
        if not api_key or not embedding_api_key:
            raise ValueError("错误：请确保 API keys 已正确设置。")
        
        self.llm_client = OpenAIChatLLM(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化嵌入函数
        if QwenEmbeddingFunction is None:
            raise ImportError("QwenEmbeddingFunction not available. Please install tongyiembedding.")
        
        try:
            embedding_function = QwenEmbeddingFunction(api_key=embedding_api_key,base_url=embedding_base_url, model=embedding_model_name)
        except Exception as e:
            print(f"  [Embedding Error] QwenEmbeddingFunction 初始化失败: {e}")
            raise
        
        # 加载文档
        print("📚 加载文档...")
        documents = self._load_documents()
        
        # 初始化知识库
        print("🗄️ 初始化知识库...")
        try:
            self.kb = ChromaKnowledgeBase(
                initial_documents=documents,
                embedding_function=embedding_function,
                force_rebuild=force_rebuild_kb,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"创建Chroma知识库失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 初始化组件
        print("⚙️ 初始化系统组件...")
        self._initialize_components()
        
        print("✅ KAG系统初始化完成！")
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        documents = []
        
        # 尝试从指定文件夹加载文档
        if os.path.exists(DOCUMENT_FOLDER_PATH):
            documents = load_documents_from_folder(DOCUMENT_FOLDER_PATH)
        
        # 如果没有加载到文档，使用默认测试文档
        if not documents:
            print("⚠️ 未找到文档，使用默认测试文档")
            documents = create_default_test_documents()
        
        print(f"📄 共加载 {len(documents)} 个文档")
        return documents
    
    def _initialize_components(self):
        """初始化系统组件"""
        # 初始化提示词模板
        planner_prompt = PlannerPrompt()
        deduce_prompt_template = DeducePrompt()
        code_exec_prompt = CodeExecutionPrompt()
        refer_generator_prompt = UserProvidedReferGeneratorPrompt(language="zh")
        
        # 初始化执行器
        retrieval_executor = RetrievalExecutor(kb=self.kb)
        deduce_executor = DeduceExecutor(self.llm_client, deduce_prompt_template)
        code_executor = CodeExecutor(self.llm_client, code_exec_prompt)
        finish_executor = FinishExecutor()
        
        executors_map = {
            "RetrievalExecutor": retrieval_executor,
            "DeduceExecutor": deduce_executor,
            "CodeExecutor": code_executor,
            "FinishAction": finish_executor
        }
        
        # 初始化规划器和生成器
        planner = Planner(self.llm_client, planner_prompt)
        generator = AnswerGenerator(self.llm_client, refer_generator_prompt)
        
        # 初始化流水线
        self.pipeline = IterativePipeline(
            planner, executors_map, generator, max_iterations=MAX_ITERATIONS
        )
    
    async def query(self, user_query: str) -> str:
        """处理用户查询"""
        if not self.pipeline:
            raise RuntimeError("系统未初始化，请先调用 initialize() 方法。")
        
        print(f"\n🚀 处理查询: \"{user_query}\"")
        final_answer = await self.pipeline.run(user_query)
        print(f"\n🏁 最终答案: {final_answer}")
        return final_answer


async def run_main_example():
    """运行主程序示例"""
    print("🎯 启动KAG系统示例...")
    
    # 创建系统实例
    kag_system = KAGSystem()
    
    # 初始化系统
    await kag_system.initialize(force_rebuild_kb=False)
    
    # 运行查询
    user_query = "财务报销制度说明中关于报销国内或国外住宿费的规定具体是什么？"
    final_answer = await kag_system.query(user_query)
    
    print(f"\n🎉 查询完成！")
    return final_answer


if __name__ == "__main__":
    print("开始执行深度类递归V2优化版主逻辑...")
    asyncio.run(run_main_example())
