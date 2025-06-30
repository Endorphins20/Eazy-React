"""
知识库模块
包含向量知识库的实现
"""
import os
import shutil
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document
from langchain_chroma import Chroma


class ChromaKnowledgeBase:
    """基于Chroma的向量知识库"""
    
    def __init__(self, embedding_function: Callable,
                 initial_documents: Optional[List[Document]] = None,
                 persist_directory: str = "chroma_db_kag_recursive",
                 collection_name: str = "kag_recursive_documents",
                 force_rebuild: bool = False):
        print(f"  [ChromaKB] 初始化知识库: {persist_directory}, 集合: {collection_name}")
        
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None
        
        # 强制重建时删除现有目录
        if force_rebuild and os.path.exists(persist_directory):
            print(f"  [ChromaKB] force_rebuild=True, 删除目录: {persist_directory}")
            try:
                shutil.rmtree(persist_directory)
            except OSError as e:
                print(f"  [ChromaKB Error] 删除目录失败: {e}.")
        
        # 尝试加载现有向量库
        if os.path.exists(persist_directory) and not force_rebuild:
            print(f"  [ChromaKB] 从 '{persist_directory}' 加载已存在向量库...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                print(f"  [ChromaKB] 成功加载向量库 '{self.collection_name}'.")
            except Exception as e:
                print(f"  [ChromaKB Error] 从 '{persist_directory}' 加载失败: {e}. 将尝试新建。")
                self.vectorstore = None
        
        # 创建新向量库
        if self.vectorstore is None:
            if initial_documents:
                print(f"  [ChromaKB] 为 {len(initial_documents)} 个文档构建新向量库...")
                self.vectorstore = Chroma.from_documents(
                    documents=initial_documents,
                    embedding=self.embedding_function,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                print(f"  [ChromaKB] 新向量库构建并持久化完成。")
            else:
                print(f"  [ChromaKB] 无初始文档，创建空的持久化集合。")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                print(f"  [ChromaKB] 空的持久化 Chroma 集合 '{self.collection_name}' 已准备就绪。")

    def add_documents(self, documents: List[Document]):
        """添加文档到知识库"""
        if not self.vectorstore:
            if documents:
                print(f"  [ChromaKB] Vectorstore 为空, 尝试从当前 {len(documents)} 个文档创建...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_function,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                print(f"  [ChromaKB] 基于新文档创建并持久化完成。")
                return
            else:
                print("  [ChromaKB Error] Vectorstore 未初始化且无文档可添加.")
                return
        
        if documents:
            print(f"  [ChromaKB] 向集合 '{self.collection_name}' 添加 {len(documents)} 个新文档...")
            self.vectorstore.add_documents(documents)
            print(f"  [ChromaKB] 文档添加完成。")

    def retrieve(self, query: str, top_k: int = 3, 
                filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """从知识库检索相关文档"""
        if not self.vectorstore:
            print("  [ChromaKB Error] Vectorstore 未初始化.")
            return []
        
        try:
            if (self.vectorstore._collection is None or 
                self.vectorstore._collection.count() == 0):
                print("  [ChromaKB] 知识库集合为空或未正确加载。")
                return []
        except Exception as e:
            print(f"  [ChromaKB Warning] 无法获取集合计数: {e}")
        
        print(f"  [ChromaKB] 检索查询 '{query}', top_k={top_k}, 过滤器: {filter_dict}...")
        
        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filter_dict
            )
        except Exception as e:
            print(f"  [ChromaKB Error] Chroma similarity search failed: {e}")
            return []
        
        processed_results = [
            {
                "id": doc.metadata.get("id", f"retrieved_{i}"),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for i, (doc, score) in enumerate(results_with_scores)
        ]
        
        print(f"  [ChromaKB] 检索到 {len(processed_results)} 个文档。")
        return processed_results
