import dashscope
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import List
from http import HTTPStatus
chroma_client = chromadb.Client()

class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str  ):
        self.api_key = api_key

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            text (str): 要生成 embedding 的文本.

        Returns:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        response = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v3,
            api_key=self.api_key,
            input=text
        )
        #print(response.output)
        if response.status_code == HTTPStatus.OK: # 假设 200 是 HTTP 状态码 OK
            return response.output['embeddings'][0]['embedding'] # 假设返回的嵌入向量在 response.output['embedding']
        else:
            raise Exception(f"Failed to get embedding for text: {text}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]

# 使用示例
if __name__ == '__main__':
    from langchain_core.documents import Document
    embedding_function = QwenEmbeddingFunction(api_key='sk-af4423da370c478abaf68b056f547c6e')
    collection = chroma_client.create_collection(name="my_collection", embedding_function=embedding_function)
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"source": "fish-pets-doc"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"source": "bird-pets-doc"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
    vectorstore = Chroma.from_documents(documents, embedding=embedding_function)
    # print(vectorstore.similarity_search("cat"))\
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    print(retriever.batch(["cat", "shark"]))