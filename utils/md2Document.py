import markdown
from langchain_core.documents import Document
import os

# 读取Markdown文件内容
def read_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    return md_content

# 将Markdown文件内容转换为Document对象
def create_document_from_md(file_path, metadata=None):
    md_content = read_md_file(file_path)
    html_content = markdown.markdown(md_content)  # 如果需要将Markdown转换为HTML
    # 确保metadata是一个字典
    if metadata is None:
        metadata = {}
    return Document(page_content=html_content, metadata=metadata)

# 示例用法
if __name__ == "__main__":
    # 设置目标文件夹路径
    folder_path = r'D:\Master\llm\database\kag\测试数据集'  # Windows路径

    # 获取文件夹中所有文件
    files = os.listdir(folder_path)

    # 筛选出所有 .md 文件
    md_files = [file for file in files if file.endswith('.md')]

    # 创建Document对象列表
    documents = []
    for file in md_files:
        file_path = os.path.join(folder_path, file)  # 拼接完整的文件路径
        document = create_document_from_md(file_path)  # 创建Document对象
        documents.append(document)

    # 打印结果
    for doc in documents:
        print(f"Page Content: {doc.page_content}...")  # 打印前100个字符
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)