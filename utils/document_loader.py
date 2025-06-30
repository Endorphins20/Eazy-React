"""
文档处理工具模块
包含文档加载和处理功能
"""
import os
from typing import List
from langchain_core.documents import Document

try:
    from utils.md2Document import create_document_from_md
except ImportError:
    print("[Warning] md2Document module not found. MD file loading may not work.")
    create_document_from_md = None


def load_documents_from_folder(folder_path: str, file_extension: str = '.md') -> List[Document]:
    """从文件夹加载指定类型的文档"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"[Warning] 文档文件夹 {folder_path} 不存在。")
        return documents
    
    # 获取文件夹中所有指定类型的文件
    files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]
    
    print(f"[DocumentLoader] 在 {folder_path} 中找到 {len(files)} 个 {file_extension} 文件")
    
    # 加载每个文件
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if file_extension == '.md' and create_document_from_md:
                document = create_document_from_md(file_path)
                documents.append(document)
            else:
                # 通用文件加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                document = Document(
                    page_content=content,
                    metadata={"source": file_path, "filename": file}
                )
                documents.append(document)
        except Exception as e:
            print(f"[DocumentLoader] 加载文件 {file_path} 失败: {e}")
    
    print(f"[DocumentLoader] 成功加载 {len(documents)} 个文档")
    return documents


def create_default_test_documents() -> List[Document]:
    """创建默认的测试文档"""
    return [
        Document(
            page_content="【正柴胡饮颗粒】药品说明书（摘要）\n【检查】应符合颗粒剂项下有关的各项规定（详见《中国药典》通则0104）。其余按品种标准执行。",
            metadata={"id": "zchy_spec_main_v3", "source_name": "正柴胡饮颗粒说明书摘要"}
        ),
        Document(
            page_content="《中国药典》通则0104 - 颗粒剂（概述）\n本通则为颗粒剂的通用质量控制要求。具体检查项目包括：【性状】、【鉴别】、【检查】（如粒度、水分、溶化性、装量差异、微生物限度等）、【含量测定】等。\n关于【微生物限度】，颗粒剂应符合现行版《中国药典》通则1105（非无菌产品微生物限度检查：微生物计数法）、通则1106（非无菌产品微生物限度检查：控制菌检查法）和通则1107（非无菌药品微生物限度标准）的相关规定。本通则0104不详述这些微生物限度标准的具体限值。",
            metadata={"id": "tg0104_overview_v3", "source_name": "药典通则0104概述"}
        ),
        Document(
            page_content="《中国药典》通则1107 - 非无菌药品微生物限度标准（节选）\n本标准规定了各类非无菌药品所需控制的微生物限度。\n对于口服固体制剂（如颗粒剂）：\n1. 需氧菌总数：每1g（或1ml）不得过1000 cfu。\n2. 霉菌和酵母菌总数：每1g（或1ml）不得过100 cfu。\n3. 控制菌：每1g（或1ml）不得检出大肠埃希菌；对于含动物脏器、组织或血液成分的制剂，每10g（或10ml）不得检出沙门菌。",
            metadata={"id": "tg1107_details_v3", "source_name": "药典通则1107-微生物限度细节"}
        ),
        Document(
            page_content="《中国药典》通则0104 - 颗粒剂（粒度与水分细节）\n【粒度】（通则0982第二法）不能通过一号筛（2.00mm）与能通过五号筛（0.250mm）的药粉总和不得超过总重量的15％。\n【水分】（通则0832第一法）中药颗粒剂不得过8.0％。",
            metadata={"id": "tg0104_sizewater_v3", "source_name": "药典通则0104-粒度与水分细节"}
        )
    ]
