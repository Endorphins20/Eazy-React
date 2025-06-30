# KAG (Knowledge-Augmented Generation) 系统

> 一个基于知识增强生成的智能问答系统，集成了检索、推理和代码执行能力

## 🚀 项目简介

KAG系统是一个先进的知识增强生成框架，通过结合向量检索、逻辑推理和代码执行，为用户提供准确、全面的问答服务。系统采用模块化设计，支持多种执行器类型，可以处理复杂的多步骤查询任务。

## ✨ 主要特性

- 🧠 **智能规划器**：自动分解复杂查询为可执行的步骤
- 🔍 **向量检索**：基于ChromaDB的高效文档检索
- 🤖 **逻辑推理**：通过LLM进行深度推理分析
- 💻 **代码执行**：支持Python代码的动态执行
- 📚 **知识库管理**：灵活的文档加载和管理机制
- 🔄 **迭代执行**：多轮次的任务执行和结果优化

## 🏗️ 系统架构

```
KAG System
├── Planning Layer (规划层)
│   └── Planner - 任务分解和执行计划
├── Execution Layer (执行层)
│   ├── RetrievalExecutor - 文档检索
│   ├── DeduceExecutor - 逻辑推理
│   ├── CodeExecutor - 代码执行
│   └── FinishExecutor - 任务完成
├── Knowledge Layer (知识层)
│   └── ChromaKnowledgeBase - 向量数据库
├── Generation Layer (生成层)
│   └── AnswerGenerator - 答案生成
└── Pipeline Layer (流水线层)
    └── IterativePipeline - 迭代执行控制
```

## 📁 项目结构

```
kag/
├── config/                  # 配置管理
│   ├── __init__.py
│   └── settings.py         # 系统配置
├── core/                   # 核心组件
│   ├── __init__.py
│   ├── context_manager.py  # 上下文管理
│   └── data_structures.py  # 数据结构定义
├── executors/              # 执行器模块
│   ├── __init__.py
│   ├── base_executor.py    # 执行器基类
│   ├── code_executor.py    # 代码执行器
│   ├── deduce_executor.py  # 推理执行器
│   ├── finish_executor.py  # 完成执行器
│   └── retrieval_executor.py # 检索执行器
├── generation/             # 答案生成
│   ├── __init__.py
│   └── answer_generator.py # 答案生成器
├── knowledge/              # 知识库管理
│   ├── __init__.py
│   └── knowledge_base.py   # 知识库实现
├── model/                  # 模型接口
│   ├── __init__.py
│   ├── client.py          # LLM客户端
│   └── embedding.py       # 嵌入模型
├── pipeline/              # 执行流水线
│   ├── __init__.py
│   └── iterative_pipeline.py # 迭代流水线
├── planning/              # 任务规划
│   ├── __init__.py
│   └── planner.py         # 任务规划器
├── prompts/               # 提示词模板
│   ├── __init__.py
│   └── templates.py       # 提示词定义
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── document_loader.py  # 文档加载器
│   └── md2Document.py     # Markdown转换
├── files/                 # 文档文件
│   ├── AT-FIN-EXPENSE-001.md
│   ├── AT-FIN-EXPENSE-002.md
│   ├── AT-HR-BENEFIT-001.md
│   ├── AT-HR-MANUAL-001.md
│   ├── AT-HR-ONBOARD-001.md
│   ├── AT-IT-GUIDE-001.md
│   └── AT-IT-SECURITY-001.md
├── chroma_db/             # 向量数据库存储
├── main.py               # 主程序入口
└── README.md             # 项目说明文档
```

## 🛠️ 安装配置

### 环境要求

- Python 3.8+
- pip 或 conda 包管理器

### 依赖安装

```bash
pip install chromadb
pip install langchain-core
pip install openai
pip install asyncio
pip install python-dotenv
```

### 配置设置

1. 在 `config/settings.py` 中配置您的API密钥：

```python
# API Keys 配置
API_KEY = "your-api-key-here"
EMBEDDING_API_KEY = "your-embedding-api-key-here"
MODEL_NAME = "qwen-plus"
EMBEDDING_MODEL_NAME = "text-embedding-v2"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

2. 将您的文档文件放置在 `files/` 目录下

## 🚀 快速开始

### 运行主程序

```bash
python main.py
```

## 🔧 核心组件详解

### 1. 规划器 (Planner)
- 自动分析用户查询的复杂度
- 生成执行计划和步骤分解
- 选择合适的执行器组合

### 2. 执行器 (Executors)
- **RetrievalExecutor**: 从知识库检索相关文档
- **DeduceExecutor**: 基于上下文进行逻辑推理
- **CodeExecutor**: 执行Python代码并返回结果
- **FinishExecutor**: 标记任务完成

### 3. 知识库 (Knowledge Base)
- 基于ChromaDB的向量存储
- 支持文档的自动索引和检索
- 使用通义千问嵌入模型

### 4. 迭代流水线 (Iterative Pipeline)
- 控制多轮次的任务执行
- 管理上下文状态和结果传递
- 自动处理执行错误和重试

## 🔍 配置说明

### 系统配置参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `MAX_ITERATIONS` | 最大迭代次数 | 10 |
| `CHROMA_PERSIST_DIRECTORY` | 数据库存储目录 | "chroma_db" |
| `DOCUMENT_FOLDER_PATH` | 文档文件夹路径 | "./files" |

### API配置
- 支持通义千问系列模型
- 兼容OpenAI API格式
- 可配置不同的base_url和模型名称

