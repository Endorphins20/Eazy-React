# KAG系统模块化重构

## 项目结构

```
kag/
├── __init__.py                    # 包初始化文件
├── main.py                       # 主程序入口，包含KAGSystem类
├── run_kag.py                    # 简化的运行入口
│
├── config/                       # 配置模块
│   ├── __init__.py
│   └── settings.py              # 系统配置和常量
│
├── core/                        # 核心数据结构和管理器
│   ├── __init__.py
│   ├── data_structures.py       # 数据结构定义
│   └── context_manager.py       # 上下文管理器
│
├── prompts/                     # 提示词模板
│   ├── __init__.py
│   └── templates.py             # 所有提示词模板
│
├── llm/                         # LLM客户端
│   ├── __init__.py
│   └── client.py                # OpenAI兼容的LLM客户端
│
├── knowledge/                   # 知识库模块
│   ├── __init__.py
│   └── knowledge_base.py        # Chroma向量知识库实现
│
├── executors/                   # 执行器模块
│   ├── __init__.py
│   ├── base_executor.py         # 执行器基类
│   ├── retrieval_executor.py    # 检索执行器
│   ├── deduce_executor.py       # 推理执行器
│   ├── code_executor.py         # 代码执行器
│   └── finish_executor.py       # 完成执行器
│
├── planning/                    # 规划模块
│   ├── __init__.py
│   └── planner.py              # 任务规划器
│
├── generation/                  # 生成模块
│   ├── __init__.py
│   └── answer_generator.py     # 答案生成器
│
├── pipeline/                    # 流水线模块
│   ├── __init__.py
│   └── iterative_pipeline.py   # 迭代式执行流水线
│
└── utils/                       # 工具模块
    ├── __init__.py
    └── document_loader.py       # 文档加载工具
```

## 模块说明

### 1. 配置模块 (config/)
- **settings.py**: 集中管理所有系统配置，包括API密钥、模型名称、文件路径等

### 2. 核心模块 (core/)
- **data_structures.py**: 定义Task、DeduceExecutorOutput等核心数据结构
- **context_manager.py**: 管理任务上下文和执行历史

### 3. 提示词模块 (prompts/)
- **templates.py**: 包含所有提示词模板，便于统一管理和修改

### 4. LLM模块 (llm/)
- **client.py**: 封装与大语言模型的交互逻辑

### 5. 知识库模块 (knowledge/)
- **knowledge_base.py**: 基于Chroma的向量知识库实现

### 6. 执行器模块 (executors/)
- **base_executor.py**: 所有执行器的基类
- **retrieval_executor.py**: 负责知识库检索
- **deduce_executor.py**: 负责推理和判断
- **code_executor.py**: 负责代码生成和执行
- **finish_executor.py**: 标记任务完成

### 7. 规划模块 (planning/)
- **planner.py**: 基于当前状态规划下一步行动

### 8. 生成模块 (generation/)
- **answer_generator.py**: 基于执行历史生成最终答案

### 9. 流水线模块 (pipeline/)
- **iterative_pipeline.py**: 协调整个任务执行流程

### 10. 工具模块 (utils/)
- **document_loader.py**: 文档加载和处理功能

## 使用方式

### 方式1: 使用主程序类
```python
import asyncio
from main import KAGSystem

async def main():
    # 创建系统实例
    kag_system = KAGSystem()
    
    # 初始化系统
    await kag_system.initialize()
    
    # 执行查询
    result = await kag_system.query("你的问题")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 方式2: 直接运行
```bash
python run_kag.py
```

## 优势

1. **模块化清晰**: 每个模块职责单一，便于理解和维护
2. **易于扩展**: 新增功能只需要在对应模块中添加，不影响其他部分
3. **便于测试**: 每个模块可以独立测试
4. **代码复用**: 公共功能抽取到独立模块，便于复用
5. **配置集中**: 所有配置项集中管理，便于修改
6. **依赖清晰**: 模块间依赖关系清晰，便于理解系统架构

## 迁移指南

原来的`kag7.0.py`文件已经被完全模块化重构。如果需要使用原有功能：

1. 所有的类和函数都已经按功能分类到对应模块
2. 主要的入口点是`KAGSystem`类
3. 配置项现在在`config/settings.py`中管理
4. 可以通过`run_kag.py`快速运行系统

这种重构使得代码更加SOTA（State-of-the-Art），符合现代软件工程的最佳实践。
