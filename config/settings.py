"""
配置管理模块
管理所有系统配置和常量
"""
import os

# Chroma 数据库配置
CHROMA_PERSIST_DIRECTORY = "chroma_db_kag_recursive1"


# API Keys 配置
API_KEY = "sk-"
EMBEDDING_API_KEY = "sk-af4423da370c478abaf68b056f547c6e"
MODEL_NAME = "qwen-plus"
EMBEDDING_MODEL_NAME = "text-embedding-v2"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Pipeline 配置
MAX_ITERATIONS = 10
TRUNCATE_LENGTH = 150

# 文档路径配置
DOCUMENT_FOLDER_PATH = r'E:\workspaceE\kag\凌云科技文档'
