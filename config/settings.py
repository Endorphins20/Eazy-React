"""
配置管理模块
管理所有系统配置和常量
"""
import os

# Chroma 数据库配置
CHROMA_PERSIST_DIRECTORY = "chroma_db"


# API Keys 配置
API_KEY = "sk-"
EMBEDDING_API_KEY = "sk-"
MODEL_NAME = "qwen-plus"
EMBEDDING_MODEL_NAME = "text-embedding-v2"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Pipeline 配置
MAX_ITERATIONS = 10
TRUNCATE_LENGTH = 150

# 文档路径配置
DOCUMENT_FOLDER_PATH = r'./files'
