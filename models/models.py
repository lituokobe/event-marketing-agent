import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.paths import ENV_PATH
load_dotenv(ENV_PATH)

ALI_API_KEY = os.getenv("ALI_API_KEY")
ALI_BASE_URL = os.getenv("ALI_BASE_URL")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")

qwen_llm = ChatOpenAI(
    model = 'qwen-plus',
    temperature = 0.8,
    api_key = ALI_API_KEY,
    base_url = ALI_BASE_URL,
    max_tokens = 200
)