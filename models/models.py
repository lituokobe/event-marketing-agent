from langchain_openai import ChatOpenAI

ALI_BASE_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_API_KEY = ""

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = ""

GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_API_KEY = ""

qwen_llm = ChatOpenAI(
    model = 'qwen-plus',
    temperature = 0.1,
    api_key = ALI_API_KEY,
    base_url = ALI_BASE_URL,
    max_tokens = 500
)

deepseek_llm = ChatOpenAI(
    model = 'deepseek-chat',
    temperature = 0.1,
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_BASE_URL,
    max_tokens = 500
)

glm_llm = ChatOpenAI(
    model = 'glm-4.6',
    temperature = 0.1,
    api_key = GLM_API_KEY,
    base_url = GLM_BASE_URL,
    max_tokens = 500
)