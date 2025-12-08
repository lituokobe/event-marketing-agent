from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from data.paths import QWEN_EMBEDDING_PATH
from functionals.log_utils import logger_chatflow

#TODO: Initialize embedding model

# """Web download approach"""
# import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
# class Qwen3CustomEmbedding(Embeddings):
#     """
#     Customize a Qwen3 Embedding class, integrated with LangChain
#     """
#     def __init__(self, model_name):
#         self.qwen3_embedding = SentenceTransformer(model_name)
#     def embed_query(self, text : str) ->list[float]:
#         return self.embed_documents([text])[0]
#     def embed_documents(self, texts : list[str]) -> list[list[float]]:
#         return self.qwen3_embedding.encode(texts)
#
# qwen3_embedding_model = Qwen3CustomEmbedding("Qwen/Qwen3-Embedding-0.6B")

"""Local model approach"""
import torch
class Qwen3CustomEmbedding(Embeddings):
    """
    Customize a Qwen3 Embedding class, integrated with LangChain
    """
    def __init__(self, model_path: str):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            logger_chatflow.info("使用GPU部署嵌入模型")
        else:
            logger_chatflow.info("使用CPU部署嵌入模型")
        self.qwen3_embedding = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device=device
        )
    def embed_query(self, text : str) ->list[float]:
        return self.embed_documents([text])[0]
    def embed_documents(self, texts : list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts).tolist()

# Initialize the embedding model
qwen3_embedding_model = Qwen3CustomEmbedding(str(QWEN_EMBEDDING_PATH))