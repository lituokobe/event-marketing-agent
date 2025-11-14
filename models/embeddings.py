import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class Qwen3CustomEmbedding(Embeddings):
    """
    Customize a Qwen3 Embedding class, integrated with LangChain
    """
    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name)
    def embed_query(self, text : str) ->list[float]:
        return self.embed_documents([text])[0]
    def embed_documents(self, texts : list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts)

qwen3_embedding_model = Qwen3CustomEmbedding("Qwen/Qwen3-Embedding-0.6B")