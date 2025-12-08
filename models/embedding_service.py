from fastapi import FastAPI, HTTPException # Create API service
from pydantic import BaseModel
from typing import Union
from models.embedding_model import qwen3_embedding_model
from functionals.log_utils import logger_chatflow

# Convert the initialized model to a web service with API

class EmbedRequest(BaseModel): # input data validation class
    input: Union[str, list[str]]

class EmbedResponse(BaseModel): # output data validation class
    embeddings: list[list[float]]
app = FastAPI(title="千问嵌入服务-李拓电脑", version="1.0")

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    try:
        if isinstance(request.input, str):
            # Single query → embed_query
            emb = qwen3_embedding_model.embed_query(request.input)
            return EmbedResponse(embeddings=[emb])
        else:
            # Batch → embed_documents
            if not request.input:
                raise HTTPException(status_code=400, detail="Input list is empty")
            embs = qwen3_embedding_model.embed_documents(request.input)
            return EmbedResponse(embeddings=embs)
    except Exception as e:
        logger_chatflow.error(f"嵌入失败: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding error")

@app.get("/health")
def health():
    return {"status": "ok", "model": "Qwen3-Embedding-0.6B"}

# This embedding model will be locally deployed and run on port 8081
# uvicorn models.embedding_service:app --host 0.0.0.0 --port 8081