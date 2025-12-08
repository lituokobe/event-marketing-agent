import requests

# TODO: call the embedding service with API
EMBED_SERVICE_URL = "http://127.0.0.1:8081" # local deployment, port 8081

def embed_query(text: str) -> list[float]:
    resp = requests.post(f"{EMBED_SERVICE_URL}/embed", json={"input": text})
    resp.raise_for_status() #cecks the HTTP status code and raises an exception if the request failed
    return resp.json()["embeddings"][0]

def embed_documents(texts: list[str]) -> list[float]:
    resp = requests.post(f"{EMBED_SERVICE_URL}/embed", json={"input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]

# # Example usage
# emb = embed_query("你哪位")
# print(f"{len(emb)}")  # Should be 1024
# print(f"{emb}")
#
# emb_doc = embed_documents(["这是谁", "这是你"])
# print(f"{len(emb_doc)}")
# print(f"{emb_doc}")