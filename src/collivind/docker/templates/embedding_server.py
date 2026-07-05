from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Collivind Embedding Service")
model = SentenceTransformer("all-MiniLM-L6-v2")


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health_check():
    return {"status": "ok", "dimension": model.get_sentence_embedding_dimension()}


@app.post("/embed")
def embed(request: EmbedRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}


@app.post("/embed_batch")
def embed_batch(request: EmbedBatchRequest):
    embeddings = model.encode(request.texts).tolist()
    return {"embeddings": embeddings}
