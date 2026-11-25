import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

app = FastAPI(title="Collivind Embedding Service")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# The search reranker. A failed load leaves embeddings up; /rerank then
# answers 503 and collivind keeps its first-stage ranking.
RERANK_MODEL = os.environ.get("COLLIVIND_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
try:
    reranker = CrossEncoder(RERANK_MODEL, max_length=256) if RERANK_MODEL else None
except Exception as e:  # noqa: BLE001
    logging.getLogger(__name__).warning("reranker %s failed to load: %s", RERANK_MODEL, e)
    reranker = None


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


class RerankRequest(BaseModel):
    query: str
    texts: List[str]


@app.post("/rerank")
def rerank(request: RerankRequest):
    if reranker is None:
        raise HTTPException(status_code=503, detail="no reranker loaded")
    scores = reranker.predict([(request.query, t) for t in request.texts], batch_size=64)
    return {"scores": [float(x) for x in scores]}
