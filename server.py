import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from search.engine import HybridSearchEngine
from agents.workflow import MultiAgentOrchestrator

app = FastAPI(title="RFP Intelligence Platform", version="1.0")

engine = HybridSearchEngine()
for folder in ["data/Bid1", "data/Bid2"]:
    if os.path.isdir(folder):
        engine.index_bid_folder(folder)
orchestrator = MultiAgentOrchestrator(engine)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    bid_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    bid_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(engine.bm25_corpus)}


@app.post("/search")
def search(req: SearchRequest):
    results = engine.search(req.query, bid_id=req.bid_id, top_k=req.top_k, mode=req.mode)
    return {
        "query": req.query,
        "mode": req.mode,
        "results": [
            {
                "bid_id": r.get("bid_id"),
                "file_name": r.get("file_name"),
                "page_number": r.get("page_number"),
                "doc_type": r.get("doc_type"),
                "snippet": str(r.get("text", ""))[:400],
            }
            for r in results
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return orchestrator.ask_question(req.question, bid_id=req.bid_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/bids/{bid_id}")
def get_bid(bid_id: str):
    if not bid_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid bid id")
    path = os.path.join("outputs", f"{bid_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No extraction output for this bid yet")
    with open(path, encoding="utf-8") as f:
        return json.load(f)