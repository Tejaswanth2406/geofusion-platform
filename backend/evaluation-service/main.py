"""
GeoFusion AI Platform — Evaluation Service
==============================================
Exposes retrieval benchmarking (F1@K, mAP, latency) over HTTP.
"""

import os
from typing import List

from evaluator import evaluate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GeoFusion Evaluation Service", version="1.0.0")

DEFAULT_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../../vector-database/faiss.index")
DEFAULT_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/eval/")


class EvalRequest(BaseModel):
    dataset_path: str = DEFAULT_DATASET_PATH
    index_path: str = DEFAULT_INDEX_PATH
    top_k: List[int] = [5, 10]


@app.get("/health")
async def health():
    return {"status": "up"}


@app.post("/evaluate")
async def run_evaluation(req: EvalRequest):
    """Run a full retrieval benchmark and return F1@K / mAP / latency."""
    try:
        report = evaluate(req.dataset_path, req.index_path, req.top_k)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
