"""
GeoFusion — Retrieval Evaluation Engine
============================================
Computes F1@K, mAP, and latency benchmarks for the retrieval
pipeline against a labelled evaluation dataset.

Usage:
    python evaluator.py --dataset_path data/eval/ \
        --index_path ../../vector-database/faiss.index --top_k 5 10
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval-service"))

try:
    from retrieval import GeoFusionRetriever
except ImportError:
    GeoFusionRetriever = None


def precision_recall_at_k(relevant_ids: set, retrieved_ids: List[str], k: int):
    """Compute precision@k and recall@k for a single query."""
    top_k_ids = retrieved_ids[:k]
    hits = sum(1 for rid in top_k_ids if rid in relevant_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(relevant_ids: set, retrieved_ids: List[str]) -> float:
    """Compute Average Precision (AP) for a single query."""
    hits = 0
    score = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_ids) if relevant_ids else 0.0


def load_eval_set(dataset_path: str) -> List[Dict]:
    """
    Loads an evaluation manifest.

    Expected format (eval_manifest.json):
        [
          {
            "query_id": "sar_001",
            "query_embedding": [...512 floats...],
            "query_sensor": "sar",
            "relevant_ids": ["optical_001", "optical_002"]
          },
          ...
        ]
    """
    manifest_path = os.path.join(dataset_path, "eval_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        return json.load(f)


def evaluate(
    dataset_path: str,
    index_path: str,
    top_k_values: List[int],
) -> Dict:
    if GeoFusionRetriever is None:
        raise RuntimeError("Could not import GeoFusionRetriever from retrieval-service.")

    retriever = GeoFusionRetriever(index_path=index_path)
    eval_set = load_eval_set(dataset_path)

    if not eval_set:
        return {
            "warning": "No evaluation manifest found. "
            f"Place eval_manifest.json under {dataset_path}",
            "results": {},
        }

    metrics = {k: {"precision": [], "recall": [], "f1": []} for k in top_k_values}
    ap_scores = []
    latencies = []

    max_k = max(top_k_values)

    for query in eval_set:
        relevant_ids = set(query["relevant_ids"])

        t0 = time.perf_counter()
        results = retriever.search(
            query_embedding=query["query_embedding"],
            sensor=query.get("query_sensor"),
            top_k=max_k,
            retrieval_mode="cross",
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["id"] for r in results]

        for k in top_k_values:
            precision, recall = precision_recall_at_k(relevant_ids, retrieved_ids, k)
            metrics[k]["precision"].append(precision)
            metrics[k]["recall"].append(recall)
            metrics[k]["f1"].append(f1_score(precision, recall))

        ap_scores.append(average_precision(relevant_ids, retrieved_ids))

    report = {
        "num_queries": len(eval_set),
        "mAP": round(float(np.mean(ap_scores)), 4) if ap_scores else 0.0,
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "metrics_by_k": {},
    }

    for k in top_k_values:
        report["metrics_by_k"][f"@{k}"] = {
            "precision": round(float(np.mean(metrics[k]["precision"])), 4),
            "recall": round(float(np.mean(metrics[k]["recall"])), 4),
            "f1": round(float(np.mean(metrics[k]["f1"])), 4),
        }

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="GeoFusion retrieval evaluation")
    parser.add_argument("--dataset_path", type=str, default="data/eval/")
    parser.add_argument(
        "--index_path", type=str, default="../../vector-database/faiss.index"
    )
    parser.add_argument("--top_k", type=int, nargs="+", default=[5, 10])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = evaluate(args.dataset_path, args.index_path, args.top_k)
    print(json.dumps(report, indent=2))
