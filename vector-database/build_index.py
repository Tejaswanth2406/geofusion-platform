"""
GeoFusion — FAISS Index Builder
====================================
Standalone utility to (re)build the FAISS index + metadata store
from a directory of pre-computed embeddings.

Usage:
    python build_index.py --embeddings_dir data/embeddings/ \
        --index_path faiss.index --metadata_path metadata.json
"""

import argparse
import json
import os

import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss is required: pip install faiss-cpu") from e


def build_index(
    embeddings_dir: str, index_path: str, metadata_path: str, dim: int = 512
):
    """
    Expects embeddings_dir to contain *.json files of the form:
        {"id": "...", "sensor": "optical", "embedding": [...], "lat": .., "lon": ..}
    """
    index = faiss.IndexFlatIP(dim)
    metadata = {}

    files = sorted(f for f in os.listdir(embeddings_dir) if f.endswith(".json"))

    vectors = []
    for i, fname in enumerate(files):
        with open(os.path.join(embeddings_dir, fname), "r") as f:
            record = json.load(f)

        vec = np.array(record["embedding"], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vectors.append(vec)
        metadata[str(i)] = {
            "id": record.get("id", fname),
            "sensor": record.get("sensor", "unknown"),
            "lat": record.get("lat"),
            "lon": record.get("lon"),
        }

    if vectors:
        matrix = np.vstack(vectors).astype("float32")
        index.add(matrix)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {index.ntotal} vectors -> {index_path}")
    print(f"Metadata -> {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GeoFusion FAISS index")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings/")
    parser.add_argument("--index_path", type=str, default="faiss.index")
    parser.add_argument("--metadata_path", type=str, default="metadata.json")
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)
    build_index(args.embeddings_dir, args.index_path, args.metadata_path, args.dim)
