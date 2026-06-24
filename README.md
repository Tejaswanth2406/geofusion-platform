<img width="1942" height="809" alt="image" src="https://github.com/user-attachments/assets/c99dfe42-dff7-42ee-a858-ef960024ea00" />

# GeoFusion AI Platform
### Multi-Sensor Satellite Intelligence Retrieval Engine
**BAH 2026 · Hack2Skill Entry**

## Overview

GeoFusion AI is an enterprise-grade geospatial intelligence platform that enables
cross-modal satellite image retrieval across optical, SAR, and multispectral
sensors using shared embedding spaces and contrastive alignment.

## Architecture

```
USERS
  |
Web / API Interface
  |
Query Management Layer
  |
GeoFusion Core Engine
  |-------- Data Pipeline (ETL, Preprocess, Tile)
  |-------- AI Engine (Embedding, Contrastive Training)
  |-------- Retrieval Engine (FAISS/Milvus, Ranking)
  |-------- Analytics (Metrics, Explainability)
  |
Results + Explainability
  |
Top-K Images + Similarity + Metadata
```

## Repository Layout

```
geofusion-platform/
├── backend/
│   ├── api-gateway/          # Entry point, routing, auth, metrics
│   ├── embedding-service/    # Image -> 512-D vector
│   ├── retrieval-service/    # FAISS vector search + ranking
│   ├── training-service/     # Contrastive dual-encoder training
│   ├── preprocessing-service/# ETL, tiling, normalization
│   └── evaluation-service/   # F1@K, mAP, latency benchmarking
├── ai-models/
│   └── checkpoints/          # Saved model weights
├── vector-database/          # FAISS index + metadata store
├── data-engineering/
│   └── ingestion/            # Raw -> processed data ingestion
├── data/                     # Sample dataset (sentinel2/sentinel1/eval)
├── frontend/                 # React dashboard (placeholder)
└── deployment/
    ├── docker/                # Per-service Dockerfiles
    └── prometheus.yml
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start all services with Docker Compose
```bash
docker-compose up --build
```

### 3. Run individually
```bash
# API Gateway (port 8000)
cd backend/api-gateway && uvicorn main:app --reload --port 8000

# Embedding Service (port 8001)
cd backend/embedding-service && uvicorn main:app --reload --port 8001

# Retrieval Service (port 8002)
cd backend/retrieval-service && uvicorn main:app --reload --port 8002

# Preprocessing Service (port 8004)
cd backend/preprocessing-service && uvicorn main:app --reload --port 8004

# Evaluation Service (port 8005)
cd backend/evaluation-service && uvicorn main:app --reload --port 8005
```

## Services

| Service | Port | Description |
|---|---|---|
| api-gateway | 8000 | Main entry point, routing, auth |
| embedding-service | 8001 | Satellite image → 512-D vector |
| retrieval-service | 8002 | Vector search + similarity ranking |
| training-service | 8003 | Contrastive model training |
| preprocessing-service | 8004 | ETL, tiling, normalization |
| evaluation-service | 8005 | F1@5, F1@10, latency benchmarks |

## Supported Sensors
- Sentinel-2 (Optical, 13 bands)
- Sentinel-1 (SAR, VV/VH polarization)
- Landsat-8 (Optical, 11 bands)
- Hyperspectral (custom ingestion)
- DEM (Digital Elevation Model)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI Framework | PyTorch |
| Encoders | ViT / ResNet50 |
| Vector DB (dev) | FAISS |
| Vector DB (prod) | Milvus |
| Storage | MinIO / S3 |
| Pipeline | Apache Airflow |
| Experiment Tracking | MLflow |
| Containers | Docker + Kubernetes |
| Monitoring | Prometheus |

## Retrieval Example

```python
from backend.retrieval_service.retrieval import GeoFusionRetriever

retriever = GeoFusionRetriever(index_path="vector-database/faiss.index")
results = retriever.search(query_embedding=[...], sensor="SAR", top_k=10)

for r in results:
    print(f"ID: {r['id']}  Sensor: {r['sensor']}  Similarity: {r['similarity']:.4f}")
```

## Training

```bash
python backend/training-service/main.py \
    --optical_dir data/sentinel2/ \
    --sar_dir data/sentinel1/ \
    --epochs 50 \
    --batch_size 32 \
    --embedding_dim 512 \
    --model vit
```

## Evaluation

```bash
python backend/evaluation-service/evaluator.py \
    --dataset_path data/eval/ \
    --index_path vector-database/faiss.index \
    --top_k 5 10
```

Sample output:
```
F1@5  : 0.874
F1@10 : 0.912
mAP   : 0.863
Latency: 38ms avg
```

## Dataset Structure

```
satellite-data/
├── sentinel2/
│   └── tile001/
│       ├── image.tiff
│       └── metadata.json
├── sentinel1/
│   └── tile001/
│       ├── sar.tiff
│       └── metadata.json
└── pairs/
    └── pairs.json
```

## Development Roadmap
- [x] Week 1 – Data pipeline, preprocessing, baseline encoder, FAISS retrieval
- [x] Week 2 – Dual encoder, contrastive training, cross-modal retrieval
- [x] Week 3 – API gateway, React dashboard, model registry
- [x] Week 4 – Explainability layer, demo, optimization

## License
MIT License — BAH 2026 Hack2Skill
