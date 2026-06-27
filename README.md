                                                                                                                                                              
  <img width="1942" height="809" alt="GeoFusion AI" src="https://github.com/user-attachments/assets/c99dfe42-dff7-42ee-a858-ef960024ea00" />

  # GeoFusion AI Platform
  ### Enterprise Multi-Sensor Satellite Intelligence Retrieval Engine

  ![CI](https://github.com/Tejaswanth2406/geofusion-platform/actions/workflows/ci.yml/badge.svg)
  ![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen)
  ![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
  ![License MIT](https://img.shields.io/badge/License-MIT-green)
  ![Deployed](https://img.shields.io/badge/Status-Live-success)

  GeoFusion AI is a research-grade, enterprise-ready geospatial intelligence platform. It enables **cross-modal satellite image retrieval** across optical, SAR, and multispectral sensors using shared
  embedding spaces, contrastive alignment, and FAISS indexing.

  ---

  ## 🌐 Live Deployment

  | Layer | URL |
  |---|---|
  | 🖥 **Frontend (Vercel)** | https://geofusion-platform.vercel.app |
  | 🔌 **API Gateway (Render)** | https://geofusion-platform.onrender.com/docs |
  | 🧠 **Embedding Service** | https://geofusion-embedding.onrender.com/docs |
  | 🔍 **Retrieval Service** | https://geofusion-retrieval.onrender.com/docs |
  | 📊 **Evaluation Service** | https://geofusion-evaluation.onrender.com/docs |

  **Demo credentials**
  - Admin (full access): `admin` / `geofusion_demo_2026`
  - Analyst (read-only): `analyst` / `analyst_pass`

  > First request may take 30–60s to cold-start the free-tier Render services.

  ---

  ## 📸 Screenshots

  | Login | Dashboard | Retrieval Results |
  |---|---|---|  
  | ![Login](https://github.com/user-attachments/assets/a86030ba-0049-4d77-8e3c-3707f9891b12) | ![Dashboard](https://github.com/user-attachments/assets/c357094d-1e9b-439b-bae3-fcb0a21a04e6) |
  ![Results](https://github.com/user-attachments/assets/1999e5fd-e787-4210-8ef6-8afe1e7193d8) |




  🎥 **Demo Video:** [Watch on YouTube](YOUR_VIDEO_URL_HERE) *(2-3 min walkthrough)*

  ---
  
  ## 🔬 Mathematical Foundation

  GeoFusion solves the cross-modal retrieval problem by hypothesizing that **images from different sensors observing the same geographic region should have nearby representations in a common latent space**.

  ### 1. Shared Embedding Space
  Let $x_o \in \mathbb{R}^{H \times W \times C_o}$ be an optical image and $x_s \in \mathbb{R}^{H \times W \times C_s}$ be a SAR image. Modality-specific encoders map these into a shared $d$-dimensional
  latent space:

  $$z_o = f_o(x_o; \theta_o), \quad z_s = f_s(x_s; \theta_s)$$
  
  where $z_o, z_s \in \mathbb{R}^{d}$.

  ### 2. GeoFusion Objective
  The system is trained end-to-end to satisfy:                                                                                                                

  $$\min_{\theta_o, \theta_s} \sum_{i=1}^{N} D\left(f_o(x_o^i), f_s(x_s^i)\right)$$

  subject to:
  
  $$D\left(f_o(x_o^i), f_s(x_s^j)\right) > m, \quad i \neq j$$

  *GeoFusion learns modality-invariant representations by minimizing embedding distance between semantically corresponding observations while maximizing distance between unrelated ones.*

  ### 3. Contrastive Learning Loss (InfoNCE)
  For a positive pair $(z_o, z_s)$:

  $$L = -\log \frac{\exp(\text{Sim}(z_o, z_s) / \tau)}{\sum_{k=1}^{N} \exp(\text{Sim}(z_o, z_k) / \tau)}$$

  where $\tau$ is the temperature parameter and $\text{Sim}(z_i, z_j) = \frac{z_i \cdot z_j}{\|z_i\| \|z_j\|}$.

  ### 4. Retrieval Complexity
  FAISS Approximate Nearest Neighbor (ANN) index reduces brute-force search complexity from $O(Nd)$ to approximately $O(\log N)$, enabling sub-50ms latency over millions of tiles.

  ---

  ## 🏗 System Architecture                                                                                                                                    

  ```mermaid
  graph TD
      User(["User or Analyst"]) -->|Bearer JWT| Frontend["React Frontend on Vercel"]
      Frontend -->|HTTPS proxy| Gateway["API Gateway on Render"]
      Gateway --> Auth["JWT Authentication and RBAC"]
      Gateway --> Embed["Embedding Service"]
      Gateway --> Search["Retrieval Service"]
      Gateway --> Eval["Evaluation Service"]

      Embed --> Encoders["ViT and ResNet50 Encoders"]
      Search --> FAISS[("FAISS Vector Index")]
      Eval --> Metrics["F1 at K, mAP, Latency"]

      Prometheus["Prometheus"] -.->|metrics| Gateway
      Prometheus -.->|metrics| Embed
      Prometheus -.->|metrics| Search
      Grafana["Grafana Dashboards"] -.-> Prometheus
  ```



  ### Deployed Microservices                                                                                                                                  
  | Service | Role | Tech |
  |---|---|---|
  | **Frontend** | React SPA, Vite build, served from Vercel edge | React 18, Vite, Vercel rewrites for backend proxy |
  | **API Gateway** | Auth, rate limiting, request routing, explainability | FastAPI, JWT, structlog, Prometheus client |
  | **Embedding Service** | Image → 512-D latent vector | PyTorch, timm (ViT) / torchvision (ResNet50) |
  | **Retrieval Service** | Vector → top-K matches | FAISS (IndexFlatIP, cosine similarity) |
  | **Evaluation Service** | Benchmarking & metrics | F1@K, mAP, latency profiling |

  ---

  ## 🗂 Dataset

  GeoFusion is designed to work with multi-sensor Earth observation data:

  | Modality | Source | Bands / Format |
  |---|---|---|
  | **Optical** | Sentinel-2 (ESA Copernicus) | 10 bands (B02–B12), 10m–60m resolution |
  | **SAR** | Sentinel-1 (ESA Copernicus) | VV + VH polarisations, ~10m resolution |
  | **Multispectral** | Landsat-8 (USGS) | 11 bands, 30m resolution |

  ### Ingestion
  Tiles are fetched via the **Microsoft Planetary Computer STAC API**:

  ```bash
  python data-engineering/ingestion/stac_ingest.py \
    --bbox -122.5 37.7 -122.3 37.8 \
    --sensor optical \
    --max_items 5
  ```

  Each tile is preprocessed (cloud masking, tiling to 224×224 patches), embedded by the encoder, and indexed in FAISS along with its geo-metadata (lat/lon, sensor, capture date).

  For the live demo, the FAISS index is seeded with 20 synthetic tiles spanning regions (Amazon, Sahara, Himalayas, Arctic, Pacific) across all three sensor modalities — the same `/index/add` endpoint scales
   to millions of tiles in production.

  ---
  
  ## 🚀 Quick Start (Local)

  ### 1. Clone & install
  ```bash
  git clone https://github.com/Jaya242/geofusion-platform.git                                                                                                 
  cd geofusion-platform
  pip install -r requirements.txt
  ```

  ### 2. Start the full stack
  ```bash
  docker-compose up --build -d
  ```
  Runs API Gateway, Embedding, Retrieval, Evaluation, MLflow, Prometheus, and Grafana.

  ### 3. Authenticate
  ```bash
  curl -X POST http://localhost:8000/auth/token \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"geofusion_demo_2026"}'
  ```
  
  ### 4. Run a cross-modal retrieval
  ```bash
  curl -X POST http://localhost:8000/api/v1/retrieve \
    -H "Authorization: Bearer <ACCESS_TOKEN>" \
    -F "sensor=optical" \
    -F "image=@data/sentinel2/tile001/image.tiff"
  ```

  ---

  ## 🌍 Production Deployment

  | Layer | Platform | Why |
  |---|---|---|
  | **Frontend** | Vercel | Edge CDN, automatic deploys from `main`, built-in HTTPS, Vite-native |
  | **Backend** | Render | Native Docker, persistent disks, auto-scaling, regional placement |

  **Inter-service communication** is over HTTPS within a Render region (Singapore for our deployment) for low latency. Vercel rewrites proxy `/auth/*`, `/api/*`, and `/health` from the frontend domain to the
   gateway, eliminating CORS preflight overhead and exposing a single origin to the browser.

  ---
  
  ## 📂 Repository Structure

  ```text
  geofusion-platform/
  ├── .github/workflows/ci.yml      # CI/CD pipeline
  ├── backend/
  │   ├── api-gateway/              # JWT auth, rate limiting, routing
  │   ├── embedding-service/        # PyTorch encoders (ViT, ResNet)
  │   ├── retrieval-service/        # FAISS indexing & search
  │   ├── training-service/         # Contrastive loss, distributed training
  │   ├── evaluation-service/       # F1@K, mAP metrics
  │   └── preprocessing-service/    # Cloud masking, tiling
  ├── frontend/                     # React SPA (Vercel)
  ├── data-engineering/             # STAC ingestion pipelines
  ├── tests/                        # Pytest unit & integration tests
  ├── deployment/docker/            # Container definitions
  ├── monitoring/                   # Prometheus + Grafana config
  ├── mlflow/                       # Experiment tracking
  └── vector-database/              # Persistent FAISS indices
  ```

  ---
  
  ## 📊 Evaluation Metrics

  The evaluation service benchmarks the retrieval pipeline using:

  $$\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{K}$$

  $$\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{|\text{Relevant}|}$$

  $$F1@K = 2 \cdot \frac{\text{Precision@K} \times \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

  Mean Average Precision (mAP) across queries and per-request latency (p50/p95/p99) are tracked in Prometheus and visualised in Grafana.

  ---

  ## 🏆 BAH 2026 Hack2Skill • MIT License                             
