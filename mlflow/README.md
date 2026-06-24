# MLflow Tracking for GeoFusion

This directory is the artifact root for MLflow.

## Start Tracking Server
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns
```

## Tracked Metrics
- `train_loss`: Contrastive loss (InfoNCE / Triplet) over epochs
- `F1@5`, `F1@10`, `Recall@10`: Retrieval evaluation metrics
