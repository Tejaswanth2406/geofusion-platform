"""
GeoFusion AI Platform — Training Service
============================================
CLI entrypoint for contrastive dual-encoder training between
optical and SAR (or any two) satellite modalities.

Usage:
    python main.py --optical_dir data/sentinel2/ --sar_dir data/sentinel1/ \
        --epochs 50 --batch_size 32 --embedding_dim 512 --model vit
"""

import argparse
import os
import sys
import time

import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "embedding-service"))

from dataset import PairedSatelliteDataset  # noqa: E402
from losses import MultiModalContrastiveLoss  # noqa: E402

try:
    from encoder import SENSOR_TRANSFORMS, SatelliteEncoder
except ImportError:
    SatelliteEncoder = None
    SENSOR_TRANSFORMS = {}


def parse_args():
    parser = argparse.ArgumentParser(description="GeoFusion contrastive training")
    parser.add_argument("--optical_dir", type=str, default="data/sentinel2/")
    parser.add_argument("--sar_dir", type=str, default="data/sentinel1/")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--pairs_file", type=str, default="pairs/pairs.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "resnet50"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="../../ai-models/checkpoints"
    )
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        f"Starting GeoFusion training | model={args.model} device={args.device}"
    )

    try:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("geofusion-contrastive-training")
        mlflow_active = True
    except Exception as e:
        logger.warning(f"MLflow unavailable, continuing without tracking: {e}")
        mlflow_active = False

    if SatelliteEncoder is None:
        logger.error(
            "Could not import SatelliteEncoder. Ensure embedding-service is on PYTHONPATH."
        )
        return

    optical_encoder = SatelliteEncoder(
        model_type=args.model, embedding_dim=args.embedding_dim, device=args.device
    )
    sar_encoder = SatelliteEncoder(
        model_type=args.model, embedding_dim=args.embedding_dim, device=args.device
    )

    dataset = PairedSatelliteDataset(
        data_root=args.data_root,
        pairs_file=args.pairs_file,
        optical_transform=SENSOR_TRANSFORMS.get("optical"),
        sar_transform=SENSOR_TRANSFORMS.get("sar"),
    )

    if len(dataset) == 0:
        logger.warning(
            "No training pairs found. Populate data/pairs/pairs.json with "
            "{id, optical, sar, lat, lon} records before training."
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    criterion = MultiModalContrastiveLoss(temperature=args.temperature)
    params = list(optical_encoder.model.parameters()) + list(
        sar_encoder.model.parameters()
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_ctx = mlflow.start_run() if mlflow_active else _NullContext()
    with run_ctx:
        if mlflow_active:
            mlflow.log_params(vars(args))

        best_loss = float("inf")
        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            epoch_loss = 0.0

            optical_encoder.model.train()
            sar_encoder.model.train()

            for batch in dataloader:
                optical_imgs = batch["optical"].to(args.device)
                sar_imgs = batch["sar"].to(args.device)

                optical_emb = torch.nn.functional.normalize(
                    optical_encoder.model(optical_imgs), p=2, dim=1
                )
                sar_emb = torch.nn.functional.normalize(
                    sar_encoder.model(sar_imgs), p=2, dim=1
                )

                loss = criterion(optical_emb, sar_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s"
            )

            if mlflow_active:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "optical_model_state": optical_encoder.model.state_dict(),
                        "sar_model_state": sar_encoder.model.state_dict(),
                        "model_state": optical_encoder.model.state_dict(),
                        "epoch": epoch,
                        "loss": avg_loss,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved new best checkpoint -> {checkpoint_path}")

        logger.info(f"Training complete. Best loss: {best_loss:.4f}")


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
