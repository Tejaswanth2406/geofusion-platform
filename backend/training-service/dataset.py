"""
GeoFusion — Paired Satellite Dataset
========================================
Loads (optical, SAR) image pairs for contrastive training.
Expects a pairs.json manifest mapping geography IDs to file paths.
"""

import json
import os

from PIL import Image
from torch.utils.data import Dataset


class PairedSatelliteDataset(Dataset):
    """
    Dataset of (optical, SAR) image pairs sharing the same geography.

    pairs.json format:
        [
          {"id": "tile001", "optical": "sentinel2/tile001/image.tiff",
           "sar": "sentinel1/tile001/sar.tiff", "lat": 16.23, "lon": 81.54},
          ...
        ]
    """

    def __init__(
        self,
        data_root: str,
        pairs_file: str = "pairs/pairs.json",
        optical_transform=None,
        sar_transform=None,
    ):
        self.data_root = data_root
        pairs_path = os.path.join(data_root, pairs_file)

        if os.path.exists(pairs_path):
            with open(pairs_path, "r") as f:
                self.pairs = json.load(f)
        else:
            self.pairs = []

        self.optical_transform = optical_transform
        self.sar_transform = sar_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        optical_path = os.path.join(self.data_root, pair["optical"])
        sar_path = os.path.join(self.data_root, pair["sar"])

        optical_img = Image.open(optical_path).convert("RGB")
        sar_img = Image.open(sar_path).convert("RGB")

        if self.optical_transform:
            optical_img = self.optical_transform(optical_img)
        if self.sar_transform:
            sar_img = self.sar_transform(sar_img)

        return {
            "id": pair["id"],
            "optical": optical_img,
            "sar": sar_img,
            "lat": pair.get("lat", 0.0),
            "lon": pair.get("lon", 0.0),
        }
