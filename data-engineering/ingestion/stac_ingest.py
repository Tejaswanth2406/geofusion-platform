"""
GeoFusion — STAC Data Ingestion Pipeline
=========================================
Downloads satellite imagery (Sentinel-2, Sentinel-1, Landsat) via the
SpatioTemporal Asset Catalog (STAC) API (e.g., Microsoft Planetary Computer).

Usage:
    python stac_ingest.py --bbox -122.5 37.7 -122.3 37.8 --sensor optical --max_items 5
"""

import argparse
import json
import os
import urllib.request

from loguru import logger

try:
    import planetary_computer as pc
    import pystac_client

    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False


def download_asset(url: str, dest_path: str):
    """Download a STAC asset to local disk."""
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))

    logger.info(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)


def ingest_stac_data(
    bbox: list[float], sensor: str, date_range: str, max_items: int, output_dir: str
):
    if not STAC_AVAILABLE:
        logger.error("pystac-client and planetary-computer are required for ingestion.")
        logger.error("Run: pip install pystac-client planetary-computer")
        return

    # Map sensor type to STAC collections
    collections = []
    if sensor == "optical":
        collections = ["sentinel-2-l2a"]
    elif sensor == "sar":
        collections = ["sentinel-1-grd"]
    elif sensor == "landsat":
        collections = ["landsat-c2-l2"]
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    logger.info(f"Searching STAC: {collections} | BBox: {bbox} | Dates: {date_range}")
    search = catalog.search(
        collections=collections,
        bbox=bbox,
        datetime=date_range,
        max_items=max_items,
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} items matching criteria.")

    metadata_records = []

    for i, item in enumerate(items):
        item_id = item.id
        logger.info(f"Processing item {i+1}/{len(items)}: {item_id}")

        # Select best asset (visual for optical, vv for SAR)
        asset_key = "visual" if sensor == "optical" else "vv"
        if asset_key not in item.assets:
            # Fallback to the first available data asset
            asset_key = list(item.assets.keys())[0]

        asset_url = item.assets[asset_key].href
        dest_filename = os.path.join(output_dir, sensor, item_id, "image.tif")

        try:
            download_asset(asset_url, dest_filename)
        except Exception as e:
            logger.error(f"Failed to download {item_id}: {e}")
            continue

        # Save metadata
        record = {
            "tile_id": item_id,
            "sensor": sensor,
            "date": item.datetime.isoformat() if item.datetime else None,
            "bbox": item.bbox,
            "cloud_cover": item.properties.get("eo:cloud_cover", 0.0),
            "stac_collection": item.collection_id,
        }

        meta_path = os.path.join(output_dir, sensor, item_id, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(record, f, indent=2)

        metadata_records.append(record)

    logger.info(f"Ingestion complete. Downloaded {len(metadata_records)} tiles.")


def parse_args():
    parser = argparse.ArgumentParser(description="GeoFusion STAC Ingestion")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        help="Bounding box: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument(
        "--sensor", type=str, default="optical", choices=["optical", "sar", "landsat"]
    )
    parser.add_argument(
        "--dates",
        type=str,
        default="2023-01-01/2024-01-01",
        help="STAC datetime range (e.g. 2023-01-01/2023-12-31)",
    )
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="../../data/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_stac_data(
        bbox=args.bbox,
        sensor=args.sensor,
        date_range=args.dates,
        max_items=args.max_items,
        output_dir=args.output_dir,
    )
