"""
Download YOLOv8 models for shape detection.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
}


def download_file(url: str, destination: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading YOLOv8 models...")

    for name, url in MODEL_URLS.items():
        destination = models_dir / f"{name}.pt"
        if destination.exists():
            print(f"✓ {name} already exists")
        else:
            print(f"Downloading {name}...")
            download_file(url, destination)
            print(f"✓ Downloaded {name}")

    print("\n✓ All models downloaded to data/models/")


if __name__ == "__main__":
    main()
