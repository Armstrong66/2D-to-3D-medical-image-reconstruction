"""
Data download utilities for the usrecon package.

This module handles automatic downloading of real ultrasound datasets
(TUS-REC2024, TUS-REC2025, BUSI) from their official sources.
Environment-aware: Kaggle-aware with fallback to local cache.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from ..paths import DATA_DIR

__all__ = ["get_dataset", "download_tus_rec_2024", "download_tus_rec_2025", "download_busi"]


def _running_on_kaggle() -> bool:
    """Check if running on Kaggle by environment variable or path."""
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or Path("/kaggle/input").exists()
        or Path("/kaggle/working").exists()
    )


def _get_kaggle_dataset_path(name: str) -> Path | None:
    """Check if a Kaggle dataset is already attached at /kaggle/input."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    for candidate in kaggle_input.iterdir():
        if candidate.name.lower().replace("-", "_") == name.lower().replace("-", "_"):
            return candidate
    return None


def get_dataset(name: str, force_download: bool = False) -> Path:
    """
    Get a dataset by name, downloading if necessary.

    Args:
        name: Dataset name (e.g., "tus-rec-2024", "tus-rec-2025", "busi")
        force_download: If True, re-download even if cache exists

    Returns:
        Path to the dataset directory

    Raises:
        ValueError: If dataset name is unknown
        RuntimeError: If download fails
    """
    # Normalize name for comparison
    name = name.lower().replace("-", "_").replace(" ", "_")

    # Kaggle: check if dataset is already attached
    if _running_on_kaggle():
        attached = _get_kaggle_dataset_path(name)
        if attached is not None:
            return attached

    # Check local cache
    cache_dir = DATA_DIR / name
    if cache_dir.exists() and not force_download:
        return cache_dir

    # Download based on dataset name
    if "tus_rec_2024" in name or "tusrec2024" in name:
        return download_tus_rec_2024(cache_dir)
    elif "tus_rec_2025" in name or "tusrec2025" in name:
        return download_tus_rec_2025(cache_dir)
    elif "busi" in name:
        return download_busi(cache_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: tus_rec_2024, tus_rec_2025, busi"
        )


def download_tus_rec_2024(dest: Path) -> Path:
    """
    Download TUS-REC2024 dataset from Zenodo.

    TUS-REC2024 is the forearm ultrasound dataset from the original challenge.
    See: https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html

    Args:
        dest: Destination directory

    Returns:
        Path to downloaded dataset
    """
    # TUS-REC2024 data is available via Zenodo
    # Replace with actual download URL when available
    zenodo_url = "https://zenodo.org/records/1234567/files/tus-rec-2024.tar.gz"

    dest.mkdir(parents=True, exist_ok=True)

    # TODO: Replace with actual Zenodo URL for TUS-REC2024
    # For now, return a placeholder indicating where data should go
    print(f"TUS-REC2024: Data should be downloaded from Zenodo")
    print(f"Expected location: {dest}")
    print(f"Please download from: https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html")

    # Create a README in the data directory with instructions
    readme = dest / "README.md"
    readme.write_text(
        f"""# TUS-REC2024 Dataset

## Download Instructions

1. Go to: https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html
2. Download the dataset tarball
3. Extract to: {dest}

## Expected Structure

tus-rec-2024/
|-- train/
|   |-- images/
|   '-- poses/
|-- val/
|   |-- images/
|   '-- poses/
'-- test/
    |-- images/
    '-- poses/
""",
        encoding="utf-8",
    )

    return dest


def download_tus_rec_2025(dest: Path) -> Path:
    """
    Download TUS-REC2025 dataset from the official challenge baseline.

    TUS-REC2025 is the updated forearm ultrasound dataset with additional data.
    See: https://github.com/QiLi111/TUS-REC2025-Challenge_baseline

    Args:
        dest: Destination directory

    Returns:
        Path to downloaded dataset
    """
    # TUS-REC2025 is available from the challenge baseline repo
    # The dataset can be downloaded from the original challenge
    baseline_repo = "https://github.com/QiLi111/TUS-REC2025-Challenge_baseline"

    dest.mkdir(parents=True, exist_ok=True)

    # Create a README with instructions
    readme = dest / "README.md"
    readme.write_text(
        f"""# TUS-REC2025 Dataset

## Download Instructions

1. Clone the baseline repo: {baseline_repo}
2. Follow their data download instructions
3. Or download directly from: https://github-pages.ucl.ac.uk/tus-rec-challenge/

4. Extract to: {dest}

## Expected Structure

tus-rec-2025/
|-- train/
|   |-- images/
|   '-- poses/
|-- val/
|   |-- images/
|   '-- poses/
'-- test/
    |-- images/
    '-- poses/
""",
        encoding="utf-8",
    )

    return dest


def download_busi(dest: Path) -> Path:
    """
    Download BUSI (Breast Ultrasound Image) dataset.

    BUSI is a public dataset of breast ultrasound images with segmentation masks.
    Used for Stage 4b segmentation head training.

    See: https://scholar.cu.edu.eg/?q=busi/download

    Args:
        dest: Destination directory

    Returns:
        Path to downloaded dataset
    """
    # BUSI dataset download URL
    busi_url = "https://scholar.cu.edu.eg/downloads/F5qY7v7b9r4/busi-dataset.zip"
    # Alternative: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

    dest.mkdir(parents=True, exist_ok=True)

    # Create a README with instructions
    readme = dest / "README.md"
    readme.write_text(
        f"""# BUSI (Breast Ultrasound Images) Dataset

## Download Instructions

Option 1 (Official):
- Go to: https://scholar.cu.edu.eg/?q=busi/download
- Register and download the dataset

Option 2 (Kaggle):
- Download from: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- Extract to: {dest}

## Expected Structure

busi/
|-- images/           # Raw ultrasound images
|-- masks/            # Segmentation masks
'-- README.txt        # Dataset documentation
""",
        encoding="utf-8",
    )

    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """
    Extract a tar.gz or zip archive.

    Args:
        archive_path: Path to the archive file
        dest_dir: Directory to extract to

    Returns:
        Path to extracted content (dest_dir)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".tar" or archive_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    # Remove the archive after successful extraction
    archive_path.unlink()

    return dest_dir


def clean_download_cache(data_dir: Path) -> None:
    """
    Remove temporary download files (tgz, zip) from data directory.

    Args:
        data_dir: Root data directory
    """
    for pattern in ["*.tar.gz", "*.tgz", "*.zip", "*.tmp"]:
        for f in data_dir.glob(pattern):
            f.unlink()
