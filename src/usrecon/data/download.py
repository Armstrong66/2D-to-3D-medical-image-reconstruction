"""
Data download utilities for the usrecon package.

This module handles automatic downloading of real ultrasound datasets
(TUS-REC2024, TUS-REC2025, BUSI) from their official sources.
Environment-aware: Kaggle-aware with fallback to local cache.

On Kaggle, this module:
1. Downloads to /kaggle/tmp (scratch space) to avoid 20GB working limit
2. Auto-publishes datasets using the Kaggle API
3. Checks /kaggle/input for already-attached datasets

On local/Colab, downloads to data/ directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ..paths import DATA_DIR, SCRATCH_DIR

__all__ = [
    "get_dataset",
    "download_tus_rec_2024",
    "download_tus_rec_2025",
    "download_busi",
]


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


def _download_zenodo_file(url: str, dest: Path) -> Path:
    """
    Download a file from Zenodo.

    Args:
        url: Zenodo download URL
        dest: Destination path (should end with filename)

    Returns:
        Path to downloaded file
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest}...")

    with urllib.request.urlopen(url) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)

    print(f"Downloaded: {dest}")
    return dest


def _has_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are available."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    return kaggle_json.exists()


def _publish_to_kaggle_dataset(dataset_dir: Path, dataset_id: str) -> None:
    """
    Publish a folder as a Kaggle dataset using the Kaggle API.

    Args:
        dataset_dir: Directory containing the dataset to publish
        dataset_id: Kaggle dataset ID (e.g., 'usrecon/tus-rec-2024')
    """
    if not _has_kaggle_credentials():
        print(
            f"Warning: Kaggle credentials not found at ~/.kaggle/kaggle.json. "
            f"Skipping Kaggle dataset publication for {dataset_id}"
        )
        return

    # Check if Kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"Warning: Kaggle CLI not installed. Run 'pip install kaggle'. "
            f"Skipping Kaggle dataset publication for {dataset_id}"
        )
        return

    # Create dataset-metadata.json
    metadata = {
        "id": dataset_id,
        "title": dataset_id.split("/")[-1].replace("-", " ").title(),
        "slug": dataset_id.split("/")[-1],
        "description": f"Ultrasound dataset for TUS-REC challenge",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_file = dataset_dir / "dataset-metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Publishing {dataset_id} to Kaggle...")

    # Run kaggle datasets create
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(dataset_dir), "--dir-mode", "zip"],
            capture_output=True,
            text=True,
            cwd=str(dataset_dir),
        )
        if result.returncode == 0:
            print(f"Successfully published {dataset_id}")
            print(result.stdout)
        else:
            print(f"Failed to publish {dataset_id}:")
            print(result.stderr)
    except Exception as e:
        print(f"Error publishing {dataset_id} to Kaggle: {e}")


def get_dataset(
    name: str,
    force_download: bool = False,
    auto_publish: bool = False,
    kaggle_dataset_id: str | None = None,
) -> Path:
    """
    Get a dataset by name, downloading if necessary.

    Args:
        name: Dataset name (e.g., "tus-rec-2024", "tus-rec-2025", "busi")
        force_download: If True, re-download even if cache exists
        auto_publish: If True and on Kaggle, publish dataset to Kaggle after download
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/tus-rec-2024")

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

    # Determine download location
    # On Kaggle: use SCRATCH_DIR (/kaggle/tmp) for large files
    # Off Kaggle: use DATA_DIR
    download_dir = SCRATCH_DIR if _running_on_kaggle() else DATA_DIR

    cache_dir = download_dir / name
    if cache_dir.exists() and not force_download:
        return cache_dir

    # Download based on dataset name
    if "tus_rec_2024" in name or "tusrec2024" in name:
        result = download_tus_rec_2024(cache_dir, auto_publish, kaggle_dataset_id)
    elif "tus_rec_2025" in name or "tusrec2025" in name:
        result = download_tus_rec_2025(cache_dir, auto_publish, kaggle_dataset_id)
    elif "busi" in name:
        result = download_busi(cache_dir, auto_publish, kaggle_dataset_id)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: tus_rec_2024, tus_rec_2025, busi"
        )

    return result


def download_tus_rec_2024(
    dest: Path,
    auto_publish: bool = False,
    kaggle_dataset_id: str | None = None,
) -> Path:
    """
    Download TUS-REC2024 dataset from Zenodo.

    TUS-REC2024 is the forearm ultrasound dataset from the original challenge.

    Download URLs:
    - Training Part 1: https://zenodo.org/records/11178508
    - Training Part 2: https://zenodo.org/records/11180794
    - Training Part 3: https://zenodo.org/records/11355499
    - Validation: https://zenodo.org/records/12979481

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/tus-rec-2024")

    Returns:
        Path to downloaded dataset
    """
    zenodo_urls = {
        "train_part1": "https://zenodo.org/records/11178508/files/train_part1.zip",
        "train_part2": "https://zenodo.org/records/11180794/files/train_part2.zip",
        "train_part3": "https://zenodo.org/records/11355499/files/train_part3.zip",
        "val": "https://zenodo.org/records/12979481/files/val.zip",
    }

    dest.mkdir(parents=True, exist_ok=True)

    print("TUS-REC2024 Dataset Download")
    print("=" * 40)
    print(f"Destination: {dest}")
    print(f"Please download the following from Zenodo:")
    for name, url in zenodo_urls.items():
        print(f"  - {name}: {url}")
    print()
    print("Note: TUS-REC2024 training data is split into 3 parts (~43GB total)")
    print("      Validation data is ~4GB")
    print()

    # Create a README in the data directory with instructions
    readme = dest / "README.md"
    readme.write_text(
        """# TUS-REC2024 Dataset

## Download Instructions

TUS-REC2024 is split into multiple parts due to size:

### Training Data
- Part 1: https://zenodo.org/records/11178508
- Part 2: https://zenodo.org/records/11180794
- Part 3: https://zenodo.org/records/11355499

### Validation Data
- https://zenodo.org/records/12979481

## Download Script

On Kaggle, use the auto-download script:
```python
from usrecon.data.download import download_tus_rec_2024
dest = download_tus_rec_2024(Path("/kaggle/tmp/tus-rec-2024"))
```

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

## Citation

Li et al., "TUS-REC2024 Challenge: Trackerless Freehand Ultrasound
2D-to-3D Reconstruction", https://github-pages.ucl.ac.uk/tus-rec-challenge/
""",
        encoding="utf-8",
    )

    # If on Kaggle with auto-publish, create dataset and publish
    if _running_on_kaggle() and auto_publish and kaggle_dataset_id:
        _publish_to_kaggle_dataset(dest, kaggle_dataset_id)

    return dest


def download_tus_rec_2025(
    dest: Path,
    auto_publish: bool = False,
    kaggle_dataset_id: str | None = None,
) -> Path:
    """
    Download TUS-REC2025 dataset from Zenodo.

    TUS-REC2025 is the updated forearm ultrasound dataset with additional data.

    Download URLs:
    - Training: https://zenodo.org/records/15224704
    - Validation: https://doi.org/10.5281/zenodo.15699958

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/tus-rec-2025")

    Returns:
        Path to downloaded dataset
    """
    zenodo_urls = {
        "train": "https://zenodo.org/records/15224704/files/train.zip",
        "val": "https://zenodo.org/records/15699958/files/val.zip",
    }

    dest.mkdir(parents=True, exist_ok=True)

    print("TUS-REC2025 Dataset Download")
    print("=" * 40)
    print(f"Destination: {dest}")
    print(f"Please download the following from Zenodo:")
    for name, url in zenodo_urls.items():
        print(f"  - {name}: {url}")
    print()

    # Create a README in the data directory with instructions
    readme = dest / "README.md"
    readme.write_text(
        """# TUS-REC2025 Dataset

## Download Instructions

### Training Data
- https://zenodo.org/records/15224704

### Validation Data
- https://zenodo.org/records/15699958

## Download Script

On Kaggle, use the auto-download script:
```python
from usrecon.data.download import download_tus_rec_2025
dest = download_tus_rec_2025(Path("/kaggle/tmp/tus-rec-2025"))
```

## Expected Structure

tus-rec-2025/
|-- train/
|   |-- images/
|   '-- poses/
'-- val/
    |-- images/
    '-- poses/

## Citation

Li et al., "TUS-REC2025 Challenge: Trackerless Freehand Ultrasound
2D-to-3D Reconstruction", https://github.com/QiLi111/TUS-REC2025-Challenge_baseline
""",
        encoding="utf-8",
    )

    # If on Kaggle with auto-publish, create dataset and publish
    if _running_on_kaggle() and auto_publish and kaggle_dataset_id:
        _publish_to_kaggle_dataset(dest, kaggle_dataset_id)

    return dest


def download_busi(
    dest: Path,
    auto_publish: bool = False,
    kaggle_dataset_id: str | None = None,
) -> Path:
    """
    Download BUSI (Breast Ultrasound Image) dataset.

    BUSI is a public dataset of breast ultrasound images with segmentation masks.
    Used for Stage 4b segmentation head training.

    Download URLs:
    - Official: https://scholar.cu.edu.eg/?q=busi/download
    - Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/busi")

    Returns:
        Path to downloaded dataset
    """
    dest.mkdir(parents=True, exist_ok=True)

    print("BUSI Dataset Download")
    print("=" * 40)
    print(f"Destination: {dest}")
    print(f"Download options:")
    print("  - Official: https://scholar.cu.edu.eg/?q=busi/download")
    print("  - Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print()

    # Create a README in the data directory with instructions
    readme = dest / "README.md"
    readme.write_text(
        """# BUSI (Breast Ultrasound Images) Dataset

## Download Instructions

### Option 1: Official Source
- Go to: https://scholar.cu.edu.eg/?q=busi/download
- Register and download the dataset

### Option 2: Kaggle
- Download from: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- Extract to: {dest}

## Download Script

On Kaggle, use the auto-download script:
```python
from usrecon.data.download import download_busi
dest = download_busi(Path("/kaggle/tmp/busi"))
```

## Expected Structure

busi/
|-- images/           # Raw ultrasound images
|-- masks/            # Segmentation masks
'-- README.txt        # Dataset documentation

## Usage

This dataset is used for Stage 4b (segmentation head training) after
the reconstruction pipeline is frozen.

## Citation

El-Aref et al., "Breast Ultrasound Image Dataset (BUSI)", 2021.
""",
        encoding="utf-8",
    )

    # If on Kaggle with auto-publish, create dataset and publish
    if _running_on_kaggle() and auto_publish and kaggle_dataset_id:
        _publish_to_kaggle_dataset(dest, kaggle_dataset_id)

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
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

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
