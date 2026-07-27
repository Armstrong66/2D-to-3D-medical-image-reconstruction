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

Environment variables:
- KAGGLE_TOKEN: Kaggle API token (used for auto-publishing to Kaggle Datasets)
- KAGGLE_USERNAME: Kaggle username (used with KAGGLE_TOKEN)
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


# ============================================================================
# TUS-REC2024 Dataset URLs (from QiLi111/TUS-REC2025-Challenge_baseline)
# ============================================================================
TUS_REC_2024_URLS = {
    # Training parts (43GB total)
    "train_part1": "https://zenodo.org/records/11178509/files/train_part1.zip?download=1",
    "train_part2": "https://zenodo.org/records/11180795/files/train_part2.zip?download=1",
    "train_part3": "https://zenodo.org/records/11355500/files/landmark.zip?download=1",
    # Validation data (4GB)
    "val": "https://zenodo.org/records/12979481/files/Freehand_US_data_val.zip?download=1",
}

# ============================================================================
# TUS-REC2025 Dataset URLs
# ============================================================================
TUS_REC_2025_URLS = {
    "train": "https://zenodo.org/records/15224704/files/Freehand_US_data_train_2025.zip?download=1",
    "val": "https://zenodo.org/records/15699958/files/Freehand_US_data_val_2025.zip?download=1",
}

# ============================================================================
# BUSI Dataset URLs
# ============================================================================
BUSI_URLS = {
    "kaggle": "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/download?datasetVersionNumber=1",
}


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


def _download_file(url: str, dest: Path, chunk_size: int = 8192, retry: int = 3) -> Path:
    """
    Download a file from a URL with progress.

    Args:
        url: File URL
        dest: Destination path
        chunk_size: Download chunk size in bytes
        retry: Number of retries on failure

    Returns:
        Path to downloaded file
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    filename = url.split("?")[0].split("/")[-1]
    print(f"Downloading {filename} to {dest}...")

    import ssl
    # Create SSL context that doesn't verify certificates (handles some Zenodo issues)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Use urllib with custom opener
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    opener.addheaders = [("User-Agent", "usrecon-download/1.0")]

    last_error = None
    for attempt in range(retry):
        try:
            with opener.open(url) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                bytes_downloaded = 0

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        if total_size > 0:
                            progress = bytes_downloaded / total_size * 100
                            print(f"  Progress: {progress:.1f}% ({bytes_downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")

            print(f"Downloaded: {dest}")
            return dest
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1}/{retry} failed: {e}")
            if attempt < retry - 1:
                import time
                time.sleep(1)  # Wait before retry

    raise RuntimeError(f"Failed to download {url} after {retry} attempts: {last_error}")


def _has_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are available."""
    return (
        "KAGGLE_TOKEN" in os.environ
        and "KAGGLE_USERNAME" in os.environ
        and os.environ.get("KAGGLE_TOKEN")
        and os.environ.get("KAGGLE_USERNAME")
    )


def _setup_kaggle_credentials() -> None:
    """Configure Kaggle CLI credentials from environment variables."""
    if not _has_kaggle_credentials():
        return

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        credentials = {
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_TOKEN"],
        }
        with open(kaggle_json, "w") as f:
            json.dump(credentials, f)

        # Set permissions (required by Kaggle CLI)
        try:
            os.chmod(kaggle_json, 0o600)
        except (OSError, NotImplementedError):
            pass  # Windows doesn't support chmod
        print(f"Kaggle credentials configured at {kaggle_json}")


def _publish_to_kaggle_dataset(dataset_dir: Path, dataset_id: str) -> None:
    """
    Publish a folder as a Kaggle dataset using the Kaggle API.

    Args:
        dataset_dir: Directory containing the dataset to publish
        dataset_id: Kaggle dataset ID (e.g., 'usrecon/tus-rec-2024')
    """
    if not _has_kaggle_credentials():
        print(
            f"Warning: Kaggle credentials not found in environment. "
            f"Set KAGGLE_USERNAME and KAGGLE_TOKEN to enable auto-publish."
        )
        return

    # Configure Kaggle CLI credentials
    _setup_kaggle_credentials()

    # Check if Kaggle CLI is installed
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError("Kaggle CLI not working")
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
        print(
            f"Warning: Kaggle CLI not available ({e}). "
            f"Install with: pip install kaggle"
        )
        return

    # Create dataset-metadata.json
    dataset_name = dataset_id.split("/")[-1]
    metadata = {
        "id": dataset_id,
        "title": dataset_name.replace("-", " ").title(),
        "slug": dataset_name,
        "description": f"Trackerless Freehand Ultrasound 2D-to-3D Reconstruction Dataset",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ultrasound", "medical-imaging", "2d-to-3d", "reconstruction"],
        "authors": [
            {"name": "usrecon", "affiliation": "ACVSS26 Hackathon"}
        ],
    }

    metadata_file = dataset_dir / "dataset-metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Publishing {dataset_id} to Kaggle...")

    # Run kaggle datasets create
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "create",
                "-p", str(dataset_dir),
                "--dir-mode", "zip",
                "--public",
            ],
            capture_output=True,
            text=True,
            cwd=str(dataset_dir),
            timeout=300,  # 5 minutes for upload
        )
        if result.returncode == 0:
            print(f"Successfully published {dataset_id}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"Failed to publish {dataset_id}:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"Timeout publishing {dataset_id}")
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
    - train_part1: https://zenodo.org/records/11178509/files/train_part1.zip
    - train_part2: https://zenodo.org/records/11180795/files/train_part2.zip
    - train_part3: https://zenodo.org/records/11355500/files/landmark.zip
    - val: https://zenodo.org/records/12979481/files/Freehand_US_data_val.zip

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/tus-rec-2024")

    Returns:
        Path to downloaded dataset
    """
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TUS-REC2024 Dataset Download")
    print("=" * 60)
    print(f"Destination: {dest}")
    print()
    print("Downloading from Zenodo:")
    for name, url in TUS_REC_2024_URLS.items():
        print(f"  - {name}: {url.split('?')[0]}")

    print()
    print("Note: TUS-REC2024 training data is split into 3 parts (~43GB total)")
    print("      Validation data is ~4GB")
    print()

    # Download each part
    downloaded_files = []
    for part_name, url in TUS_REC_2024_URLS.items():
        zip_path = dest / f"{part_name}.zip"
        if not zip_path.exists():
            _download_file(url, zip_path)
        downloaded_files.append(zip_path)

    # Extract archives
    print()
    print("Extracting archives...")
    for zip_path in downloaded_files:
        if zip_path.exists():
            print(f"Extracting {zip_path.name}...")
            extract_archive(zip_path, dest / "extracted")
            zip_path.unlink()  # Remove zip after extraction

    print(f"Download complete: {dest}")

    # Create README
    _create_tus_rec_readme(dest, "2024")

    # Auto-publish to Kaggle if enabled
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
    - train: https://zenodo.org/records/15224704/files/Freehand_US_data_train_2025.zip
    - val: https://zenodo.org/records/15699958/files/Freehand_US_data_val_2025.zip

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/tus-rec-2025")

    Returns:
        Path to downloaded dataset
    """
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TUS-REC2025 Dataset Download")
    print("=" * 60)
    print(f"Destination: {dest}")
    print()
    print("Downloading from Zenodo:")
    for name, url in TUS_REC_2025_URLS.items():
        print(f"  - {name}: {url.split('?')[0]}")

    print()

    # Download each part
    downloaded_files = []
    for part_name, url in TUS_REC_2025_URLS.items():
        zip_path = dest / f"{part_name}.zip"
        if not zip_path.exists():
            _download_file(url, zip_path)
        downloaded_files.append(zip_path)

    # Extract archives
    print()
    print("Extracting archives...")
    for zip_path in downloaded_files:
        if zip_path.exists():
            print(f"Extracting {zip_path.name}...")
            extract_archive(zip_path, dest / "extracted")
            zip_path.unlink()

    print(f"Download complete: {dest}")

    # Create README
    _create_tus_rec_readme(dest, "2025")

    # Auto-publish to Kaggle if enabled
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
    - Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

    Args:
        dest: Destination directory
        auto_publish: If True and on Kaggle, publish dataset to Kaggle
        kaggle_dataset_id: Dataset ID for Kaggle (e.g., "usrecon/busi")

    Returns:
        Path to downloaded dataset
    """
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUSI Dataset Download")
    print("=" * 60)
    print(f"Destination: {dest}")
    print()
    print("Download options:")
    print("  - Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print()

    # Download from Kaggle
    busi_zip = dest / "busi.zip"
    if not busi_zip.exists():
        _download_file(BUSI_URLS["kaggle"], busi_zip)

    # Extract
    print()
    print("Extracting...")
    extract_archive(busi_zip, dest)
    busi_zip.unlink()

    print(f"Download complete: {dest}")

    # Create README
    _create_busi_readme(dest)

    # Auto-publish to Kaggle if enabled
    if _running_on_kaggle() and auto_publish and kaggle_dataset_id:
        _publish_to_kaggle_dataset(dest, kaggle_dataset_id)

    return dest


def _create_tus_rec_readme(dest: Path, year: str) -> None:
    """Create README for TUS-REC dataset."""
    readme = dest / "README.md"
    readme.write_text(
        f"""# TUS-REC{year} Dataset

## Description
Trackerless Freehand Ultrasound 2D-to-3D Reconstruction Challenge dataset.

## Download Source
- Official: https://github.com/QiLi111/TUS-REC2025-Challenge_baseline
- Zenodo: https://zenodo.org/records/15224704 (TUS-REC2025)
- Zenodo: https://zenodo.org/records/11178509 (TUS-REC2024 train_part1)

## Dataset Contents
### Training Data
- train/ - Training images and poses
- Size: ~43GB (TUS-REC2024), ~50GB (TUS-REC2025)

### Validation Data
- val/ - Validation images and poses
- Size: ~4GB

## Expected Structure
tus-rec-{year}/
|-- train/
|   |-- images/     # Raw ultrasound frames
|   '-- poses/      # Ground-truth poses
|-- val/
|   |-- images/
|   '-- poses/
'-- test/
    |-- images/
    '-- poses/

## Citation
Li et al., "TUS-REC2025 Challenge: Trackerless Freehand Ultrasound
2D-to-3D Reconstruction", https://github.com/QiLi111/TUS-REC2025-Challenge_baseline
""",
        encoding="utf-8",
    )


def _create_busi_readme(dest: Path) -> None:
    """Create README for BUSI dataset."""
    readme = dest / "README.md"
    readme.write_text(
        """# BUSI (Breast Ultrasound Images) Dataset

## Description
Public dataset of breast ultrasound images with segmentation masks.
Used for Stage 4b (segmentation head training) after reconstruction pipeline.

## Download Source
- Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

## Dataset Contents
- images/ - Raw ultrasound images
- masks/ - Segmentation masks
- README.txt - Dataset documentation

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
