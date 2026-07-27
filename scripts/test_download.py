#!/usr/bin/env python
"""Test script for data download utilities."""

from usrecon.data import get_dataset, download_tus_rec_2024, download_busi
from usrecon.paths import DATA_DIR

def test_get_dataset():
    """Test get_dataset returns a path."""
    # This will show instructions but not actually download
    path = get_dataset("tus-rec-2024")
    print(f"TUS-REC2024 path: {path}")
    assert path.exists(), f"Expected {path} to exist"
    assert (path / "README.md").exists(), "Expected README.md with instructions"
    print("get_dataset test passed!")

def test_download_functions():
    """Test individual download functions."""
    tus_rec_path = download_tus_rec_2024(DATA_DIR / "tus-rec-2024-test")
    print(f"TUS-REC2024 download path: {tus_rec_path}")

    busi_path = download_busi(DATA_DIR / "busi-test")
    print(f"BUSI download path: {busi_path}")
    print("Download functions test passed!")

if __name__ == "__main__":
    test_get_dataset()
    test_download_functions()
    print("\nAll tests passed!")
