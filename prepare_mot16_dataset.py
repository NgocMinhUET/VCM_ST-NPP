#!/usr/bin/env python
"""
Script to download and prepare the MOT16 dataset.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                 reporthook=t.update_to)

def main():
    # Set paths
    dataset_dir = Path("datasets/MOTChallenge/MOT16")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for MOT16 dataset
    mot16_train_url = "https://motchallenge.net/data/MOT16.zip"
    
    # Download paths
    mot16_zip_path = dataset_dir / "MOT16.zip"
    
    # Download MOT16 dataset
    print("Downloading MOT16 dataset...")
    if not mot16_zip_path.exists():
        download_url(mot16_train_url, mot16_zip_path)
    
    # Extract dataset
    print("Extracting MOT16 dataset...")
    with zipfile.ZipFile(mot16_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Clean up
    mot16_zip_path.unlink()
    
    print("MOT16 dataset preparation completed!")
    print(f"Dataset is available at: {dataset_dir.absolute()}")

if __name__ == "__main__":
    main() 