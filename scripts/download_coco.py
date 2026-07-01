#!/usr/bin/env python3
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Ensure data directory exists
DATA_DIR = Path("data/coco")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URL and path configurations
VAL_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
VAL_ZIP_PATH = DATA_DIR / "val2017.zip"
VAL_DIR = DATA_DIR / "val2017"

ANNO_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNO_ZIP_PATH = DATA_DIR / "annotations.zip"
ANNO_DIR = DATA_DIR / "annotations"

def reporthook(blocknum, blocksize, totalsize):
    """Callback for progress bar."""
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = f"\rDownloading: {percent:.2f}% ({readsofar / 1024 / 1024:.2f} MB / {totalsize / 1024 / 1024:.2f} MB)"
        sys.stdout.write(s)
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rRead {readsofar} bytes")

def download_file(url, target_path):
    print(f"\nStarting download: {url} -> {target_path}")
    if target_path.exists():
        print(f"File already exists at {target_path}, skipping download.")
        return
    urllib.request.urlretrieve(url, str(target_path), reporthook)
    print("\nDownload complete!")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get count of files for progress print
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        for i, file in enumerate(file_list):
            zip_ref.extract(file, extract_dir)
            if i % 500 == 0:
                print(f" Extracted {i}/{total_files} files...")
    print(f"Finished extraction of {zip_path}.")
    # Clean up zip file
    zip_path.unlink()
    print(f"Deleted zip file: {zip_path}")

def main():
    try:
        # 1. Download & extract annotations
        if not (ANNO_DIR / "instances_val2017.json").exists():
            download_file(ANNO_ZIP_URL, ANNO_ZIP_PATH)
            extract_zip(ANNO_ZIP_PATH, DATA_DIR)
        else:
            print("COCO 2017 validation annotations already exist. Skipping.")

        # 2. Download & extract validation images
        if not VAL_DIR.exists():
            download_file(VAL_ZIP_URL, VAL_ZIP_PATH)
            extract_zip(VAL_ZIP_PATH, DATA_DIR)
        else:
            print("COCO 2017 validation images directory already exists. Skipping.")

        print("\nCOCO 2017 validation dataset is successfully set up at 'data/coco'!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
