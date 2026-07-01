#!/usr/bin/env python3
import os
import sys
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path("data/coco")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URL and path configurations
VAL_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
VAL_ZIP_PATH = DATA_DIR / "val2017.zip"
VAL_DIR = DATA_DIR / "val2017"

ANNO_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNO_ZIP_PATH = DATA_DIR / "annotations.zip"
ANNO_DIR = DATA_DIR / "annotations"

NUM_THREADS = 16

def download_range(url, start, end, part_path):
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    with open(part_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

def download_file_parallel(url, target_path, num_threads=NUM_THREADS):
    print(f"\nAnalyzing file size for: {url}")
    # Get total size of the file
    r = requests.head(url, timeout=30)
    r.raise_for_status()
    
    # If range requests are not supported, fallback to standard download
    if r.headers.get("accept-ranges") != "bytes" and "Content-Length" not in r.headers:
        print("Range requests not supported, falling back to sequential download.")
        r_file = requests.get(url, stream=True)
        with open(target_path, "wb") as f:
            for chunk in r_file.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        return

    total_size = int(r.headers.get("content-length", 0))
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

    chunk_size = total_size // num_threads
    futures = []
    part_paths = []

    print(f"Downloading in parallel using {num_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
            part_path = target_path.with_name(f"{target_path.name}.part{i}")
            part_paths.append(part_path)
            futures.append(executor.submit(download_range, url, start, end, part_path))

        # Monitor progress
        completed_count = 0
        for future in as_completed(futures):
            future.result()
            completed_count += 1
            print(f" Progress: {completed_count}/{num_threads} parts completed...")

    # Combine parts
    print(f"Merging parts into {target_path}...")
    with open(target_path, "wb") as outfile:
        for part_path in part_paths:
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
            part_path.unlink() # Delete part file
            
    print("Download and merge complete!")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        for i, file in enumerate(file_list):
            zip_ref.extract(file, extract_dir)
            if i % 1000 == 0:
                print(f" Extracted {i}/{total_files} files...")
    print(f"Finished extraction of {zip_path}.")
    zip_path.unlink()
    print(f"Deleted zip file: {zip_path}")

def main():
    try:
        # 1. Download & extract annotations
        if not (ANNO_DIR / "instances_val2017.json").exists():
            download_file_parallel(ANNO_ZIP_URL, ANNO_ZIP_PATH)
            extract_zip(ANNO_ZIP_PATH, DATA_DIR)
        else:
            print("COCO 2017 validation annotations already exist. Skipping.")

        # 2. Download & extract validation images
        if not VAL_DIR.exists():
            download_file_parallel(VAL_ZIP_URL, VAL_ZIP_PATH)
            extract_zip(VAL_ZIP_PATH, DATA_DIR)
        else:
            print("COCO 2017 validation images directory already exists. Skipping.")

        print("\nCOCO 2017 validation dataset is successfully set up at 'data/coco'!")

    except Exception as e:
        print(f"\nError: {e}")
        # Clean up any leftover parts if error occurs
        for part in DATA_DIR.glob("*.part*"):
            try:
                part.unlink()
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
