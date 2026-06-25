"""
download_dataset.py

Downloads the "Plant Diseases" classification dataset from Roboflow
Universe.

Dataset: https://universe.roboflow.com/plant-diseases/plant-diseases-oqkrq
Task: Image Classification (10 classes covering citrus fruit/leaf
      diseases and healthy fruit/leaf):
        black_spot_fruit, black_spot_leave, canker_fruit, canker_leave,
        greening__fruit, greening_leave, healthy_fruit, healthy_leave,
        melanose_leave, scab_fruit
Size: 963 images
License: Public Domain (CC0)

You need a free Roboflow account + API key to download:
  1. Sign up at https://roboflow.com
  2. Get your API key from https://app.roboflow.com/settings/api
  3. Either:
       export ROBOFLOW_API_KEY="your_key_here"
     or pass it directly with --api_key

Usage:
    python src/download_dataset.py --api_key YOUR_KEY
    python src/download_dataset.py   # uses ROBOFLOW_API_KEY env var

If you'd rather use a different classification dataset from Roboflow
Universe, just change WORKSPACE / PROJECT / VERSION below to match the
dataset's "Download Code" snippet on its Universe page.
"""

import argparse
import os
import sys

from roboflow import Roboflow

# ---- Dataset config: change these to swap datasets -------------------
WORKSPACE = "plant-diseases"
PROJECT = "plant-diseases-oqkrq"
VERSION = 1
FORMAT = "folder"  # "folder" gives a simple train/valid/test ImageFolder layout
# -----------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download(api_key: str, output_dir: str):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} ({FORMAT} format)...")
    dataset = version.download(FORMAT, location=output_dir)
    print(f"Done. Dataset saved to: {dataset.location}")
    return dataset.location


def main():
    parser = argparse.ArgumentParser(description="Download a Roboflow Universe classification dataset.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the downloaded dataset (default: ../data)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: No Roboflow API key found.\n"
            "Pass one with --api_key YOUR_KEY or set the ROBOFLOW_API_KEY "
            "environment variable.\n"
            "Get a free key at https://app.roboflow.com/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)

    download(args.api_key, args.output_dir)


if __name__ == "__main__":
    main()
