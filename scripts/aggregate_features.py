"""
Aggregate per-frame Parquet features into clip_features.csv for ML.

Usage (from project root):
    python scripts/aggregate_features.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.clip_aggregate import aggregate_clip_features


def find_feature_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".parquet"):
                yield os.path.join(root, file)


def run_aggregation(
    input_base="data/processed/avcaffe/features",
    output_path="data/processed/avcaffe/clip_features.csv",
    label_path="data/processed/avcaffe/labels/mental_demand.txt",
    fps=25,
    sample_every=3,
):
    print("=" * 60)
    print("FEATURE AGGREGATION")
    print("=" * 60)
    print(f"Input base: {input_base}")
    print(f"Output: {output_path}")

    labels = {}
    if os.path.exists(label_path):
        print(f"Loading labels from: {label_path}")
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    parts = line.split(",")
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    labels[key] = value
        print(f"Loaded {len(labels)} labels")

    files = list(find_feature_files(input_base))
    print(f"Feature files found: {len(files)}")
    print("=" * 60)

    if not files:
        print("No feature files found.")
        return

    rows = []
    failed = 0

    for file_path in tqdm(files, desc="Aggregating"):
        try:
            df = pd.read_parquet(file_path)
            features = aggregate_clip_features(df, fps=fps, sample_every=sample_every)

            clip_id = features.get("clip_id", os.path.splitext(os.path.basename(file_path))[0])

            parts = clip_id.split("_")
            if len(parts) >= 3:
                label_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                if label_key in labels:
                    features["label"] = labels[label_key]

            rows.append(features)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Error processing {file_path}: {e}")

    result_df = pd.DataFrame(rows)

    cols = list(result_df.columns)
    priority = ["clip_id", "label", "n_frames", "duration_seconds"]
    ordered = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    result_df = result_df[ordered]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"Total clips processed: {len(rows)}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_path}")
    print(f"Output shape: {result_df.shape}")

    if "label" in result_df.columns:
        print(f"Clips with labels: {result_df['label'].notna().sum()}")

    print(f"\nFeature columns ({len(result_df.columns)}):")
    print(list(result_df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per-frame features to per-clip")
    parser.add_argument("--input-base", type=str, default="data/processed/avcaffe/features")
    parser.add_argument("--output", type=str, default="data/processed/avcaffe/clip_features.csv")
    parser.add_argument("--label-path", type=str, default="data/processed/avcaffe/labels/mental_demand.txt")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample-every", type=int, default=3)

    args = parser.parse_args()

    run_aggregation(
        input_base=args.input_base,
        output_path=args.output,
        label_path=args.label_path,
        fps=args.fps,
        sample_every=args.sample_every,
    )
