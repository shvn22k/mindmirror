"""
Batch per-frame feature extraction from landmark .npy files.

Usage (from project root):
    python scripts/run_feature_extraction.py --workers 6
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from features.feature_extractor import FeatureExtractor


def get_output_path(input_path, input_base, output_base):
    rel_path = os.path.relpath(input_path, input_base)
    rel_path = rel_path.replace(".npy", ".parquet")
    return os.path.join(output_base, rel_path)


def process_single_file(args):
    input_path, output_path, extractor_kwargs = args

    if os.path.exists(output_path):
        try:
            import pandas as pd

            df = pd.read_parquet(output_path)
            return (input_path, True, len(df), "skipped")
        except Exception:
            pass

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        landmarks = np.load(input_path)

        if landmarks.size == 0:
            return (input_path, False, 0, "empty_landmarks")

        clip_id = os.path.splitext(os.path.basename(input_path))[0]
        extractor = FeatureExtractor(**extractor_kwargs)
        df = extractor.extract_from_landmarks(landmarks, clip_id)

        df.to_parquet(output_path, index=False)

        return (input_path, True, len(df), None)

    except Exception as e:
        return (input_path, False, 0, str(e))


def find_landmark_files(base_path, participant=None):
    for root, dirs, files in os.walk(base_path):
        if participant and participant not in root:
            continue

        for file in files:
            if file.endswith(".npy"):
                yield os.path.join(root, file)


def run_feature_extraction(
    input_base="data/processed/avcaffe/landmarks",
    output_base="data/processed/avcaffe/features",
    num_workers=6,
    participant=None,
    limit=None,
    fps=25,
    sample_every=3,
):
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Input base: {input_base}")
    print(f"Output base: {output_base}")
    print(f"Workers: {num_workers}")

    extractor_kwargs = {
        "fps": fps,
        "sample_every": sample_every,
    }

    tasks = []
    for input_path in find_landmark_files(input_base, participant):
        output_path = get_output_path(input_path, input_base, output_base)
        tasks.append((input_path, output_path, extractor_kwargs))

        if limit and len(tasks) >= limit:
            break

    print(f"Files to process: {len(tasks)}")
    print("=" * 60)

    if not tasks:
        print("No files to process.")
        return

    results = {"success": 0, "skipped": 0, "failed": 0, "total_frames": 0}

    failed_files = []

    with Pool(num_workers) as pool:
        for input_path, success, num_frames, error in tqdm(
            pool.imap_unordered(process_single_file, tasks),
            total=len(tasks),
            desc="Extracting features",
        ):
            if success:
                if error == "skipped":
                    results["skipped"] += 1
                else:
                    results["success"] += 1
                results["total_frames"] += num_frames
            else:
                results["failed"] += 1
                failed_files.append((input_path, error))

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {results['success']}")
    print(f"Skipped (already done): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"Total frames processed: {results['total_frames']}")

    if failed_files and len(failed_files) <= 20:
        print("\nFailed files:")
        for path, error in failed_files[:20]:
            print(f"  - {os.path.basename(path)}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from landmarks")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--participant", type=str, help="Process only this participant")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--input-base", type=str, default="data/processed/avcaffe/landmarks")
    parser.add_argument("--output-base", type=str, default="data/processed/avcaffe/features")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    parser.add_argument("--sample-every", type=int, default=3, help="Frame sampling rate")

    args = parser.parse_args()

    run_feature_extraction(
        input_base=args.input_base,
        output_base=args.output_base,
        num_workers=args.workers,
        participant=args.participant,
        limit=args.limit,
        fps=args.fps,
        sample_every=args.sample_every,
    )
