"""
Batch landmark extraction with multiprocessing.

Usage (from project root):
    python scripts/run_extraction.py
    python scripts/run_extraction.py --workers 4 --participant aiim001
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

from cognitive_load.dataset import AVCaffeDataset
from cognitive_load.landmark_extractor import LandmarkExtractor

_extractor = None


def init_worker():
    global _extractor
    _extractor = LandmarkExtractor()


def process_single_video(args):
    global _extractor
    video_path, output_path, sample_every = args

    if os.path.exists(output_path):
        try:
            landmarks = np.load(output_path)
            return (video_path, True, landmarks.shape[0], "skipped")
        except Exception:
            pass

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        landmarks = _extractor.extract_from_video(video_path, sample_every)

        if landmarks.size == 0:
            return (video_path, False, 0, "no_face_detected")

        np.save(output_path, landmarks)
        return (video_path, True, landmarks.shape[0], None)

    except Exception as e:
        return (video_path, False, 0, str(e))


def get_output_path(video_path, output_base):
    parts = video_path.replace("\\", "/").split("/")

    try:
        videos_idx = parts.index("videos")
        rel_parts = parts[videos_idx + 1:]
    except ValueError:
        rel_parts = parts[-3:]

    rel_parts[-1] = rel_parts[-1].replace(".avi", ".npy")

    return os.path.join(output_base, *rel_parts)


def run_extraction(
    video_base="data/processed/avcaffe/videos",
    label_path="data/processed/avcaffe/labels/mental_demand.txt",
    output_base="data/processed/avcaffe/landmarks",
    num_workers=6,
    sample_every=3,
    participant=None,
    limit=None,
):
    print("=" * 60)
    print("LANDMARK EXTRACTION")
    print("=" * 60)
    print(f"Video base: {video_base}")
    print(f"Output base: {output_base}")
    print(f"Workers: {num_workers}")
    print(f"Sample every: {sample_every} frames")

    dataset = AVCaffeDataset(video_base, label_path)

    tasks = []
    for video_path, label in dataset:
        if participant and participant not in video_path:
            continue

        output_path = get_output_path(video_path, output_base)
        tasks.append((video_path, output_path, sample_every))

        if limit and len(tasks) >= limit:
            break

    print(f"Videos to process: {len(tasks)}")
    print("=" * 60)

    if not tasks:
        print("No videos to process.")
        return

    results = {
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "no_face": 0,
        "total_frames": 0,
    }

    failed_videos = []

    with Pool(num_workers, initializer=init_worker) as pool:
        for video_path, success, num_frames, error in tqdm(
            pool.imap_unordered(process_single_video, tasks),
            total=len(tasks),
            desc="Extracting landmarks",
        ):
            if success:
                if error == "skipped":
                    results["skipped"] += 1
                else:
                    results["success"] += 1
                results["total_frames"] += num_frames
            else:
                results["failed"] += 1
                if error == "no_face_detected":
                    results["no_face"] += 1
                failed_videos.append((video_path, error))

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {results['success']}")
    print(f"Skipped (already done): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"  - No face detected: {results['no_face']}")
    print(f"Total frames extracted: {results['total_frames']}")

    if failed_videos and len(failed_videos) <= 20:
        print("\nFailed videos:")
        for video, error in failed_videos[:20]:
            print(f"  - {os.path.basename(video)}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facial landmarks from videos")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample every Nth frame")
    parser.add_argument("--participant", type=str, help="Process only this participant")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--video-base", type=str, default="data/processed/avcaffe/videos")
    parser.add_argument("--output-base", type=str, default="data/processed/avcaffe/landmarks")
    parser.add_argument("--label-path", type=str, default="data/processed/avcaffe/labels/mental_demand.txt")

    args = parser.parse_args()

    run_extraction(
        video_base=args.video_base,
        label_path=args.label_path,
        output_base=args.output_base,
        num_workers=args.workers,
        sample_every=args.sample_every,
        participant=args.participant,
        limit=args.limit,
    )
