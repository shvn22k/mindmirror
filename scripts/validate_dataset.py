"""
Validate video-to-label mapping for AVCAffe.

Usage (from project root):
    python scripts/validate_dataset.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

from cognitive_load.load_labels import load_labels
from cognitive_load.dataset import get_video_label


def validate_dataset(video_base, label_path):
    labels = load_labels(label_path)

    total_videos = 0
    matched = 0
    missing_labels = []

    participants = sorted(os.listdir(video_base))

    for participant in participants:
        participant_path = os.path.join(video_base, participant)
        if not os.path.isdir(participant_path):
            continue

        tasks = sorted(os.listdir(participant_path))

        for task in tasks:
            task_path = os.path.join(participant_path, task)
            if not os.path.isdir(task_path):
                continue

            videos = [f for f in os.listdir(task_path) if f.endswith(".avi")]

            for video_file in videos:
                video_path = os.path.join(task_path, video_file)
                total_videos += 1

                label = get_video_label(video_path, labels)

                if label is not None:
                    matched += 1
                else:
                    key = f"{participant}_{task}"
                    if key not in [m[0] for m in missing_labels]:
                        missing_labels.append((key, video_path))

    return {
        "total_videos": total_videos,
        "matched": matched,
        "missing_count": total_videos - matched,
        "missing_labels": missing_labels,
        "total_label_keys": len(labels),
    }


def get_all_video_label_pairs(video_base, label_path):
    labels = load_labels(label_path)

    for participant in sorted(os.listdir(video_base)):
        participant_path = os.path.join(video_base, participant)
        if not os.path.isdir(participant_path):
            continue

        for task in sorted(os.listdir(participant_path)):
            task_path = os.path.join(participant_path, task)
            if not os.path.isdir(task_path):
                continue

            for video_file in sorted(os.listdir(task_path)):
                if not video_file.endswith(".avi"):
                    continue

                video_path = os.path.join(task_path, video_file)
                label = get_video_label(video_path, labels)

                if label is not None:
                    yield video_path, label


if __name__ == "__main__":
    video_base = "data/processed/avcaffe/videos"
    label_path = "data/processed/avcaffe/labels/mental_demand.txt"

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    stats = validate_dataset(video_base, label_path)

    print(f"\nTotal videos found: {stats['total_videos']}")
    print(f"Videos with labels: {stats['matched']}")
    print(f"Videos missing labels: {stats['missing_count']}")
    print(f"Total label keys in file: {stats['total_label_keys']}")

    match_rate = (stats["matched"] / stats["total_videos"] * 100) if stats["total_videos"] > 0 else 0
    print(f"\nMatch rate: {match_rate:.1f}%")

    if stats["missing_labels"]:
        print("\nMissing label keys (first 10):")
        for key, path in stats["missing_labels"][:10]:
            print(f"  - {key}")

    print("\n" + "=" * 60)
    print("SAMPLE VIDEO-LABEL PAIRS")
    print("=" * 60)

    count = 0
    for video_path, label in get_all_video_label_pairs(video_base, label_path):
        print(f"  {os.path.basename(video_path)} -> {label}")
        count += 1
        if count >= 10:
            break

    print("\n[Validation Complete]")
