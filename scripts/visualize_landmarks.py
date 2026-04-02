"""
Visualize facial landmarks on video frames.

Usage (from project root):
    python scripts/visualize_landmarks.py --save artifacts/debug_landmarks.png
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import argparse
import os

from cognitive_load.video_utils import read_video_frames, get_video_info
from cognitive_load.landmark_extractor import LandmarkExtractor

LANDMARK_GROUPS = {
    "face_oval": list(range(0, 17)) + list(range(17, 27)),
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "nose": [1, 2, 98, 327, 4, 5, 6, 168],
    "mouth_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "mouth_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    "left_iris": list(range(468, 473)),
    "right_iris": list(range(473, 478)),
}

GROUP_COLORS = {
    "face_oval": (200, 200, 200),
    "left_eye": (0, 255, 0),
    "right_eye": (0, 255, 0),
    "left_eyebrow": (255, 200, 0),
    "right_eyebrow": (255, 200, 0),
    "nose": (0, 255, 255),
    "mouth_outer": (0, 0, 255),
    "mouth_inner": (0, 100, 255),
    "left_iris": (255, 0, 255),
    "right_iris": (255, 0, 255),
}


def draw_landmarks(frame, landmarks, draw_all=True, draw_connections=False):
    h, w = frame.shape[:2]
    result = frame.copy()

    if draw_all:
        for i, (x, y, z) in enumerate(landmarks):
            px, py = int(x * w), int(y * h)
            cv2.circle(result, (px, py), 1, (0, 255, 0), -1)

    for group_name, indices in LANDMARK_GROUPS.items():
        color = GROUP_COLORS.get(group_name, (255, 255, 255))

        points = []
        for idx in indices:
            if idx < len(landmarks):
                x, y, z = landmarks[idx]
                px, py = int(x * w), int(y * h)
                points.append((px, py))
                cv2.circle(result, (px, py), 2, color, -1)

        if draw_connections and len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(result, points[i], points[i + 1], color, 1)

    return result


def visualize_video(video_path, num_frames=5, save_path=None, show_live=False):
    print(f"Video: {video_path}")

    info = get_video_info(video_path)
    if info:
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Total frames: {info['frame_count']}")

    extractor = LandmarkExtractor()

    if show_live:
        print("\nPlaying video with landmarks (press 'q' to quit)...")
        for frame_idx, frame in read_video_frames(video_path, sample_every=1):
            landmarks = extractor.extract_from_frame(frame)

            if landmarks is not None:
                frame = draw_landmarks(frame, landmarks)

            cv2.imshow("Landmarks", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    else:
        frames_collected = []

        for frame_idx, frame in read_video_frames(video_path, sample_every=3):
            landmarks = extractor.extract_from_frame(frame)

            if landmarks is not None:
                annotated = draw_landmarks(frame, landmarks, draw_connections=True)
                frames_collected.append((frame_idx, annotated, landmarks))

            if len(frames_collected) >= num_frames:
                break

        if not frames_collected:
            print("No faces detected in video!")
            return

        print(f"\nCollected {len(frames_collected)} frames with landmarks")

        grid_cols = min(len(frames_collected), 3)
        grid_rows = (len(frames_collected) + grid_cols - 1) // grid_cols

        frame_h, frame_w = frames_collected[0][1].shape[:2]
        grid = np.zeros((grid_rows * frame_h, grid_cols * frame_w, 3), dtype=np.uint8)

        for i, (frame_idx, frame, landmarks) in enumerate(frames_collected):
            row, col = i // grid_cols, i % grid_cols
            grid[row * frame_h : (row + 1) * frame_h, col * frame_w : (col + 1) * frame_w] = frame

            cv2.putText(
                grid,
                f"Frame {frame_idx}",
                (col * frame_w + 10, row * frame_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if save_path:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            cv2.imwrite(save_path, grid)
            print(f"Saved visualization to: {save_path}")
        else:
            cv2.imshow("Landmark Visualization", grid)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        _, _, sample_landmarks = frames_collected[0]
        print("\nSample landmark coordinates (frame 0):")
        print(f"  Nose tip (idx 1):      x={sample_landmarks[1, 0]:.4f}, y={sample_landmarks[1, 1]:.4f}")
        print(f"  Left eye (idx 33):     x={sample_landmarks[33, 0]:.4f}, y={sample_landmarks[33, 1]:.4f}")
        print(f"  Right eye (idx 263):   x={sample_landmarks[263, 0]:.4f}, y={sample_landmarks[263, 1]:.4f}")
        print(f"  Left iris (idx 468):   x={sample_landmarks[468, 0]:.4f}, y={sample_landmarks[468, 1]:.4f}")
        print(f"  Right iris (idx 473):  x={sample_landmarks[473, 0]:.4f}, y={sample_landmarks[473, 1]:.4f}")


def visualize_from_npy(npy_path, video_path=None):
    landmarks = np.load(npy_path)
    print(f"Loaded landmarks: {npy_path}")
    print(f"  Shape: {landmarks.shape}")
    print(f"  Frames: {landmarks.shape[0]}")

    if video_path and os.path.exists(video_path):
        print(f"  Overlaying on video: {video_path}")

    print("\nLandmark statistics:")
    print(f"  X range: [{landmarks[:, :, 0].min():.4f}, {landmarks[:, :, 0].max():.4f}]")
    print(f"  Y range: [{landmarks[:, :, 1].min():.4f}, {landmarks[:, :, 1].max():.4f}]")
    print(f"  Z range: [{landmarks[:, :, 2].min():.4f}, {landmarks[:, :, 2].max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize facial landmarks")
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--save", type=str, help="Save visualization to file")
    parser.add_argument("--live", action="store_true", help="Show live video with landmarks")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to visualize")
    parser.add_argument("--npy", type=str, help="Visualize from .npy file instead")

    args = parser.parse_args()

    if args.npy:
        visualize_from_npy(args.npy, args.video)
    else:
        if args.video:
            video_path = args.video
        else:
            from cognitive_load.dataset import AVCaffeDataset

            dataset = AVCaffeDataset(
                video_base="data/processed/avcaffe/videos",
                label_path="data/processed/avcaffe/labels/mental_demand.txt",
            )
            video_path, _ = dataset[0]
            print("Using first video from dataset...")

        visualize_video(
            video_path,
            num_frames=args.frames,
            save_path=args.save,
            show_live=args.live,
        )
