import os
from pathlib import Path

import cv2
import numpy as np

from cognitive_load.video_utils import get_video_info, read_video_frames

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _mediapipe_model_path() -> str:
    """Project root / models/mediapipe/face_landmarker.task"""
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "mediapipe"
    model_path = model_dir / "face_landmarker.task"

    if not model_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading face landmarker model...")
        import urllib.request

        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")

    return str(model_path)


class LandmarkExtractor:
    """MediaPipe FaceLandmarker: 478 landmarks per frame (x, y, z normalized)."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        model_path = _mediapipe_model_path()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_from_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks],
            dtype=np.float32,
        )

        return landmarks

    def extract_from_video(self, video_path, sample_every=3):
        all_landmarks = []

        for frame_idx, frame in read_video_frames(video_path, sample_every):
            landmarks = self.extract_from_frame(frame)

            if landmarks is not None:
                all_landmarks.append(landmarks)

        if not all_landmarks:
            return np.array([], dtype=np.float32)

        return np.stack(all_landmarks, axis=0)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def extract_from_video(video_path, output_path, sample_every=3):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with LandmarkExtractor() as extractor:
        landmarks = extractor.extract_from_video(video_path, sample_every)

        if landmarks.size == 0:
            print(f"Warning: No landmarks extracted from {video_path}")
            return False

        np.save(output_path, landmarks)
        return True


def process_single_video(args):
    video_path, output_path, sample_every = args

    if os.path.exists(output_path):
        landmarks = np.load(output_path)
        return (video_path, True, landmarks.shape[0])

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        extractor = LandmarkExtractor()
        landmarks = extractor.extract_from_video(video_path, sample_every)
        extractor.close()

        if landmarks.size == 0:
            return (video_path, False, 0)

        np.save(output_path, landmarks)
        return (video_path, True, landmarks.shape[0])

    except Exception:
        return (video_path, False, 0)


if __name__ == "__main__":
    from cognitive_load.dataset import AVCaffeDataset

    print("Testing LandmarkExtractor...")

    dataset = AVCaffeDataset(
        video_base="data/processed/avcaffe/videos",
        label_path="data/processed/avcaffe/labels/mental_demand.txt",
    )

    video_path, label = dataset[0]
    print(f"\nVideo: {video_path}")
    print(f"Label: {label}")

    info = get_video_info(video_path)
    print(f"Total frames: {info['frame_count']}")

    with LandmarkExtractor() as extractor:
        landmarks = extractor.extract_from_video(video_path, sample_every=3)

        print(f"\nExtracted landmarks shape: {landmarks.shape}")
        if landmarks.size > 0:
            print(
                f"  Frame 0 nose (idx 1): x={landmarks[0, 1, 0]:.4f}, y={landmarks[0, 1, 1]:.4f}"
            )

    print("\n[Test Complete]")
