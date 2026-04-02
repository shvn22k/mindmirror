"""
Main feature extractor that combines all feature modules.

Extracts per-frame features from landmark arrays.
"""

import numpy as np
import pandas as pd

from .eye_features import compute_eye_features, is_blink_frame
from .head_features import compute_head_features
from .gaze_features import compute_gaze_features
from .mouth_features import compute_mouth_features, is_yawn_frame


class FeatureExtractor:
    """
    Extract behavioral features from facial landmarks.
    
    Usage:
        extractor = FeatureExtractor()
        df = extractor.extract_from_landmarks(landmarks_array, clip_id="aiim001_task_1_clip_001")
    """
    
    def __init__(self, fps=25, sample_every=3, blink_threshold=0.2, yawn_threshold=0.6):
        """
        Initialize feature extractor.
        
        Args:
            fps: Original video FPS (for timestamp calculation)
            sample_every: Frame sampling rate used during landmark extraction
            blink_threshold: EAR threshold for blink detection
            yawn_threshold: MAR threshold for yawn detection
        """
        self.fps = fps
        self.sample_every = sample_every
        self.effective_fps = fps / sample_every
        self.blink_threshold = blink_threshold
        self.yawn_threshold = yawn_threshold
    
    def extract_frame_features(self, landmarks):
        """
        Extract features from a single frame's landmarks.
        
        Args:
            landmarks: Array of shape (478, 3)
            
        Returns:
            Dict with all features for this frame
        """
        features = {}
        
        # Eye features
        eye = compute_eye_features(landmarks)
        features.update(eye)
        features["is_blink"] = is_blink_frame(eye["ear_avg"], self.blink_threshold)
        
        # Head features
        head = compute_head_features(landmarks)
        features.update(head)
        
        # Gaze features
        gaze = compute_gaze_features(landmarks)
        features.update(gaze)
        
        # Mouth features
        mouth = compute_mouth_features(landmarks)
        features.update(mouth)
        features["is_yawn"] = is_yawn_frame(mouth["mar"], self.yawn_threshold)
        
        return features
    
    def extract_from_landmarks(self, landmarks_array, clip_id=None):
        """
        Extract per-frame features from a landmarks array.
        
        Args:
            landmarks_array: Array of shape (n_frames, 478, 3)
            clip_id: Optional clip identifier
            
        Returns:
            DataFrame with one row per frame
        """
        n_frames = landmarks_array.shape[0]
        
        rows = []
        for i in range(n_frames):
            landmarks = landmarks_array[i]
            features = self.extract_frame_features(landmarks)
            
            # Add metadata
            features["frame_idx"] = i
            features["timestamp_ms"] = (i * self.sample_every * 1000) / self.fps
            
            if clip_id:
                features["clip_id"] = clip_id
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        cols = ["clip_id", "frame_idx", "timestamp_ms"] if clip_id else ["frame_idx", "timestamp_ms"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        
        return df
    
    def extract_from_file(self, npy_path, clip_id=None):
        """
        Extract features from a .npy landmarks file.
        
        Args:
            npy_path: Path to .npy file with landmarks
            clip_id: Optional clip identifier (defaults to filename)
            
        Returns:
            DataFrame with per-frame features
        """
        import os
        
        landmarks = np.load(npy_path)
        
        if clip_id is None:
            clip_id = os.path.splitext(os.path.basename(npy_path))[0]
        
        return self.extract_from_landmarks(landmarks, clip_id)


def extract_features_from_file(npy_path, output_path=None, **kwargs):
    """
    Convenience function to extract features from a .npy file.
    
    Args:
        npy_path: Path to input .npy landmarks file
        output_path: Optional path to save Parquet output
        **kwargs: Arguments passed to FeatureExtractor
        
    Returns:
        DataFrame with features (also saved to output_path if provided)
    """
    import os
    
    extractor = FeatureExtractor(**kwargs)
    df = extractor.extract_from_file(npy_path)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
    
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        npy_path = sys.argv[1]
    else:
        npy_path = "data/processed/avcaffe/landmarks/aiim001/task_1/aiim001_task_1_clip_001.npy"
    
    print(f"Testing feature extraction on: {npy_path}")
    
    extractor = FeatureExtractor()
    df = extractor.extract_from_file(npy_path)
    
    print(f"\nExtracted {len(df)} frames")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nFeature statistics:")
    print(df.describe())
