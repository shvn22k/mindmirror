"""Aggregate per-frame feature rows into one clip-level feature dict (for ML / API)."""

import numpy as np
import pandas as pd

from features.eye_features import compute_blink_features_for_sequence
from features.gaze_features import compute_gaze_dispersion, compute_fixation_features
from features.mouth_features import compute_yawn_features
from features.head_features import compute_head_movement


def aggregate_clip_features(df: pd.DataFrame, fps: float = 25, sample_every: int = 3) -> dict:
    effective_fps = fps / sample_every

    features = {}

    if "clip_id" in df.columns:
        features["clip_id"] = df["clip_id"].iloc[0]
    features["n_frames"] = len(df)
    features["duration_seconds"] = len(df) / effective_fps

    ear_sequence = df["ear_avg"].values

    features["ear_mean"] = float(np.mean(ear_sequence))
    features["ear_std"] = float(np.std(ear_sequence))
    features["ear_min"] = float(np.min(ear_sequence))
    features["ear_max"] = float(np.max(ear_sequence))

    blink_stats = compute_blink_features_for_sequence(ear_sequence, fps=effective_fps)
    features.update(blink_stats)

    head_positions = df[["head_x", "head_y", "head_z"]].values
    head_movement = compute_head_movement(head_positions)
    features.update(head_movement)

    for col in ["head_pitch", "head_yaw", "head_roll"]:
        if col in df.columns:
            features[f"{col}_mean"] = float(df[col].mean())
            features[f"{col}_std"] = float(df[col].std())

    gaze_sequence = df[["gaze_x", "gaze_y"]].values

    gaze_dispersion = compute_gaze_dispersion(gaze_sequence)
    features.update(gaze_dispersion)

    fixation_stats = compute_fixation_features(gaze_sequence, fps=effective_fps)
    features.update(fixation_stats)

    features["gaze_x_mean"] = float(df["gaze_x"].mean())
    features["gaze_y_mean"] = float(df["gaze_y"].mean())
    features["gaze_x_std"] = float(df["gaze_x"].std())
    features["gaze_y_std"] = float(df["gaze_y"].std())

    mar_sequence = df["mar"].values

    features["mar_mean"] = float(np.mean(mar_sequence))
    features["mar_std"] = float(np.std(mar_sequence))
    features["mar_max"] = float(np.max(mar_sequence))

    yawn_stats = compute_yawn_features(mar_sequence, fps=effective_fps)
    features.update(yawn_stats)

    return features
