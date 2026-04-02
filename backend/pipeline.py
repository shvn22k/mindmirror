"""
Video → landmarks → per-frame features → clip aggregate → model predictions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _resolve_models_dir(models_dir: str | None) -> Path:
    raw = models_dir or os.environ.get("MODELS_DIR", "models/trained")
    p = Path(raw)
    return p if p.is_absolute() else _ROOT / p


class CognitiveLoadPredictor:
    """
    Loads preprocess.joblib plus regression/classification model files (joblib xgboost by default).
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        regression_filename: str | None = None,
        classification_filename: str | None = None,
        fps: float = 25.0,
        sample_every: int = 3,
    ):
        self.models_dir = _resolve_models_dir(str(models_dir) if models_dir else None)
        self.fps = fps
        self.sample_every = sample_every

        preprocess_path = self.models_dir / "preprocess.joblib"
        if not preprocess_path.exists():
            raise FileNotFoundError(
                f"Missing {preprocess_path}. Run training (scripts/train_model.py) or "
                f"scripts/export_preprocess.py to create preprocess.joblib."
            )

        blob = joblib.load(preprocess_path)
        self.feature_names: list[str] = list(blob["feature_names"])
        self.class_names: list[str] = list(blob["class_names"])
        self.scaler = blob.get("scaler")

        reg_name = regression_filename or os.environ.get(
            "REGRESSION_MODEL", "xgboost_regression.joblib"
        )
        cls_name = classification_filename or os.environ.get(
            "CLASSIFICATION_MODEL", "xgboost_classification.joblib"
        )

        reg_path = self.models_dir / reg_name
        cls_path = self.models_dir / cls_name
        if not reg_path.exists():
            raise FileNotFoundError(f"Regression model not found: {reg_path}")
        if not cls_path.exists():
            raise FileNotFoundError(f"Classification model not found: {cls_path}")

        self._reg = joblib.load(reg_path)
        self._cls = joblib.load(cls_path)

        from cognitive_load.landmark_extractor import LandmarkExtractor
        from features.feature_extractor import FeatureExtractor
        from features.clip_aggregate import aggregate_clip_features

        self._landmarks = LandmarkExtractor()
        self._feature_extractor = FeatureExtractor(
            fps=self.fps, sample_every=self.sample_every
        )
        self._aggregate_clip_features = aggregate_clip_features

    @classmethod
    def from_env(cls) -> CognitiveLoadPredictor:
        fps = float(os.environ.get("INFERENCE_FPS", "25"))
        sample_every = int(os.environ.get("INFERENCE_SAMPLE_EVERY", "3"))
        return cls(fps=fps, sample_every=sample_every)

    def _feats_dict_to_matrix(self, feats: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        row: list[float] = []
        missing: list[str] = []
        for name in self.feature_names:
            if name not in feats:
                missing.append(name)
                row.append(0.0)
                continue
            v = feats[name]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append(0.0)
            else:
                row.append(float(v))
        X = np.asarray(row, dtype=np.float64).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        return X, missing

    def predict_video_path(self, video_path: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(str(video_path))

        landmarks = self._landmarks.extract_from_video(
            str(video_path), sample_every=self.sample_every
        )
        return self.predict_from_landmarks(landmarks)

    def predict_from_landmarks(self, landmarks: np.ndarray) -> dict[str, Any]:
        if landmarks is None or landmarks.size == 0:
            raise ValueError("No face landmarks extracted (empty video or no detections).")

        df = self._feature_extractor.extract_from_landmarks(
            landmarks, clip_id="inference"
        )
        feats = self._aggregate_clip_features(
            df, fps=self.fps, sample_every=self.sample_every
        )

        X, missing_features = self._feats_dict_to_matrix(feats)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        reg_score = float(self._reg.predict(X)[0])
        cls_idx = int(self._cls.predict(X)[0])
        label = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else str(cls_idx)

        probs: dict[str, float] = {}
        if hasattr(self._cls, "predict_proba"):
            proba = self._cls.predict_proba(X)[0]
            for i, name in enumerate(self.class_names):
                if i < len(proba):
                    probs[name] = float(proba[i])

        out: dict[str, Any] = {
            "regression_score": reg_score,
            "classification_label": label,
            "classification_class_index": cls_idx,
            "probabilities": probs,
            "n_frames_used": int(landmarks.shape[0]),
            "warnings": [],
        }
        if missing_features:
            out["warnings"].append(
                f"Missing feature keys (filled with 0): {missing_features[:10]}"
                + ("..." if len(missing_features) > 10 else "")
            )
        return out


def predict_uploaded_video(
    file_bytes: bytes,
    suffix: str = ".mp4",
    predictor: CognitiveLoadPredictor | None = None,
) -> dict[str, Any]:
    pred = predictor or CognitiveLoadPredictor.from_env()
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(path).write_bytes(file_bytes)
        return pred.predict_video_path(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
