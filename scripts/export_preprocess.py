"""
Export preprocess.joblib (scaler + feature_names + class_names) without full training.

Fits StandardScaler on all labeled rows in clip_features.csv. This matches training
only approximately (training fits the scaler on the train split). Prefer running
scripts/train_model.py once so preprocess.joblib is saved from the same run as the models.

Usage (from project root):
    python scripts/export_preprocess.py
    python scripts/export_preprocess.py --output models/trained/preprocess.joblib
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from training.data_prep import load_features, get_feature_columns, create_classification_labels


def main():
    parser = argparse.ArgumentParser(description="Export preprocess artifact for inference API")
    parser.add_argument("--data", type=str, default="data/processed/avcaffe/clip_features.csv")
    parser.add_argument("--output", type=str, default="models/trained/preprocess.joblib")
    args = parser.parse_args()

    df = load_features(args.data)
    feature_cols = get_feature_columns(df)
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0)
    _, class_names = create_classification_labels(df["label"], n_classes=3)

    scaler = StandardScaler()
    scaler.fit(X)

    payload = {
        "feature_names": list(feature_cols),
        "class_names": list(class_names),
        "scaler": scaler,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Wrote {out} ({len(feature_cols)} features, scaler fitted on {len(X)} rows)")


if __name__ == "__main__":
    main()
