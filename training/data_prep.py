"""
Data preparation for model training.

Handles:
- Loading clip_features.csv
- Creating classification labels from regression targets
- Train/validation/test splits
- Feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def load_features(path="data/processed/avcaffe/clip_features.csv"):
    """
    Load feature dataset.
    
    Args:
        path: Path to clip_features.csv
        
    Returns:
        DataFrame with features and labels
    """
    df = pd.read_csv(path)
    
    # Drop rows without labels
    df = df.dropna(subset=["label"])
    
    print(f"Loaded {len(df)} samples with labels")
    print(f"Label range: {df['label'].min():.1f} - {df['label'].max():.1f}")
    
    return df


def create_classification_labels(labels, n_classes=3):
    """
    Convert continuous labels to classification labels.
    
    For 3 classes (LOW/MEDIUM/HIGH):
        - LOW: 0-7
        - MEDIUM: 8-14
        - HIGH: 15-21
    
    Args:
        labels: Array of continuous labels (0-21 scale)
        n_classes: Number of classes (3 or 5)
        
    Returns:
        Array of class labels (0, 1, 2, ...)
    """
    labels = np.array(labels)
    
    if n_classes == 3:
        # LOW: 0-7, MEDIUM: 8-14, HIGH: 15-21
        class_labels = np.zeros(len(labels), dtype=int)
        class_labels[labels > 7] = 1   # MEDIUM
        class_labels[labels > 14] = 2  # HIGH
        class_names = ["LOW", "MEDIUM", "HIGH"]
    elif n_classes == 5:
        # Very Low: 0-4, Low: 5-8, Medium: 9-12, High: 13-17, Very High: 18-21
        class_labels = np.zeros(len(labels), dtype=int)
        class_labels[labels > 4] = 1
        class_labels[labels > 8] = 2
        class_labels[labels > 12] = 3
        class_labels[labels > 17] = 4
        class_names = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    else:
        raise ValueError(f"Unsupported n_classes: {n_classes}")
    
    return class_labels, class_names


def get_feature_columns(df):
    """
    Get list of feature columns (exclude metadata and labels).
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of feature column names
    """
    exclude = ["clip_id", "label", "label_class", "n_frames", "duration_seconds"]
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def prepare_data(
    df,
    task="both",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    scale_features=False
):
    """
    Prepare data for training.
    
    Args:
        df: DataFrame with features and labels
        task: "regression", "classification", or "both"
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        scale_features: Whether to standardize features
        
    Returns:
        Dict with train/val/test splits and metadata
    """
    # Get feature columns
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Regression target
    y_reg = df["label"].values
    
    # Classification target
    y_cls, class_names = create_classification_labels(df["label"], n_classes=3)
    
    # First split: train+val vs test
    X_trainval, X_test, y_reg_trainval, y_reg_test, y_cls_trainval, y_cls_test = train_test_split(
        X, y_reg, y_cls,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cls
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X_trainval, y_reg_trainval, y_cls_trainval,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_cls_trainval
    )
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_val": y_reg_val,
        "y_reg_test": y_reg_test,
        "y_cls_train": y_cls_train,
        "y_cls_val": y_cls_val,
        "y_cls_test": y_cls_test,
        "feature_names": feature_cols,
        "class_names": class_names,
        "scaler": scaler,
        "n_features": len(feature_cols),
        "n_classes": len(class_names),
    }
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {len(feature_cols)}")
    
    print(f"\nClass distribution (train):")
    for i, name in enumerate(class_names):
        count = np.sum(y_cls_train == i)
        pct = count / len(y_cls_train) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    return data


def save_preprocess_artifact(data, path):
    """
    Persist scaler, feature column order, and class names for inference (must match training).

    Args:
        data: Dict returned by prepare_data(..., scale_features=True)
        path: Output path (e.g. models/trained/preprocess.joblib)
    """
    import joblib

    payload = {
        "feature_names": list(data["feature_names"]),
        "class_names": list(data["class_names"]),
        "scaler": data["scaler"],
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(payload, path)


if __name__ == "__main__":
    df = load_features()
    data = prepare_data(df)
