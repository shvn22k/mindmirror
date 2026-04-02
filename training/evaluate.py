"""
Evaluation utilities for model training.

Includes metrics, visualization, and reporting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import os


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict with metrics
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred, class_names=None):
    """
    Evaluate classification predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dict with metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }
    
    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None)
    if class_names:
        for i, name in enumerate(class_names):
            metrics[f"f1_{name}"] = f1_per_class[i]
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label"
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
    return cm


def plot_feature_importance(importance, feature_names, top_n=20, title="Feature Importance", save_path=None):
    """
    Plot feature importance.
    
    Args:
        importance: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance[indices])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Scatter plot of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()


def generate_report(results, save_path="results/training_report.md"):
    """
    Generate markdown report of training results.
    
    Args:
        results: Dict with model results
        save_path: Path to save report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    lines = [
        "# Model Training Report",
        "",
        "## Regression Results",
        "",
        "| Model | MAE | RMSE | R² |",
        "|-------|-----|------|-----|",
    ]
    
    for name, res in results.get("regression", {}).items():
        lines.append(
            f"| {name} | {res['mae']:.3f} | {res['rmse']:.3f} | {res['r2']:.3f} |"
        )
    
    lines.extend([
        "",
        "## Classification Results",
        "",
        "| Model | Accuracy | F1 (Macro) | F1 (Weighted) |",
        "|-------|----------|------------|---------------|",
    ])
    
    for name, res in results.get("classification", {}).items():
        lines.append(
            f"| {name} | {res['accuracy']:.3f} | {res['f1_macro']:.3f} | {res['f1_weighted']:.3f} |"
        )
    
    if "best_model" in results:
        lines.extend([
            "",
            "## Best Model",
            "",
            f"**Regression:** {results['best_model'].get('regression', 'N/A')}",
            "",
            f"**Classification:** {results['best_model'].get('classification', 'N/A')}",
        ])
    
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {save_path}")


def create_comparison_table(results):
    """
    Create comparison DataFrame from results.
    
    Args:
        results: Dict with model results
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for task in ["regression", "classification"]:
        if task not in results:
            continue
        for model_name, metrics in results[task].items():
            row = {"task": task, "model": model_name}
            row.update(metrics)
            rows.append(row)
    
    return pd.DataFrame(rows)
