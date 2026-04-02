"""
Train cognitive load models (regression + classification + ensembles).

Usage (from project root):
    python scripts/train_model.py --skip-tabnet
    python scripts/train_model.py --output-dir models/trained
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

from training.data_prep import load_features, prepare_data, save_preprocess_artifact
from training.models import get_model
from training.evaluate import (
    evaluate_regression,
    evaluate_classification,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_predictions_vs_actual,
    generate_report,
    create_comparison_table,
)
from training.ensemble import WeightedAverageEnsemble, optimize_ensemble_weights


def train_single_model(model_name, task, data, verbose=True):
    if verbose:
        print(f"\n  Training {model_name} ({task})...", end=" ", flush=True)

    start = time.time()

    if task == "regression":
        y_train, y_val, y_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
    else:
        y_train, y_val, y_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]

    model = get_model(model_name, task=task)
    model.fit(data["X_train"], y_train, data["X_val"], y_val)

    y_pred = model.predict(data["X_test"])

    if task == "regression":
        metrics = evaluate_regression(y_test, y_pred)
    else:
        metrics = evaluate_classification(y_test, y_pred, data["class_names"])

    elapsed = time.time() - start
    metrics["train_time"] = elapsed

    if verbose:
        if task == "regression":
            print(f"MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f} ({elapsed:.1f}s)")
        else:
            print(f"Acc={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f} ({elapsed:.1f}s)")

    return model, metrics


def train_all_models(data, model_names=None, tasks=None, skip_tabnet=False, verbose=True):
    if model_names is None:
        model_names = ["xgboost", "lightgbm", "catboost"]
        if not skip_tabnet:
            model_names.append("tabnet")

    if tasks is None:
        tasks = ["regression", "classification"]

    results = {
        "regression": {},
        "classification": {},
        "models": {
            "regression": {},
            "classification": {},
        },
    }

    for task in tasks:
        print(f"\n{'=' * 50}")
        print(f"Training {task.upper()} models")
        print(f"{'=' * 50}")

        for model_name in model_names:
            try:
                model, metrics = train_single_model(model_name, task, data, verbose)
                results[task][model_name] = metrics
                results["models"][task][model_name] = model
            except Exception as e:
                print(f"\n  Error training {model_name}: {e}")

    return results


def create_ensembles(results, data, verbose=True):
    for task in ["regression", "classification"]:
        if task not in results["models"] or not results["models"][task]:
            continue

        base_models = {k: v for k, v in results["models"][task].items() if k != "ensemble"}

        if len(base_models) < 2:
            continue

        if verbose:
            print(f"\n  Creating {task} ensemble...", end=" ", flush=True)

        if task == "regression":
            y_val, y_test = data["y_reg_val"], data["y_reg_test"]
        else:
            y_val, y_test = data["y_cls_val"], data["y_cls_test"]

        weights = optimize_ensemble_weights(base_models, data["X_val"], y_val, task)

        ensemble = WeightedAverageEnsemble(base_models, weights, task)
        y_pred = ensemble.predict(data["X_test"])

        if task == "regression":
            metrics = evaluate_regression(y_test, y_pred)
        else:
            metrics = evaluate_classification(y_test, y_pred, data["class_names"])

        results[task]["ensemble"] = metrics
        results["models"][task]["ensemble"] = ensemble

        if verbose:
            if task == "regression":
                print(f"MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
            else:
                print(f"Acc={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")
            print(f"    Weights: {weights}")

    return results


def save_models(results, output_dir="models/trained"):
    os.makedirs(output_dir, exist_ok=True)

    for task in ["regression", "classification"]:
        if task not in results["models"]:
            continue

        for name, model in results["models"][task].items():
            if name == "ensemble":
                path = os.path.join(output_dir, f"ensemble_{task}.joblib")
                model.save(path)
            elif name == "catboost":
                path = os.path.join(output_dir, f"{name}_{task}.cbm")
                model.save(path)
            elif name == "tabnet":
                path = os.path.join(output_dir, f"{name}_{task}")
                model.save(path)
            else:
                path = os.path.join(output_dir, f"{name}_{task}.joblib")
                model.save(path)

    print(f"\nModels saved to: {output_dir}/")


def save_results(results, data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    best = {"regression": None, "classification": None}

    for task in ["regression", "classification"]:
        if task not in results or not results[task]:
            continue

        if task == "regression":
            best_name = min(results[task], key=lambda x: results[task][x]["mae"])
        else:
            best_name = max(results[task], key=lambda x: results[task][x]["f1_macro"])

        best[task] = best_name

    results["best_model"] = best

    comparison_df = create_comparison_table(results)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    generate_report(results, os.path.join(output_dir, "training_report.md"))

    if "classification" in results["models"]:
        for name, model in results["models"]["classification"].items():
            y_pred = model.predict(data["X_test"])
            plot_confusion_matrix(
                data["y_cls_test"],
                y_pred,
                data["class_names"],
                title=f"Confusion Matrix - {name}",
                save_path=os.path.join(output_dir, f"confusion_{name}.png"),
            )

    if best["regression"] and best["regression"] != "ensemble":
        model = results["models"]["regression"][best["regression"]]
        try:
            importance = model.feature_importance()
            plot_feature_importance(
                importance,
                data["feature_names"],
                title=f"Feature Importance - {best['regression']}",
                save_path=os.path.join(output_dir, "feature_importance.png"),
            )
        except Exception:
            pass

    if best["regression"]:
        model = results["models"]["regression"][best["regression"]]
        y_pred = model.predict(data["X_test"])
        plot_predictions_vs_actual(
            data["y_reg_test"],
            y_pred,
            title=f"Predictions vs Actual - {best['regression']}",
            save_path=os.path.join(output_dir, "predictions_vs_actual.png"),
        )

    print(f"Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train cognitive load estimation models")
    parser.add_argument("--data", type=str, default="data/processed/avcaffe/clip_features.csv")
    parser.add_argument("--models", type=str, nargs="+", help="Models to train")
    parser.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    parser.add_argument("--skip-tabnet", action="store_true", help="Skip TabNet (faster)")
    parser.add_argument("--output-dir", type=str, default="models/trained")
    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    print("=" * 60)
    print("COGNITIVE LOAD MODEL TRAINING")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_features(args.data)
    data = prepare_data(df, scale_features=True)

    preprocess_path = os.path.join(args.output_dir, "preprocess.joblib")
    save_preprocess_artifact(data, preprocess_path)
    print(f"Preprocess artifact saved: {preprocess_path}")

    tasks = ["regression", "classification"] if args.task == "both" else [args.task]

    print("\n[2/4] Training models...")
    results = train_all_models(
        data,
        model_names=args.models,
        tasks=tasks,
        skip_tabnet=args.skip_tabnet,
    )

    print("\n[3/4] Creating ensembles...")
    results = create_ensembles(results, data)

    print("\n[4/4] Saving results...")
    save_models(results, args.output_dir)
    save_results(results, data, args.results_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nRegression Results (Test Set):")
    print("-" * 40)
    for name, metrics in sorted(results.get("regression", {}).items(), key=lambda x: x[1]["mae"]):
        print(f"  {name:15} MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.3f}")

    print("\nClassification Results (Test Set):")
    print("-" * 40)
    for name, metrics in sorted(results.get("classification", {}).items(), key=lambda x: -x[1]["f1_macro"]):
        print(f"  {name:15} Acc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}")

    print(f"\nBest Regression Model: {results['best_model']['regression']}")
    print(f"Best Classification Model: {results['best_model']['classification']}")


if __name__ == "__main__":
    main()
