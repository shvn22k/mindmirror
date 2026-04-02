"""
Ensemble methods for combining multiple models.

Includes:
- Stacking ensemble
- Weighted average ensemble
"""

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
import joblib


class StackingEnsemble:
    """
    Stacking ensemble that uses base model predictions as features.
    
    For regression: Ridge regression as meta-learner
    For classification: Logistic regression as meta-learner
    """
    
    def __init__(self, base_models, task="regression"):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: Dict of {name: model} for base models
            task: "regression" or "classification"
        """
        self.base_models = base_models
        self.task = task
        self.meta_model = None
    
    def _get_meta_features(self, X, fit=False):
        """Get predictions from base models as meta-features."""
        meta_features = []
        
        for name, model in self.base_models.items():
            if self.task == "classification":
                # Use probabilities for classification
                try:
                    proba = model.predict_proba(X)
                    meta_features.append(proba)
                except:
                    pred = model.predict(X)
                    meta_features.append(pred.reshape(-1, 1))
            else:
                pred = model.predict(X)
                meta_features.append(pred.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def fit(self, X, y):
        """
        Fit the meta-learner on base model predictions.
        
        Note: Base models should already be trained!
        
        Args:
            X: Validation features (used to generate meta-features)
            y: Validation labels
        """
        meta_X = self._get_meta_features(X)
        
        if self.task == "regression":
            self.meta_model = Ridge(alpha=1.0)
        else:
            self.meta_model = LogisticRegression(max_iter=1000, multi_class="multinomial")
        
        self.meta_model.fit(meta_X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble."""
        meta_X = self._get_meta_features(X)
        return self.meta_model.predict(meta_X)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        meta_X = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_X)
    
    def save(self, path):
        """Save ensemble to file."""
        joblib.dump({
            "meta_model": self.meta_model,
            "task": self.task,
        }, path)
    
    def load(self, path):
        """Load ensemble from file."""
        data = joblib.load(path)
        self.meta_model = data["meta_model"]
        self.task = data["task"]


class WeightedAverageEnsemble:
    """
    Simple weighted average ensemble.
    
    For regression: Weighted average of predictions
    For classification: Weighted average of probabilities, then argmax
    """
    
    def __init__(self, base_models, weights=None, task="regression"):
        """
        Initialize weighted ensemble.
        
        Args:
            base_models: Dict of {name: model} for base models
            weights: Dict of {name: weight} or None for equal weights
            task: "regression" or "classification"
        """
        self.base_models = base_models
        self.task = task
        
        if weights is None:
            n = len(base_models)
            self.weights = {name: 1.0/n for name in base_models}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {name: w/total for name, w in weights.items()}
    
    def predict(self, X):
        """Make predictions using weighted average."""
        if self.task == "regression":
            pred = np.zeros(len(X))
            for name, model in self.base_models.items():
                pred += self.weights[name] * model.predict(X)
            return pred
        else:
            # For classification, average probabilities then take argmax
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        
        proba = None
        for name, model in self.base_models.items():
            p = model.predict_proba(X)
            if proba is None:
                proba = self.weights[name] * p
            else:
                proba += self.weights[name] * p
        
        return proba
    
    def save(self, path):
        """Save weights to file."""
        joblib.dump({
            "weights": self.weights,
            "task": self.task,
        }, path)


def optimize_ensemble_weights(base_models, X_val, y_val, task="regression"):
    """
    Find optimal weights for weighted ensemble using validation set.
    
    Uses simple grid search over weight combinations.
    
    Args:
        base_models: Dict of trained models
        X_val: Validation features
        y_val: Validation labels
        task: "regression" or "classification"
        
    Returns:
        Dict of optimal weights
    """
    from sklearn.metrics import mean_absolute_error, f1_score
    
    model_names = list(base_models.keys())
    n_models = len(model_names)
    
    # Get base predictions
    predictions = {}
    for name, model in base_models.items():
        if task == "regression":
            predictions[name] = model.predict(X_val)
        else:
            try:
                predictions[name] = model.predict_proba(X_val)
            except:
                pred = model.predict(X_val)
                predictions[name] = np.eye(len(np.unique(y_val)))[pred]
    
    # Grid search over weights (simplified)
    best_score = float("inf") if task == "regression" else float("-inf")
    best_weights = None
    
    # Generate weight combinations (sum to 1)
    steps = 5
    from itertools import product
    
    for combo in product(range(steps + 1), repeat=n_models):
        if sum(combo) == 0:
            continue
        
        weights = {name: c / sum(combo) for name, c in zip(model_names, combo)}
        
        # Compute weighted prediction
        if task == "regression":
            pred = sum(weights[name] * predictions[name] for name in model_names)
            score = mean_absolute_error(y_val, pred)
            if score < best_score:
                best_score = score
                best_weights = weights
        else:
            proba = sum(weights[name] * predictions[name] for name in model_names)
            pred = np.argmax(proba, axis=1)
            score = f1_score(y_val, pred, average="macro")
            if score > best_score:
                best_score = score
                best_weights = weights
    
    return best_weights
