"""
Model definitions and wrappers for training.

Provides unified interface for:
- XGBoost
- LightGBM
- CatBoost
- TabNet
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, task="regression", **kwargs):
        """
        Initialize model.
        
        Args:
            task: "regression" or "classification"
            **kwargs: Model-specific parameters
        """
        self.task = task
        self.model = None
        self.params = kwargs
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        raise NotImplementedError("predict_proba not implemented for this model")
    
    @abstractmethod
    def feature_importance(self):
        """Get feature importance scores."""
        pass
    
    def save(self, path):
        """Save model to file."""
        pass
    
    def load(self, path):
        """Load model from file."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, task="regression", **kwargs):
        super().__init__(task, **kwargs)
        
        import xgboost as xgb
        
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = xgb.XGBRegressor(**default_params)
        else:
            self.model = xgb.XGBClassifier(**default_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only for classification")
    
    def feature_importance(self):
        return self.model.feature_importances_
    
    def save(self, path):
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path):
        import joblib
        self.model = joblib.load(path)


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, task="regression", **kwargs):
        super().__init__(task, **kwargs)
        
        import lightgbm as lgb
        
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = lgb.LGBMRegressor(**default_params)
        else:
            self.model = lgb.LGBMClassifier(**default_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        callbacks = []
        if eval_set:
            import lightgbm as lgb
            callbacks = [lgb.early_stopping(50, verbose=False)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else None
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only for classification")
    
    def feature_importance(self):
        return self.model.feature_importances_
    
    def save(self, path):
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path):
        import joblib
        self.model = joblib.load(path)


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(self, task="regression", **kwargs):
        super().__init__(task, **kwargs)

        import os
        from catboost import CatBoostRegressor, CatBoostClassifier

        train_dir = kwargs.pop("train_dir", "artifacts/catboost")
        os.makedirs(train_dir, exist_ok=True)

        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "random_state": 42,
            "verbose": False,
            "train_dir": train_dir,
        }
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = CatBoostRegressor(**default_params)
        else:
            self.model = CatBoostClassifier(**default_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = (X_val, y_val) if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=False
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only for classification")
    
    def feature_importance(self):
        return self.model.feature_importances_
    
    def save(self, path):
        self.model.save_model(path)
    
    def load(self, path):
        self.model.load_model(path)


class TabNetModel(BaseModel):
    """TabNet model wrapper."""
    
    def __init__(self, task="regression", **kwargs):
        super().__init__(task, **kwargs)
        
        self.default_params = {
            "n_d": 16,
            "n_a": 16,
            "n_steps": 3,
            "gamma": 1.3,
            "lambda_sparse": 1e-3,
            "optimizer_params": {"lr": 2e-2},
            "scheduler_params": {"step_size": 10, "gamma": 0.9},
            "scheduler_fn": None,
            "mask_type": "sparsemax",
            "verbose": 0,
        }
        self.default_params.update(kwargs)
        self._feature_importances = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.task == "regression":
            from pytorch_tabnet.tab_model import TabNetRegressor
            self.model = TabNetRegressor(**self.default_params)
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1) if y_val is not None else None
        else:
            from pytorch_tabnet.tab_model import TabNetClassifier
            self.model = TabNetClassifier(**self.default_params)
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        eval_name = ["val"] if eval_set else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
        )
        
        self._feature_importances = self.model.feature_importances_
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        if self.task == "regression":
            return preds.flatten()
        return preds
    
    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only for classification")
    
    def feature_importance(self):
        return self._feature_importances
    
    def save(self, path):
        self.model.save_model(path)
    
    def load(self, path):
        if self.task == "regression":
            from pytorch_tabnet.tab_model import TabNetRegressor
            self.model = TabNetRegressor()
        else:
            from pytorch_tabnet.tab_model import TabNetClassifier
            self.model = TabNetClassifier()
        self.model.load_model(path)


def get_model(name, task="regression", **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        name: Model name ("xgboost", "lightgbm", "catboost", "tabnet")
        task: "regression" or "classification"
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "tabnet": TabNetModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name](task=task, **kwargs)
