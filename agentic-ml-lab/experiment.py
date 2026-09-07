"""
experiment.py
-------------
Dataset loading, fixed train/val/test split, and controlled experiment execution.

The test set is loaded here but is NEVER returned to any caller outside this
module during the agent loop. Only `run_experiment` (train+val) and
`inspect_dataset` (metadata about train/val only) are exposed to the agent.
A separate `evaluate_on_test` function exists purely for the harness to use
once, at the very end, after the agent has finished its budget.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Fixed data split (loaded once, module-level, deterministic)
# ---------------------------------------------------------------------------

_data = load_breast_cancer()
_X, _y = _data.data, _data.target
_feature_names = list(_data.target_names)  # class names actually
_feature_cols = list(_data.feature_names)

# 60% train / 20% val / 20% test, stratified, fixed random_state
_X_train, _X_temp, _y_train, _y_temp = train_test_split(
    _X, _y, test_size=0.4, random_state=RANDOM_STATE, stratify=_y
)
_X_val, _X_test, _y_val, _y_test = train_test_split(
    _X_temp, _y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=_y_temp
)

_scaler = StandardScaler().fit(_X_train)
_X_train_scaled = _scaler.transform(_X_train)
_X_val_scaled = _scaler.transform(_X_val)
_X_test_scaled = _scaler.transform(_X_test)


# ---------------------------------------------------------------------------
# Allowed configuration space (used for validation)
# ---------------------------------------------------------------------------

ALLOWED_CONFIGS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [2, 3, 5],
    },
}


class ConfigValidationError(ValueError):
    pass


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an experiment config against the allowed search space.

    Returns a normalized config (missing hyperparams filled with defaults =
    the first allowed value) or raises ConfigValidationError.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("config must be a JSON object")

    model_type = config.get("model_type")
    if model_type not in ALLOWED_CONFIGS:
        raise ConfigValidationError(
            f"model_type must be one of {list(ALLOWED_CONFIGS.keys())}, got {model_type!r}"
        )

    allowed_params = ALLOWED_CONFIGS[model_type]
    hyperparams = config.get("hyperparams", {}) or {}
    if not isinstance(hyperparams, dict):
        raise ConfigValidationError("hyperparams must be a JSON object")

    normalized: Dict[str, Any] = {}
    for param, choices in allowed_params.items():
        value = hyperparams.get(param, choices[0])
        if value not in choices:
            raise ConfigValidationError(
                f"{model_type}.{param} must be one of {choices}, got {value!r}"
            )
        normalized[param] = value

    unknown = set(hyperparams.keys()) - set(allowed_params.keys())
    if unknown:
        raise ConfigValidationError(
            f"Unknown hyperparams for {model_type}: {sorted(unknown)}"
        )

    return {"model_type": model_type, "hyperparams": normalized}


def _build_model(model_type: str, hyperparams: Dict[str, Any]):
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=hyperparams["C"],
            penalty=hyperparams["penalty"],
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=hyperparams["n_estimators"],
            max_depth=hyperparams["max_depth"],
            random_state=RANDOM_STATE,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=hyperparams["n_estimators"],
            learning_rate=hyperparams["learning_rate"],
            max_depth=hyperparams["max_depth"],
            random_state=RANDOM_STATE,
        )
    raise ConfigValidationError(f"Unsupported model_type: {model_type}")


def inspect_dataset() -> Dict[str, Any]:
    """Return metadata about the dataset and the train/val split ONLY.
    The test set's contents are never surfaced here.
    """
    n_pos = int(_y_train.sum())
    n_neg = int(len(_y_train) - n_pos)
    return {
        "dataset": "sklearn breast_cancer",
        "n_features": _X.shape[1],
        "feature_names": _feature_cols,
        "class_names": _feature_names,
        "train_size": int(len(_y_train)),
        "val_size": int(len(_y_val)),
        "test_size_held_out": "hidden from agent",
        "train_class_balance": {"malignant(0)": n_neg, "benign(1)": n_pos},
        "notes": "Features are standardized (StandardScaler fit on train only).",
    }


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the requested model on the TRAIN split and evaluate on the
    VALIDATION split. The test split is never touched here.
    """
    normalized = validate_config(config)
    model = _build_model(normalized["model_type"], normalized["hyperparams"])
    model.fit(_X_train_scaled, _y_train)
    preds = model.predict(_X_val_scaled)

    metrics = {
        "f1": float(f1_score(_y_val, preds)),
        "accuracy": float(accuracy_score(_y_val, preds)),
        "precision": float(precision_score(_y_val, preds)),
        "recall": float(recall_score(_y_val, preds)),
    }
    return {"config": normalized, "metrics": metrics}


def evaluate_on_test(config: Dict[str, Any]) -> Dict[str, Any]:
    """Harness-only: evaluate a given (already validated) config on the held
    out TEST split. Never called by the agent / tool layer.
    """
    normalized = validate_config(config)
    model = _build_model(normalized["model_type"], normalized["hyperparams"])
    model.fit(_X_train_scaled, _y_train)
    preds = model.predict(_X_test_scaled)
    metrics = {
        "f1": float(f1_score(_y_test, preds)),
        "accuracy": float(accuracy_score(_y_test, preds)),
        "precision": float(precision_score(_y_test, preds)),
        "recall": float(recall_score(_y_test, preds)),
    }
    return {"config": normalized, "metrics": metrics}


def baseline_score() -> Dict[str, Any]:
    """A fixed baseline experiment (default logistic regression) used for
    reporting comparison at the end of the run.
    """
    return run_experiment({"model_type": "logistic_regression", "hyperparams": {"C": 1.0, "penalty": "l2"}})
