#!/usr/bin/env python3
"""Compute per-class accuracies for all models in a structured format.

Outputs a dict like:
{
    "Rule-Based": {"Nitrogen": 90.0, "Phosphorus": 70.0, "Potassium": 80.0, "Overall": 81.2},
    "RandomForest": { ... },
    "SVM": { ... },
    "XGBoost": { ... },
    "EfficientNetB0": { ... }
}

Usage (from backend/):
  python -m src.utils.evaluate_per_class_accuracies
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import joblib
import tensorflow as tf

from .data_preprocessor import DataPreprocessor
from ..classical_ml.feature_extractor import FeatureExtractor
from ..rule_based.color_analyzer import RiceLeafAnalyzer


CLASS_NAMES: List[str] = ["Nitrogen", "Phosphorus", "Potassium"]


def compute_per_class_accuracies(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    overall = float(np.mean(y_true == y_pred) * 100.0)
    metrics["Overall"] = overall
    for idx, name in enumerate(CLASS_NAMES):
        class_mask = (y_true == idx)
        if np.sum(class_mask) == 0:
            metrics[name] = float("nan")
        else:
            acc = float(np.mean(y_pred[class_mask] == idx) * 100.0)
            metrics[name] = acc
    return metrics


def evaluate_rule_based(X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    analyzer = RiceLeafAnalyzer()
    preds: List[int] = []
    for img in X_test:
        pred_name = analyzer.detect_deficiency(img)
        try:
            pred_idx = CLASS_NAMES.index(pred_name)
        except ValueError:
            # Map non-N/P/K (e.g., "Healthy") to the nearest by features? fallback to -1
            pred_idx = -1
        preds.append(pred_idx)
    y_pred = np.array(preds)
    # Remove invalid (-1) predictions from accuracy computation
    valid_mask = (y_pred >= 0)
    return compute_per_class_accuracies(y_test[valid_mask], y_pred[valid_mask])


def load_scaler_if_available(models_dir: str = "models"):
    path = os.path.join(models_dir, "feature_scaler.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def evaluate_classical_ml(X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    scaler = load_scaler_if_available()
    if scaler is not None:
        X_test_ml = scaler.transform(X_test)
    else:
        X_test_ml = X_test

    candidates: List[Tuple[str, str]] = [
        ("RandomForest", "models/ml_model_RandomForest.joblib"),
        ("SVM", "models/ml_model_SVM.joblib"),
        ("XGBoost", "models/ml_model_XGBoost.joblib"),
    ]
    for name, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            y_pred = model.predict(X_test_ml)
            metrics = compute_per_class_accuracies(y_test, y_pred)
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": f"{e}"}
    return results


def evaluate_deep_learning(X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    model_path = "models/best_efficientnetb0_improved.h5"
    if not os.path.exists(model_path):
        model_path = "models/best_efficientnetb0.h5"
    if not os.path.exists(model_path):
        return {"error": "DL model not found"}
    try:
        model = tf.keras.models.load_model(model_path)
        X = X_test.astype("float32") / 255.0
        preds = model.predict(X, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        return compute_per_class_accuracies(y_test, y_pred)
    except Exception as e:
        return {"error": f"{e}"}


def main() -> Dict[str, Dict[str, float]]:
    # Load images as arrays for classical ML features and DL
    prep = DataPreprocessor(base_dir="data/rice_plant_lacks_nutrients", target_size=(224, 224))
    images, labels = prep.load_images()

    # Prepare features for classical ML
    extractor = FeatureExtractor()
    X_features = np.array([extractor.extract_all_features(img) for img in images])
    y = labels

    # Split consistently for both pipelines
    (X_train_f, y_train), (X_val_f, y_val), (X_test_f, y_test) = prep.split_data(X_features, y)
    (X_train_i, _), (X_val_i, _), (X_test_i, _) = prep.split_data(images, y)

    results: Dict[str, Dict[str, float]] = {}

    # Rule-based (on image arrays)
    results["Rule-Based"] = evaluate_rule_based(X_test_i, y_test)

    # Classical ML (on feature arrays)
    results.update(evaluate_classical_ml(X_test_f, y_test))

    # Deep Learning (on image arrays)
    results["EfficientNetB0"] = evaluate_deep_learning(X_test_i, y_test)

    # Print nicely and also return
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()


