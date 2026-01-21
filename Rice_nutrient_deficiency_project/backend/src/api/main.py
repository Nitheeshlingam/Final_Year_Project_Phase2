#!/usr/bin/env python3
"""FastAPI server for Rice Nutrient Deficiency Detection predictions."""

import io
import os
import sys
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf
import joblib
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

# Import our custom modules
from src.classical_ml.feature_extractor import FeatureExtractor
from src.rule_based.color_analyzer import RiceLeafAnalyzer
from src.ensemble.ensemble_predictor import EnsemblePredictor

CLASSES = ['Nitrogen', 'Phosphorus', 'Potassium']
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'best_efficientnetb0.h5')

app = FastAPI(title="Rice Nutrient Deficiency Detection API")

# CORS for local dev (React Vite default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
MODEL = None
ML_MODELS = {}
RULE_ANALYZER = None
FEATURE_EXTRACTOR = None
ENSEMBLE_PREDICTOR = None


def _resize_to_model(img_array: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    try:
        expected_h, expected_w = model.input_shape[1], model.input_shape[2]
    except Exception:
        expected_h, expected_w = 224, 224
    if img_array.shape[:2] != (expected_h, expected_w):
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((expected_w, expected_h))
        return np.array(img_pil)
    return img_array


def load_ml_models():
    """Load all available classical ML models."""
    global ML_MODELS
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = [
        ("RandomForest", os.path.join(base_path, 'models', 'ml_model_RandomForest.joblib')),
        ("SVM", os.path.join(base_path, 'models', 'ml_model_SVM.joblib')),
        ("XGBoost", os.path.join(base_path, 'models', 'ml_model_XGBoost.joblib')),
    ]
    
    for name, path in candidates:
        if os.path.exists(path):
            try:
                ML_MODELS[name] = joblib.load(path)
                print(f"✅ Loaded ML model: {name}")
            except Exception as e:
                print(f"⚠️ Failed to load {name}: {e}")

@app.on_event("startup")
def load_models_on_startup():
    global MODEL, RULE_ANALYZER, FEATURE_EXTRACTOR, ENSEMBLE_PREDICTOR
    
    # Initialize Rule-based analyzer and Feature extractor
    try:
        RULE_ANALYZER = RiceLeafAnalyzer()
        FEATURE_EXTRACTOR = FeatureExtractor()
        print("✅ Initialized Rule-based analyzer and Feature extractor")
    except Exception as e:
        print(f"⚠️ Failed to initialize analyzers: {e}")
        RULE_ANALYZER = None
        FEATURE_EXTRACTOR = None
    
    # Initialize Ensemble Predictor
    try:
        ENSEMBLE_PREDICTOR = EnsemblePredictor()
        ENSEMBLE_PREDICTOR.load_models()
        print("✅ Initialized Ensemble Predictor")
    except Exception as e:
        print(f"⚠️ Failed to initialize Ensemble Predictor: {e}")
        ENSEMBLE_PREDICTOR = None
    
    # Load Deep Learning model (fallback for individual predictions)
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Loaded Deep Learning model: EfficientNetB0")
        except Exception as e:
            print(f"⚠️ Failed to load Deep Learning model: {e}")
            MODEL = None
    else:
        print("⚠️ Deep Learning model not found")
        MODEL = None
    # Load Classical ML models (fallback for individual predictions)
    load_ml_models()


@app.get("/health")
def health():
    return {
        "status": "ok", 
        "models_loaded": {
            "deep_learning": MODEL is not None,
            "classical_ml": len(ML_MODELS) > 0,
            "rule_based": RULE_ANALYZER is not None
        }
    }

@app.get("/training-accuracies")
def get_training_accuracies():
    """Return static training accuracy data for all models."""
    return {
        "Rule-Based": {
            "Nitrogen": 100.0,
            "Phosphorus": 60.0,
            "Potassium": 50.0,
            "Overall": 71.91
        },
        "Random Forest": {
            "Nitrogen": 88.0,
            "Phosphorus": 85.0,
            "Potassium": 90.0,
            "Overall": 88.20
        },
        "SVM": {
            "Nitrogen": 86.0,
            "Phosphorus": 80.0,
            "Potassium": 92.0,
            "Overall": 86.00
        },
        "XGBoost": {
            "Nitrogen": 90.0,
            "Phosphorus": 87.0,
            "Potassium": 93.0,
            "Overall": 89.50
        },
        "EfficientNetB0": {
            "Nitrogen": 96.0,
            "Phosphorus": 92.0,
            "Potassium": 95.0,
            "Overall": 94.00
        }
    }


@app.post("/predict-ensemble")
async def predict_ensemble(file: UploadFile = File(...)):
    """Predict using ensemble of all models for maximum accuracy."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        img_array = np.array(image)
        
        # Rule-based prediction
        rule_pred = None
        rule_features = None
        if RULE_ANALYZER is not None:
            try:
                rule_pred = RULE_ANALYZER.detect_deficiency(img_array)
                rule_features = RULE_ANALYZER.analyze_color_features(img_array)
            except Exception as e:
                print(f"Rule-based prediction error: {e}")
        
        # Ensemble prediction
        ensemble_result = None
        if ENSEMBLE_PREDICTOR is not None:
            try:
                ensemble_result = ENSEMBLE_PREDICTOR.predict_ensemble(
                    img_array, rule_pred, rule_features
                )
            except Exception as e:
                print(f"Ensemble prediction error: {e}")
        
        # Fallback to individual predictions if ensemble fails
        if ensemble_result is None:
            return await predict_all_models(file)
        
        return {
            "ensemble_prediction": ensemble_result['ensemble_prediction'],
            "ensemble_confidence": ensemble_result['ensemble_confidence'],
            "ensemble_probabilities": ensemble_result['ensemble_probabilities'],
            "model_agreement": ensemble_result['model_agreement'],
            "individual_predictions": ensemble_result['individual_predictions'],
            "individual_confidences": ensemble_result['individual_confidences'],
            "individual_probabilities": ensemble_result['individual_probabilities'],
            "image_info": {
                "shape": img_array.shape,
                "filename": file.filename
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ensemble prediction failed: {e}")


@app.post("/predict-all")
async def predict_all_models(file: UploadFile = File(...)):
    """Predict using all available models and return comprehensive results."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        img_array = np.array(image)
        
        results = {
            "rule_based": None,
            "classical_ml": {},
            "deep_learning": None,
            "image_info": {
                "shape": img_array.shape,
                "filename": file.filename
            }
        }
        
        # Rule-Based Prediction
        if RULE_ANALYZER is not None:
            try:
                prediction = RULE_ANALYZER.detect_deficiency(img_array)
                features = RULE_ANALYZER.analyze_color_features(img_array)
                results["rule_based"] = {
                    "prediction": prediction,
                    "features": features,
                    "confidence": "N/A"  # Rule-based doesn't have traditional confidence
                }
            except Exception as e:
                results["rule_based"] = {"error": str(e)}
        
        # Classical ML Predictions
        if FEATURE_EXTRACTOR is not None and ML_MODELS:
            try:
                features = FEATURE_EXTRACTOR.extract_all_features(img_array)
                features_array = np.array(features).reshape(1, -1)
                
                for name, model in ML_MODELS.items():
                    try:
                        idx = model.predict(features_array)[0]
                        pred = CLASSES[idx]
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(features_array)[0]
                            confidence = float(np.max(probs))
                        else:
                            probs = None
                            confidence = "N/A"
                        
                        results["classical_ml"][name] = {
                            "prediction": pred,
                            "confidence": confidence,
                            "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs.tolist())} if probs is not None else None
                        }
                    except Exception as e:
                        results["classical_ml"][name] = {"error": str(e)}
            except Exception as e:
                results["classical_ml"] = {"error": str(e)}
        
        # Deep Learning Prediction
        if MODEL is not None:
            try:
                img_resized = _resize_to_model(img_array, MODEL)
                img = img_resized.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                
                preds = MODEL.predict(img, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = CLASSES[pred_idx]
                confidence = float(preds[pred_idx])
                
                results["deep_learning"] = {
                    "prediction": pred_class,
                    "confidence": confidence,
                    "probabilities": {cls: float(p) for cls, p in zip(CLASSES, preds.tolist())}
                }
            except Exception as e:
                results["deep_learning"] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Original single model prediction endpoint (EfficientNetB0 only)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        img_array = np.array(image)
        img_array = _resize_to_model(img_array, MODEL)
        img = img_array.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        preds = MODEL.predict(img, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx])

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "probs": {cls: float(p) for cls, p in zip(CLASSES, preds.tolist())}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


