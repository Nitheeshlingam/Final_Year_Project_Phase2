#!/usr/bin/env python3
"""Ensemble Prediction System for Rice Nutrient Deficiency Detection"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class EnsemblePredictor:
    """Ensemble predictor that combines all model predictions for better accuracy."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.classes = ['Nitrogen', 'Phosphorus', 'Potassium']
        self.model_weights = {
            'rule_based': 0.15,      # Lower weight due to lower accuracy
            'RandomForest': 0.20,     # Good balance
            'SVM': 0.15,             # Lower weight due to lower accuracy
            'XGBoost': 0.25,         # Higher weight due to good performance
            'EfficientNetB0': 0.25   # Higher weight due to best performance
        }
        
    def load_models(self, base_path: str = "models"):
        """Load all available models."""
        # Load classical ML models
        ml_models = ['RandomForest', 'SVM', 'XGBoost']
        for model_name in ml_models:
            model_path = f"{base_path}/ml_model_{model_name}.joblib"
            try:
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
        
        # Load scaler
        try:
            self.scaler = joblib.load(f"{base_path}/feature_scaler.joblib")
            print("✅ Loaded feature scaler")
        except Exception as e:
            print(f"⚠️ Failed to load scaler: {e}")
        
        # Load deep learning model
        dl_model_path = f"{base_path}/best_efficientnetb0_improved.h5"
        if not os.path.exists(dl_model_path):
            dl_model_path = f"{base_path}/best_efficientnetb0.h5"
        
        try:
            self.models['EfficientNetB0'] = tf.keras.models.load_model(dl_model_path)
            print(f"✅ Loaded EfficientNetB0")
        except Exception as e:
            print(f"⚠️ Failed to load EfficientNetB0: {e}")
    
    def predict_ensemble(self, image_array: np.ndarray, rule_based_pred: str = None, 
                        rule_based_features: Dict = None) -> Dict[str, Any]:
        """Make ensemble prediction combining all models."""
        
        predictions = {}
        confidences = {}
        probabilities = {}
        
        # Rule-based prediction
        if rule_based_pred:
            predictions['rule_based'] = rule_based_pred
            rb_probs = self._rule_based_probabilities(rule_based_features)
            probabilities['rule_based'] = rb_probs
            # Confidence aligned to displayed probabilities
            try:
                confidences['rule_based'] = float(max(rb_probs.values()))
            except Exception:
                confidences['rule_based'] = self._calculate_rule_confidence(rule_based_features)
        
        # Classical ML predictions
        if self.scaler and any(model in self.models for model in ['RandomForest', 'SVM', 'XGBoost']):
            try:
                from src.classical_ml.feature_extractor import FeatureExtractor
                extractor = FeatureExtractor()
                features = extractor.extract_all_features(image_array)
                features_scaled = self.scaler.transform([features])
                
                for model_name in ['RandomForest', 'SVM', 'XGBoost']:
                    if model_name in self.models:
                        model = self.models[model_name]
                        pred_idx = model.predict(features_scaled)[0]
                        pred_class = self.classes[pred_idx]
                        predictions[model_name] = pred_class
                        
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(features_scaled)[0]
                            confidences[model_name] = float(np.max(probs))
                            probabilities[model_name] = {cls: float(p) for cls, p in zip(self.classes, probs)}
                        else:
                            confidences[model_name] = 1.0
                            probabilities[model_name] = {cls: 1.0 if cls == pred_class else 0.0 for cls in self.classes}
            except Exception as e:
                print(f"⚠️ Classical ML prediction failed: {e}")
        
        # Deep Learning prediction
        if 'EfficientNetB0' in self.models:
            try:
                model = self.models['EfficientNetB0']
                
                # Resize image to model input size
                if image_array.shape[:2] != (224, 224):
                    from PIL import Image
                    img_pil = Image.fromarray(image_array)
                    img_pil = img_pil.resize((224, 224))
                    image_array = np.array(img_pil)
                
                # Preprocess
                img = image_array.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Predict
                preds = model.predict(img, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = self.classes[pred_idx]
                
                predictions['EfficientNetB0'] = pred_class
                confidences['EfficientNetB0'] = float(preds[pred_idx])
                probabilities['EfficientNetB0'] = {cls: float(p) for cls, p in zip(self.classes, preds)}
                
            except Exception as e:
                print(f"⚠️ Deep Learning prediction failed: {e}")
        
        # Ensemble voting
        ensemble_result = self._weighted_voting(predictions, confidences, probabilities)
        
        return {
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'individual_probabilities': probabilities,
            'ensemble_prediction': ensemble_result['prediction'],
            'ensemble_confidence': ensemble_result['confidence'],
            'ensemble_probabilities': ensemble_result['probabilities'],
            'model_agreement': self._calculate_agreement(predictions),
            'weighted_scores': ensemble_result['weighted_scores']
        }
    
    def _calculate_rule_confidence(self, features: Dict) -> float:
        """Calculate confidence for rule-based prediction."""
        if not features:
            return 0.5
        
        # Prefer combined deficiency scores if available
        if 'nitrogen_score' in features or 'phosphorus_score' in features or 'potassium_score' in features:
            max_score = max(
                float(features.get('nitrogen_score', 0.0)),
                float(features.get('phosphorus_score', 0.0)),
                float(features.get('potassium_score', 0.0)),
            )
            return float(max(0.0, min(1.0, max_score)))

        # Fallback: use the maximum color ratio as a weak confidence proxy
        max_ratio = max(
            float(features.get('yellow_ratio', 0.0)),
            float(features.get('purple_ratio', 0.0)),
            float(features.get('brown_ratio', 0.0)),
        )
        return float(min(max_ratio * 10.0, 1.0))
    
    def _rule_based_probabilities(self, features: Dict) -> Dict[str, float]:
        """Convert rule-based features to probabilities."""
        if not features:
            return {cls: 1.0/3 for cls in self.classes}
        
        # Prefer combined deficiency scores if available
        combined = [
            float(features.get('nitrogen_score', 0.0)),
            float(features.get('phosphorus_score', 0.0)),
            float(features.get('potassium_score', 0.0)),
        ]
        if sum(combined) > 0:
            total = sum(combined)
            return {self.classes[i]: combined[i] / total for i in range(3)}

        # Fallback: normalize raw color ratios to probabilities
        ratios = [
            float(features.get('yellow_ratio', 0.0)),
            float(features.get('purple_ratio', 0.0)),
            float(features.get('brown_ratio', 0.0)),
        ]
        ratios = [r + 1e-6 for r in ratios]  # avoid divide-by-zero
        total = sum(ratios)
        return {self.classes[i]: ratios[i] / total for i in range(3)}
    
    def _weighted_voting(self, predictions: Dict, confidences: Dict, probabilities: Dict) -> Dict:
        """Perform weighted voting based on model weights and confidences."""
        
        # Calculate weighted probabilities
        weighted_probs = {cls: 0.0 for cls in self.classes}
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            if model_name in self.model_weights and model_name in probabilities:
                weight = self.model_weights[model_name]
                confidence = confidences.get(model_name, 0.5)
                
                # Adjust weight by confidence
                adjusted_weight = weight * confidence
                total_weight += adjusted_weight
                
                # Add weighted probabilities
                for cls in self.classes:
                    weighted_probs[cls] += probabilities[model_name][cls] * adjusted_weight
        
        # Normalize probabilities
        if total_weight > 0:
            for cls in self.classes:
                weighted_probs[cls] /= total_weight
        
        # Find best prediction
        best_class = max(weighted_probs, key=weighted_probs.get)
        best_confidence = weighted_probs[best_class]
        
        return {
            'prediction': best_class,
            'confidence': best_confidence,
            'probabilities': weighted_probs,
            'weighted_scores': weighted_probs
        }
    
    def _calculate_agreement(self, predictions: Dict) -> Dict[str, Any]:
        """Calculate agreement between models."""
        if not predictions:
            return {'agreement_score': 0.0, 'majority_prediction': None}
        
        # Count predictions
        pred_counts = Counter(predictions.values())
        total_models = len(predictions)
        
        # Calculate agreement score
        max_count = max(pred_counts.values())
        agreement_score = max_count / total_models
        
        # Find majority prediction
        majority_prediction = pred_counts.most_common(1)[0][0] if pred_counts else None
        
        return {
            'agreement_score': agreement_score,
            'majority_prediction': majority_prediction,
            'prediction_counts': dict(pred_counts),
            'total_models': total_models
        }

# Import os for path operations
import os
