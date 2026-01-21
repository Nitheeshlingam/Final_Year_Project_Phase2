#!/usr/bin/env python3
"""Improved Classical ML Training Script with Better Parameters and Feature Engineering"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from .feature_extractor import FeatureExtractor
from ..utils.data_preprocessor import DataPreprocessor

BASE_DIR = "data/rice_plant_lacks_nutrients"

def train_improved_classical_ml():
    """Train classical ML models with improved parameters and feature engineering."""
    print("🌾 IMPROVED CLASSICAL ML TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    prep = DataPreprocessor(BASE_DIR, target_size=(224, 224))
    images, labels = prep.load_images()
    
    print(f"Loaded {len(images)} images")
    print(f"Classes: {prep.classes}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Extract features with improved feature extractor
    extractor = FeatureExtractor()
    print("Extracting features...")
    X = np.array([extractor.extract_all_features(img) for img in images])
    y = labels
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.split_data(X, y)
    
    # Feature scaling for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler.joblib")
    
    # Improved model configurations
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        ),
    }
    
    best_name, best_model, best_score = None, None, -1.0
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        print(f"{'='*40}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_score = model.score(X_val_scaled, y_val)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"{name} Results:")
        print(f"  Train Accuracy: {model.score(X_train_scaled, y_train):.4f}")
        print(f"  Val Accuracy: {val_score:.4f}")
        print(f"  Test Accuracy: {test_score:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred, target_names=prep.classes))
        
        print("Confusion Matrix (Test):")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        joblib.dump(model, f"models/ml_model_{name}.joblib")
        
        # Track best model
        if val_score > best_score:
            best_name, best_model, best_score = name, model, val_score
        
        results[name] = {
            'train_acc': model.score(X_train_scaled, y_train),
            'val_acc': val_score,
            'test_acc': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Save best model
    if best_model is not None:
        joblib.dump(best_model, f"models/best_ml_model_{best_name}.joblib")
        print(f"\n🏆 Best model: {best_name} with validation accuracy: {best_score:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"{name:12} | Train: {result['train_acc']:.3f} | Val: {result['val_acc']:.3f} | Test: {result['test_acc']:.3f} | CV: {result['cv_mean']:.3f}±{result['cv_std']:.3f}")
    
    return results

if __name__ == "__main__":
    train_improved_classical_ml()
