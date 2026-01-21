#!/usr/bin/env python3
"""Comprehensive Model Retraining Script for Rice Nutrient Deficiency Detection"""

import os
import sys
import subprocess

def main():
    """Retrain all models with improved parameters."""
    print("🌾 COMPREHENSIVE MODEL RETRAINING")
    print("=" * 60)
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("1️⃣ Retraining Classical ML Models...")
    print("-" * 40)
    try:
        from src.classical_ml.train_improved_ml_models import train_improved_classical_ml
        train_improved_classical_ml()
        print("✅ Classical ML models retrained successfully")
    except Exception as e:
        print(f"❌ Classical ML training failed: {e}")
    
    print("\n2️⃣ Retraining Deep Learning Model...")
    print("-" * 40)
    try:
        from src.deep_learning.train_improved_efficientnet import train_improved_efficientnet
        train_improved_efficientnet()
        print("✅ Deep Learning model retrained successfully")
    except Exception as e:
        print(f"❌ Deep Learning training failed: {e}")
    
    print("\n3️⃣ Testing Improved Models...")
    print("-" * 40)
    try:
        # Test the improved models
        from test_rice_deficiency import RiceDeficiencyTester
        tester = RiceDeficiencyTester()
        results = tester.test_all_images_in_folder()
        print("✅ Model testing completed")
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
    
    print("\n🎉 Retraining completed!")
    print("=" * 60)
    print("Next steps:")
    print("1. Restart the API server to load new models")
    print("2. Test the frontend with improved predictions")
    print("3. Check model performance in the web interface")

if __name__ == "__main__":
    main()
