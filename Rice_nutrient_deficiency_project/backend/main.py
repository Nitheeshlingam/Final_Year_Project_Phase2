#!/usr/bin/env python3
"""Main application entry point for Rice Nutrient Deficiency Detection."""

import os
import sys
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ...
    return {
        "prediction": pred_class,
        "confidence": confidence,
        "probs": {cls: float(p) for cls, p in zip(CLASSES, preds.tolist())}
    }