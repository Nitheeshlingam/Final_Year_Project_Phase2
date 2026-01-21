#!/usr/bin/env python3
"""Train EfficientNetB0 on rice nutrient dataset and save model."""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "rice_plant_lacks_nutrients")
MODEL_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_efficientnetb0.h5")


def train(batch_size: int = 8, target_size=(192, 192), epochs: int = 30):

    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {BASE_DIR}")

    # Constrain TF threads for low-spec CPUs and ensure channels_last
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
    # Ensure channels_last to match ImageNet weights
    K.set_image_data_format('channels_last')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        BASE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_gen.num_classes

    # Build EfficientNetB0; try ImageNet weights, fallback to random init if mismatch
    try:
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(target_size[0], target_size[1], 3),
        )
    except Exception as e:
        print(f"Warning: Failed to load ImageNet weights due to: {e}. Falling back to weights=None.")
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(target_size[0], target_size[1], 3),
        )

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    # Compute class weights to handle imbalance
    labels = train_gen.classes
    unique_labels = np.unique(labels)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    class_weight = {int(k): float(v) for k, v in zip(unique_labels, class_weights_arr)}

    # Phase 1: freeze backbone, train head with higher LR
    base_model.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    warmup_cbs = [
        callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=8,
        callbacks=warmup_cbs,
        verbose=1,
        class_weight=class_weight,
    )

    # Phase 2: unfreeze last 20 layers and fine-tune with low LR
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    finetune_cbs = [
        callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=finetune_cbs,
        verbose=1,
        class_weight=class_weight,
    )

    print(f"Saved best model to: {MODEL_OUT}")


if __name__ == "__main__":
    train()


