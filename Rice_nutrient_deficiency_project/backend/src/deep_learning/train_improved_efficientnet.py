#!/usr/bin/env python3
"""Improved EfficientNetB0 Training Script with Better Data Augmentation and Parameters"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_improved_efficientnet_model(input_shape=(224, 224, 3), num_classes=3):
    """Create improved EfficientNetB0 model with better architecture."""
    
    # Load EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights=None,  # Train from scratch for better domain adaptation
        include_top=False,
        input_shape=input_shape
    )
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_improved_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """Create improved data generators with better augmentation."""
    
    # Training data generator with aggressive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,           # Increased rotation
        width_shift_range=0.2,       # Increased shift
        height_shift_range=0.2,
        shear_range=0.2,             # Increased shear
        zoom_range=0.2,              # Increased zoom
        horizontal_flip=True,
        vertical_flip=True,          # Added vertical flip
        brightness_range=[0.8, 1.2], # Brightness variation
        channel_shift_range=0.1,     # Color channel shift
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_improved_efficientnet():
    """Train improved EfficientNetB0 model."""
    print("🌾 IMPROVED EFFICIENTNETB0 TRAINING")
    print("=" * 60)
    
    # Configuration
    data_dir = "data/rice_plant_lacks_nutrients"
    target_size = (224, 224)
    batch_size = 16  # Reduced batch size for better gradient updates
    epochs = 50       # Increased epochs
    num_classes = 3
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return None
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = create_improved_data_generators(
        data_dir, target_size, batch_size
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Create model
    print("Creating model...")
    model = create_improved_efficientnet_model(target_size + (3,), num_classes)
    
    # Compile model with improved optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Higher initial learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_efficientnetb0_improved.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,  # Reduced patience for faster learning rate reduction
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test data
    print("\nEvaluating model...")
    test_loss, test_acc, test_top_k = model.evaluate(val_generator, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Top-K Accuracy: {test_top_k:.4f}")
    
    # Generate predictions for detailed analysis
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Classification report
    class_names = list(val_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('models/final_efficientnetb0_improved.h5')
    print(f"\n✅ Model saved to: models/final_efficientnetb0_improved.h5")
    
    return model, history

def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_improved_efficientnet()
