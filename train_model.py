"""
Template script for training a cattle/buffalo breed classification model.
Replace this with your actual training code based on your dataset structure.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100  # Increased for better accuracy

# Root folder for your dataset:
# breeds/
#   Gir/
#   Murrah/
#   ...
DATA_DIR = "breeds"

def load_data(data_dir: str):
    """
    Load images from directory structure with enhanced augmentation for better accuracy.
    """
    # Enhanced augmentation for training - more variations to improve generalization
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=30,  # Increased from 20
        width_shift_range=0.25,  # Increased
        height_shift_range=0.25,  # Increased
        shear_range=0.15,  # Added shear transformation
        zoom_range=0.25,  # Increased
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],  # Added brightness variation
        fill_mode='nearest'
    )
    
    # Minimal augmentation for validation - just rescaling
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_model(num_classes, fine_tune=False):
    """Create a mobile-optimized CNN model with better architecture for accuracy"""
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Full width version for better accuracy
    )
    
    # First train with base frozen
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),  # Slightly increased dropout
        layers.Dense(512, activation='relu'),  # Increased from 128 to 512
        layers.BatchNormalization(),  # Added batch normalization
        layers.Dropout(0.4),  # Increased dropout
        layers.Dense(256, activation='relu'),  # Added another layer
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model, base_model

def fine_tune_model(model, base_model, train_gen, val_gen, initial_epochs=50):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze top layers of base model for fine-tuning
    base_model.trainable = True
    
    # Freeze bottom layers, unfreeze top layers
    fine_tune_at = len(base_model.layers) // 3  # Unfreeze top 2/3 of layers
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # 10x smaller LR
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def train():
    """Main training function"""
    # Use the 'breeds' folder the user already has
    data_dir = DATA_DIR  # Path to your dataset root (breeds/)
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found!")
        print("Please organize your images in the following structure:")
        print("dataset/")
        print("  breed1/")
        print("    img1.jpg")
        print("    img2.jpg")
        print("  breed2/")
        print("    ...")
        return
    
    print("Loading data...")
    train_gen, val_gen = load_data(data_dir)
    
    num_classes = len(train_gen.class_indices)
    print(f"Found {num_classes} classes")
    print(f"Class indices: {train_gen.class_indices}")
    
    print("Creating model...")
    model, base_model = create_model(num_classes)

    # Class weights (helps with imbalance across breeds)
    # Use balanced weights: total / (num_classes * count_i)
    class_counts = np.bincount(train_gen.classes)
    total = float(np.sum(class_counts))
    num_cls = float(len(class_counts))
    class_weight = {i: float(total / (num_cls * max(1, class_counts[i]))) for i in range(len(class_counts))}
    
    # Add callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Stop if no improvement for 15 epochs
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_breed_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("Training initial model (frozen base)...")
    initial_epochs = min(50, EPOCHS // 2)  # First half with frozen base
    
    history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    
    # Fine-tune with unfrozen layers
    print("\nFine-tuning model (unfrozen top layers)...")
    model = fine_tune_model(model, base_model, train_gen, val_gen, initial_epochs)
    
    fine_tune_epochs = EPOCHS - initial_epochs
    if fine_tune_epochs > 0:
        history_finetune = model.fit(
            train_gen,
            initial_epoch=initial_epochs,
            epochs=EPOCHS,
            validation_data=val_gen,
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        
        # Combine histories
        for key in history.history.keys():
            history.history[key].extend(history_finetune.history[key])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/breed_classifier.h5'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save class indices for inference
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    print("Class indices saved to models/class_indices.json")

if __name__ == '__main__':
    print("=" * 50)
    print("Cattle & Buffalo Breed Classification Model Training")
    print("=" * 50)
    train()
