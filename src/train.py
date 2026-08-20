"""
Trains a small CNN (from scratch, no pretrained weights needed) to classify
concrete surface images as crack / no_crack.

Note: transfer learning with a pretrained backbone (e.g. MobileNetV2) would
typically generalize better to real-world photos and is the standard
production choice for small datasets -- see the commented-out block below
for how to switch to it once you have unrestricted internet access to
download ImageNet weights.
"""

import os
import json
import tensorflow as tf

from model import build_model

IMG_SIZE = 128
BATCH_SIZE = 32
DATA_DIR = "data/images"
MODELS_DIR = "models"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="binary",
    )

    class_names = train_ds.class_names  # e.g. ['crack', 'no_crack']
    print("Class mapping (0/1):", class_names)

    model = build_model(img_size=IMG_SIZE)
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=8, verbose=2)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.3f}")

    model.save(os.path.join(MODELS_DIR, "crack_classifier.keras"))
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump({
            "img_size": IMG_SIZE,
            "class_names": class_names,  # index 0 = class_names[0], etc.
            "val_accuracy": float(val_acc),
        }, f, indent=2)
    print(f"Saved model + config to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
