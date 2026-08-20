"""
CNN architecture for crack/no_crack classification, trained from scratch.

Note: transfer learning with a pretrained backbone (e.g. MobileNetV2) would
typically generalize better to real-world photos and is the standard
production choice for small datasets. See the commented-out block below for
how to switch to it once you have internet access to download ImageNet
weights (from storage.googleapis.com, which this build environment couldn't
reach).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size=128):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="crack_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

    # --- To use transfer learning instead (needs internet access to
    #     download ImageNet weights): ---
    # base = keras.applications.MobileNetV2(
    #     input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    # )
    # base.trainable = False
    # x = layers.Rescaling(1.0 / 255)(inputs)
    # x = base(x, training=False)
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(32, activation="relu")(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)
