"""
Generates synthetic concrete-surface images (crack / no_crack), standing in
for the real dataset:
https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

Why synthetic data:
This sandbox can't reach kaggle.com, so this script builds textured
grayscale "concrete-like" backgrounds and draws random irregular crack lines
on half of them. This lets the full pipeline (data -> CNN -> SQL logging ->
FastAPI -> Docker) be built and verified end-to-end right now.

To use the REAL dataset instead:
1. Download the Kaggle "Surface Crack Detection" dataset (40,000 images,
   Positive/Negative folders).
2. Replace this script's output folders with the real images, keeping the
   same directory structure: data/images/crack/*.jpg, data/images/no_crack/*.jpg
3. Nothing downstream needs to change -- src/train.py uses
   tf.keras.utils.image_dataset_from_directory, which just needs the folder
   structure, not this specific generator.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RNG = np.random.default_rng(7)
IMG_SIZE = 128


def make_textured_background():
    """Noisy gray 'concrete' base texture."""
    base = RNG.normal(160, 18, (IMG_SIZE, IMG_SIZE)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def add_crack(img, n_segments=None):
    """Draws a random irregular dark line (a 'crack') across the image."""
    draw = ImageDraw.Draw(img)
    n_segments = n_segments or RNG.integers(4, 9)

    x, y = RNG.integers(0, IMG_SIZE), 0
    points = [(x, y)]
    for _ in range(n_segments):
        x = np.clip(x + RNG.integers(-25, 25), 0, IMG_SIZE - 1)
        y = np.clip(y + RNG.integers(10, 30), 0, IMG_SIZE - 1)
        points.append((x, y))

    width = RNG.integers(1, 3)
    dark = int(RNG.integers(15, 60))
    draw.line(points, fill=(dark, dark, dark), width=int(width))
    # slight blur so the crack blends into the texture like a real photo
    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def generate_dataset(n_per_class=300):
    for label in ["crack", "no_crack"]:
        out_dir = f"data/images/{label}"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n_per_class):
            img = make_textured_background()
            if label == "crack":
                img = add_crack(img)
            img.save(f"{out_dir}/{label}_{i:04d}.png")
    print(f"Generated {n_per_class} images per class in data/images/")


if __name__ == "__main__":
    generate_dataset(n_per_class=300)
