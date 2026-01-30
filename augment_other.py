"""
Augment images in breeds/Other/ to create more negative examples.

Why: With only ~18 "Other" images, the classifier will still map humans/text/random images to a breed.
This script generates many augmented variants (crop/rotate/brightness/contrast/noise) and saves them
back into breeds/Other/ with new filenames.

Usage:
  python augment_other.py --in_dir breeds/Other --out_dir breeds/Other --copies 40
"""

import argparse
import os
import random
from PIL import Image, ImageEnhance


def list_images(in_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for name in os.listdir(in_dir):
        path = os.path.join(in_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            yield path


def random_crop(img: Image.Image):
    w, h = img.size
    if w < 64 or h < 64:
        return img
    scale = random.uniform(0.65, 0.95)
    nw, nh = int(w * scale), int(h * scale)
    x0 = random.randint(0, max(0, w - nw))
    y0 = random.randint(0, max(0, h - nh))
    return img.crop((x0, y0, x0 + nw, y0 + nh))


def add_noise(img: Image.Image, amount: float = 0.02):
    # lightweight noise via pixel jitter (no numpy dependency)
    if amount <= 0:
        return img
    img = img.convert("RGB")
    w, h = img.size
    px = img.load()
    for _ in range(int(w * h * amount)):
        x = random.randrange(w)
        y = random.randrange(h)
        r, g, b = px[x, y]
        jitter = random.randint(-35, 35)
        px[x, y] = (max(0, min(255, r + jitter)), max(0, min(255, g + jitter)), max(0, min(255, b + jitter)))
    return img


def augment(img: Image.Image):
    img = img.convert("RGB")
    if random.random() < 0.9:
        img = random_crop(img)
    if random.random() < 0.9:
        angle = random.uniform(-18, 18)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.5))
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.4))
    if random.random() < 0.35:
        img = add_noise(img, amount=random.uniform(0.01, 0.03))
    # keep reasonable size
    img.thumbnail((900, 900))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="breeds/Other")
    ap.add_argument("--out_dir", default="breeds/Other")
    ap.add_argument("--copies", type=int, default=40, help="Number of augmented copies per source image")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = list(list_images(args.in_dir))
    if not src:
        raise SystemExit(f"No images found in {args.in_dir}")

    made = 0
    for path in src:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            img = Image.open(path)
        except Exception:
            continue
        for i in range(args.copies):
            out = augment(img)
            out_name = f"{base}__aug_{i:03d}.jpg"
            out_path = os.path.join(args.out_dir, out_name)
            out.save(out_path, "JPEG", quality=88, optimize=True)
            made += 1

    print(f"Generated {made} augmented images into {args.out_dir}")


if __name__ == "__main__":
    main()

