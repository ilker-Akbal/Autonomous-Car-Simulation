from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


RAW_DATASET_DIR = Path(
    "~/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/dataset_v2/raw"
).expanduser()

OUTPUT_DIR = Path(
    "~/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/dataset_v2"
).expanduser()

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

TARGET_PER_CLASS = 800
IMAGE_SIZE = 224
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


augment_pipeline = A.Compose([
    A.OneOf([
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.SafeRotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    ], p=0.80),

    A.OneOf([
        A.Perspective(scale=(0.03, 0.10), p=1.0),
        A.Affine(
            scale=(0.80, 1.20),
            translate_percent=(-0.10, 0.10),
            shear=(-12, 12),
            rotate=(-15, 15),
            fit_output=False,
            p=1.0,
        ),
    ], p=0.75),

    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=1.0,
        ),
        A.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.08,
            p=1.0,
        ),
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=1.0,
        ),
    ], p=0.85),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 9), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.Defocus(radius=(2, 6), alias_blur=(0.1, 0.5), p=1.0),
    ], p=0.70),

    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        A.ISONoise(
            color_shift=(0.01, 0.08),
            intensity=(0.10, 0.40),
            p=1.0,
        ),
    ], p=0.65),

    A.OneOf([
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=1.0,
        ),
        A.RandomFog(
            fog_coef_range=(0.05, 0.20),
            alpha_coef=0.08,
            p=1.0,
        ),
        A.RandomRain(
            brightness_coefficient=0.9,
            drop_width=1,
            blur_value=3,
            p=1.0,
        ),
    ], p=0.45),

    A.OneOf([
        A.ImageCompression(
            quality_range=(25, 70),
            compression_type="jpeg",
            p=1.0,
        ),
        A.Downscale(
            scale_range=(0.40, 0.85),
            interpolation_pair={
                "downscale": cv2.INTER_AREA,
                "upscale": cv2.INTER_LINEAR,
            },
            p=1.0,
        ),
    ], p=0.60),

    A.OneOf([
        A.Sharpen(alpha=(0.15, 0.40), lightness=(0.8, 1.2), p=1.0),
        A.UnsharpMask(
            blur_limit=(3, 7),
            sigma_limit=(0.2, 1.0),
            alpha=(0.2, 0.5),
            threshold=10,
            p=1.0,
        ),
        A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.4), p=1.0),
    ], p=0.30),

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])


def reset_output_dirs():
    for split in ["train", "val"]:
        split_path = OUTPUT_DIR / split
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)


def get_image_files(directory: Path):
    return sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image, path: Path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def choose_split():
    if random.random() < TRAIN_RATIO:
        return "train"
    return "val"


def generate_dataset():
    reset_output_dirs()

    class_dirs = sorted([
        d for d in RAW_DATASET_DIR.iterdir()
        if d.is_dir()
    ])

    print("\n=== TRAFFIC SIGN DATASET AUGMENTATION ===")
    print(f"RAW     : {RAW_DATASET_DIR}")
    print(f"OUTPUT  : {OUTPUT_DIR}")
    print(f"PER CLS : {TARGET_PER_CLASS}")
    print("SPLIT   : train 85% / val 15%\n")

    total_generated = 0
    skipped_classes = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = get_image_files(class_dir)

        if len(image_files) == 0:
            skipped_classes.append(class_name)
            print(f"[SKIP] {class_name}: görsel yok")
            continue

        print(f"\n[CLASS] {class_name} | raw={len(image_files)}")

        generated = 0

        for i in tqdm(range(TARGET_PER_CLASS), desc=class_name):
            source_path = random.choice(image_files)
            image = load_image(source_path)

            if image is None:
                continue

            try:
                aug = augment_pipeline(image=image)["image"]
            except Exception as exc:
                print(f"[AUG ERROR] {source_path}: {exc}")
                continue

            split = choose_split()
            out_dir = OUTPUT_DIR / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{class_name}_{i:05d}.jpg"
            save_image(aug, out_path)

            generated += 1
            total_generated += 1

        print(f"[OK] {class_name}: generated={generated}")

    print("\n=== GENERATION DONE ===")
    print(f"TOTAL GENERATED: {total_generated}")

    if skipped_classes:
        print("\nSKIPPED EMPTY CLASSES:")
        for cls in skipped_classes:
            print(f"- {cls}")


def print_stats():
    print("\n=== FINAL DATASET STATS ===\n")

    for split in ["train", "val"]:
        split_dir = OUTPUT_DIR / split

        print(f"[{split.upper()}]")
        total = 0

        if not split_dir.exists():
            print("YOK\n")
            continue

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue

            count = len(get_image_files(cls_dir))
            total += count
            print(f"{cls_dir.name:<35} {count}")

        print(f"TOTAL {split}: {total}\n")


if __name__ == "__main__":
    generate_dataset()
    print_stats()