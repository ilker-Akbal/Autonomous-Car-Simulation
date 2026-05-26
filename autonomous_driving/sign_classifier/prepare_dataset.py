from pathlib import Path
import shutil
import random
import json
import pandas as pd

SOURCE_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\autonomous_driving_project\Trafik\Trafik")
LABELS_CSV = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\autonomous_driving_project\Trafik\labels.csv")
OUTPUT_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\autonomous_driving_project\sign_classifier\dataset")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SELECTED_CLASSES = {
    0: "hiz_siniri_20",
    1: "hiz_siniri_30",
    2: "hiz_siniri_40",
    3: "hiz_siniri_50",
    4: "yaya_gecidi",
    11: "dur",
    13: "girisi_olmayan_yol",
    21: "okul_gecidi",
    24: "tasit_giremez",
    25: "saga_donulmez",
    26: "sola_donulmez",
    48: "yol_calismasi",
    49: "isikli_isaret_cihazi",
    67: "dikkat",
    77: "yol_ver",
    81: "park_etmek_yasaktir",
    82: "duraklamak_park_yasaktir",
}

random.seed(SEED)

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    for class_name in SELECTED_CLASSES.values():
        (OUTPUT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

class_map = {}

for new_id, (old_id, class_name) in enumerate(SELECTED_CLASSES.items()):
    src_dir = SOURCE_ROOT / str(old_id)

    if not src_dir.exists():
        print(f"Eksik klasör: {src_dir}")
        continue

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(list(src_dir.glob(ext)))

    images = sorted(images)
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, files in split_files.items():
        dst_dir = OUTPUT_ROOT / split / class_name

        for img_path in files:
            dst_path = dst_dir / img_path.name
            shutil.copy2(img_path, dst_path)

    class_map[new_id] = {
        "old_class_id": old_id,
        "class_name": class_name,
        "total": n,
        "train": len(split_files["train"]),
        "val": len(split_files["val"]),
        "test": len(split_files["test"]),
    }

    print(
        f"{old_id:02d} -> {class_name}: "
        f"total={n}, train={len(split_files['train'])}, "
        f"val={len(split_files['val'])}, test={len(split_files['test'])}"
    )

with open(OUTPUT_ROOT / "class_map.json", "w", encoding="utf-8") as f:
    json.dump(class_map, f, ensure_ascii=False, indent=2)

print("\nDataset hazırlandı:")
print(OUTPUT_ROOT)