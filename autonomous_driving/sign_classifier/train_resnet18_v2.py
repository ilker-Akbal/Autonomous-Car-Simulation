from pathlib import Path
import json
import copy
import time
import random
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


PROJECT_ROOT = Path(
    "~/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier"
).expanduser()

DATA_ROOT = PROJECT_ROOT / "dataset_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.RandomRotation(10),
    ], p=0.35),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
        )
    ], p=0.35),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ], p=0.20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.10),
        ratio=(0.3, 3.3),
        value="random",
    ),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

class_names = train_ds.classes
num_classes = len(class_names)

print("Sınıf sayısı:", num_classes)
print("Train:", len(train_ds))
print("Val:", len(val_ds))
print("Sınıflar:", class_names)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(in_features, num_classes),
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def run_epoch(model, loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)

    return avg_loss, acc, all_preds, all_labels


def class_accuracy(preds, labels, class_names):
    stats = {}

    for idx, name in enumerate(class_names):
        total = 0
        correct = 0

        for p, y in zip(preds, labels):
            if y == idx:
                total += 1
                if p == y:
                    correct += 1

        acc = correct / total if total > 0 else 0.0
        stats[name] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }

    return stats


best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0
bad_epochs = 0
history = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    train_loss, train_acc, _, _ = run_epoch(model, train_loader, train=True)
    val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, train=False)

    scheduler.step()

    elapsed = time.time() - start
    lr_now = optimizer.param_groups[0]["lr"]

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": lr_now,
        "time_sec": elapsed,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
        f"LR: {lr_now:.6f} | "
        f"{elapsed:.1f}s"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        bad_epochs = 0

        checkpoint = {
            "model_state_dict": best_model_wts,
            "class_names": class_names,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
            "val_acc": best_val_acc,
            "epoch": epoch,
            "arch": "resnet18",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

        torch.save(checkpoint, OUTPUT_DIR / "sign_classifier_resnet18_v2_best.pt")

        with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        class_stats = class_accuracy(val_preds, val_labels, class_names)
        with open(OUTPUT_DIR / "val_class_accuracy.json", "w", encoding="utf-8") as f:
            json.dump(class_stats, f, ensure_ascii=False, indent=2)

        print(f"Yeni best model kaydedildi. Val Acc: {best_val_acc:.4f}")

    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(f"Early stopping. {PATIENCE} epoch iyileşme yok.")
        break


model.load_state_dict(best_model_wts)

with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "training_history.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
            "time_sec",
        ],
    )
    writer.writeheader()
    writer.writerows(history)

print("\nEğitim tamamlandı.")
print("Best epoch:", best_epoch)
print("Best val acc:", best_val_acc)
print("Model:", OUTPUT_DIR / "sign_classifier_resnet18_v2_best.pt")
print("Class names:", OUTPUT_DIR / "class_names.json")