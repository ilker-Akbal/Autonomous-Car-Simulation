from pathlib import Path
import json
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


PROJECT_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\autonomous_driving_project\sign_classifier")
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 0
SEED = 42

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=eval_tfms)
test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=eval_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_ds.classes
num_classes = len(class_names)

print("Sınıf sayısı:", num_classes)
print("Sınıflar:", class_names)
print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
)

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

history = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        train_loss += loss.item() * images.size(0)
        train_correct += (preds == labels).sum().item()

    train_loss /= len(train_ds)
    train_acc = train_correct / len(train_ds)

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_loss += loss.item() * images.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(val_ds)
    val_acc = val_correct / len(val_ds)

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_model_wts,
            "class_names": class_names,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
        }, OUTPUT_DIR / "sign_classifier_resnet18_best.pt")

    elapsed = time.time() - start

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "time_sec": elapsed,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
        f"{elapsed:.1f}s"
    )

model.load_state_dict(best_model_wts)
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("\nEn iyi validation accuracy:", best_acc)
print("Test accuracy:", test_acc)
print("Model kaydedildi:", OUTPUT_DIR / "sign_classifier_resnet18_best.pt")
print("Class names kaydedildi:", OUTPUT_DIR / "class_names.json")