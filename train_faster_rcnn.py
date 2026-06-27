"""
train_faster_rcnn.py

Обучение Faster R-CNN (день 8-9 плана / раздел 8.3-8.4 методички).

Использует стандартный цикл обучения PyTorch (в отличие от YOLOv8/ultralytics,
у torchvision detection-моделей нет встроенного метода .train()).

Запуск:
    python train_faster_rcnn.py
"""

import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import TARGET_CLASSES
from src.dataset.torchvision_dataset import TorchvisionCocoDataset, collate_fn
from src.models.faster_rcnn import build_faster_rcnn_model
from src.evaluation.metrics import evaluate_torchvision_model

RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_DIR = "data/processed"

NUM_CLASSES = len(TARGET_CLASSES) + 1  # +1 для фона
EPOCHS = 15
BATCH_SIZE = 4  # Faster R-CNN требовательнее к памяти GPU, чем YOLOv8n
LEARNING_RATE = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    train_ds = TorchvisionCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_train.json")
    val_ds = TorchvisionCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_val.json")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    print(f"Train: {len(train_ds)} изображений, Val: {len(val_ds)} изображений")

    model = build_faster_rcnn_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for images, targets in progress:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress.set_postfix(loss=f"{losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | {elapsed:.1f}s\n")

    print("\nОбучение завершено. Оценка на val...")
    metrics = evaluate_torchvision_model(model, val_loader, DEVICE, TARGET_CLASSES)
    print(metrics)

    import os
    os.makedirs("results/logs/faster_rcnn", exist_ok=True)
    torch.save(model.state_dict(), "results/logs/faster_rcnn/faster_rcnn_weights.pt")
    print("Веса сохранены: results/logs/faster_rcnn/faster_rcnn_weights.pt")


if __name__ == "__main__":
    main()
