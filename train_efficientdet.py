"""
train_efficientdet.py

Обучение EfficientDet-D0 (день 8-9 плана / раздел 8.3-8.4 методички).
Использует библиотеку effdet (Ross Wightman) + timm для backbone.

Запуск:
    python train_efficientdet.py
"""

import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import TARGET_CLASSES
from src.dataset.effdet_dataset import EffDetCocoDataset, effdet_collate_fn
from src.models.efficientdet import build_efficientdet_model, effdet_predictions_to_standard_format
from src.evaluation.metrics import compute_map, compute_precision_recall_f1

RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_DIR = "data/processed"

NUM_CLASSES = len(TARGET_CLASSES)
IMAGE_SIZE = 512
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    train_ds = EffDetCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_train.json",
                                  image_size=IMAGE_SIZE)
    val_ds = EffDetCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_val.json",
                                image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=effdet_collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=effdet_collate_fn, num_workers=2)

    print(f"Train: {len(train_ds)} изображений, Val: {len(val_ds)} изображений")

    model = build_efficientdet_model(num_classes=NUM_CLASSES, pretrained=True,
                                      image_size=IMAGE_SIZE, mode="train")
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                 momentum=0.9, weight_decay=0.0005)

    def warmup_lambda(step):
        return min(1.0, step / WARMUP_STEPS) if WARMUP_STEPS > 0 else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for images, targets, _ in progress:
            images = images.to(DEVICE)
            targets = {
                "bbox": [b.to(DEVICE) for b in targets["bbox"]],
                "cls": [c.to(DEVICE) for c in targets["cls"]],
                "img_size": targets["img_size"].to(DEVICE),
                "img_scale": targets["img_scale"].to(DEVICE),
            }

            optimizer.zero_grad()
            output = model(images, targets)
            loss = output["loss"]

            if not torch.isfinite(loss):
                progress.set_postfix(loss="skipped (non-finite)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if global_step < WARMUP_STEPS:
                warmup_scheduler.step()
            global_step += 1

            loss_value = loss.item()
            epoch_loss += loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        if global_step >= WARMUP_STEPS:
            main_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | {elapsed:.1f}s\n")

    print("\nОбучение завершено. Оценка на val...")
    predict_model = build_efficientdet_model(num_classes=NUM_CLASSES, pretrained=False,
                                              image_size=IMAGE_SIZE, mode="predict")
    predict_model.model.load_state_dict(model.model.state_dict())
    predict_model.to(DEVICE)
    predict_model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(DEVICE)
            img_info = {
                "img_size": targets["img_size"].to(DEVICE),
                "img_scale": targets["img_scale"].to(DEVICE),
            }
            raw_output = predict_model(images, img_info)
            preds = effdet_predictions_to_standard_format(raw_output, score_thresh=0.2)
            all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])

            for bbox, cls in zip(targets["bbox"], targets["cls"]):
                if len(bbox) > 0:
                    y1, x1, y2, x2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                else:
                    boxes_xyxy = bbox
                all_targets.append({"boxes": boxes_xyxy, "labels": cls.long()})

    map_metrics = compute_map(all_preds, all_targets, iou_thresholds=[0.5])
    prf1_metrics = compute_precision_recall_f1(all_preds, all_targets, iou_threshold=0.5)
    metrics = {
        "mAP50": map_metrics["map_50"],
        "mAP": map_metrics["map"],
        "precision": prf1_metrics["precision"],
        "recall": prf1_metrics["recall"],
        "f1": prf1_metrics["f1"],
    }
    print(metrics)

    import os
    os.makedirs("results/logs/efficientdet", exist_ok=True)
    torch.save(model.model.state_dict(), "results/logs/efficientdet/efficientdet_weights.pt")
    print("Веса сохранены: results/logs/efficientdet/efficientdet_weights.pt")


if __name__ == "__main__":
    main()
