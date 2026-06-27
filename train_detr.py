"""
train_detr.py

Обучение DETR (день 8-9 плана / раздел 8.3-8.4 методички).

Использует библиотеку transformers (HuggingFace). DETR — трансформерная
архитектура, обычно требует больше эпох для сходимости, чем CNN-based
модели (см. раздел 3.2 отчёта — известное ограничение DETR на мелких
объектах и при недостатке данных/эпох обучения).

Запуск:
    python train_detr.py
"""

import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrImageProcessor

from src.dataset.dataset import TARGET_CLASSES
from src.dataset.detr_dataset import DetrCocoDataset, detr_collate_fn
from src.models.detr import build_detr_model
from src.evaluation.metrics import compute_map, compute_precision_recall_f1

RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_DIR = "data/processed"

NUM_CLASSES = len(TARGET_CLASSES)  # БЕЗ фона (DETR обрабатывает "no object" отдельно)
EPOCHS = 8  # DETR медленный; уменьшено с 15 для разумного времени обучения
BATCH_SIZE = 8  # увеличен благодаря уменьшенному разрешению (480/640 вместо 800/1333)
LEARNING_RATE = 1e-5  # transfer learning -> малый LR, как рекомендуется для DETR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 480, "longest_edge": 640},
    )

    train_ds = DetrCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_train.json", processor)
    val_ds = DetrCocoDataset(RAW_IMAGES_DIR, f"{PROCESSED_DIR}/instances_val.json", processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=detr_collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=detr_collate_fn, num_workers=2)

    print(f"Train: {len(train_ds)} изображений, Val: {len(val_ds)} изображений")

    model = build_detr_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for pixel_values, pixel_mask, labels in progress:
            pixel_values = pixel_values.to(DEVICE)
            pixel_mask = pixel_mask.to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in labels]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            if not torch.isfinite(loss):
                progress.set_postfix(loss="skipped (non-finite)")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | {elapsed:.1f}s\n")

    print("\nОбучение завершено. Оценка на val...")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for pixel_values, pixel_mask, labels in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            pixel_mask = pixel_mask.to(DEVICE)

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.stack([t["orig_size"] for t in labels]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=target_sizes
            )
            all_preds.extend([{k: v.cpu() for k, v in r.items()} for r in results])

            for t in labels:
                # labels из processor хранят боксы в нормализованном cxcywh —
                # конвертируем в абсолютные xyxy для сопоставимости с предсказаниями
                h, w = t["orig_size"].tolist()
                boxes_cxcywh = t["boxes"]
                cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                all_targets.append({"boxes": boxes_xyxy.cpu(), "labels": t["class_labels"].cpu()})

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
    os.makedirs("results/logs/detr", exist_ok=True)
    model.save_pretrained("results/logs/detr/detr_model")
    print("Веса сохранены: results/logs/detr/detr_model")


if __name__ == "__main__":
    main()
