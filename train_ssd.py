"""
train_ssd.py

Обучение SSD300 (день 8-9 плана / раздел 8.3-8.4 методички).

SSD работает с фиксированным разрешением 300x300 — заметно меньше, чем
Faster R-CNN, поэтому обучение быстрее и можно использовать больший batch_size.

Запуск:
    python train_ssd.py
"""

import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import TARGET_CLASSES
from src.dataset.torchvision_dataset import TorchvisionCocoDataset, collate_fn
from src.models.ssd import build_ssd_model
from src.evaluation.metrics import evaluate_torchvision_model

RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_DIR = "data/processed"

NUM_CLASSES = len(TARGET_CLASSES) + 1  # +1 для фона
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0015  # типичный диапазон для SSD300+SGD: 0.001-0.0026
WARMUP_STEPS = 200  # плавный разогрев LR в начале обучения для стабильности
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

    model = build_ssd_model(num_classes=NUM_CLASSES, pretrained=True,
                             score_thresh=0.2, detections_per_img=100)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Линейный разогрев LR с малого значения до LEARNING_RATE за WARMUP_STEPS
    # шагов — стандартный приём против NaN/расходимости в начале обучения,
    # когда градиенты от случайно инициализированной головы ещё нестабильны.
    def warmup_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    global_step = 0

    # ВАЖНО: AMP (mixed precision) отключён для SSD — известная проблема
    # numerical instability (NaN loss) у SSD loss-функции при float16.
    # Используем полную точность float32 — немного медленнее, но стабильно.
    use_amp = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for images, targets in progress:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                # Пропускаем нестабильный шаг вместо того, чтобы "заразить"
                # веса модели NaN-значениями навсегда.
                progress.set_postfix(loss="skipped (non-finite)")
                continue

            losses.backward()
            # gradient clipping — дополнительная защита от расходящихся градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if global_step < WARMUP_STEPS:
                warmup_scheduler.step()
            global_step += 1

            loss_value = losses.item()
            epoch_loss += loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        if global_step >= WARMUP_STEPS:
            main_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | {elapsed:.1f}s\n")

    print("\nОбучение завершено. Оценка на val...")
    metrics = evaluate_torchvision_model(model, val_loader, DEVICE, TARGET_CLASSES)
    print(metrics)

    import os
    os.makedirs("results/logs/ssd", exist_ok=True)
    torch.save(model.state_dict(), "results/logs/ssd/ssd_weights.pt")
    print("Веса сохранены: results/logs/ssd/ssd_weights.pt")


if __name__ == "__main__":
    main()
