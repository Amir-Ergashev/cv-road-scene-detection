"""
train_yolo.py

Обучение baseline-модели YOLOv8 (день 7 плана / раздел 8.3-8.4 методички).

Параметры подобраны для обучения на GPU (RTX 4070):
- модель yolov8n (nano);
- полный размер изображений 640;
- больше эпох, т.к. GPU значительно быстрее CPU.

Для обучения на CPU верните DEVICE = "cpu", уменьшите IMG_SIZE до 416
и EPOCHS до ~20 (иначе обучение будет идти очень долго).

Запуск:
    python train_yolo.py
"""

from src.dataset.coco_to_yolo import coco_to_yolo_format, write_yolo_yaml
from src.models.yolo import build_yolo_model

RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_DIR = "data/processed"
YOLO_DATA_ROOT = "data/yolo_format"

# Параметры обучения — настроены под GPU (RTX 4070). При обучении на CPU
# верните DEVICE = "cpu" и уменьшите EPOCHS/IMG_SIZE.
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda"


def prepare_yolo_data():
    """Конвертирует train/val/test сплиты из COCO-формата в формат YOLO."""
    print("Конвертация данных в формат YOLO...")
    for split in ["train", "val", "test"]:
        coco_to_yolo_format(
            coco_json_path=f"{PROCESSED_DIR}/instances_{split}.json",
            images_src_dir=RAW_IMAGES_DIR,
            output_dir=f"{YOLO_DATA_ROOT}/{split}",
        )
    write_yolo_yaml(f"{YOLO_DATA_ROOT}/data.yaml", YOLO_DATA_ROOT)


def train_baseline():
    """Обучает YOLOv8n на подготовленных данных."""
    print(f"\nОбучение YOLOv8n: epochs={EPOCHS}, imgsz={IMG_SIZE}, "
          f"batch={BATCH_SIZE}, device={DEVICE}\n")

    model = build_yolo_model(model_size="n", pretrained=True)
    results = model.train(
        data=f"{YOLO_DATA_ROOT}/data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="results/logs",
        name="yolov8n_baseline_gpu",
        seed=42,
        patience=10,  # ранняя остановка, если нет улучшений 10 эпох
    )
    # Примечание: ultralytics может сохранить результаты по пути
    # runs/detect/results/logs/yolov8n_baseline/ (в зависимости от версии
    # библиотеки) — точный путь будет показан в выводе как "Results saved to ...".
    print("\nОбучение завершено. Смотрите точный путь к результатам в строке "
          "'Results saved to ...' выше.")
    return results


if __name__ == "__main__":
    prepare_yolo_data()
    train_baseline()
