"""
src/models/yolo.py

Обёртка над YOLOv8 (ultralytics) — baseline-модель проекта.
Реализуется на этапе "Baseline-модель" (день 7 плана).
"""

from ultralytics import YOLO


def build_yolo_model(model_size: str = "n", pretrained: bool = True) -> YOLO:
    """
    Создаёт модель YOLOv8 для детекции объектов.

    model_size: "n" (nano, самая лёгкая, рекомендуется для CPU),
                "s" (small), "m" (medium) и т.д.
    pretrained: если True — загружает веса, предобученные на полном COCO
                (transfer learning); если False — обучение архитектуры с нуля.
    """
    if pretrained:
        return YOLO(f"yolov8{model_size}.pt")
    return YOLO(f"yolov8{model_size}.yaml")
