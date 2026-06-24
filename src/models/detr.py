"""
src/models/detr.py

Обёртка над DETR (Transformer-based detector, HuggingFace transformers).
Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""


def build_detr_model(num_classes: int, pretrained: bool = True):
    """
    Создаёт модель DETR для детекции объектов.

    TODO: transformers.DetrForObjectDetection.from_pretrained(...)
    """
    raise NotImplementedError
