"""
src/models/ssd.py

Обёртка над SSD (torchvision).
Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def build_ssd_model(num_classes: int, pretrained: bool = True,
                     score_thresh: float = 0.2, detections_per_img: int = 100):
    """
    Создаёт модель SSD300 (VGG16 backbone) для детекции объектов.

    num_classes: число классов включая фон (background), для наших
                 6 классов дорожной обстановки num_classes = 7.
    pretrained: если True — backbone предобучен на ImageNet.
    score_thresh: порог уверенности для сохранения предсказания. Дефолт
                  torchvision (0.01) слишком низкий — модель выдаёт сотни
                  слабых предсказаний на изображение.
    detections_per_img: максимум предсказаний на изображение (дефолт
                  torchvision — 200, здесь 100, как у Faster R-CNN).

    SSD300 работает с фиксированным разрешением 300x300 — меньше, чем
    у Faster R-CNN (480-800px), что даёт быстрее обучение и инференс.
    """
    weights_backbone = "DEFAULT" if pretrained else None
    return ssd300_vgg16(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )
