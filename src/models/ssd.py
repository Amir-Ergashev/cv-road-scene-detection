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

    num_classes: число классов ВКЛЮЧАЯ фон (background), т.е. для наших
                 6 классов дорожной обстановки num_classes = 7.
    pretrained: если True — backbone предобучен на ImageNet (модель в целом
                не предобучена на COCO с нужным числом классов, поэтому
                голова обучается с нуля в любом случае).
    score_thresh: порог уверенности для сохранения предсказания. Дефолт
                  torchvision (0.01) слишком низкий для нашей задачи — модель
                  выдаёт сотни предсказаний с минимальной уверенностью на
                  изображение, что обнуляет Precision при оценке метрик.
    detections_per_img: максимум предсказаний на одно изображение (дефолт
                  torchvision — 200, мы используем 100, как у Faster R-CNN).

    Примечание: SSD300 работает с фиксированным разрешением 300x300 —
    оно заметно меньше, чем у Faster R-CNN (480-800px), что даёт более
    быстрое обучение и инференс за счёт уступки в точности на мелких
    объектах (известное ограничение SSD, см. раздел 3.2 отчёта).
    """
    weights_backbone = "DEFAULT" if pretrained else None
    return ssd300_vgg16(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )
