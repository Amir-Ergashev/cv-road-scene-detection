"""
src/models/faster_rcnn.py

Обёртка над Faster R-CNN (torchvision).
Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_model(num_classes: int, pretrained: bool = True,
                             min_size: int = 480, max_size: int = 640):
    """
    Создаёт модель Faster R-CNN для детекции объектов.

    num_classes: число классов ВКЛЮЧАЯ фон (background), т.е. для наших
                 6 классов дорожной обстановки num_classes = 7.
    pretrained: если True — backbone и веса предобучены на полном COCO
                (transfer learning), голова классификатора всегда
                заменяется под наше число классов.
    min_size/max_size: к какому разрешению ресайзятся изображения внутри
                модели. По умолчанию в torchvision это 800/1333 — заметно
                больше, чем у YOLOv8 (640), что сильно замедляет обучение.
                Уменьшение до 480/640 даёт значительное ускорение почти
                без потери качества для большинства задач детекции.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
