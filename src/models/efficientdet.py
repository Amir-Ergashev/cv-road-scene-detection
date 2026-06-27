"""
src/models/efficientdet.py

Обёртка над EfficientDet (effdet + timm).
Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


def build_efficientdet_model(num_classes: int, pretrained: bool = True,
                              model_name: str = "efficientdet_d0", image_size: int = 512,
                              mode: str = "train"):
    """
    Создаёт модель EfficientDet для детекции объектов.

    num_classes: число классов БЕЗ фона (в отличие от torchvision-моделей,
                 effdet не использует отдельный класс для фона), т.е. для
                 наших 6 классов дорожной обстановки num_classes = 6.
    pretrained: если True — backbone (EfficientNet) предобучен на ImageNet.
    model_name: вариант EfficientDet, d0 — самый лёгкий (рекомендуется
                для обучения без мощного GPU-кластера).
    image_size: разрешение входных изображений (должно делиться на 128
                согласно архитектуре EfficientDet).
    mode: "train" — возвращает DetBenchTrain (нужны targets при forward,
          возвращает loss); "predict" — возвращает DetBenchPredict (только
          изображения на вход, возвращает [x1,y1,x2,y2,score,class] по
          каждому изображению).
    """
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=pretrained)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    if mode == "train":
        return DetBenchTrain(net, config)
    return DetBenchPredict(net)


def effdet_predictions_to_standard_format(raw_output, score_thresh: float = 0.2):
    """
    Конвертирует вывод DetBenchPredict (тензор [batch, 100, 6] с колонками
    [x1, y1, x2, y2, score, class]) в список словарей {boxes, scores, labels}
    — тот же формат, что используют Faster R-CNN/SSD (torchvision), что
    позволяет применить общую функцию evaluate_torchvision_model.
    """
    results = []
    for img_preds in raw_output:
        mask = img_preds[:, 4] >= score_thresh
        filtered = img_preds[mask]
        results.append({
            "boxes": filtered[:, :4],
            "scores": filtered[:, 4],
            "labels": filtered[:, 5].long(),
        })
    return results
