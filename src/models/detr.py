"""
src/models/detr.py

Обёртка над DETR (Transformer-based detector, HuggingFace transformers).
Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig


def build_detr_model(num_classes: int, pretrained: bool = True,
                      checkpoint: str = "facebook/detr-resnet-50"):
    """
    Создаёт модель DETR для детекции объектов.

    num_classes: число классов без фона (DETR обрабатывает "no object"
                 отдельно внутри головы), для 6 классов дорожной
                 обстановки num_classes = 6.
    pretrained: если True — загружает веса, предобученные на COCO,
                с заменой головы классификации под наше число классов.
    checkpoint: предобученный чекпойнт с HuggingFace Hub.

    Конфиг facebook/detr-resnet-50 на Hub хранит "dilation": null, что
    не проходит проверку типов в новых версиях transformers (ожидается
    bool) — поэтому конфиг загружается отдельно и поле исправляется
    перед созданием DetrConfig.
    """
    if pretrained:
        config_dict, _ = DetrConfig.get_config_dict(checkpoint, revision="no_timm")
        if config_dict.get("dilation") is None:
            config_dict["dilation"] = False
        config_dict["num_labels"] = num_classes
        config = DetrConfig.from_dict(config_dict)

        return DetrForObjectDetection.from_pretrained(
            checkpoint,
            revision="no_timm",
            config=config,
            ignore_mismatched_sizes=True,
        )

    config = DetrConfig(num_labels=num_classes)
    return DetrForObjectDetection(config)


def build_detr_processor(checkpoint: str = "facebook/detr-resnet-50"):
    """Создаёт DetrImageProcessor для подготовки данных (pixel_values, labels)."""
    return DetrImageProcessor.from_pretrained(checkpoint)
