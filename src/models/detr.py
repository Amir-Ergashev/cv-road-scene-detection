from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig


def build_detr_model(num_classes: int, pretrained: bool = True,
                      checkpoint: str = "facebook/detr-resnet-50"):

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
    return DetrImageProcessor.from_pretrained(checkpoint)
