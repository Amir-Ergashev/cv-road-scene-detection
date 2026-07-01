from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


def build_efficientdet_model(num_classes: int, pretrained: bool = True,
                              model_name: str = "efficientdet_d0", image_size: int = 512,
                              mode: str = "train"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=pretrained)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    if mode == "train":
        return DetBenchTrain(net, config)
    return DetBenchPredict(net)


def effdet_predictions_to_standard_format(raw_output, score_thresh: float = 0.2):
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
