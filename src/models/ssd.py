from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def build_ssd_model(num_classes: int, pretrained: bool = True,
                     score_thresh: float = 0.2, detections_per_img: int = 100):
    weights_backbone = "DEFAULT" if pretrained else None
    return ssd300_vgg16(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )
