from ultralytics import YOLO


def build_yolo_model(model_size: str = "n", pretrained: bool = True) -> YOLO:
    if pretrained:
        return YOLO(f"yolov8{model_size}.pt")
    return YOLO(f"yolov8{model_size}.yaml")
