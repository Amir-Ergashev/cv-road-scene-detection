"""
src/dataset/coco_to_yolo.py

Конвертация отфильтрованных COCO-аннотаций (data/processed/instances_*.json)
в формат, который понимает YOLOv8 (ultralytics):
    images/*.jpg
    labels/*.txt  (class_id x_center y_center width height, нормализовано 0..1)

Реализуется на этапе "Baseline-модель" (день 7 плана).
"""

import json
import shutil
from pathlib import Path

from src.dataset.dataset import TARGET_CLASSES


def coco_to_yolo_format(coco_json_path: str, images_src_dir: str, output_dir: str,
                         class_list: list = None) -> None:
    """
    Конвертирует один COCO-сплит (train/val/test) в формат YOLO.

    images_src_dir — папка с исходными jpg (data/raw/images/val2017)
    output_dir — куда положить результат (например data/yolo_format/train)
    """
    class_list = class_list or TARGET_CLASSES

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_yolo_idx = {name: i for i, name in enumerate(class_list)}

    images_out = Path(output_dir) / "images"
    labels_out = Path(output_dir) / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    for img in coco["images"]:
        img_id = img["id"]
        w, h = img["width"], img["height"]
        file_name = img["file_name"]

        src = Path(images_src_dir) / file_name
        dst = images_out / file_name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

        label_lines = []
        for a in anns_by_image.get(img_id, []):
            cat_name = cat_id_to_name[a["category_id"]]
            if cat_name not in name_to_yolo_idx:
                continue
            yolo_cls = name_to_yolo_idx[cat_name]
            x, y, bw, bh = a["bbox"]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            label_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = labels_out / (Path(file_name).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print(f"Сконвертировано {len(coco['images'])} изображений: {coco_json_path} -> {output_dir}")


def write_yolo_yaml(output_path: str, dataset_root: str, class_list: list = None) -> None:
    """Создаёт data.yaml — конфигурационный файл датасета для ultralytics YOLO."""
    class_list = class_list or TARGET_CLASSES
    lines = [
        f"path: {dataset_root}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(class_list)}",
        f"names: {class_list}",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"YAML сохранён: {output_path}")
