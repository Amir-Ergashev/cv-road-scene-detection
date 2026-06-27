"""
src/dataset/detr_dataset.py

Датасет в формате, который ожидает DetrForObjectDetection (HuggingFace
transformers): использует DetrImageProcessor для подготовки pixel_values,
pixel_mask и labels (нормализованные боксы в формате cxcywh).

Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.dataset.dataset import TARGET_CLASSES


class DetrCocoDataset(Dataset):
    """
    Оборачивает отфильтрованные COCO-аннотации (data/processed/instances_*.json)
    в формат, ожидаемый DetrImageProcessor. category_id переиндексируются
    в диапазон [0, num_classes-1] (без отдельного класса фона —
    "no object" обрабатывается DETR автоматически внутри головы).
    """

    def __init__(self, images_dir: str, annotations_path: str,
                 processor, class_list: list = None):
        self.images_dir = Path(images_dir)
        self.processor = processor
        class_list = class_list or TARGET_CLASSES

        with open(annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        # DETR: классы с индексации 0 (без отдельного фона)
        self.name_to_label = {name: i for i, name in enumerate(class_list)}
        self.cat_id_to_label = {
            cid: self.name_to_label[name]
            for cid, name in cat_id_to_name.items()
            if name in self.name_to_label
        }

        self.anns_by_image = {}
        for a in coco["annotations"]:
            if a["category_id"] in self.cat_id_to_label:
                self.anns_by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")

        anns = self.anns_by_image.get(image_id, [])
        coco_annotations = [
            {
                "bbox": a["bbox"],  # [x, y, w, h] — формат, который ожидает processor
                "category_id": self.cat_id_to_label[a["category_id"]],
                "area": a["bbox"][2] * a["bbox"][3],
                "iscrowd": 0,
            }
            for a in anns
        ]

        target = {"image_id": image_id, "annotations": coco_annotations}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return pixel_values, labels


def detr_collate_fn(batch):
    """
    DETR-изображения после ресайза могут иметь разный размер (processor
    сохраняет соотношение сторон) — нужен паддинг до общего максимального
    размера в батче. pixel_mask показывает модели, какие пиксели реальные
    (1), а какие — паддинг (0).

    Реализован вручную (без processor.pad), так как API паддинга менялся
    между версиями библиотеки transformers.
    """
    pixel_values_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)

    padded_pixel_values = torch.zeros(len(batch), 3, max_h, max_w)
    pixel_mask = torch.zeros(len(batch), max_h, max_w, dtype=torch.long)

    for i, pv in enumerate(pixel_values_list):
        h, w = pv.shape[1], pv.shape[2]
        padded_pixel_values[i, :, :h, :w] = pv
        pixel_mask[i, :h, :w] = 1

    return padded_pixel_values, pixel_mask, labels
