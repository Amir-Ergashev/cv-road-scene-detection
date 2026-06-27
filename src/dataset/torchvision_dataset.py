"""
src/dataset/torchvision_dataset.py

Датасет в формате, который ожидают torchvision detection-модели
(Faster R-CNN, SSD): target — словарь с тензорами boxes [x1,y1,x2,y2],
labels, image_id.

Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from src.dataset.dataset import TARGET_CLASSES


class TorchvisionCocoDataset(Dataset):
    """
    Оборачивает отфильтрованные COCO-аннотации (data/processed/instances_*.json)
    в формат torchvision detection API.

    Важно: метки классов сдвинуты на +1, так как класс 0 зарезервирован
    под фон (background) в torchvision Faster R-CNN / SSD.
    """

    def __init__(self, images_dir: str, annotations_path: str, class_list: list = None):
        self.images_dir = Path(images_dir)
        class_list = class_list or TARGET_CLASSES

        with open(annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        self.name_to_label = {name: i + 1 for i, name in enumerate(class_list)}
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}
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
        image_tensor = F.to_tensor(image)

        anns = self.anns_by_image.get(image_id, [])
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[a["category_id"]])

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    """Нужен для DataLoader: изображения разного размера, нельзя стэкать в тензор."""
    return tuple(zip(*batch))
