"""
src/dataset/effdet_dataset.py

Датасет в формате, который ожидает библиотека effdet (EfficientDet):
target — словарь с ключами bbox (список тензоров в формате [y1,x1,y2,x2]!),
cls (1-индексированные классы без фона), img_size, img_scale.

Реализуется на этапе "Остальные модели" (день 8-9 плана).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from src.dataset.dataset import TARGET_CLASSES


class EffDetCocoDataset(Dataset):
    """
    Оборачивает отфильтрованные COCO-аннотации (data/processed/instances_*.json)
    в формат effdet.

    effdet ожидает боксы в формате [y1, x1, y2, x2] (YXYX), а не привычном
    [x1, y1, x2, y2]. Классы 1-индексированы (без отдельного класса фона,
    в отличие от torchvision-моделей).
    """

    def __init__(self, images_dir: str, annotations_path: str,
                 class_list: list = None, image_size: int = 512):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        class_list = class_list or TARGET_CLASSES

        with open(annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        self.name_to_label = {name: i + 1 for i, name in enumerate(class_list)}
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
        orig_w, orig_h = img_info["width"], img_info["height"]

        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_tensor = F.to_tensor(image)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        anns = self.anns_by_image.get(image_id, [])
        boxes_yxyx, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1, y1 = x * scale_x, y * scale_y
            x2, y2 = (x + w) * scale_x, (y + h) * scale_y
            boxes_yxyx.append([y1, x1, y2, x2])  # effdet: YXYX
            labels.append(self.cat_id_to_label[a["category_id"]])

        if boxes_yxyx:
            boxes_t = torch.as_tensor(boxes_yxyx, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "bbox": boxes_t,
            "cls": labels_t,
            "img_size": torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            "img_scale": torch.tensor(1.0, dtype=torch.float32),
            "image_id": image_id,
        }
        return image_tensor, target


def effdet_collate_fn(batch):
    """
    effdet ожидает: images как единый тензор [B,C,H,W], а targets — словарь,
    где каждый ключ (bbox, cls, img_size, img_scale) содержит СПИСОК
    значений по изображениям (не стэкнутый тензор для bbox/cls, так как
    число объектов на изображении переменное).
    """
    images = torch.stack([item[0] for item in batch])
    targets = {
        "bbox": [item[1]["bbox"] for item in batch],
        "cls": [item[1]["cls"] for item in batch],
        "img_size": torch.stack([item[1]["img_size"] for item in batch]),
        "img_scale": torch.stack([item[1]["img_scale"] for item in batch]),
    }
    image_ids = [item[1]["image_id"] for item in batch]
    return images, targets, image_ids
