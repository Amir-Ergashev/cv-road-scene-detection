"""
src/dataset/dataset.py

Загрузка и предобработка данных (подмножество COCO для детекции
объектов дорожной обстановки).

Реализуется на этапе "Подготовка данных" (день 6-7 плана).
"""

import json
import random
from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image

# Классы, относящиеся к дорожной обстановке (раздел 3.1 темы проекта)
TARGET_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]


def filter_coco_by_classes(annotations_path: str, target_classes: list) -> dict:
    """
    Загружает COCO-аннотации и оставляет только изображения и объекты,
    относящиеся к заданному списку классов.

    Возвращает словарь в том же формате COCO (images, annotations, categories),
    но отфильтрованный.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    name_to_id = {c["name"]: c["id"] for c in coco["categories"] if c["name"] in target_classes}
    target_cat_ids = set(name_to_id.values())

    missing = set(target_classes) - set(name_to_id.keys())
    if missing:
        print(f"ВНИМАНИЕ: классы не найдены в датасете: {missing}")

    filtered_annotations = [a for a in coco["annotations"] if a["category_id"] in target_cat_ids]
    image_ids_with_target = {a["image_id"] for a in filtered_annotations}
    filtered_images = [img for img in coco["images"] if img["id"] in image_ids_with_target]
    filtered_categories = [c for c in coco["categories"] if c["id"] in target_cat_ids]

    return {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }


def print_dataset_stats(filtered: dict) -> None:
    """Печатает базовую статистику по отфильтрованному датасету."""
    cat_id_to_name = {c["id"]: c["name"] for c in filtered["categories"]}
    counts = Counter(cat_id_to_name[a["category_id"]] for a in filtered["annotations"])

    print(f"Изображений после фильтрации: {len(filtered['images'])}")
    print(f"Аннотаций после фильтрации: {len(filtered['annotations'])}")
    print("Распределение объектов по классам:")
    for name, cnt in counts.most_common():
        print(f"  {name}: {cnt}")


def save_filtered_annotations(filtered: dict, output_path: str) -> None:
    """Сохраняет отфильтрованные аннотации в data/processed/ в формате COCO."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f)
    print(f"Сохранено: {output_path}")


def split_dataset(filtered: dict, train_ratio: float = 0.7, val_ratio: float = 0.15,
                   test_ratio: float = 0.15, seed: int = 42) -> dict:
    """
    Разбивает отфильтрованный COCO-датасет на train/val/test по изображениям
    (раздел 3.4 методички: train_split=0.7, val_split=0.15, test_split=0.15).

    Разбиение выполняется на уровне изображений (не аннотаций), чтобы все
    объекты одного изображения попадали в один и тот же сплит. seed
    фиксирован для воспроизводимости экспериментов.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Доли должны суммироваться в 1.0"

    rng = random.Random(seed)
    image_ids = [img["id"] for img in filtered["images"]]
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])

    def make_subset(ids: set) -> dict:
        images = [img for img in filtered["images"] if img["id"] in ids]
        annotations = [a for a in filtered["annotations"] if a["image_id"] in ids]
        return {"images": images, "annotations": annotations, "categories": filtered["categories"]}

    return {
        "train": make_subset(train_ids),
        "val": make_subset(val_ids),
        "test": make_subset(test_ids),
    }


def print_split_stats(splits: dict) -> None:
    """Печатает статистику по получившимся train/val/test сплитам."""
    for name, subset in splits.items():
        print(f"{name}: {len(subset['images'])} изображений, {len(subset['annotations'])} аннотаций")


class RoadSceneDataset(Dataset):
    """
    Датасет объектов дорожной обстановки на основе отфильтрованного COCO subset.

    Ожидаемая структура:
        data/raw/images/val2017/*.jpg
        data/processed/instances_filtered.json  (создаётся filter_coco_by_classes)
    """

    def __init__(self, images_dir: str, annotations_path: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        with open(annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        # Группируем аннотации по image_id
        self.annotations_by_image = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")

        anns = self.annotations_by_image.get(image_id, [])
        boxes = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]

        if self.transform:
            image = self.transform(image)

        return image, {"boxes": boxes, "labels": labels, "image_id": image_id}
