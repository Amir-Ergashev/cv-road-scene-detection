"""
Запуск:
python prepare_data.py
"""

import json
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.dataset.dataset import (
    TARGET_CLASSES,
    filter_coco_by_classes,
    print_dataset_stats,
    save_filtered_annotations,
    split_dataset,
    print_split_stats,
)

RAW_ANNOTATIONS = "data/raw/annotations/instances_val2017.json"
RAW_IMAGES_DIR = "data/raw/images/val2017"
PROCESSED_ANNOTATIONS = "data/processed/instances_filtered.json"
EXAMPLE_OUTPUT = "results/plots/example_with_bbox.png"


def visualize_example(filtered: dict, images_dir: str, output_path: str, seed: int = 42):
    random.seed(seed)
    cat_id_to_name = {c["id"]: c["name"] for c in filtered["categories"]}

    img_info = random.choice(filtered["images"])
    anns = [a for a in filtered["annotations"] if a["image_id"] == img_info["id"]]

    img_path = f"{images_dir}/{img_info['file_name']}"
    image = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    for a in anns:
        x, y, w, h = a["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, max(y - 5, 0), cat_id_to_name[a["category_id"]],
                 color="white", fontsize=10, backgroundcolor="green")
    ax.set_title(f"Пример: {img_info['file_name']} ({len(anns)} объектов)")
    ax.axis("off")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Визуализация сохранена: {output_path}")


def main():
    print(f"Фильтрация по классам: {TARGET_CLASSES}\n")

    filtered = filter_coco_by_classes(RAW_ANNOTATIONS, TARGET_CLASSES)
    print_dataset_stats(filtered)
    print()

    save_filtered_annotations(filtered, PROCESSED_ANNOTATIONS)
    visualize_example(filtered, RAW_IMAGES_DIR, EXAMPLE_OUTPUT)

    print("\nРазбиение на train/val/test (70/15/15)...")
    splits = split_dataset(filtered, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    print_split_stats(splits)

    for split_name, subset in splits.items():
        save_filtered_annotations(subset, f"data/processed/instances_{split_name}.json")


if __name__ == "__main__":
    main()
