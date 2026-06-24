"""
main.py — точка входа в проект.

Использование (см. раздел 11.4 методички):
    python main.py --model yolo
    python main.py --model faster_rcnn
    python main.py --model ssd
    python main.py --model efficientdet
    python main.py --model detr
"""

import argparse

from src.utils.utils import load_config

MODEL_BUILDERS = {
    "yolo": "src.models.yolo.build_yolo_model",
    "faster_rcnn": "src.models.faster_rcnn.build_faster_rcnn_model",
    "ssd": "src.models.ssd.build_ssd_model",
    "efficientdet": "src.models.efficientdet.build_efficientdet_model",
    "detr": "src.models.detr.build_detr_model",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск обучения CV-модели")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_BUILDERS.keys()),
        help="Какую модель обучать",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Путь к файлу конфигурации",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"Запуск модели: {args.model}")
    print(f"Конфигурация: {config['project']['name']}")

    # TODO по мере реализации:
    # 1. Создать датасет (src/dataset/dataset.py)
    # 2. Построить модель (MODEL_BUILDERS[args.model])
    # 3. Обучить (src/training/train.py)
    # 4. Оценить (src/evaluation/metrics.py)
    raise NotImplementedError("Пайплайн будет реализован поэтапно согласно плану")


if __name__ == "__main__":
    main()
