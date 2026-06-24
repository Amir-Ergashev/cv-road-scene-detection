"""
src/utils/utils.py

Вспомогательные функции: визуализация предсказаний, логирование,
работа с конфигами.
"""

import yaml


def load_config(path: str) -> dict:
    """Загружает конфигурацию эксперимента из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def visualize_predictions(image, predictions, ground_truth=None, save_path=None):
    """
    Визуализирует предсказанные bounding box'ы (и опционально ground truth)
    на изображении. Используется на этапе анализа ошибок (день 11).
    """
    raise NotImplementedError
