"""
src/training/train.py

Логика обучения моделей: цикл обучения, оптимизатор, логирование.
Реализуется поэтапно начиная с baseline (день 7) и далее для всех моделей.
"""


def train_model(model, train_loader, val_loader, config):
    """
    Обучает переданную модель согласно конфигурации (configs/default.yaml).

    TODO:
    - настройка optimizer (Adam/SGD) и lr_scheduler по config
    - цикл по эпохам с логированием loss
    - сохранение чекпоинтов в results/logs/
    - валидация после каждой эпохи
    """
    raise NotImplementedError
