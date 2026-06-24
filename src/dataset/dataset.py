"""
src/dataset/dataset.py

Загрузка и предобработка данных (подмножество COCO для детекции
объектов дорожной обстановки).

Реализуется на этапе "Подготовка данных" (день 6-7 плана).
"""

from pathlib import Path
from torch.utils.data import Dataset


class RoadSceneDataset(Dataset):
    """
    Датасет объектов дорожной обстановки на основе COCO subset.

    Ожидаемая структура data/raw/:
        data/raw/images/*.jpg
        data/raw/annotations/instances.json  (COCO format)
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []  # TODO: заполняется при парсинге аннотаций COCO

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: загрузка изображения + bbox-аннотаций, применение аугментаций
        raise NotImplementedError
