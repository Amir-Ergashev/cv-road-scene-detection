"""
src/evaluation/metrics.py

Вычисление метрик качества: mAP, Precision, Recall, F1-score.
Реализуется на этапе "Эксперименты" / "Анализ результатов" (день 10-11 плана).
"""


def compute_map(predictions, ground_truth, iou_threshold: float = 0.5):
    """Вычисляет mean Average Precision (mAP) при заданном IoU threshold."""
    raise NotImplementedError


def compute_precision_recall_f1(predictions, ground_truth, iou_threshold: float = 0.5):
    """Вычисляет Precision, Recall и F1-score."""
    raise NotImplementedError
