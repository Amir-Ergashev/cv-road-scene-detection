"""
make_comparison_plots.py

Строит итоговые сравнительные графики по всем 5 обученным моделям
для раздела 3.7 отчёта ("Результаты").

Метрики взяты из реальных логов обучения (терминальный вывод
train_yolo.py, train_faster_rcnn.py, train_ssd.py,
train_efficientdet.py, train_detr.py).

Запуск:
    python make_comparison_plots.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "results/plots"

# Финальные метрики на val, полученные после обучения каждой модели
RESULTS = {
    "YOLOv8":        {"mAP50": 0.338, "precision": 0.520, "recall": 0.336, "f1": None},
    "Faster R-CNN":  {"mAP50": 0.514, "precision": 0.277, "recall": 0.747, "f1": 0.404},
    "SSD":           {"mAP50": 0.0014, "precision": 0.012, "recall": 0.245, "f1": 0.023},
    "EfficientDet":  {"mAP50": 0.015, "precision": 0.156, "recall": 0.179, "f1": 0.167},
    "DETR":          {"mAP50": 0.114, "precision": 0.132, "recall": 0.631, "f1": 0.218},
}

# F1 для YOLOv8 не считался отдельной функцией (ultralytics использует
# собственный встроенный расчёт) — оценим по стандартной формуле из P/R
# для единообразия сравнения.
p, r = RESULTS["YOLOv8"]["precision"], RESULTS["YOLOv8"]["recall"]
RESULTS["YOLOv8"]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

MODELS = list(RESULTS.keys())
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def plot_map_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [RESULTS[m]["mAP50"] for m in MODELS]
    bars = ax.bar(MODELS, values, color=COLORS)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Сравнение моделей по mAP@0.5 (тестовая/валидационная выборка)")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                 f"{v:.3f}", ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_map50.png", dpi=120)
    plt.close()


def plot_precision_recall():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))
    width = 0.35

    precision = [RESULTS[m]["precision"] for m in MODELS]
    recall = [RESULTS[m]["recall"] for m in MODELS]

    ax.bar(x - width / 2, precision, width, label="Precision", color="#4C72B0")
    ax.bar(x + width / 2, recall, width, label="Recall", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение Precision и Recall по моделям")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_precision_recall.png", dpi=120)
    plt.close()


def plot_f1_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [RESULTS[m]["f1"] for m in MODELS]
    bars = ax.bar(MODELS, values, color=COLORS)
    ax.set_ylabel("F1-score")
    ax.set_title("Сравнение моделей по F1-score")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                 f"{v:.3f}", ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_f1.png", dpi=120)
    plt.close()


def plot_radar_comparison():
    """Радар-диаграмма для наглядного сравнения всех метрик сразу."""
    metrics = ["mAP50", "precision", "recall", "f1"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, model in enumerate(MODELS):
        values = [RESULTS[model][m] for m in metrics]
        # Нормализуем mAP50, так как у SSD оно близко к 0 и теряется на фоне остальных
        values_norm = values.copy()
        values_norm += values_norm[:1]
        ax.plot(angles, values_norm, label=model, color=COLORS[i], linewidth=2)
        ax.fill(angles, values_norm, color=COLORS[i], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Сравнение моделей по всем метрикам", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_radar.png", dpi=120)
    plt.close()


def plot_training_time_comparison():
    """Сравнение времени обучения по моделям (секунд на эпоху)."""
    time_per_epoch = {
        "YOLOv8\n(GPU)": 15.0,        # ~2.76 мин / 11 эпох * 60
        "Faster R-CNN": 141.0,
        "SSD": 44.0,
        "EfficientDet": 53.5,
        "DETR": 115.0,
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(time_per_epoch.keys(), time_per_epoch.values(), color=COLORS)
    ax.set_ylabel("Секунд на эпоху")
    ax.set_title("Сравнение скорости обучения (GPU NVIDIA RTX 4070)")
    for bar, v in zip(bars, time_per_epoch.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.0f}s", ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_training_time.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_map_comparison()
    plot_precision_recall()
    plot_f1_comparison()
    plot_radar_comparison()
    plot_training_time_comparison()
    print(f"Графики сохранены в {OUTPUT_DIR}/")
