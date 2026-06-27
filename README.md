# CV Project: детекция и классификация объектов дорожной обстановки

Учебный проект полного цикла в области компьютерного зрения.
Выполняется в рамках учебной практики БВТ.

## Предметная область

Интеллектуальные транспортные системы — детектирование и классификация
объектов на изображениях дорожной обстановки.

**Классы:** person, car, bicycle, motorcycle, bus, traffic light.

## Датасет

Подмножество COCO 2017 (val2017), отфильтрованное по 6 целевым классам:
**2972 изображения**, **14545 объектов**. Разбиение train/val/test:
2080 / 445 / 447 изображений (`seed=42`, раздел 3.4 отчёта).

## Используемые модели и результаты

Обучено и сравнено 5 современных моделей детекции объектов:

| Модель | mAP@0.5 | Precision | Recall | F1-score |
|---|---|---|---|---|
| **YOLOv8n** | 0.338 | **0.520** | 0.336 | **0.405** |
| **Faster R-CNN** | **0.514** | 0.277 | **0.747** | 0.404 |
| SSD300 | 0.001 | 0.012 | 0.245 | 0.023 |
| EfficientDet-D0 | 0.015 | 0.156 | 0.179 | 0.167 |
| DETR | 0.114 | 0.132 | 0.631 | 0.218 |

Лучший результат по F1-score — **YOLOv8n** и **Faster R-CNN** (практически
равный итог разными архитектурными путями). Подробный анализ — в разделе
"Обсуждение" отчёта (`report/07_discussion.md`).

## Структура проекта

```
cv-project-template/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   └── default.yaml          # гиперпараметры и пути к данным
├── data/
│   ├── raw/                   # исходные изображения и аннотации COCO
│   └── processed/             # отфильтрованные данные, train/val/test
├── src/
│   ├── dataset/
│   │   ├── dataset.py             # фильтрация COCO, train/val/test split
│   │   ├── coco_to_yolo.py        # конвертация в формат YOLO
│   │   ├── torchvision_dataset.py # датасет для Faster R-CNN/SSD
│   │   ├── effdet_dataset.py      # датасет для EfficientDet
│   │   └── detr_dataset.py        # датасет для DETR
│   ├── models/
│   │   ├── yolo.py
│   │   ├── faster_rcnn.py
│   │   ├── ssd.py
│   │   ├── efficientdet.py
│   │   └── detr.py
│   ├── training/          # вспомогательная логика обучения
│   ├── evaluation/
│   │   └── metrics.py     # mAP, Precision, Recall, F1
│   └── utils/
├── notebooks/
│   └── exploration.ipynb  # исследовательский анализ данных (EDA)
├── results/
│   ├── plots/              # графики сравнения моделей
│   └── logs/                # логи и веса обученных моделей
├── report/                 # полный текстовый отчёт по разделам
│   ├── 02_problem_statement.md
│   ├── 03_literature_review.md
│   ├── 04_methods_and_models.md
│   ├── 05_experiments.md
│   ├── 06_results.md
│   ├── 07_discussion.md
│   └── 08_conclusion.md
├── prepare_data.py          # фильтрация датасета + train/val/test split
├── train_yolo.py
├── train_faster_rcnn.py
├── train_ssd.py
├── train_efficientdet.py
├── train_detr.py
└── make_comparison_plots.py # итоговые сравнительные графики
```

## Установка

```bash
git clone https://github.com/Amir-Ergashev/cv-road-scene-detection.git
cd cv-road-scene-detection
pip install -r requirements.txt
```

Для обучения на GPU (рекомендуется) дополнительно установите PyTorch с
поддержкой CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Подготовка данных

1. Скачайте COCO val2017 images и аннотации (instructions in
   `data/raw/`), либо используйте собственное подмножество в том же
   формате.
2. Запустите фильтрацию и разбиение датасета:

```bash
python prepare_data.py
```

Создаст `data/processed/instances_{filtered,train,val,test}.json` и
визуализацию примера с bounding box'ами в `results/plots/`.

## Обучение моделей

Каждая модель обучается отдельным скриптом (использует общие данные из
`data/processed/`):

```bash
python train_yolo.py           # YOLOv8n, ~3 мин (GPU)
python train_faster_rcnn.py    # Faster R-CNN, ~35 мин (GPU)
python train_ssd.py            # SSD300, ~11 мин (GPU)
python train_efficientdet.py   # EfficientDet-D0, ~13 мин (GPU)
python train_detr.py           # DETR, ~16 мин (GPU)
```

Время приведено для NVIDIA RTX 4070 Laptop GPU. На CPU обучение займёт
существенно дольше — рекомендуется GPU или Google Colab.

## Построение сравнительных графиков

После обучения всех моделей (или с готовыми метриками из отчёта):

```bash
python make_comparison_plots.py
```

Сохранит 5 сравнительных графиков (mAP@0.5, Precision/Recall, F1,
радар-диаграмма, скорость обучения) в `results/plots/`.

## Отчёт

Полный научный отчёт расположен в папке `report/` по разделам:

1. Постановка задачи — `02_problem_statement.md`
2. Обзор литературы (17 источников) — `03_literature_review.md`
3. Методы и модели — `04_methods_and_models.md`
4. Эксперименты — `05_experiments.md`
5. Результаты — `06_results.md`
6. Обсуждение — `07_discussion.md`
7. Заключение — `08_conclusion.md`

## Окружение

- Python 3.13
- PyTorch 2.6.0 (CUDA 12.4)
- torchvision, ultralytics, effdet, transformers, torchmetrics
- Обучение проводилось на NVIDIA RTX 4070 Laptop GPU
