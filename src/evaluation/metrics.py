import torch
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_precision_recall_f1(all_preds, all_targets, iou_threshold: float = 0.5) -> dict:
    tp, fp, fn = 0, 0, 0

    for preds, targets in zip(all_preds, all_targets):
        pred_boxes = preds["boxes"]
        pred_labels = preds["labels"]
        pred_scores = preds.get("scores")
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        matched_gt = set()

        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        # Сортируем индексы предсказаний по убыванию score (если score нет —
        # оставляем исходный порядок).
        if pred_scores is not None:
            order = torch.argsort(pred_scores, descending=True).tolist()
        else:
            order = list(range(len(pred_boxes)))

        ious = box_iou(pred_boxes, gt_boxes)

        for pred_idx in order:
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if pred_labels[pred_idx] != gt_labels[gt_idx]:
                    continue
                iou_val = ious[pred_idx, gt_idx].item()
                if iou_val > best_iou:
                    best_iou, best_gt_idx = iou_val, gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_map(all_preds, all_targets, iou_thresholds: list = None) -> dict:
    iou_thresholds = iou_thresholds or [0.5]
    metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)
    metric.update(all_preds, all_targets)
    result = metric.compute()
    return {
        "map": result["map"].item(),
        "map_50": result["map_50"].item(),
    }


@torch.no_grad()
def evaluate_torchvision_model(model, data_loader, device: str, class_list: list) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            all_preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
            all_targets.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

    map_metrics = compute_map(all_preds, all_targets, iou_thresholds=[0.5])
    prf1_metrics = compute_precision_recall_f1(all_preds, all_targets, iou_threshold=0.5)

    return {
        "mAP50": map_metrics["map_50"],
        "mAP": map_metrics["map"],
        "precision": prf1_metrics["precision"],
        "recall": prf1_metrics["recall"],
        "f1": prf1_metrics["f1"],
    }
