from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


def resolve_random_seed(seed: Optional[int]) -> int:
    if seed is None or seed < 0:
        return random.SystemRandom().randint(1, 2**31 - 1)
    return seed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    if logits.ndim == 2 and logits.size(-1) == 2:
        positive_scores = torch.softmax(logits, dim=-1)[:, 1]
    elif logits.ndim == 1:
        positive_scores = logits
    else:
        raise ValueError(f"Unsupported logits shape for binary metrics: {tuple(logits.shape)}")

    predictions = (positive_scores >= threshold).long()
    labels = labels.long()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    precision_neg = tn / max(tn + fn, 1)
    recall_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * precision_neg * recall_neg / max(precision_neg + recall_neg, 1e-12)
    macro_f1 = (f1 + f1_neg) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "rumor_precision": precision,
        "rumor_recall": recall,
        "rumor_f1": f1,
        "rumor_support": tp + fn,
        "non_rumor_precision": precision_neg,
        "non_rumor_recall": recall_neg,
        "non_rumor_f1": f1_neg,
        "non_rumor_support": tn + fp,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def compute_multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    if logits.ndim != 2 or logits.size(-1) < 2:
        raise ValueError(f"Unsupported logits shape for multiclass metrics: {tuple(logits.shape)}")

    predictions = torch.argmax(logits, dim=-1).long()
    labels = labels.long()
    num_classes = int(max(logits.size(-1), labels.max().item() + 1 if labels.numel() else 0))

    correct = int((predictions == labels).sum().item())
    total = int(labels.numel())
    accuracy = correct / max(total, 1)

    precisions = []
    recalls = []
    f1s = []
    metrics: Dict[str, float] = {"accuracy": accuracy}
    for class_index in range(num_classes):
        pred_is_class = predictions == class_index
        label_is_class = labels == class_index
        tp = int((pred_is_class & label_is_class).sum().item())
        fp = int((pred_is_class & ~label_is_class).sum().item())
        fn = int((~pred_is_class & label_is_class).sum().item())
        support = int(label_is_class.sum().item())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        metrics[f"class_{class_index}_precision"] = precision
        metrics[f"class_{class_index}_recall"] = recall
        metrics[f"class_{class_index}_f1"] = f1
        metrics[f"class_{class_index}_support"] = float(support)

    macro_precision = float(sum(precisions) / max(len(precisions), 1))
    macro_recall = float(sum(recalls) / max(len(recalls), 1))
    macro_f1 = float(sum(f1s) / max(len(f1s), 1))
    metrics.update(
        {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "macro_f1": macro_f1,
        }
    )
    return metrics


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 2):
        return compute_binary_metrics(logits, labels, threshold=threshold)
    return compute_multiclass_metrics(logits, labels)


def save_json(data: Dict[str, object], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

