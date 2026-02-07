import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve
)

# Global metrics
def compute_global_metrics(labels, preds,task_type="val"):
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    if task_type=="val":
        return {
            "val/precision_macro": p,
            "val/recall_macro": r,
            "val/f1_macro": f1
        }
    elif task_type=="test_tta":
        return {
            "test_tta/precision_macro": p,
            "test_tta/recall_macro": r,
            "test_tta/f1_macro": f1
        }
    elif task_type=="test":
        return {
            "test/precision_macro": p,
            "test/recall_macro": r,
            "test/f1_macro": f1
        }
    else:
        raise(ValueError)

# Per-class metrics
def compute_per_class_metrics(labels, probs, class_names):
    """
    Calcule les métriques par classe et prépare les courbes PR.
    
    Returns:
        scalars (dict): Dictionnaire plat pour wandb.log()
                        ex: {'class/anger/f1': 0.5, ...}
        curves (dict): Données brutes pour tracer les courbes
                       ex: {'anger': (precision_array, recall_array)}
    """
    scalars = {}
    curves = {}
    
    preds = probs.argmax(axis=1)
    
    p, r, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        zero_division=0
    )

    for i, name in enumerate(class_names):
        scalars[f"class/{name}/precision"] = p[i]
        scalars[f"class/{name}/recall"] = r[i]
        scalars[f"class/{name}/f1"] = f1[i]

        y_true = (labels == i).astype(int)
        y_score = probs[:, i]

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    
        curves[name] = (precision_curve, recall_curve)

    return scalars, curves


# Top_k accuracy
def topk_accuracy(logits, targets, topk=(3,5),task_type="test"):
    """
    Calcule la précision Top-K.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            if task_type=="test":
                res[f"test/top{k}_acc"] = (correct_k / batch_size).item()
            elif task_type=="test_tta":
                res[f"test_tta/top{k}_acc"] = (correct_k / batch_size).item()
            else:
                raise(ValueError)

        return res