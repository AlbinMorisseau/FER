import yaml
import torch
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Imports projet
from src.data.datamodule import EmotionDataModule
from src.models.convnextiny import build_convnext
from src.models.resnet50 import build_resnet
from src.models.vggface import build_vggface
from src.training.metrics import (
    compute_per_class_metrics, 
    compute_global_metrics, 
    topk_accuracy
)

EMOTION_LABELS = {
    0: "surprised",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral",
}

# Utils
def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "convnext_tiny": return build_convnext(cfg)
    if name == "resnet50": return build_resnet(cfg)
    if name == "vggface": return build_vggface(cfg)
    raise ValueError(f"Unknown model: {name}")

# Evaluation
def evaluate(args):

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_cfg = cfg["logging"]
    use_tta = log_cfg.get("log_tta", False) 

    ckpt_path = Path("checkpoints") / f"{cfg['experiment']['name']}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")
    
    dm = EmotionDataModule(**cfg["data"])
    dm.setup()
    test_loader = dm.test_dataloader()
    class_labels = dm.test_dataset.classes
    class_names = [EMOTION_LABELS[int(lab)] for lab in class_labels]

    model = build_model(cfg).to(device)
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    wandb_id = checkpoint.get("wandb_id", None)


    run = wandb.init(
        project=cfg["experiment"]["project"],
        name=f"{cfg['experiment']['name']}",
        config=cfg,
        job_type="evaluation",
        tags=["eval", cfg["model"]["name"]],
        resume="must" if wandb_id else None,
        id=wandb_id
    )
    
    all_probs, all_preds, all_labels = [], [], []
    all_probs_tta, all_preds_tta = [], [] 
    losses = []
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Running Inference (TTA: {use_tta})...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

            losses.append(loss.item())
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            #TTA
            if use_tta:
                x_flipped = torch.flip(x, [3]) 
                logits_flipped = model(x_flipped)
                probs_flipped = torch.softmax(logits_flipped, dim=1)

                avg_probs = (probs + probs_flipped) / 2.0
                all_probs_tta.append(avg_probs.cpu().numpy())
                all_preds_tta.extend(avg_probs.argmax(1).cpu().numpy())


    all_probs = np.concatenate(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics_log = {}
    
    metrics_log["test/loss"] = np.mean(losses)
    metrics_log["test/accuracy"] = accuracy_score(all_labels, all_preds)
    metrics_log.update(compute_global_metrics(all_labels, all_preds,"test"))

    if use_tta:
        all_probs_tta = np.concatenate(all_probs_tta)
        all_preds_tta = np.array(all_preds_tta)
        metrics_log["test_tta/accuracy"] = accuracy_score(all_labels, all_preds_tta)
        metrics_log.update(compute_global_metrics(all_labels, all_preds_tta, "test_tta"))
    
    #Per Class
    per_class_scalars, per_class_curves = compute_per_class_metrics(
        all_labels, all_probs, class_names
    )
    metrics_log.update(per_class_scalars)

    #Top-K 
    if log_cfg["log_topk"]:
        topk_res = topk_accuracy(torch.tensor(all_probs), torch.tensor(all_labels), topk=(3,5))
        metrics_log.update(topk_res)
        if use_tta:
            topk_tta = topk_accuracy(torch.tensor(all_probs_tta), torch.tensor(all_labels), topk=(3,5),task_type="test_tta")
            metrics_log.update({f"{k}_tta": v for k, v in topk_tta.items()})

    wandb.log(metrics_log)
    wandb.run.summary.update(metrics_log)
    
    def log_cm(labels, preds, name,task_type="test"):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if task_type=="test":
            wandb.log({f"test/{name}": wandb.Image(fig)})
        elif task_type=="test_tta":
            wandb.log({f"test_tta/{name}": wandb.Image(fig)})
        else:
            raise(ValueError)
        plt.close(fig)

    log_cm(all_labels, all_preds, "confusion_matrix",task_type="test")
    if use_tta:
        log_cm(all_labels, all_preds_tta, "confusion_matrix_tta",task_type="test_tta")

    #PR Curves 
    if log_cfg["log_pr_curves"]:
        print("Logging PR Curves...")
        for cls_name, (prec, rec) in per_class_curves.items():
            stride = max(1, len(prec) // 1000)
            data = [[x, y] for (x, y) in zip(rec[::stride], prec[::stride])]
            table = wandb.Table(data=data, columns=["Recall", "Precision"])
            wandb.log({
                f"pr_curve/{cls_name}": wandb.plot.line(
                    table, "Recall", "Precision", title=f"PR Curve - {cls_name}"
                )
            })

    wandb.finish()
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args)