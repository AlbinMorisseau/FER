import yaml
import torch
import wandb
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.data.datamodule import EmotionDataModule
from src.models.convnextiny import build_convnext
from src.models.resnet50 import build_resnet
from src.models.vggface import build_vggface
from src.training.optim import build_optimizer, build_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.ema import EMA
from src.utils.freeze import freeze_backbone, unfreeze_backbone,freeze_backbone_partial
from src.utils.seed import set_seed

#### Utils #####
def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "convnext_tiny": return build_convnext(cfg)
    if name == "resnet50": return build_resnet(cfg)
    if name == "vggface": return build_vggface(cfg)
    raise ValueError(f"Unknown model: {name}")

def update_cfg_from_wandb(cfg):
    """Permet aux Sweeps W&B d'écraser la config YAML."""
    if wandb.run and wandb.run.sweep_id:
        for k, v in wandb.config.items():
            keys = k.split('.')
            sub_cfg = cfg
            for key in keys[:-1]:
                sub_cfg = sub_cfg.setdefault(key, {})
            sub_cfg[keys[-1]] = v
    return cfg

def mixup_data(x, y, alpha, device):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def run_validation(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return np.mean(losses), acc, f1

#### Train ####
def train(args):
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialisation W&B
    run = wandb.init(
        project=cfg["experiment"]["project"],
        name=cfg["experiment"]["name"],
        config=cfg,
        job_type="train"
    )

    # Update of config for sweeps
    cfg = update_cfg_from_wandb(cfg)
    set_seed(cfg["experiment"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Data & Model
    dm = EmotionDataModule(**cfg["data"])
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = build_model(cfg).to(device)
    
    transfer_cfg = cfg["transfer"]
    
    if transfer_cfg["mode"] == "freeze" and transfer_cfg["freeze_epochs"] > 0:
        
        freeze_level = transfer_cfg.get("freeze_level", "all") 
        model_name = cfg["model"]["name"]

        if freeze_level == "partial":
            freeze_backbone_partial(model, model_name)
        else:
            freeze_backbone(model)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg["regularization"]["label_smoothing"])

    # EMA
    ema_cfg = cfg["regularization"]["ema"]
    ema = (
        EMA(model, decay=ema_cfg["decay"])
        if ema_cfg["enabled"]
        else None
    )

    # Mixup
    mixup_cfg = cfg["regularization"].get("mixup", None)

    early_stopping = EarlyStopping(**cfg["early_stopping"])

    if cfg["early_stopping"]["monitor"]=="val/loss":
        best_metric = float('inf')
    else:
        best_metric = -1.0
    
    # Training Loop
    for epoch in range(cfg["training"]["epochs"]):
        
        model.train()
        train_loss = 0.0
        
        # Unfreeze backbone if needed
        if (cfg["transfer"]["mode"] == "freeze" and cfg["transfer"]["freeze_epochs"] > 0 
            and epoch == cfg["transfer"]["freeze_epochs"]):

            print(f"Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model)

            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, len(train_loader))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            use_mixup = (
                mixup_cfg is not None
                and mixup_cfg["alpha"] > 0
                and np.random.rand() < mixup_cfg.get("prob", 1.0)
            )

            if use_mixup:
                x, y_a, y_b, lam = mixup_data(
                    x, y,
                    alpha=mixup_cfg["alpha"],
                    device=device
                )
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            
            #UPDATE LR (batch-based schedulers)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # EMA after weights update
            if ema: 
                ema.update(model)
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Validation
        eval_model = ema.model if ema else model
        val_loss, val_acc, val_f1 = run_validation(eval_model, val_loader, criterion, device)

        if scheduler and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)
        
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "lr/backbone": optimizer.param_groups[0]["lr"],
            "lr/head": optimizer.param_groups[1]["lr"],
        }
        wandb.log(metrics)

        monitor_metric = metrics.get(cfg["early_stopping"]["monitor"], val_f1)
        
        if cfg["early_stopping"]["monitor"]=="val/loss":
            if monitor_metric < best_metric:
                best_metric = monitor_metric
                fname = f"{cfg['experiment']['name']}_best.pth"
                path = ckpt_dir / fname
                
                torch.save({
                    "epoch": epoch,
                    "model_state": eval_model.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_metric,
                    "wandb_id": run.id
                }, path)

                print(f"New best val/loss: {monitor_metric}. New checkpoint saved.")
        else:
            if monitor_metric > best_metric:
                best_metric = monitor_metric
                fname = f"{cfg['experiment']['name']}_best.pth"
                path = ckpt_dir / fname
                
                torch.save({
                    "epoch": epoch,
                    "model_state": eval_model.state_dict(),
                    "config": cfg,
                    "best_f1": best_metric,
                    "wandb_id": run.id
                }, path)

                print(f"New best val/f1-score: {monitor_metric}. New checkpoint saved.")

        if early_stopping(monitor_metric):
            print("Early stopping triggered.")
            break

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args)