import torch
import torch.optim as optim
from src.utils.freeze import get_param_groups

def build_optimizer(model, cfg):
    opt_cfg = cfg["optimizer"]
    tr_cfg = cfg["transfer"]

    if tr_cfg["mode"] in ["freeze", "finetune"]:
        param_groups = get_param_groups(
            model,
            backbone_lr=tr_cfg["backbone_lr"],
            head_lr=tr_cfg["head_lr"]
        )
    else:
        param_groups = model.parameters()

    if opt_cfg["name"] == "adam":
        return optim.AdamW(
            param_groups,
            weight_decay=opt_cfg["weight_decay"]
        )

    elif opt_cfg["name"] == "sgd":
        return optim.SGD(
            param_groups,
            momentum=opt_cfg["momentum"],
        )

    else:
        raise ValueError(f"Unknown optimizer {opt_cfg['name']}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    name = cfg["scheduler"]["name"].lower()

    if name == "onecycle":
        max_lr_factor = float(cfg["scheduler"].get("max_lr_factor", 10))

        max_lr = [
            float(group["lr"]) * max_lr_factor
            for group in optimizer.param_groups
        ]

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg["training"]["epochs"],
            pct_start=0.3,
        )

    elif name == "cosine":

        total_steps = cfg["training"]["epochs"]
        tmax_ratio = cfg["scheduler"].get("tmax_ratio", 1.0)

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(tmax_ratio * total_steps),
        )

    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg["scheduler"].get("mode", "min"),
            patience=cfg["scheduler"].get("patience", 3),
            factor=cfg["scheduler"].get("factor", 0.5)
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")

