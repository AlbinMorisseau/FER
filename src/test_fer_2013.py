import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


# Imports projet
from src.models.convnextiny import build_convnext
from src.models.resnet50 import build_resnet
from src.models.vggface import build_vggface

EMOTION_LABELS = {
    0: "surprised",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral",
}


# ==========================================
# UTILS
# ==========================================
def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "convnext_tiny":
        return build_convnext(cfg)
    if name == "resnet50":
        return build_resnet(cfg)
    if name == "vggface":
        return build_vggface(cfg)
    raise ValueError(f"Unknown model: {name}")

def gray_to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ==========================================
# MAIN EVALUATION
# ==========================================
def evaluate(config_path: str, fer_test_path: str):

    # -------- Load config --------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load checkpoint --------
    ckpt_path = Path("checkpoints") / f"{cfg['experiment']['name']}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # -------- Load FER-2013 dataset brut --------
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.Lambda(gray_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_dataset = ImageFolder(fer_test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class_names = [EMOTION_LABELS[i] for i in range(7)]

    # -------- Model --------
    model = build_model(cfg).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # -------- Evaluation loop --------
    all_preds, all_labels = [], []
    losses = []

    criterion = torch.nn.CrossEntropyLoss()

    print("Running evaluation...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # -------- Metrics --------
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print("\n========== TEST METRICS ==========")
    print(f"Loss (mean)     : {np.mean(losses):.4f}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-score  : {f1:.4f}")

    # -------- Confusion Matrix --------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("FER-2013 Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("Evaluation complete.")


# ==========================================
# ENTRY
# ==========================================
if __name__ == "__main__":
    CONFIG_PATH = "configs/resnet50_hyper_optimized_2.yaml" 
    FER_TEST_PATH = "data/aligned/test_fer_2013"
    evaluate(CONFIG_PATH, FER_TEST_PATH)
