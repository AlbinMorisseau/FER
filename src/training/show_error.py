import os
import yaml
import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

from src.data.datamodule import EmotionDataModule
from src.models.convnextiny import build_convnext
from src.models.resnet50 import build_resnet

EMOTION_LABELS = {0: "surprised", 1: "fear", 2: "disgust", 3: "happy", 4: "sad", 5: "angry", 6: "neutral"}

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean

def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "convnext_tiny": return build_convnext(cfg)
    if name == "resnet50": return build_resnet(cfg)
    raise ValueError(f"Unknown model: {name}")

def run_analysis(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg['experiment']['name']
    output_dir = Path("errors") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(args.metadata)
    meta_df['fn_key'] = meta_df['filename'].apply(lambda x: x.lower())

    dm = EmotionDataModule(**cfg["data"])
    dm.setup()
    test_loader = dm.test_dataloader()

    all_filepaths = [Path(s[0]).name.lower() for s in dm.test_dataset.samples]

    model = build_model(cfg).to(device)
    ckpt_path = Path("checkpoints") / f"{run_name}_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    error_data = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Analyse")):
            x_dev, y_dev = x.to(device), y.to(device)
            logits = model(x_dev)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(x.size(0)):
                idx_in_dataset = batch_idx * test_loader.batch_size + i
                true_idx, pred_idx = y[i].item(), preds[i].item()

                if true_idx != pred_idx:
                    true_label = EMOTION_LABELS[true_idx]
                    pred_label = EMOTION_LABELS[pred_idx]
                    conf = probs[i][pred_idx].item()
                    
                    filename = all_filepaths[idx_in_dataset]
                    row = meta_df[meta_df['fn_key'] == filename]
                    source = row['source'].values[0] if not row.empty else "unknown"
                    pair = sorted([true_label, pred_label])
                    folder_name = f"{pair[0]}_vs_{pair[1]}"
                    
                    sub_dir = output_dir / folder_name
                    sub_dir.mkdir(exist_ok=True)

                    img_name = f"{true_label}_as_{pred_label}_{filename}_{conf:.2f}.jpg"
                    save_image(denormalize(x[i]), sub_dir / img_name)

                    error_data.append({
                        "filename": filename,
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": conf,
                        "source": source,
                        "pair": folder_name
                    })

    df_results = pd.DataFrame(error_data)
    df_results.to_csv(output_dir / "error_report.csv", index=False)

    print(f"\n --- ERORS STATS ({run_name}) ---")
    print(f"Mean confidence : {df_results['confidence'].mean():.2f}")
    print("\nTop 5 confusions :")
    print(df_results['pair'].value_counts().head(5))
    print("\nSource dataset :")
    print(df_results['source'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="data/aligned/metadata.csv")
    args = parser.parse_args()
    run_analysis(args)