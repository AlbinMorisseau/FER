import yaml
import torch
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.datamodule import EmotionDataModule
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

# Utils
def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "convnext_tiny": return build_convnext(cfg)
    if name == "resnet50": return build_resnet(cfg)
    if name == "vggface": return build_vggface(cfg)
    raise ValueError(f"Unknown model: {name}")


def get_target_layer(model, model_name):

    if "convnext" in model_name:
        return model.backbone.features[-1]

    if "resnet" in model_name:
        return model.layer4[-1]

    if "vggface" in model_name:
        return model.features[-1]

    raise ValueError(f"Target layer not defined for {model_name}")

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Forward → activations
        self.target_layer.register_forward_hook(self._forward_hook)
        # Backward → gradients
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # ConvNeXt peut retourner un tuple
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self.gradients = grad.detach()

    def __call__(self, x, class_idx):
        logits = self.model(x)

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # GAP sur les gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), logits

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run_gradcam(args):

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path("checkpoints") / f"{cfg['experiment']['name']}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    dm = EmotionDataModule(**cfg["data"])
    dm.setup()
    transform = dm.test_dataset.transform

    # Image
    img = Image.open(args.image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # Model
    model = build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Grad-CAM
    target_layer = get_target_layer(model, cfg["model"]["name"])
    gradcam = GradCAM(model, target_layer)

    # Prediction
    with torch.enable_grad():
        logits = model(x)
        pred_class = logits.argmax(1).item()

        cam, logits = gradcam(x, pred_class)

    # Resize CAM
    cam = cv2.resize(cam, img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(img)
    overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM saved to {out_path}")
    print(f"Predicted class: {EMOTION_LABELS[pred_class]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="gradcam.png")
    args = parser.parse_args()

    run_gradcam(args)
