import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtTinyEmotion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_p=0.0):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features

        head = [nn.LayerNorm(in_features)]
        if dropout_p > 0:
            head.append(nn.Dropout(p=dropout_p))
        head.append(nn.Linear(in_features, num_classes))

        self.backbone.classifier[2] = nn.Sequential(*head)

    def forward(self, x):
        return self.backbone(x)
    
def build_convnext(cfg):
    num_classes = cfg["model"].get("num_classes", 7)
    pretrained = cfg["model"].get("pretrained", True)
    dropout_p = cfg["model"].get("dropout", 0.0)
    return ConvNeXtTinyEmotion(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)
