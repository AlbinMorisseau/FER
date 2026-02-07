import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Emotion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_p=0.0):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        head = []
        if dropout_p > 0:
            head.append(nn.Dropout(p=dropout_p))
        head.append(nn.Linear(in_features, num_classes))

        self.backbone.fc = nn.Sequential(*head)

    def forward(self, x):
        return self.backbone(x)


# Fonction factory
def build_resnet(cfg):
    num_classes = cfg["model"].get("num_classes", 7)
    pretrained = cfg["model"].get("pretrained", True)
    dropout_p = cfg["model"].get("dropout", 0.0)
    return ResNet50Emotion(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)
