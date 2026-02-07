import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class VGGFaceEmotion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_p=0.0):
        super().__init__()
        
        weights = 'vggface2' if pretrained else None
        
        self.backbone = InceptionResnetV1(pretrained=weights, classify=True, num_classes=num_classes)
        in_features = self.backbone.logits.in_features

        self.backbone.logits = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def build_vggface(cfg):
    num_classes = cfg["model"].get("num_classes", 7)
    pretrained = cfg["model"].get("pretrained", True)
    dropout_p = cfg["model"].get("dropout", 0.0)
    return VGGFaceEmotion(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)