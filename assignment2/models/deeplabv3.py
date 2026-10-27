import torch.nn as nn

from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from torchvision.models import ResNet50_Weights


def build_deeplabv3(num_classes=19, pretrained_backbone=True):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=(
            ResNet50_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        ),
        num_classes=num_classes,
        aux_loss=False,
    )

    return model


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super().__init__()

        self.model = build_deeplabv3(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )

    def forward(self, x):
        return self.model(x)["out"]