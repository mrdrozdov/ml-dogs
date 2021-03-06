import torch
import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, num_classes=120, version='resnet18', pretrained=True):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', version, pretrained=pretrained)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

        self.backbone = model

    def forward(self, x):
        return self.backbone(x)


model_class_name = Resnet
