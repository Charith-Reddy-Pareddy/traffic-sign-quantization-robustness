import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class BaselineCNN(nn.Module):
    """Small convolutional classifier for 32x32 traffic sign images."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.classifier(features)


class MobileNetV3Small(nn.Module):
    """Compact MobileNetV3 model with a traffic-sign output layer."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.model = mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def create_model(model_name: str, num_classes: int = 43) -> nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if model_name == "mobilenet_v3_small":
        return MobileNetV3Small(num_classes)
    raise ValueError("model_name must be 'baseline_cnn' or 'mobilenet_v3_small'")
