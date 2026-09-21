"""The neural networks."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def conv_block(in_channels, out_channels):
    """One pattern-finding stage.

    Conv2d:      slides 3x3 filters over the image, each looking for a pattern
    BatchNorm2d: keeps the numbers in a steady range so training is stable
    ReLU:        keeps positive responses, zeroes the rest
    MaxPool2d:   halves width and height, keeping the strongest responses
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A small convolutional network trained from scratch.

    Four blocks find increasingly complex patterns (16 -> 128 filters) while
    shrinking the image 224 -> 14. Average pooling turns each of the 128
    pattern maps into one number, and a final linear layer turns those 128
    numbers into one score per tumour type.
    """

    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ResNet18Grey(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted to single-channel MRI slices.

    ResNet-18 learned its filters from 1.28 million colour photographs. Those
    early edge and texture detectors transfer to MRI, so the whole network is
    fine-tuned from those weights rather than trained from scratch.

    Two mismatches are fixed here rather than in MRIDataset, so that the
    dataset keeps returning plain 1-channel tensors in [0, 1] and SmallCNN is
    unaffected:
      - the grey channel is repeated three times, which leaves ResNet's
        pretrained first convolution exactly as ImageNet trained it;
      - ImageNet's per-channel mean and standard deviation are applied,
        because pretrained weights expect inputs on that scale.
    """

    def __init__(self, n_classes=3, weights=None):
        super().__init__()
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def prepare(self, x):
        """Turn 1-channel [0, 1] scans into the 3-channel input ResNet expects."""
        return (x.expand(-1, 3, -1, -1) - self.mean) / self.std

    def forward(self, x):
        return self.backbone(self.prepare(x))


def build_model(name, n_classes=3, pretrained=True):
    """Create a model from its name, as used on the command line.

    pretrained only affects resnet18: True fetches the ImageNet weights
    (~45 MB, downloaded once and cached), False starts from random weights
    so tests can run offline.
    """
    if name == "small_cnn":
        return SmallCNN(in_channels=1, n_classes=n_classes)
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return ResNet18Grey(n_classes=n_classes, weights=weights)
    raise ValueError(f"unknown model: {name}")
