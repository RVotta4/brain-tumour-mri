"""The neural networks."""

from torch import nn


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


def build_model(name, n_classes=3):
    """Create a model from its name, as used on the command line."""
    if name == "small_cnn":
        return SmallCNN(in_channels=1, n_classes=n_classes)
    raise ValueError(f"unknown model: {name}")
