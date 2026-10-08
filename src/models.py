"""Model definitions: a from-scratch baseline and the transfer-learning model."""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SmallCNN(nn.Module):
    """Minimal CNN trained from scratch.

    This exists as a controlled reference point, not as a competitive model: it
    is trained on the same split, with the same loss, schedule and early-stopping
    criterion as the transfer-learning model, so the difference between the two
    isolates the contribution of ImageNet pre-training on a dataset this small.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_efficientnet_b0(num_classes=3, unfreeze_last_n_blocks=3):
    """ImageNet-pretrained EfficientNet-B0 with partial unfreezing.

    With ~360 training images, fine-tuning the whole backbone overfits within a
    few epochs. Freezing everything except the classifier head and the last N
    feature blocks keeps the trainable parameter count low while still letting
    the high-level features adapt to VIA imagery.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    n_blocks = len(model.features)
    start = max(0, n_blocks - unfreeze_last_n_blocks)
    for i in range(start, n_blocks):
        for p in model.features[i].parameters():
            p.requires_grad = True

    return model


def count_parameters(model):
    """Return (total, trainable, trainable_fraction)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, trainable / total
