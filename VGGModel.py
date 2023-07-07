from typing import cast, Dict, List, Union

import torch
from torch import nn

__all__ = [
            'VGG',
            'vgg16',
            'vgg19'
]


ARC: Dict[str, List[Union[str, int]]] = {
    'V16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16

    'V19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}  # VGG19


class VGG(nn.Module):
    """Make VGG architecture."""

    def __init__(self, architecture: List[Union[str, int]], num_classes: int = 1000) -> None:
        """Create model."""
        super().__init__()
        self.features = self._make_layers(architecture)
        self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.classifier = nn.Sequential(
                                        nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(inplace = True),
                                        nn.Dropout(p = 0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace = True),
                                        nn.Dropout(p = 0.5),
                                        nn.Linear(4096, num_classes))

    def _make_layers(self, arc: List[Union[str, int]]) -> nn.Sequential:
        """Construct model layers."""
        layers: List[nn.Module] = []
        in_ch = 3
        for l in arc:
            if l == 'M':
                layers += [nn.MaxPool2d(3, stride = 2, padding = 1)]
            else:
                l = cast(int, l)
                layers += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1),
                           nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make forward path."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(num_classes: int = 1000) -> VGG:
    """Make VGG16 model."""
    return VGG(ARC['V16'], num_classes)


def vgg19(num_classes: int = 1000) -> VGG:
    """Make VGG19 model."""
    return VGG(ARC['V19'], num_classes)
