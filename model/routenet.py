"""Modules for RouteNet.

Refer to https://ieeexplore.ieee.org/document/8587655.
"""


import torch
import torch.nn as nn
from torchvision import models


class RouteNet(nn.Module):
    """RouteNet model for regression task."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        if pretrained:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 1)
            nn.init.xavier_normal_(self.model.fc)
        else:
            self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_layer_(self, layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._initialize_layer_(m)

class RouteNetFCN(nn.Module):
    """RouteNet model for segmentation task."""

    def __init__(self, in_channels):
        super().__init__()
        self.layer_input_bn = nn.BatchNorm2d(in_channels)

        self.layer_cp1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer_cp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # no more pooling
        self.layer_c1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer_c2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer_d1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=9, stride=2, padding=4,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer_c3 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())

        # W = (W-1)*stride - 2*P + Kernel
        self.layer_d2 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU())

        # W2=(W1-F+2P)/S+1
        self.layer_c4 = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.layer_input_bn(x)
        save = self.layer_cp1(out)
        out = self.layer_cp2(save)
        out = self.layer_c1(out)
        out = self.layer_c2(out)
        out = self.layer_d1(out)
        out = torch.cat([out, save], dim=1)
        out = self.layer_c3(out)
        out = self.layer_d2(out)
        out = self.layer_c4(out)
        return out.squeeze(dim=1)
