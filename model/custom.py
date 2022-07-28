"""Custom models.
"""


import torch
import torch.nn as nn


class CompactFCN(nn.Module):
    """Compact FCN model."""

    def __init__(self, in_channels):
        super().__init__()
        self.layer_cp1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=9, padding=4, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer_cp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer_c1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
        )
        self.layer_tc1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, bias=True, stride=2, output_padding=0),
            nn.ReLU(),
        )
        self.layer_c2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=3, bias=True),
            nn.ReLU(),
        )
        self.layer_tc2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1, bias=True, stride=2),
        )
 
    def forward(self, x):
        x = self.layer_cp1(x)
        x = self.layer_cp2(x)
        x = self.layer_c1(x)
        x = self.layer_tc1(x)
        x = self.layer_c2(x)
        out = self.layer_tc2(x)
        return out[...,:-1,:-1].squeeze(dim=1)


class CompactCNN(nn.Module):
    """Compact CNN model."""

    def __init__(self, in_channels):
        super().__init__()
        self.layer_cp1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=9, padding=4, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer_cp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer_c1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
        )
        self.layer_c2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2, bias=True, stride=2),
            nn.ReLU(),
        )
        self.layer_c3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True, stride=2),
            nn.ReLU(),
        )

        self.layer_l1 = nn.Sequential(
            nn.Linear(1568, 256, bias=True),
            nn.ReLU(),
        )
        self.layer_l2 = nn.Linear(256, 1, bias=True)

    def forward(self, x):
        x = self.layer_cp1(x)
        x = self.layer_cp2(x)
        x = self.layer_c1(x)
        x = self.layer_c2(x)
        x = self.layer_c3(x)
        x = x.flatten(start_dim=1)
        x = self.layer_l1(x)
        out = self.layer_l2(x)
        return out

class CompactCNN2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4, bias=True),
            nn.ReLU(),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=9, padding=4, bias=True),
        )

    def forward(self, x):
        x = self.input_conv(x)
        out = self.output_conv(x)
        return out.squeeze(dim=1)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_channels = 17
    net = CompactCNN2(in_channels=n_channels).to(device)
    x = torch.randn(16, n_channels, 224, 224).to(device)
    out = net(x)
    print(out.size())
