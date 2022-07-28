"""Custom models.
"""


import torch
import torch.nn as nn


class CompactCNN56(nn.Module):
    """Compact CNN model with 56x56 output."""

    def __init__(self, in_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=9, padding=4, stride=2, bias=True),
        )

    def forward(self, x):
        x = self.input_conv(x)
        out = self.output_conv(x)
        return out.squeeze(dim=1)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_channels = 17
    net = CompactCNN56(in_channels=n_channels).to(device)
    x = torch.randn(16, n_channels, 224, 224).to(device)
    out = net(x)
    print(out.size())
