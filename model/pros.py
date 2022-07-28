"""Modules for PROS.

Refer to https://ieeexplore.ieee.org/document/9256565.
"""


import torch.cuda.amp
import torch.nn as nn


def _conv1x1(in_channels, out_channels, stride=1, bias=False) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=bias)


def _conv3x3(in_channels, out_channels,
             stride=1, padding=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with 'same' padding of zeros."""
    padding += dilation - 1
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


def _conv7x7(in_channels, out_channels, stride=1, padding=3) -> nn.Conv2d:
    """7x7 convolution with 'same' padding of zeros."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride,
                     padding=padding, bias=False)


class ResBlock(nn.Module):
    """Basic block for ResNet."""

    def __init__(self, in_channels, channels, stride, dilation=1):
        super().__init__()
        self.conv1_bn = nn.Sequential(
            _conv3x3(in_channels, channels, stride=stride, dilation=dilation),
            nn.BatchNorm2d(channels))
        self.conv2_bn = nn.Sequential(
            _conv3x3(channels, channels, dilation=dilation),
            nn.BatchNorm2d(channels))

        self.activation = nn.ReLU(inplace=True)

        if in_channels != channels:
            self.shortcut = nn.Sequential(
                _conv1x1(in_channels, channels, stride=stride),
                nn.BatchNorm2d(channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1_bn(x)
        out = self.activation(out)

        out = self.conv2_bn(out)
        out += self.shortcut(x)
        out = self.activation(out)

        return out


class RefinementBlock(nn.Module):
    """Refinement Block defined in Fig.5 of the PROS paper."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1x1 = _conv1x1(in_channels, channels)
        self.conv3x3_bn_relu = nn.Sequential(
            _conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv3x3 = _conv3x3(channels, channels)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out_1x1 = self.conv1x1(x)
        out_3x3 = self.conv3x3_bn_relu(out_1x1)
        out_3x3 = self.conv3x3(out_3x3)
        out = self.final_relu(out_1x1 + out_3x3)
        return out


class PROS(nn.Module):
    """RouteNet model for segmentation task."""

    def __init__(self, in_channels):
        super().__init__()
        # Encoder
        self.stem_conv = nn.Sequential(
            _conv7x7(in_channels, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(2, 2)
        self.res_1 = nn.Sequential(
            ResBlock(64, 256, stride=1),
            ResBlock(256, 256, stride=1)
        )
        self.res_2 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
        )
        self.res_3 = nn.Sequential(
            ResBlock(512, 1024, stride=1, dilation=2),
            ResBlock(1024, 1024, stride=1, dilation=2),
        )
        self.res_4 = nn.Sequential(
            ResBlock(1024, 2048, stride=1, dilation=4),
            ResBlock(2048, 2048, stride=1, dilation=4),
        )

        # Decoder
        self.rb_128 = RefinementBlock(256, 128)
        self.rb_512_sub = nn.Sequential(
            RefinementBlock(2048, 512),
            nn.PixelShuffle(2),
        )
        self.rb_32 = RefinementBlock(64, 32)
        self.rb_128_sub = nn.Sequential(
            RefinementBlock(128, 128),
            nn.PixelShuffle(2),
        )
        self.rb_8_sub = nn.Sequential(
            RefinementBlock(32, 8),
            nn.PixelShuffle(2),
        )

        self.final_classifier = _conv1x1(2, 1, bias=True)

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        # Encoder
        # x = x.half()
        stem_out = self.stem_conv(x)

        res_1_out = self.max_pool(stem_out)
        res_1_out = self.res_1(res_1_out)

        res_out = self.res_2(res_1_out)
        res_out = self.res_3(res_out)
        res_out = self.res_4(res_out)

        # Decoder
        decoder_stage_1 = self.rb_128(res_1_out)
        decoder_stage_1 += self.rb_512_sub(res_out)

        decoder_stage_2 = self.rb_128_sub(decoder_stage_1)
        decoder_stage_2 += self.rb_32(stem_out)

        out = self.rb_8_sub(decoder_stage_2)
        out = self.final_classifier(out)

        return out.squeeze(dim=1)


if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.memory_allocated(0))
    net = PROS(in_channels=64).half().to(device)
    print(torch.cuda.memory_allocated(0))
    import time
    start = time.time()
    while True:
        x = torch.randn(4, 64, 224, 224, dtype=torch.float, device=device)
        out = net(x)
        # net.to('cpu')
        # m = torch.cuda.memory_allocated(0)
        # while m > 1000:
        #     print(m)
        #     new_net = net.cpu()
        #     del net
        #     net = new_net
        #     torch.cuda.empty_cache()
        #     time.sleep(1)
        #     m = torch.cuda.memory_allocated(0)
        # exit()
