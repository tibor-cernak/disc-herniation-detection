import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with two 3D convolutions, batch norm, and ReLU."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling and ConvBlock."""

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution and ConvBlock."""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VnetModel(nn.Module):
    """U-Net architecture for 3D heatmap regression."""

    def __init__(self, in_channels=1, out_channels=1):
        super(VnetModel, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 16)
        self.encoder2 = DownBlock(16, 32)
        self.encoder3 = DownBlock(32, 64)
        self.encoder4 = DownBlock(64, 128)
        self.encoder5 = DownBlock(128, 256)

        self.decoder4 = UpBlock(256, 128)
        self.decoder3 = UpBlock(128, 64)
        self.decoder2 = UpBlock(64, 32)
        self.decoder1 = UpBlock(32, 16)

        # Final convolution to produce a single-channel heatmap
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for heatmap values in [0, 1]

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        # Final output
        out = self.final_conv(d1)
        out = self.sigmoid(out)
        return out
