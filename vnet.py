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


class AttentionBlock(nn.Module):
    """Attention block that refines skip connections in U-Net."""

    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(f_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(f_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi


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
    """Upsampling block with optional attention."""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        if use_attention:
            self.attention = AttentionBlock(
                out_channels, out_channels, out_channels // 2
            )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.use_attention:
            x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VNetModelBase(nn.Module):
    """Base class for VNet models with configurable upsampling blocks."""

    def __init__(self, in_channels=1, out_channels=1, use_attention=False):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, 16)
        self.encoder2 = DownBlock(16, 32)
        self.encoder3 = DownBlock(32, 64)
        self.encoder4 = DownBlock(64, 128)
        self.encoder5 = DownBlock(128, 256)

        self.decoder4 = UpBlock(256, 128, use_attention)
        self.decoder3 = UpBlock(128, 64, use_attention)
        self.decoder2 = UpBlock(64, 32, use_attention)
        self.decoder1 = UpBlock(32, 16, use_attention)

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


class VNetModel(VNetModelBase):
    """Standard V-Net without attention."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels, out_channels, use_attention=False)


class AttentionVNetModel(VNetModelBase):
    """V-Net with attention in upsampling blocks."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels, out_channels, use_attention=True)
