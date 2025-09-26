import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.in_channels == self.out_channels:
            out = x + x2
        else:
            out = x1 + x2
        return out / math.sqrt(2)


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    """
    Note: This module expects you to concatenate skip BEFORE the block.
    The first layer is a ConvTranspose2d that takes the concatenated channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        return self.model(x)


class EmbedBlock(nn.Module):
    """
    Accepts either:
      - indices tensor of shape [B]  -> one-hot internally
      - vectors tensor of shape [B, input_dim]
    Outputs: [B, embed_dim]
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = F.one_hot(x.long(), num_classes=self.input_dim).float()
        return self.layers(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, t, c):
        h, w = x.shape[-2:]
        return self.layers(torch.cat([x, t.repeat(1, 1, h, w), c.repeat(1, 1, h, w)], dim=1))


class ConditionalUnet(nn.Module):
    def __init__(self, in_channels, n_feat: int = 128, n_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        # embeddings (time uses a single scalar feature per sample)
        self.timeembed1 = EmbedBlock(1, 2 * n_feat)
        self.timeembed2 = EmbedBlock(1, 1 * n_feat)
        self.conditionembed1 = EmbedBlock(n_classes, 2 * n_feat)
        self.conditionembed2 = EmbedBlock(n_classes, 1 * n_feat)

        # encoder
        self.init_conv = ResConvBlock(in_channels, n_feat)
        self.downblock1 = UnetDown(n_feat, n_feat)
        self.downblock2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # decoder
        self.upblock0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.upblock1 = UnetUp(4 * n_feat, n_feat)
        self.upblock2 = UnetUp(2 * n_feat, n_feat)

        self.outblock = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        # feature fusion at multiple scales
        self.fusion1 = FusionBlock(3 * self.n_feat, self.n_feat)
        self.fusion2 = FusionBlock(6 * self.n_feat, 2 * self.n_feat)
        self.fusion3 = FusionBlock(3 * self.n_feat, self.n_feat)
        self.fusion4 = FusionBlock(3 * self.n_feat, self.n_feat)

    def forward(self, x, t, c):
        device = x.device

        t = t.to(device).float().unsqueeze(1)
        c = c.to(device).long()

        # time embeddings
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # condition embeddings
        cemb1 = self.conditionembed1(c).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.conditionembed2(c).view(-1, self.n_feat, 1, 1)

        # encoder
        down0 = self.init_conv(x)
        down0 = self.fusion1(down0, temb2, cemb2)

        down1 = self.downblock1(down0)

        down2 = self.downblock2(down1)
        down2 = self.fusion2(down2, temb1, cemb1)

        # bottleneck
        vec = self.to_vec(down2)

        # decoder
        up0 = self.upblock0(vec)

        up1 = self.upblock1(up0, down2)
        up1 = self.fusion3(up1, temb2, cemb2)

        up2 = self.upblock2(up1, down1)
        up2 = self.fusion4(up2, temb2, cemb2)

        # output
        out = self.outblock(torch.cat((up2, down0), dim=1))
        return out
