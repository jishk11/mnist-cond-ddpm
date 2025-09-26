import torch
from src.config import DMConfig
from src.models.unet import ConditionalUnet

def test_unet_shapes():
    cfg = DMConfig()
    net = ConditionalUnet(cfg.in_channels, cfg.num_feat, cfg.num_classes)
    x = torch.randn(2, cfg.in_channels, *cfg.input_dim)
    t = torch.randint(0, cfg.T, (2,))
    y = torch.randint(0, cfg.num_classes, (2,))
    out = net(x, t, y)
    assert out.shape == x.shape
