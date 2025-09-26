import torch
from src.config import DMConfig
from src.ddpm.model import ConditionalDDPM

def test_ddpm_loss_runs():
    cfg = DMConfig()
    m = ConditionalDDPM(cfg)
    m.set_device(torch.device("cpu"))
    x = torch.randn(4, cfg.in_channels, *cfg.input_dim)
    y = torch.randint(0, cfg.num_classes, (4,))
    loss = m(x, y)
    assert torch.isfinite(loss)
