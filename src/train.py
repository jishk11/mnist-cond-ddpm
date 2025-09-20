import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid

from .config import DMConfig
from .ddpm.model import ConditionalDDPM
from .data.mnist import make_dataloader
from .utils.io import save_checkpoint
from .utils.plots import plot_images

def _select_device(cfg: DMConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)

def _set_seed(seed: int):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_loop():
    cfg = DMConfig()
    device = _select_device(cfg)
    _set_seed(cfg.seed)

    model = ConditionalDDPM(cfg).to(device)
    model.set_device(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = MultiStepLR(optimizer, milestones=list(cfg.mult_lr_milestones), gamma=cfg.mult_lr_gamma)

    train_loader = make_dataloader(cfg, train=True)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()

            running += loss.item()

        sched.step()
        avg = running / len(train_loader)
        print(f"epoch {epoch}: loss={avg:.4f}")

        if epoch == 1 or epoch % 5 == 0:
            with torch.no_grad():
                y_fixed = torch.arange(0, cfg.num_classes, device=device)
                y_fixed = y_fixed.repeat_interleave(8)[:64] 
                imgs = model.sample(n=len(y_fixed), device=device, y=y_fixed, omega=cfg.omega)
            os.makedirs("results", exist_ok=True)
            plot_images(imgs, path=f"results/samples_e{epoch}.png", nrow=8)

        os.makedirs(cfg.out_dir, exist_ok=True)
        save_checkpoint(model, optimizer, cfg, epoch, os.path.join(cfg.out_dir, "last.ckpt"))

if __name__ == "__main__":
    train_loop()
