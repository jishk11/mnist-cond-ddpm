import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

from .config import DMConfig
from .ddpm.model import ConditionalDDPM
from .utils.plots import plot_images

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def main(checkpoint="checkpoints/last.ckpt", out="results/samples_grid.png", omega=5.0):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ckpt = load_checkpoint(checkpoint)
    cfg = DMConfig(**ckpt["cfg"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalDDPM(cfg).to(device)
    model.set_device(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y = torch.arange(0, cfg.num_classes, device=device).repeat(6)[:48]  
    imgs = model.sample(n=len(y), device=device, y=y, omega=omega)
    plot_images(imgs, path=out, nrow=8, title=f"ω={omega}")

if __name__ == "__main__":
    main()
