import os
import argparse
import torch

from .config import DMConfig
from .ddpm.model import ConditionalDDPM
from .utils.plots import plot_images

def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")

def run(checkpoint: str, out: str, omega: float, n: int, per_class: int):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ckpt = load_checkpoint(checkpoint)
    cfg = DMConfig(**ckpt["cfg"])
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    model = ConditionalDDPM(cfg).to(device)
    model.set_device(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # build labels: e.g., 10 classes × per_class
    if n > 0:
        y = torch.randint(0, cfg.num_classes, (n,), device=device)
    else:
        y = torch.arange(0, cfg.num_classes, device=device).repeat_interleave(per_class)

    imgs = model.sample(n=len(y), device=device, y=y, omega=omega)
    plot_images(imgs, path=out, nrow=per_class, title=f"ω={omega}")
    print(f"[save] {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    p.add_argument("--out", type=str, default="results/samples_grid.png")
    p.add_argument("--omega", type=float, default=5.0)
    p.add_argument("--n", type=int, default=0, help="if >0, random labels of length n")
    p.add_argument("--per_class", type=int, default=8, help="images per class row")
    args = p.parse_args()
    run(args.checkpoint, args.out, args.omega, args.n, args.per_class)

if __name__ == "__main__":
    main()
