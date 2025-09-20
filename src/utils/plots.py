from typing import Optional
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def plot_images(x: torch.Tensor, path: Optional[str] = None, nrow: int = 8, title: str = None):
    """
    x in [-1,1], shape [B,C,H,W]
    """
    x = (x + 1) / 2.0
    grid = make_grid(x, nrow=nrow)
    plt.figure()
    if title: plt.title(title)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
