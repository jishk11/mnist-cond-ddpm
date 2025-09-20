import os
import torch
from ..config import DMConfig

def save_checkpoint(model, optimizer, cfg: DMConfig, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "epoch": epoch,
    }, path)
