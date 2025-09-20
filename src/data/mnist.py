import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..config import DMConfig

def make_dataloader(cfg: DMConfig, train: bool = True) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),               
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    ds = datasets.MNIST(root="data", train=train, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader
