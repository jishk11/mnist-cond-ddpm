import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field, asdict, replace
from typing import List, Tuple
from tqdm import tqdm
from DDPM import ConditionalDDPM, DMConfig
from torchvision.utils import make_grid
from torchvision import datasets
from torch.utils.data import DataLoader

# force deterministic seed and pick device
torch.manual_seed(0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# whitelist DMConfig for safe unpickling
from torch.serialization import add_safe_globals
add_safe_globals([DMConfig])

@dataclass
class DMConfig:
    '''
    Model and experiment settings
    '''
    input_dim: Tuple[int, int] = (28, 28)
    num_channels: int = 1
    condition_mask_value: int = -1
    num_classes: int = 10
    T: int = 400
    beta_1: float = 1e-4
    beta_T: float = 2e-2
    mask_p: float = 0.1
    num_feat: int = 128
    omega: float = 2.0

    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-4
    multi_lr_milestones: List[int] = field(default_factory=lambda: [20])
    multi_lr_gamma: float = 0.1


def make_dataloader(transform, batch_size, dir='./data', train=True):
    dataset = datasets.MNIST(root=dir, train=train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def make_dataloader_cifar10(transform, batch_size, dir='./data', train=True):
    dataset = datasets.CIFAR10(root=dir, train=train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def save_checkpoint(save_path, epoch, model, optimizer, dmconfig):
    '''
    Save model + optimizer states, plus a plain dict of dmconfig.
    '''
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dmconfig_dict': asdict(dmconfig)
    }
    torch.save(ckpt, save_path)

class Averager:
    def __init__(self): self.n = self.v = 0.0
    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    def item(self): return self.v


def check_forward(dataloader, dmconfig, device):
    model = ConditionalDDPM(dmconfig).to(device)
    optimizer = optim.Adam(model.network.parameters(), lr=dmconfig.learning_rate)
    model.train()
    losses = []
    for imgs, cond in tqdm(dataloader, desc='train', leave=False):
        imgs, cond = imgs.to(device), cond.to(device)
        loss = model(imgs, cond)
        losses.append(loss.item())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    plt.plot(losses); plt.xlabel('iter'); plt.ylabel('loss'); plt.title('Noise Loss'); plt.show()
    return model


def check_sample(model, dmconfig, device):
    model.eval()
    c = torch.arange(dmconfig.num_classes, device=device)
    c = torch.tile(c, (10,1)).T.reshape(-1,)
    with torch.no_grad(): imgs = model.sample(c, dmconfig.omega).cpu()
    fig, axes = plt.subplots(dmconfig.num_classes, 10, figsize=(6,6), gridspec_kw={'hspace':0,'wspace':0})
    for ax, img in zip(axes.flatten(), imgs): ax.imshow(img.permute(1,2,0), cmap='gray'); ax.axis('off')
    return fig


def sample_images(config, checkpoint_path):
    # load full checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    # reconstruct original config
    saved = ckpt['dmconfig_dict']
    dc = DMConfig(**saved)
    # override only omega
    dc = replace(dc, omega=config.omega)

    model = ConditionalDDPM(dc).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    c = torch.arange(dc.num_classes, device=device)
    c = torch.tile(c, (10,1)).T.reshape(-1,)
    with torch.no_grad(): imgs = model.sample(c, dc.omega)
    return make_grid(imgs, nrow=10, padding=0).permute(1,2,0).cpu().numpy()


def plot_images(model, num_classes, omega, T):
    model.eval()
    c = torch.arange(num_classes, device=device)
    c = torch.tile(c, (10,1)).T.reshape(-1,)
    with torch.no_grad(): imgs = model.sample(c, omega).cpu()
    fig, axes = plt.subplots(num_classes,10,figsize=(6,6), gridspec_kw={'hspace':0,'wspace':0})
    for ax, img in zip(axes.flatten(), imgs): ax.imshow(img.permute(1,2,0), cmap='gray'); ax.axis('off')
    return fig


def train(loader, model, opt):
    model.train(); avg = Averager()
    for imgs, cond in tqdm(loader, desc='train', leave=False):
        imgs, cond = imgs.to(device), cond.to(device)
        loss = model(imgs, cond); avg.add(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    return avg.item()


def test(loader, model):
    model.eval(); avg = Averager()
    with torch.no_grad():
        for imgs, cond in tqdm(loader, desc='test', leave=False):
            imgs, cond = imgs.to(device), cond.to(device)
            avg.add(model(imgs, cond).item())
    return avg.item()


def solver(dmconfig, exp_name, train_loader, test_loader):
    os.makedirs(f'./save/{exp_name}/images', exist_ok=True)
    model = ConditionalDDPM(dmconfig).to(device)
    opt   = optim.Adam(model.network.parameters(), lr=dmconfig.learning_rate)
    sched = MultiStepLR(opt, milestones=dmconfig.multi_lr_milestones, gamma=dmconfig.multi_lr_gamma)

    best = float('inf')
    for epoch in range(1, dmconfig.epochs+1):
        print(f'epoch {epoch}/{dmconfig.epochs}')
        tr = train(train_loader, model, opt)
        sched.step()
        te = test(test_loader, model)
        print(f'train: {tr:.4f}  test: {te:.4f}')
        if te < best:
            best = te
            save_checkpoint(f'./save/{exp_name}/best_checkpoint.pth', epoch, model, opt, dmconfig)
        fig = plot_images(model, dmconfig.num_classes, dmconfig.omega, dmconfig.T)
        fig.savefig(f'./save/{exp_name}/images/generate_epoch_{epoch}.png')
        plt.close(fig)
