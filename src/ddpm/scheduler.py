import torch
from ..config import DMConfig

class Scheduler:
    def __init__(self, cfg: DMConfig, device: torch.device):
        T = cfg.T
        betas = torch.linspace(cfg.beta_1, cfg.beta_T, T, device=device)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alphas)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars)

    def lookup(self, t: torch.LongTensor):
        return {
            "beta_t": self.betas[t],
            "alpha_t": self.alphas[t],
            "alpha_bar_t": self.alpha_bars[t],
            "sqrt_alpha_t": self.sqrt_alpha[t],
            "sqrt_one_minus_alpha_t": self.sqrt_one_minus_alpha[t],
            "sqrt_alpha_bar_t": self.sqrt_alpha_bar[t],
            "sqrt_one_minus_alpha_bar_t": self.sqrt_one_minus_alpha_bar[t],
        }
