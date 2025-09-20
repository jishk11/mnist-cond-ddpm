import torch
import torch.nn as nn

from ..config import DMConfig
from ..models.unet import ConditionalUnet
from .scheduler import Scheduler

class ConditionalDDPM(nn.Module):

    def __init__(self, cfg: DMConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.in_channels if hasattr(cfg, "in_channels") else 1
        self.net = ConditionalUnet(in_ch, cfg.num_feat, cfg.num_classes)
        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        self.sched = None  

    def set_device(self, device: torch.device):
        self._dummy = self._dummy.to(device)
        self.sched = Scheduler(self.cfg, device)

    def forward(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.cfg.T, (B,), device=device, dtype=torch.long)

        # noise
        eps = torch.randn_like(x0)

        # q(x_t | x0)
        look = self.sched.lookup(t)
        x_t = (look["sqrt_alpha_bar_t"].view(-1,1,1,1) * x0 +
               look["sqrt_one_minus_alpha_bar_t"].view(-1,1,1,1) * eps)



        eps_pred = self.net(x_t, t, y)  

        loss = torch.mean((eps - eps_pred) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, y: torch.Tensor, omega: float = 5.0):
        self.eval()
        x_t = torch.randn(n, self.cfg.in_channels, *self.cfg.input_dim, device=device)
        T = self.cfg.T

        for t_step in reversed(range(T)):
            t = torch.full((n,), t_step, device=device, dtype=torch.long)
            look = self.sched.lookup(t)

            eps_pred = self.net(x_t, t, y)

            # DDPM update
            beta_t = look["beta_t"].view(-1,1,1,1)
            sqrt_alpha_t = look["sqrt_alpha_t"].view(-1,1,1,1)
            sqrt_one_minus_alpha_t = look["sqrt_one_minus_alpha_t"].view(-1,1,1,1)
            alpha_t = look["alpha_t"].view(-1,1,1,1)

            x0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
            mean = (torch.sqrt(look["alpha_bar_t"]).view(-1,1,1,1) * x0_pred +
                    torch.sqrt(1.0 - look["alpha_bar_t"]).view(-1,1,1,1) * eps_pred)

            if t_step > 0:
                z = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * z
            else:
                x_t = x0_pred

        return x_t.clamp(-1, 1)
