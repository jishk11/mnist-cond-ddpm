from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DMConfig:
    input_dim: Tuple[int, int] = (28, 28)
    in_channels: int = 1
    num_classes: int = 10
    num_feat: int = 128

    T: int = 400
    beta_1: float = 1e-4
    beta_T: float = 2e-2
    omega: float = 5.0

    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-4
    mult_lr_milestones: List[int] = (20,)
    mult_lr_gamma: float = 0.1

    seed: int = 42
    device: str = "auto"   # "cuda"|"mps"|"cpu"
    out_dir: str = "checkpoints"
