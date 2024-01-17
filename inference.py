from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

def inference(
    save_path: str = "./contents", samples: int = 8, device: str = "cuda:0", load_path: str = "ddpm.pth"
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    ddpm.load_state_dict(torch.load(load_path))

    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(samples, (3, 32, 32), device)
        xset = torch.cat([xh, x[:samples]], dim=0)
        grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
        save_image(grid, os.path.join(save_path, 'sample_'+str(i)+'.png'))

if __name__ == "__main__":
    inference(save_path="./contents", samples=8, device="mps", load_path="ddpm.pth")