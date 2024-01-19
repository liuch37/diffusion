from typing import Dict, Optional, Tuple
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
    save_path: str = "./generations", samples: int = 8, device: str = "cuda:0", load_path: str = "ddpm.pth", sampler: str = 'ddpm'
) -> None:

    model = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    model.load_state_dict(torch.load(load_path))

    model.to(device)

    model.eval()
    with torch.no_grad():
        xh = model.sample(samples, (3, 32, 32), device)
        grid = make_grid(xh, normalize=True, nrow=4)
        save_image(grid, os.path.join(save_path, 'sample.png'))

if __name__ == "__main__":
    os.makedirs('generations',exist_ok=True)
    inference(save_path="./generations", samples=16, device="cuda", load_path="./models/ddpm_cifar.pth", sampler='ddpm')