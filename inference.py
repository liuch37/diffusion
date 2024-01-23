from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet, ContextUnet
from mindiffusion.ddpm import DDPM, DDPM_Context

def inference(
    save_path: str = "./generations", samples: int = 8, device: str = "cuda:0", load_path: str = "ddpm.pth", sampler: str = 'ddpm'
) -> None:

    # uncomment to select one of below model
    #model = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    #model = DDPM_Context(eps_model=ContextUnet(3, 3, n_feat=128, encoding='onehot', nc_feat=10), betas=(1e-4, 0.02), n_T=1000)
    model = DDPM_Context(eps_model=ContextUnet(3, 3, n_feat=128, encoding='clip', nc_feat=512), betas=(1e-4, 0.02), n_T=1000)

    model.load_state_dict(torch.load(load_path))

    model.to(device)

    model.eval()
    with torch.no_grad():
        c = torch.tensor([1, 1, 1, 1,
                          3, 3, 3, 3,
                          5, 5, 5, 5,
                          8, 8, 8, 8]) # ['automobile', 'cat', 'dog', 'ship']
        xh = model.sample(samples, (3, 32, 32), device, c)
        grid = make_grid(xh, normalize=True, value_range=(-1, 1), nrow=4)
        save_image(grid, os.path.join(save_path, 'sample.png'))

if __name__ == "__main__":
    os.makedirs('generations',exist_ok=True)
    inference(save_path="./generations", samples=16, device="cuda", load_path="./models/ddpm_context_clip_cifar.pth", sampler='ddpm')