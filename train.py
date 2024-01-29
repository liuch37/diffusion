from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10, MovingMNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet, ContextUnet
from mindiffusion.ddpm import DDPM, DDPM_Context, DDPM_Dual

def train_cifar10(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    # uncomment to select one of below model
    #ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    #ddpm = DDPM_Context(eps_model=ContextUnet(3, 3, n_feat=128, encoding='onehot', nc_feat=10), betas=(1e-4, 0.02), n_T=1000)
    ddpm = DDPM_Context(eps_model=ContextUnet(3, 3, n_feat=128, encoding='clip', nc_feat=512), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = ddpm(x, y)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_cifar_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./models/ddpm_context_clip_cifar.pth")

def train_dual(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM_Dual(eps_model1=NaiveUnet(1, 1, n_feat=128), eps_model2=NaiveUnet(1, 1, n_feat=128), rho=0.99, betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.Resize((32, 32))]
    )

    dataset = MovingMNIST(
        "./data",
        download=True,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)
    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            x = x.float() / 255.0
            x1 = x[:, 0, :, :, :]
            x2 = x[:, -1, :, :, :]
            x1, x2 = tf(x1), tf(x2)
            loss = ddpm(x1, x2)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        ddpm.eval()
        with torch.no_grad():
            xh, xh_dual = ddpm.sample(8, (1, 32, 32), device)
            xset = torch.cat([xh, xh_dual], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_dual_mnist_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./models/ddpm_dual_mnist.pth")

if __name__ == "__main__":
    os.makedirs('contents',exist_ok=True)
    os.makedirs('models', exist_ok=True)
    #train_cifar10(n_epoch=100, device="cuda", load_pth=None)
    train_dual(n_epoch=100, device="cuda", load_pth=None)