# Diffusion
A basic diffusion model - Denoising Diffusion Probabilistic Models (DDPM) pipeline. Build on top of https://github.com/cloneofsimo/minDiffusion/tree/master.

# Installation
1. Create a conda environment.
```
conda create --name diffusion python=3.9 -y
conda activate diffusion
```

2. Install PyTorch.
```
pip install torch torchvision torchaudio
```

3. Install additional libraries.
```
pip install tqdm
```

# Data Preparation
1. CIFAR10

No preparation needed. Simply create ```./contents``` and ```./models``` folders.

2. Customized dataset

Download your own images into ```./data```.

# Train

# Inference
