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
pip install tqdm openai-clip
```

# Data Preparation
1. CIFAR10

No preparation needed.

2. Customized dataset

Download your own images into ```./data``` and write your customized data loader.

# Train
Run the below commad. You can specify different models (NaiveUnet, ContextUnet) and other hyperparamters.
```
python train.py
```

# Inference
Run the below command.
```
python inference.py
```

# Results
1. DDPM 1000 steps CIFAR10 without conditions and trained with 100 epochs.

![Generated images](https://github.com/liuch37/diffusion/blob/main/misc/ddpm_sample_cifar_99_naiveunet.png)

2. DDPM 1000 steps CIFAR10 with one-hot encoding class conditions and trained with 100 epochs.

![Generated images](https://github.com/liuch37/diffusion/blob/main/misc/ddpm_sample_cifar_99_contextunet.png)

3. DDPM 1000 steps CIFAR10 with text embedding class conditions and trained with 100 epochs.

# Checkpoints
1. DDPM without conditions trained with 100 epochs on CIFAR10.

2. DDPM with one-hot encoding class conditions and trained with 100 epochs on CIFAR10.

3. DDPM with text embedding class conditions and trained with 100 epochs on CIFAR10.
