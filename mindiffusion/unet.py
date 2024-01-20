"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.

        Code reference: https://github.com/hanyun2019/difussion-model-code-implementation/blob/dm-project-haowen-mac/diffusion_utilities.py
        '''
        self.input_dim = input_dim

        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)

class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))

        return out

class ContextUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256, encoding: str = 'onehot', nc_feat: int = 10) -> None:
        super(ContextUnet, self).__init__()
        if encoding == 'clip':
            self.clip_model, _ = clip.load("ViT-B/32")
            # freeze model
            for param in self.clip_model.parameters():
                param.requires_grad = False
            # construct a codebook for CIFAR10
            self.codebook = {0: 'a photo of an airplane',
                             1: 'a photo of an automobile',
                             2: 'a photo of a bird',
                             3: 'a photo of a cat',
                             4: 'a photo of a deer',
                             5: 'a photo of a dog',
                             6: 'a photo of a frog',
                             7: 'a photo of a horse',
                             8: 'a photo of a ship',
                             9: 'a photo of a truck'
                             }
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat
        self.encoding = encoding
        self.nc_feat = nc_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed1 = TimeSiren(2 * n_feat)
        self.timeembed2 = TimeSiren(1 * n_feat)
        self.contextembed1 = EmbedFC(nc_feat, 2 * n_feat)
        self.contextembed2 = EmbedFC(nc_feat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def text_feature_extractor(self, label):
        '''
        Extract text feature through 1) One-hot encoding and 2) CLIP encoder.
        Ref: https://github.com/openai/CLIP

        Input:
        label: torch tensor with shape (batch, )
        encoding: string options of ['onehot', 'clip']

        Output:
        feature: torch tensor with shape (batch, number of features)
        '''
        if self.encoding == 'onehot':
            vec = F.one_hot(label, self.nc_feat).float() # (batch, 10)
        elif self.encoding == 'clip':
            text_batch = []
            for l in label:
                text_batch.append(self.codebook[int(l)])
            text = clip.tokenize(text_batch).to(label)
            with torch.no_grad():
                vec = self.clip_model.encode_text(text) # (batch, 512)
        else:
            print("Encoding method not supported.")
            exit(-1)

        return vec

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

        x = self.init_conv(x)
        c = self.text_feature_extractor(c)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.nc_feat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 1, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 1, 1, 1)

        thro = self.up0(thro)

        up1 = self.up1(cemb1*thro, down3) + temb1
        up2 = self.up2(up1, down2)
        up3 = self.up3(cemb2*up2, down1) + temb2

        out = self.out(torch.cat((up3, x), 1))

        return out