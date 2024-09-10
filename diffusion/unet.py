import torch
from torch import nn

from .blocks import create_downblock, create_resblock, create_upblock
from .noise_embedding import create_sin_embedding_fun


class UNet(nn.Module):
    def __init__(self, device:str="cuda"):
        super(UNet, self).__init__()

        self.device = device

        self.noise_upsample = nn.Upsample(size=64, mode='bilinear')
        self.c_start = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=1, device=device
        )

        kwargs = dict(rblocks=2, device=device)

        self.d1 = create_downblock(in_channels=64, out_channels=32, **kwargs)
        self.d2 = create_downblock(in_channels=32, out_channels=64, **kwargs)
        self.d3 = create_downblock(in_channels=64, out_channels=96, **kwargs)

        self.r1 = create_resblock(
            in_channels=96, out_channels=128, device=device
        )
        self.r2 = create_resblock(
            in_channels=128, out_channels=128, device=device
        )

        # upblock take both x and downblock and concat as input
        self.u3 = create_upblock(
            in_channels=128, downblock_channels=96, out_channels=96, **kwargs
        )
        self.u2 = create_upblock(
            in_channels=96, downblock_channels=64, out_channels=64, **kwargs
        )
        self.u1 = create_upblock(
            in_channels=64, downblock_channels=32, out_channels=32, **kwargs
        )

        self.c_end = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=1, device=device
        )

        self.embedding_fun = create_sin_embedding_fun(
            embedding_dim=32,
            tmax=1000,
            device=device,
        )

    def forward(
        self,
        x_input:torch.tensor,  # (batch, 3, 64, 64)
        noise_var:torch.tensor,  # (batch, 1, 1, 1)
    ) -> torch.tensor:

        assert x_input.shape[1] == 3
        assert x_input.shape[2] == 64
        assert x_input.shape[3] == 64

        assert noise_var.shape[0] == x_input.shape[0]
        assert noise_var.shape[1] == 1
        assert noise_var.shape[2] == 1
        assert noise_var.shape[3] == 1

        x_input, noise_var = x_input.to(self.device), noise_var.to(self.device)

        noise_emb = self.embedding_fun(noise_var)
        noise_emb = self.noise_upsample(noise_emb)

        # (batch, 64, 64, 64)
        x = torch.cat([self.c_start(x_input), noise_emb], dim=1)

        x, d_res_1 = self.d1(x)
        x, d_res_2 = self.d2(x)
        x, d_res_3 = self.d3(x)

        x = self.r1(x)
        x = self.r2(x)

        x = self.u3(x, d_res_3)
        x = self.u2(x, d_res_2)
        x = self.u1(x, d_res_1)

        x = self.c_end(x)

        return x







