import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, *, in_channels:int, out_channels:int, device:str="cuda"):
        """
        Create an operation that passes a minibatch through residual and conv
        layers. Return this new operation.
        """
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        if in_channels != out_channels:
            self.reshape_residue = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                device=device,
            )
        else:
            self.reshape_residue = lambda x: x

        self.bn_layer = nn.BatchNorm2d(num_features=in_channels, device=device)
        self.c1_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            device=device,
        )
        self.a1_layer = nn.SiLU()
        self.c2_layer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            device=device,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Instantiates multiple layers and apsses the input tensor through.
        input: tensor of shape (batch, channels, height, width)
        output: tensor of shape (batch, num_channels_out, height, width)
        """
        # x.shape (batch, channels, height, width)
        # residue = x
        # if x.shape[1] != self.out_channels:
        residue = self.reshape_residue(x)
        x = self.bn_layer(x)
        x = self.a1_layer(self.c1_layer(x))
        x = self.c2_layer(x)
        x = x + residue

        return x
