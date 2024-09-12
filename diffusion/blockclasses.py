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


class DownBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels:int,
        out_channels:int,
        rblocks:int,
        device:str="cuda",
    ) -> callable:
        """
        Create an operation that passes a minibatch through a ddownward block,
        reducing the input (H, W) -> (H/2, W/2)
        num_channels_out:int the number of channels output from the block
        num_rblocks:int the nbumebr of residual blocks tp repeat in the downblock
        """
        super(DownBlock, self).__init__()
        # first block chanegs channels
        self.r_blocks = nn.ModuleList()
        self.r_blocks.append(
            ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                device=device
            )
        )

        # subsequent blocks preserve channels
        for _ in range(rblocks-1):
            self.r_blocks.append(
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    device=device
                )
            )
        self.pooling_block = nn.AvgPool2d(kernel_size=2)

    def forward(
        self,
        x: torch.tensor
    ) -> tuple[torch.tensor, list[torch.tensor]]:
        """ compute the output and all the residuals """
        rblock_outputs = []
        for r_block_i in self.r_blocks:
            x = r_block_i(x)
            rblock_outputs.append(x)
        x = self.pooling_block(x)
        return x, rblock_outputs


class UpBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels:int,
        out_channels:int,
        downblock_channels:int,
        rblocks:int=2,
        device:str="cuda"
    ) -> callable:
        """
        Create an operation that passes a minibatch through a ddownward block,
        reducing the input (H, W) -> (H/2, W/2)
        num_channels_out:int the number of channels output from the block
        num_rblocks:int the nbumebr of residual blocks tp repeat in the downblock
        """
        super(UpBlock, self).__init__()

        self.r_blocks = nn.ModuleList()

        # first block chanegs channels
        self.r_blocks.append(
            ResBlock(
                in_channels=in_channels + downblock_channels,
                out_channels=out_channels,
                device=device
            )
        )

        # subsequent blocks preserve channels
        for _ in range(rblocks - 1):
            self.r_blocks.append(
                ResBlock(
                    in_channels=out_channels + downblock_channels,
                    out_channels=out_channels,
                    device=device
                )
            )

    def forward(
        self,
        x:torch.tensor,
        dblock_outputs: list[torch.tensor],
    ) -> torch.tensor:

        assert len(dblock_outputs) == len(self.r_blocks)

        size = dblock_outputs[0].shape[2:]
        us_1 = nn.Upsample(size=size, mode='bilinear')
        x = us_1(x)

        for r_block_i, dblock_output_i in zip(self.r_blocks, dblock_outputs):
            x = torch.cat([x, dblock_output_i], dim=1)
            x = r_block_i(x)

        return x


