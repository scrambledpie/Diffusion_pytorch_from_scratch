import torch
from torch import nn

from .residualblock import ResBlock


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

        ARGS:
        num_channels_out:int the number of channels output from the block
        num_rblocks: int the number of residual blocks to repeat in downblock
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
