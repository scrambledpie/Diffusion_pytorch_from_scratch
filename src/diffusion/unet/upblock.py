import torch
from torch import nn

from .residualblock import ResBlock


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
        reducing the input (H, W) -> (H*2, W*2). The output size is determined
        by the corresponding downblock.

        ARGS
        num_channels_out:int the number of channels output from the block
        num_rblocks:int the number of residual blocks to repeat in the upblock
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