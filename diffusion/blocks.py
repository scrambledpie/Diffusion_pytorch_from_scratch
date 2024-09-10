import torch
import torch.nn as nn


def create_resblock(
        *, in_channels:int, out_channels:int, device:str="cuda"
    ) -> callable:
    """
    Create an operation that passes a minibatch through residual and conv
    layers. Return this new operation.
    """
    reshape_residue = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        device=device,
    )
    bn_layer = nn.BatchNorm2d(num_features=in_channels, device=device)
    c1_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding="same",
        device=device,
    )
    a1_layer = nn.SiLU()
    c2_layer = nn.Conv2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding="same",
        device=device,
    )

    def residual_block(x: torch.tensor) -> torch.tensor:
        """
        Instantiates multiple layers and apsses the input tensor through.
        input: tensor of shape (batch, channels, height, width)
        output: tensor of shape (batch, num_channels_out, height, width)
        """
        # x.shape (batch, channels, height, width)
        residue = x
        if x.shape[1] != out_channels:
            residue = reshape_residue(residue)
        x = bn_layer(x)
        x = a1_layer(c1_layer(x))
        x = c2_layer(x)
        x = x + residue

        return x

    return residual_block


def create_downblock(
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
    # first block chanegs channels
    r_blocks = [create_resblock(
            in_channels=in_channels,
            out_channels=out_channels,
            device=device
        )]

    # subsequent blocks preserve channels
    r_blocks += [
        create_resblock(
            in_channels=out_channels,
            out_channels=out_channels,
            device=device
        )
        for _ in range(rblocks-1)
    ]
    pooling_block = nn.AvgPool2d(kernel_size=2)

    def downblock(x: torch.tensor) -> tuple[torch.tensor, list[torch.tensor]]:
        """ compute the output and all the residuals """
        rblock_outputs = []
        for r_block_i in r_blocks:
            x = r_block_i(x)
            rblock_outputs.append(x)
        x = pooling_block(x)
        return x, rblock_outputs

    return downblock


def create_upblock(
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

    # first block chanegs channels
    r_blocks = [create_resblock(
            in_channels=in_channels + downblock_channels,
            out_channels=out_channels,
            device=device
        )]

    # subsequent blocks preserve channels
    r_blocks += [
        create_resblock(
            in_channels=out_channels + downblock_channels,
            out_channels=out_channels,
            device=device
        )
        for _ in range(rblocks-1)
    ]

    def upblock(
        x:torch.tensor,
        down_rblock_outputs: list[torch.tensor],
    ) -> torch.tensor:

        assert len(down_rblock_outputs) == len(r_blocks)

        size = down_rblock_outputs[0].shape[2]
        us_1 = nn.Upsample(size=size, mode='bilinear')
        x = us_1(x)

        for r_block_i, downblock_output in zip(r_blocks, down_rblock_outputs):
            x = torch.cat([x, downblock_output], dim=1)
            x = r_block_i(x)

        return x

    return upblock


