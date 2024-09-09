import torch
import torch.nn as nn


def create_residualblock(*, out_channels:int, device:str="cuda") -> callable:
    """
    Create an operation that passes a minibatch through residual and conv
    layers. Return this new operation.
    """

    def residual_block(x: torch.tensor) -> torch.tensor:
        """
        Instantiates multiple layers and apsses the input tensor through.
        input: tensor of shape (batch, channels, height, width)
        output: tensor of shape (batch, num_channels_out, height, width)
        """
        # x.shape (batch, channels height, width)
        num_channels_in = x.shape[1]
        if num_channels_in == out_channels:
            residue = x
        else:
            layer = nn.Conv2d(
                in_channels=num_channels_in,
                out_channels=out_channels,
                kernel_size=1,
                device=device,
            )
            residue = layer(x)

        bn_layer = nn.BatchNorm2d(num_features=num_channels_in, device=device)
        c1_layer = nn.Conv2d(
            in_channels=num_channels_in,
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

        x = bn_layer(x)
        x = a1_layer(c1_layer(x))
        x = c2_layer(x)
        x = x + residue

        return x

    return residual_block


def create_downblock(
    *,
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

    def downblock(x: torch.tensor) -> tuple[torch.tensor, list[torch.tensor]]:
        """ compute the output and all the residuals """
        rblock_outputs = []
        for _ in range(rblocks):
            r_block_i = create_residualblock(
                out_channels=out_channels,
                device=device
            )
            x = r_block_i(x)
            rblock_outputs.append(x)

        pooling_block = nn.AvgPool2d(kernel_size=2)
        x = pooling_block(x)
        return x, rblock_outputs

    return downblock


def create_upblock(
    *,
    out_channels:int,
    device:str="cuda"
) -> callable:
    """
    Create an operation that passes a minibatch through a ddownward block,
    reducing the input (H, W) -> (H/2, W/2)
    num_channels_out:int the number of channels output from the block
    num_rblocks:int the nbumebr of residual blocks tp repeat in the downblock
    """
    def upblock(
        x:torch.tensor,
        down_rblock_outputs: list[torch.tensor],
    ) -> torch.tensor:

        size = down_rblock_outputs[0].shape[2]
        us_1 = nn.Upsample(size=size, mode='bilinear')
        x = us_1(x)

        for downblock_output in down_rblock_outputs:
            x = torch.cat([x, downblock_output], dim=1)
            r_block_i = create_residualblock(
                out_channels=out_channels,
                device=device,
            )
            x = r_block_i(x)

        return x

    return upblock


