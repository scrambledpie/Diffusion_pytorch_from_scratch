import unittest

import torch

from diffusion.blockclasses import (
    DownBlock,
    ResBlock,
    UpBlock,
)


class TestBlocks(unittest.TestCase):
    def test_residual_block(self):
        """ Make sure the residual block preserves shape """
        x_minibatch = torch.rand((128, 3, 63, 63))

        r1_block = ResBlock(in_channels=3, out_channels=3, device="cpu")
        x_1 = r1_block(x_minibatch)
        assert x_minibatch.shape == x_1.shape

        r1_block = ResBlock(in_channels=3, out_channels=6, device="cpu")
        x_1 = r1_block(x_minibatch)
        for i in range(4):
            if i ==1:
                continue
            assert x_minibatch.shape[i] == x_1.shape[i]


    def test_downblock(self):
        """ Make sure the downblock actually works """
        batch_size = 128
        num_channels = 3
        height = 64
        width = 64
        x_minibatch = torch.rand((batch_size, num_channels, height, width))

        # instantaite layers
        d1_block = DownBlock(
            in_channels=3, out_channels=6, rblocks=2, device="cpu"
        )
        u1_block = UpBlock(
            in_channels=6,
            downblock_channels=6,
            out_channels=num_channels,
            device="cpu",
        )

        # execute layers
        x_1, rblock_outputs_1 = d1_block(x_minibatch)
        x_minibatch_recon = u1_block(x_1, rblock_outputs_1)

        # ansurte input and output size match
        assert x_minibatch.shape == x_minibatch_recon.shape


if __name__=="__main__":
    tc = TestBlocks()
    tc.test_downblock()
