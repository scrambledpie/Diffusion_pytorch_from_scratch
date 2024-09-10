import unittest

import torch

from diffusion.unet import UNet

DEVICE = "cpu"


class TestUNet(unittest.TestCase):
    def test_unet(self):
        """ perform one forward pass and ensure shapes are preserved """

        # inputs
        x_minibatch = torch.rand((128, 3, 64, 64)).to(DEVICE)
        noise_var = torch.rand((128, 1, 1, 1)).to(DEVICE)

        # model
        unet = UNet(device=DEVICE)

        # output
        # x_predict = unet.forward(x_minibatch, noise_var)
        x_predict = unet(x_minibatch, noise_var)


        assert x_predict.shape == x_minibatch.shape


if __name__ == "__main__":
    tc = TestUNet()
    tc.test_unet()
