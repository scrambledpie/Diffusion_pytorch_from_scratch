import unittest
from collections import OrderedDict
from pprint import pprint

import torch

from diffusion.unet.unet import UNet

DEVICE = "cpu"

def get_shape(obj, printout:bool=True):
    """
    Scan a nested dict/list for tensors and print out all their shapes in the
    same nested structure.
    """
    if isinstance(obj, list):
        output = [get_shape(x, False) for x in obj]

    elif isinstance(obj, dict) or isinstance(obj, OrderedDict):
        output = {k:get_shape(v, False) for k, v in obj.items()}

    elif isinstance(obj, torch.Tensor):
        output = obj.shape

    else:
        raise TypeError(f"WTF if this?? {type(obj)}")

    if printout:
        pprint(output)
    else:
        return output


class TestUNet(unittest.TestCase):
    def test_unet(self):
        """ perform one forward pass and ensure shapes are preserved """

        # inputs
        x_minibatch = torch.rand((128, 3, 103, 91)).to(DEVICE)
        noise_var = torch.rand((128, 1, 1, 1)).to(DEVICE)

        # model
        unet = UNet(device=DEVICE)

        # output
        # x_predict = unet.forward(x_minibatch, noise_var)
        x_predict = unet(x_minibatch, noise_var)

        assert x_predict.shape == x_minibatch.shape

        get_shape(unet.state_dict())


if __name__ == "__main__":
    tc = TestUNet()
    tc.test_unet()
