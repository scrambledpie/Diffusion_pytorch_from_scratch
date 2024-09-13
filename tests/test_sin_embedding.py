import unittest

import torch

from diffusion.unet.noise_embedding import create_sin_embedding_fun


class TestSinEmb(unittest.TestCase):
    def test_embedding(self):
        emb_fun = create_sin_embedding_fun(device="cpu", embedding_dim=32)

        x = torch.tensor([[[[0.2]]]]).to("cpu")
        emb = emb_fun(x)

        assert tuple(emb.shape) == (1,32,1,1)


if __name__=="__main__":
    tc = TestSinEmb()
    tc.test_embedding()
