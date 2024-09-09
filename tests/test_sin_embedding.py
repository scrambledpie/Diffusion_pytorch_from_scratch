import unittest

import torch

from diffusion.noise_embedding import create_sin_embedding_fun


class TestSinEmb(unittest.TestCase):
    def test_embedding(self):
        emb_fun = create_sin_embedding_fun(device="cpu")

        x = torch.tensor([[[[0.2]]]]).to("cpu")
        emb = emb_fun(x)
        print(x.shape, emb.shape)


if __name__=="__main__":
    tc = TestSinEmb()
    tc.test_embedding()
