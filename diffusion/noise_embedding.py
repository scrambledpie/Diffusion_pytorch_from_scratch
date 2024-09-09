import torch


def create_sin_embedding_fun(
    *,
    embedding_dim:int=32,
    tmax:int=1000,
    device:str="cuda"
) -> callable:
    """ Create a function to encode any scalar into sin embeddings """
    assert embedding_dim % 2 ==0, "embedding dim must be even"

    freqs = torch.exp(
        torch.linspace(
            start=torch.log(torch.tensor(1.0)),
            end=torch.log(torch.tensor(tmax)),
            steps=embedding_dim // 2,
        ),
    ).to(device)

    freqs = 2.0 * torch.pi * freqs

    freqs = freqs.view((1, embedding_dim//2, 1, 1))

    def encode_x(x:torch.tensor) -> torch.tensor:
        """ (batch, 1, 1, 1) -> (batch, emb_dim, 1, 1) """
        sin_x = torch.sin(freqs * x)
        cos_x = torch.cos(freqs * x)
        return torch.cat([sin_x, cos_x], dim=1)

    return encode_x

