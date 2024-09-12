import torch
from torch.utils.data import Dataset

from prepare_datasets import (
    FLOWERS_64_PTFILE,
    CELEBA_SMALL_PTFILE,
    CELEBA_SMALL_10K_PTFILE,
)


class TensorDataset(Dataset):
    """ All datasets are stored as tensors and fully loaded into RAM """
    _data : torch.tensor = None

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, index) -> torch.tensor:
        return self._data[index, :, :, :]


class Flowers(TensorDataset):
    """ Oxford Flowers (dataset_size, C, H, W) = (8189, 3, 64, 64) """
    _data = torch.load(FLOWERS_64_PTFILE, weights_only=True)


class CelebA(TensorDataset):
    """ Full CelebA dataset (dataset_size, C, H, W) = (201000, 3, 109, 89) """
    _data = torch.load(CELEBA_SMALL_PTFILE, weights_only=True)


class CelebA10k(TensorDataset):
    """ First 10k of CelebA (dataset_size, C, H, W) = (10000, 3, 109, 89) """
    _data = torch.load(CELEBA_SMALL_10K_PTFILE, weights_only=True)
