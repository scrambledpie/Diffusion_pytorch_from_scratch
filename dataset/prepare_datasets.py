from pathlib import Path

import torch

from .reshape_images import resize_images, stack_images_into_tensor

DATASET_DIR = Path(__file__).parent

FLOWERS_DIR = DATASET_DIR / "flowers"
FLOWERS_64_DIR = DATASET_DIR / "flowers_64"
FLOWERS_64_PTFILE = DATASET_DIR / "flowers_64_tensor.pt"

CELEBA_DIR = DATASET_DIR / "celebA"
CELEBA_SMALL_DIR = DATASET_DIR / "celebA_small"
CELEBA_SMALL_PTFILE = DATASET_DIR / "celebA_small_tensor.pt"
CELEBA_SMALL_10K_PTFILE = DATASET_DIR / "celebA_small_10k.pt"


def main():
    """
    Construct the Tensors that contain the full size datasets
    """

    if not FLOWERS_64_PTFILE.exists():
        resize_images(
            src_dir=FLOWERS_DIR,
            dest_dir=FLOWERS_64_DIR,
            height=64,
            width=64
        )
        stack_images_into_tensor(
            src_dir=FLOWERS_64_DIR,
            output_file=FLOWERS_64_PTFILE,
        )

    if not CELEBA_SMALL_PTFILE.exists():
        resize_images(
            src_dir=CELEBA_DIR,
            dest_dir=CELEBA_SMALL_DIR,
            height=109,
            width=89,
        )
        stack_images_into_tensor(
            src_dir=CELEBA_SMALL_DIR,
            output_file=CELEBA_SMALL_PTFILE,
        )

        celeba_tensor = torch.load(CELEBA_SMALL_PTFILE, weights_only=True)

        # save first the 10k as a smaller training set
        torch.save(
            celeba_tensor[:10000, :, :, :].detach().clone(),
            CELEBA_SMALL_10K_PTFILE,
        )


if __name__ == "__main__":
    main()
