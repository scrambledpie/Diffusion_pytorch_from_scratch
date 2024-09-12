import multiprocessing as mp
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def reshape_file(
    src_file:Path,
    dst_file:Path,
    height:int,
    width:int,
) -> None:
    """ Load an image file, reshape it and save it to a new file """
    img = Image.open(src_file)
    resize_transform = transforms.Resize((height, width))
    resized_img = resize_transform(img)
    resized_img.save(dst_file)


def resize_images(
    src_dir:Path,
    dest_dir:Path,
    height:int=64,
    width:int=64,
    n_threads:int=32,
) -> None:
    """ Read all images from a folder, resize and save to a new folder """
    files = list(src_dir.glob("*.jpg"))

    print(f"Found {len(files)} files, resizing....", end="")

    args_tups = [(f, dest_dir / f.name, height, width) for f in files]
    with mp.Pool(n_threads) as p:
        p.starmap(reshape_file, args_tups)

    print("Done")


def stack_images_into_tensor(
    src_dir : Path,
    output_file: Path,
) -> None:
    """
    From a folder, load all images and save as a pytorch tensor
    """
    to_tensor = transforms.ToTensor()
    files = list(src_dir.glob("*.jpg"))

    print(f"Found {len(files)} images, loading as a tensor...", end="")
    tensors = []
    for i, image_path in enumerate(files):
        image = Image.open(image_path)
        image_tensor = to_tensor(image)
        tensors.append(image_tensor)

        if i % 1000 == 0:
            print(i)

    x = torch.stack(tensors)
    torch.save(x, output_file)

    prod = 1
    for d in x.shape:
        prod *= d

    memory_size = prod * 4 / (1024 * 1024)
    print(f"Saved {output_file} {x.shape=} {x.dtype=} {memory_size}MB")
