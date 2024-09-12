# Datasets

- Flowers, 8189 pictures from [https://www.robots.ox.ac.uk/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)

- CelebA faces, 201,000 pictures [kaggle celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)


We preprocess the data by reducing the size of the images (to make training easier). We could simply point a pytorch dataloader at any shrunken image folder, but we go one step further and loading all of the images and save a single ready-to-go massive tensor. This massive tensor is then loaded into cpu RAM and used as the source for the pytorch dataset class and dataloader.
- download the images and save them into local folders "flowers" and "celebA" (the folder names are in `prepare_datasets.py`)
- create the tensors `python prepare_datasets.py`

Then we may use the datasets `from datasets import Flowers` or `from datasets import CelebA10k` in the pytorch dataloader.



