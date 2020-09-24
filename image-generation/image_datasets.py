#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

from os import getenv, path

import numpy as np
import torch
import torchvision


class ImageGenerationDataset(object):
    def _transform(self, x):
        x = np.array(x, dtype=np.int64).ravel()
        y = (x.astype(np.float32) / 255) * 2 - 1

        return torch.from_numpy(x[:-1]), torch.from_numpy(y[1:])

    @property
    def sequence_length(self):
        return len(self[0][0])


class Imagenet64(ImageGenerationDataset):
    def __init__(self, root, train):
        if train:
            self._data = np.load(
                path.join(root, "imagenet-64x64", "train_data.npy"),
                mmap_mode="r"
            )
        else:
            self._data = np.load(
                path.join(root, "imagenet-64x64", "val_data.npy"),
                mmap_mode="r"
            )

    def __getitem__(self, i):
        return  self._transform(self._data[i])

    def __len__(self):
        return len(self._data)


class TorchvisionDataset(ImageGenerationDataset):
    def __init__(self, dset, root, train):
        self._dataset = dset(root, download=True, train=train)

    def __getitem__(self, i):
        return self._transform(self._dataset[i][0])

    def __len__(self):
        return len(self._dataset)


def add_dataset_arguments(parser):
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "imagenet64"],
        default="mnist",
        help="Choose the dataset"
    )
    parser.add_argument(
        "--dataset_directory",
        default=getenv("DATASET_DIRECTORY", "./data"),
        help="Where to find or place the datasets"
    )


def get_dataset(args):
    root = args.dataset_directory
    dsets = {
        "mnist": torchvision.datasets.MNIST,
        "cifar10": torchvision.datasets.CIFAR10
    }
    if args.dataset == "imagenet64":
        return (
            Imagenet64(root, True),
            Imagenet64(root, False)
        )
    if args.dataset in dsets:
        return (
            TorchvisionDataset(dsets[args.dataset], root, True),
            TorchvisionDataset(dsets[args.dataset], root, False)
        )
    else:
        raise RuntimeError("Dataset {} not available".format(args.dataset))
