"""The data module: Provides two core Dataset subclasses, and the method for loading the data."""

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from src.config.config import get_config, ROOT
from src.data.datasets import IndexedDataset


def get_transform(
        apply_augmentation: bool,
        config: Dict[Any]
) -> transforms.Compose:
    """For getting the transformation to the training and test sets."""
    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    return train_transform


def worker_init_fn(worker_id):
    """Set the seed for workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def get_dataloader(
        dataset: IndexedDataset,
        batch_size: int,
        shuffle: bool = False
):
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, worker_init_fn=worker_init_fn)


def load_dataset(
        dataset_name: str,
        shuffle: bool,
        apply_augmentation: bool
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset]:
    """Load the dataset giving control over shuffling and augmentation. Currently only supports CIFAR10 and CIFAR100.

    :param dataset_name: Name of the dataset to load (only accepts `CIFAR10` and `CIFAR100`).
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.

    :return: Tuple containing DataLoader for the training set, training set, DataLoader for the test set, and test set.
    """
    config = get_config(dataset_name)

    train_transform = get_transform(apply_augmentation, config)
    if dataset_name == 'CIFAR10':
        training_set = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, 'data'), download=True,
                                                    transform=train_transform)
    else:
        training_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), download=True,
                                                     transform=train_transform)

    training_set = IndexedDataset(training_set, apply_augmentation)

    training_loader = get_dataloader(training_set, config['batch_size'], shuffle)

    return training_loader, training_set
