"""The data module: Provides two core Dataset subclasses, and the method for loading the data."""

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from src.config.config import get_config, ROOT
from src.data.datasets import IndexedDataset, TinyImageNet


def get_transform(
        apply_augmentation: bool,
        config: Dict[str, Any]
) -> transforms.Compose:
    """For getting the transformation to the training and test sets."""
    if apply_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config['crop_size'], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    return transform


def inject_label_noise(
        dataset: Dataset,
        noise_ratio: float,
        num_classes: int
) -> Dataset:
    """
    Inject label noise by randomly flipping a fraction of labels.

    Args:
        dataset: PyTorch dataset with .targets attribute (list of labels)
        noise_ratio: Fraction of labels to corrupt (e.g., 0.1 = 10% noise)
        num_classes: Number of classes in the dataset

    Returns:
        Dataset with corrupted labels (modifies in-place)
    """
    targets = dataset.targets
    num_samples = len(targets)
    num_noisy = int(num_samples * noise_ratio)
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

    for idx in noisy_indices:
        current_label = targets[idx]
        # Choose a different random class (not the current one)
        possible_labels = list(range(num_classes))
        possible_labels.remove(current_label)
        new_label = np.random.choice(possible_labels)
        targets[idx] = new_label

    dataset.targets = targets

    return dataset


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


def load_training_dataset(
        dataset_name: str,
        label_noise_ratio: float,
        shuffle: bool,
        apply_augmentation: bool
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset]:
    """Load the training dataset giving control over shuffling and augmentation.

    :param dataset_name: Name of the dataset to load.
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.
    :param label_noise_ratio: Fraction of labels to corrupt (e.g., 0.1 = 10% noise).

    :return: Tuple containing DataLoader for the training set, and the training set.
    """
    config = get_config(dataset_name)

    transform = get_transform(apply_augmentation, config)
    if dataset_name == 'CIFAR10':
        training_set = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, 'data'), download=True,
                                                    transform=transform)
    elif dataset_name == 'CIFAR100':
        training_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), download=True,
                                                     transform=transform)
    else:
        training_set = TinyImageNet(root=os.path.join(ROOT, 'data'), download=True, transform=transform)

    if label_noise_ratio > 0:
        training_set = inject_label_noise(training_set, label_noise_ratio, config['num_classes'])

    training_set = IndexedDataset(training_set, apply_augmentation)
    training_loader = get_dataloader(training_set, config['batch_size'], shuffle)

    return training_loader, training_set


def load_holdout_dataset(
        dataset_name: str,
        shuffle: bool,
        apply_augmentation: bool
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset]:
    """Load the test dataset giving control over shuffling and augmentation.

    :param dataset_name: Name of the dataset to load.
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.

    :return: Tuple containing DataLoader for the holdout set, and the holdout set.
    """
    config = get_config(dataset_name)

    transform = get_transform(apply_augmentation, config)
    if dataset_name == 'CIFAR10':
        holdout_set = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, 'data'), train=False, download=True,
                                                   transform=transform)
    elif dataset_name == 'CIFAR100':
        holdout_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), train=False, download=True,
                                                    transform=transform)
    else:
        holdout_set = TinyImageNet(root=os.path.join(ROOT, 'data'), split='val', download=True, transform=transform)

    holdout_set = IndexedDataset(holdout_set, apply_augmentation)
    holdout_loader = get_dataloader(holdout_set, config['batch_size'], shuffle)

    return holdout_loader, holdout_set
