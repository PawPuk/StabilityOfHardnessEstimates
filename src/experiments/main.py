"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.config.config import get_config
from src.data.loading import load_training_dataset, load_holdout_dataset
from src.training.train import ModelTrainer


def main(dataset_name: str):
    config = get_config(dataset_name)

    training_loaders, training_set_size = [], 0
    load_holdout_dataset(dataset_name, shuffle=False, apply_augmentation=False)
    """for _ in range(config['num_datasets']):
        training_loader, training_set = load_training_dataset(dataset_name, shuffle=False, apply_augmentation=False)
        training_set_size = len(training_set)
        training_loaders.append(training_loader)

    holdout_loader, holdout_set = load_holdout_dataset(dataset_name, shuffle=False, apply_augmentation=False)

    trainer = ModelTrainer(training_set_size, training_loaders, holdout_loader, dataset_name, noise_ratio=10)

    trainer.train_ensemble()"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10, CIFAR-100, or TinyImageNet.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'],
                        help='Dataset name')

    args = parser.parse_args()
    main(args.dataset_name)
