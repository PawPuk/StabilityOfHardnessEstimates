"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.config.config import get_config
from src.data.loading import load_training_dataset, load_holdout_dataset
from src.training.train import ModelTrainer
from src.utils.reproducibility import set_reproducibility


def main(dataset_name: str, label_noise_ratio: float):
    config = get_config(dataset_name)
    set_reproducibility()  # For noise injection

    training_loaders, training_set_size = [], 0
    for _ in range(config['num_datasets']):
        training_loader, training_set = load_training_dataset(dataset_name, label_noise_ratio, shuffle=False,
                                                              apply_augmentation=False)
        training_set_size = len(training_set)
        training_loaders.append(training_loader)

    holdout_loader, holdout_set = load_holdout_dataset(dataset_name, shuffle=False, apply_augmentation=False)

    trainer = ModelTrainer(training_set_size, training_loaders, holdout_loader, dataset_name, label_noise_ratio)

    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10, CIFAR-100, or TinyImageNet.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'],
                        help='Dataset name')
    parser.add_argument('--noise_ratio', type=float, required=True,
                        help='Ratio of noise to be injected to the training dataset.')

    args = parser.parse_args()
    main(args.dataset_name, args.noise_ratio)
