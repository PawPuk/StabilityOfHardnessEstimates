"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.data.loading import load_dataset


def main(dataset_name: str):
    training_loader, training_set = load_dataset(dataset_name, shuffle=False, apply_augmentation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset name: CIFAR10 or CIFAR100')

    args = parser.parse_args()
    main(args.dataset_name)
