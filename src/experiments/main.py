"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.data.loading import load_dataset


def main(dataset_name: str):
    training_loader, training_set = load_dataset(dataset_name, shuffle=False, apply_augmentation=False)
    print(f"Loaded {dataset_name} with {len(training_set)} samples")
    # Check first batch
    for batch_idx, (data, labels, indices) in enumerate(training_loader):
        print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10, CIFAR-100, or TinyImageNet.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'],
                        help='Dataset name')

    args = parser.parse_args()
    main(args.dataset_name)
