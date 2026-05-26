#!/bin/bash

set -e

echo "========================================"
echo "Running Case Study 1 Experiments"
echo "========================================"

echo "Training models on balanced dataset (the baseline)..."

dataset_names=('CIFAR10' 'CIFAR100' 'TinyImageNet')
noise_ratios=(0.1 0.2 0.3)

for dataset_name in "${dataset_names[@]}"
do
    for noise_ratio in "${noise_ratios[@]}"
    do
        if [ "$dataset_name" == "CIFAR10" ]; then
            dataset_code="C0"
        elif [ "$dataset_name" == "CIFAR100" ]; then
            dataset_code="C00"
        else
            dataset_code='TIN'
        fi

        job_name="${dataset_code}${noise_ratio}"
        log_file="Output/output_estimate_hardness_on_${dataset_name}_with_${noise_ratio}_mislabeled_samples.out"

        sbatch \
            --job-name="$job_name" \
            --output="$log_file" \
            scripts/slurm/train_baseline_models.sh "$dataset_name" "$noise_ratio"
    done
done