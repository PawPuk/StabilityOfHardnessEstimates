"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
from typing import Dict, List, Tuple, Union

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.hardness.estimators import estimate_instance_hardness, second_split_forgetting_computation
from src.models.neural_networks import ResNetLowRes
from src.utils.io import save_results
from src.utils.reproducibility import compute_current_seed, get_latest_model_index, set_reproducibility


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(
            self,
            training_set_size: int,
            training_loaders: List[DataLoader],
            test_loader: DataLoader,
            dataset_name: str,
            pruning_type: str = 'none',
            resnet_depth: int = 18
    ):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set. This is only useful for experiment1.py.
        :param training_loaders: List of DataLoaders for the training datasets. For experiment1.py where only one
        dataset is used pass the DataLoader in a List.
        :param test_loader: Holdout set used to compute SSFT.
        :param dataset_name: The name of the dataset being used.
        :param pruning_type: Type of pruning being applied (default: 'none'). This is used in experiment2.py and
        experiment3.py to ensure unique saving directories.
        :param resnet_depth: Specifies the depth of the ResNet used for hardness estimation. We use 20 as default.
        """
        self.training_set_size = training_set_size
        self.training_loaders = training_loaders
        self.test_loader = test_loader
        self.pruning_type = pruning_type
        self.dataset_name = dataset_name
        self.resnet_depth = resnet_depth

        self.config = get_config(self.dataset_name)

        self.num_epochs = self.config['num_epochs']
        self.num_models_to_train_per_dataset = self.config['num_models_per_dataset']
        self.dataset_count = self.config['num_datasets']

        self.save_dir = os.path.join(self.config['save_dir'], pruning_type, dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_model(
            self,
            current_dataset_index: int,
            current_model_index: int,
            hardness_estimates: Union[Dict[Tuple[int, int], Dict], None]
    ):
        """Train a single model."""
        dataset_model_id = (current_dataset_index, current_model_index)
        seed = compute_current_seed(self.config, current_dataset_index, current_model_index)
        set_reproducibility(seed)

        model = ResNetLowRes(self.resnet_depth, num_classes=self.config['num_classes']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        for estimator in ['Data Cartography', 'AUM', 'Action Scores', 'EL2N']:
            # hardness_estimates[dataset_model_id][estimator][epoch_index][sample_index]: float
            hardness_estimates[dataset_model_id][estimator] = [[0.0 for _ in range(self.num_epochs)]
                                                               for _ in range(self.training_set_size)]
        perfect_epoch_counter, convergence_epoch = 0, -1

        for epoch in range(self.config['num_epochs']):
            model.train()
            epoch_perfect = True

            for inputs, labels, indices in self.training_loaders[current_dataset_index]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_perfect = estimate_instance_hardness(indices, inputs, outputs, labels, hardness_estimates, epoch,
                                                           dataset_model_id)
                if not batch_perfect:
                    epoch_perfect = False

            if epoch_perfect:
                perfect_epoch_counter += 1
                if perfect_epoch_counter == 5:
                    convergence_epoch = epoch
                    print(f'Convergence achieved at epoch {epoch}.')
            else:
                perfect_epoch_counter = 0

            scheduler.step()

        hardness_estimates[dataset_model_id]['convergence_epoch'] = convergence_epoch
        return model

    def train_ensemble(
            self
    ):
        """Train an ensemble of models."""

        latest_model_indices = get_latest_model_index(self.save_dir, self.dataset_count)

        print(f"Starting training {self.dataset_count} ensembles of {self.num_models_to_train_per_dataset} models each "
              f"on {self.dataset_name}.")

        for dataset_id in tqdm(range(self.dataset_count)):
            for model_id in tqdm(range(latest_model_indices[dataset_id] + 1, self.num_models_to_train_per_dataset)):
                hardness_estimates = {(dataset_id, model_id): {}}
                model = self.train_model(dataset_id, model_id, hardness_estimates)
                second_split_forgetting_computation(model, self.config, self.training_loaders[dataset_id],
                                                    self.test_loader, hardness_estimates, (dataset_id, model_id),
                                                    self.training_set_size)
                save_results(hardness_estimates, (dataset_id, model_id), self.dataset_name)
