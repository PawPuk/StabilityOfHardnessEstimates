"""Contains all the functions relating to setting or computing seeds."""

from collections import defaultdict
import os
import random
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def compute_current_seed(
        config: Dict[str, Union[int, float, str, List[int], List[float], List[str], Tuple[float, float, float]]],
        current_dataset_index: int,
        current_model_index: int
) -> int:
    """Compute the seed for training the current model."""
    base_seed = config['probe_base_seed']
    seed_step = config['probe_seed_step']
    dataset_step = config['probe_dataset_step']

    seed = base_seed + current_dataset_index * dataset_step + current_model_index * seed_step
    return seed


def set_reproducibility(seed: int = 42):
    """Sets the seed to specific value for all random events."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True


def get_latest_model_index(save_dir: str, max_dataset_count: int) -> List[int]:
    """Find the latest trained model index for each version of the dataset. This makes it easier to add more models
    to the ensembles, as we don't have to retrain from scratch."""
    max_indices = defaultdict(lambda: -1)  # -1 means that the next index is 0
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            match = re.search(rf'dataset_(\d+)_model_(\d+)\.pth$', filename)
            if match:
                dataset_idx = int(match.group(1))
                model_idx = int(match.group(2))
                max_indices[dataset_idx] = max(max_indices[dataset_idx], model_idx)
    return [max_indices[i] for i in range(max_dataset_count)]
