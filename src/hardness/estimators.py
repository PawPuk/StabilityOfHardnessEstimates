from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config.config import DEVICE
from src.models.neural_networks import ResNetLowRes


def estimate_instance_hardness(
        batch_indices: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[int, List[float]]]]],
        epoch: int,
        dataset_model_id: Tuple[int, int]
) -> bool:
    """Estimate hardness through confidence, AUM, DataIQ, Loss, and Forgetting. In our work we use AUM as the default
    estimator for resampling and pruning. This function is used in train_ensemble.py and is called only when running
    train_baseline_models.py."""

    # Check if batch is 100% correct
    _, predicted = torch.max(outputs, 1)
    batch_correct = (predicted == labels).sum().item()
    batch_total = len(labels)

    for index_within_batch, (i, x, logits, correct_label) in enumerate(zip(batch_indices, inputs, outputs, labels)):
        i = i.item()
        correct_label = correct_label.item()

        logits = logits.detach()
        correct_logit = logits[correct_label].item()
        probs = torch.nn.functional.softmax(logits, dim=0)
        # Data Cartography (Confidence)
        hardness_estimates[dataset_model_id]['Data Cartography'][i][epoch] = correct_logit
        # AUM
        max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
        hardness_estimates[dataset_model_id]['AUM'][i][epoch] = correct_logit - max_other_logit
        # Action Scores (Loss)
        label_tensor = torch.tensor([correct_label], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
        hardness_estimates[dataset_model_id]['Action Scores'][i][epoch] = loss
        # EL2N
        one_hot = torch.zeros_like(probs)
        one_hot[correct_label] = 1.0
        el2n = torch.norm(probs - one_hot, p=2).item()
        hardness_estimates[dataset_model_id]['EL2N'][i][epoch] = el2n

    return batch_correct == batch_total


@torch.no_grad()
def evaluate_and_update_forgetting(
    model: ResNetLowRes,
    train_loader: DataLoader,
    epoch: int,
    hardness_estimates: Dict[Tuple[int, int], Dict],
    dataset_model_id: Tuple[int, int]
) -> None:
    """
    For each training sample:
      - If it is misclassified AND remembrance[i] == 1, then set SSFT[i] = epoch and remembrance[i] = 0.
    """
    model.eval()
    for inputs, labels, indices in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)
        misclassified_mask = (pred != labels).cpu().numpy()

        for idx, is_mis in zip(indices.cpu().numpy(), misclassified_mask):
            if is_mis and hardness_estimates[dataset_model_id]['remembrance'][idx] == 1:
                hardness_estimates[dataset_model_id]['SSFT'][idx] = epoch
                hardness_estimates[dataset_model_id]['remembrance'][idx] = 0


def second_split_forgetting_computation(
        model: ResNetLowRes,
        config: Dict[str, Any],
        train_loader: DataLoader,
        holdout_loader: DataLoader,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, Union[int, List[List[float]]]]],
        dataset_model_id: Tuple[int, int],
        training_set_size: int
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                          weight_decay=config['weight_decay'], nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_milestones'], gamma=0.2)

    hardness_estimates[dataset_model_id]['SSFT'] = [-1 for _ in range(training_set_size)]
    hardness_estimates[dataset_model_id]['remembrance'] = [1 for _ in range(training_set_size)]
    perfect_epoch_counter = 0

    for epoch in range(config['num_epochs']):
        evaluate_and_update_forgetting(model, train_loader, epoch, hardness_estimates, dataset_model_id)

        model.train()
        epoch_perfect = True

        for inputs, labels, _ in holdout_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = len(labels)

            batch_perfect = batch_total == batch_correct
            if not batch_perfect:
                epoch_perfect = False

        if epoch_perfect:
            perfect_epoch_counter += 1
            if perfect_epoch_counter == 5:
                print(f'Convergence on holdout set achieved at epoch {epoch}.')
                return None
        else:
            perfect_epoch_counter = 0

        scheduler.step()
