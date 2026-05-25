from typing import Dict, List, Tuple, Union

import torch


def estimate_instance_hardness(
        batch_indices: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[int, List[float]]]]],
        epoch: int,
        dataset_model_id: Tuple[int, int]
):
    """Estimate hardness through confidence, AUM, DataIQ, Loss, and Forgetting. In our work we use AUM as the default
    estimator for resampling and pruning. This function is used in train_ensemble.py and is called only when running
    train_baseline_models.py."""

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
        hardness_estimates[dataset_model_id]['Loss'][i][epoch] = loss
        # EL2N
        one_hot = torch.zeros_like(probs)
        one_hot[correct_label] = 1.0
        el2n = torch.norm(probs - one_hot, p=2).item()
        hardness_estimates[dataset_model_id]['EL2N'][i][epoch] = el2n
