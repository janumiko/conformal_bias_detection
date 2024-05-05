import torch
from torch.utils.data import DataLoader
from typing import Callable
import tqdm


def adaptive_conformal_score(logits: torch.Tensor) -> torch.Tensor:
    """Get the adaptive conformal score

    Args:
        logits: The logits

    Returns:
        The adaptive conformal score
    """
    softmax = torch.nn.functional.softmax(logits, dim=1)
    sorted_softmax, indices = torch.sort(softmax, descending=True, dim=1)
    cumsum = torch.cumsum(sorted_softmax, dim=1)
    
    # Use the original indices to put the scores back in the original order
    reordered_scores = cumsum.gather(1, indices)
    
    return reordered_scores


def predict_conformal_set(
    model: torch.nn.Module,
    data: DataLoader,
    score_fn,
    threshold: torch.Tensor,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Get the conformal set for the data

    Args:
        model: The model to use
        data: The data to use
        score_fn: The score function to use
        threshold: The threshold to use
        device: The device to use

    Returns:
        The conformal set
    """
    model.eval()
    with torch.no_grad():
        results = []
        for inputs, _ in tqdm.tqdm(data):
            inputs = inputs.to(device)
            logits = model(inputs)
            score = score_fn(logits)

            score = score <= threshold

            for i in range(inputs.size(0)):
                indices = torch.nonzero(score[i]).squeeze()
                results.append(indices.cpu().numpy())

    return results


def get_calibration_scores(
    model: torch.nn.Module,
    calibrate_data: DataLoader,
    score_fn: Callable,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Get the calibration scores for the calibration data

    Args:
        model: The model to use
        calibrate_data: The calibration data
        score_fn: The score function to use
        device: The device to use

    Returns:
        The calibration scores
    """
    model.eval()
    with torch.no_grad():
        scores = []
        for inputs, targets in tqdm.tqdm(calibrate_data):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            score = score_fn(logits).gather(1, targets.unsqueeze(1))
            scores.append(score)

    return torch.cat(scores)
