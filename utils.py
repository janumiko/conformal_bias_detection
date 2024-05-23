from collections import Counter

import numpy as np
import torch

from constants import IMAGENET_CLASSES_DICT
# ImageClassification
from torchvision.models import ResNet18_Weights


def unnormalize(img: torch.Tensor) -> np.ndarray:
    """Unnormalize the image.

    Args:
        img: The image to unnormalize

    Returns:
        The unnormalized image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.numpy().transpose((1, 2, 0))  # Convert to numpy and change dimensions
    img = std * img + mean
    img = np.clip(img, 0, 1)  # Clip values to range [0, 1]
    return img


def calculate_distances(
    latent_vectors: dict[int, torch.Tensor], target_vector: torch.Tensor, normalize=True
) -> dict[int, float]:
    """Calculate the distances between the latent vectors and the target vector using the L2 norm.

    Args:
        latent_vectors: The latent vectors to compare
        target_vector: The target vector to compare to
        normalize: Whether to normalize the distances

    Returns:
        The distances between the latent vectors and the target vector
    """
    distances = {}
    for key, value in latent_vectors.items():
        distances[key] = round((value - target_vector).norm().item(), 4)

    if not normalize:
        return distances

    min_distance = min(distances.values())
    max_distance = max(distances.values())
    for key in distances:
        distances[key] = (distances[key] - min_distance) / (max_distance - min_distance)

    return distances


def calculate_cosine_similarity(
    latent_vectors: dict[int, torch.Tensor], target_vector: torch.Tensor
) -> dict[int, float]:
    """Calculate the cosine similarity between the latent vectors and the target vector.

    Args:
        latent_vectors: The latent vectors to compare
        target_vector: The target vector to compare to

    Returns:
        The cosine similarities between the latent vectors and the target vector
    """
    similarities = {}
    for key, value in latent_vectors.items():
        cosine_similarity = torch.nn.functional.cosine_similarity(
            value.unsqueeze(0), target_vector.unsqueeze(0)
        ).item()
        similarities[key] = round(cosine_similarity, 4)
    return similarities


def name_mapping_fn(label: int) -> str:
    """Map the label to the name of the class.

    Args:
        label: The label to map

    Returns:
        The name of the class
    """
    if label not in IMAGENET_CLASSES_DICT:
        return label

    return f"{IMAGENET_CLASSES_DICT[label].split(',')[0]} [{label}]"


def get_top_k(counter: Counter, k: int, scale: int = 1) -> Counter:
    """Get the top k items from the counter.

    Args:
        counter: The counter to get the top k items from
        k: The number of items to get
        scale: The scale to divide by

    Returns:
        The top k items from the counter
    """
    return Counter({label: value / scale for label, value in counter.most_common(k)})
