from math import ceil
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import Counter
from pathlib import Path


def unnormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.numpy().transpose((1, 2, 0))  # Convert to numpy and change dimensions
    img = std * img + mean
    img = np.clip(img, 0, 1)  # Clip values to range [0, 1]
    return img


def visualize_imagenet_images(
    dataset: torch.utils.data.Dataset | torch.utils.data.Subset,
    number_of_images: int = 10,
) -> None:
    number_of_plots = ceil(number_of_images / 10)

    for plot_index in range(number_of_plots):
        start_index = plot_index * 10
        end_index = min((plot_index + 1) * 10, number_of_images)

        # Get the images and their labels for the current plot
        images_and_labels = list(
            zip(*[dataset[i] for i in range(start_index, end_index)])
        )
        images, labels = images_and_labels[0], images_and_labels[1]

        # Create a grid of subplots
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        # For each image and its corresponding label
        for i, (img, _) in enumerate(zip(images, labels)):
            img = unnormalize(img)
            # Plot the image in the corresponding subplot
            ax = axs[i // 5, i % 5]
            ax.imshow(img)
            ax.axis("off")

        fig.show()


def get_top_k(
    counter: Counter, k: int, name_mapping: dict[int, str]
) -> list[tuple[str, int]]:
    """Get the top k items from the counter."""
    top_k = [
        (name_mapping[label] if label in name_mapping else "EMPTY", value)
        for label, value in counter.most_common(k)
    ]
    return top_k


def plot_top_k_predictions(
    counters: dict[float, Counter],
    k: int,
    name_mapping: dict[int, str],
    model_name: str,
    class_idx: int,
    save_root: str = None,
):
    """Plot the top k predictions for each alpha value."""
    _, axs = plt.subplots(1, len(counters), figsize=(4 * len(counters), 4))

    for i, (alpha, counter) in enumerate(counters.items()):
        labels, values = zip(*get_top_k(counter, k, name_mapping))
        axs[i].bar(labels, values)
        axs[i].set_title(f"Alpha: {alpha}")
        axs[i].set_xticks(range(len(labels)))
        axs[i].set_xticklabels(labels, rotation=90)

    plt.suptitle(
        f'Top {k} predictions for class "{name_mapping[class_idx]}" using {model_name}'
    )
    
    if save_root:
        Path(save_root).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_root}/{model_name}_top_{k}_{name_mapping[class_idx]}.png")

    plt.show()
