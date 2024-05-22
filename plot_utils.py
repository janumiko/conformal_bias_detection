from collections import Counter
from math import ceil
from pathlib import Path
from typing import Callable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset

from constants import IMAGENET_CLASSES_DICT
from utils import (
    calculate_cosine_similarity,
    calculate_distances,
    name_mapping_fn,
    unnormalize,
)


def plot_experiment_results(
    target_class: int,
    counters: dict[float, Counter],
    latent_vectors: dict[int, torch.Tensor],
    model_name: str,
    class_dataset: Dataset | Subset = None,
    save_path: str = None,
) -> None:
    """Plot the predictions, distances and similarities of the model for the target class.

    Args:
        target_class: The target class to compare the predictions to
        counters: The counters of the predictions for different alpha values
        latent_vectors: The latent vectors of the dataset
        model_name: The name of the model
        class_dataset: The dataset of the classes
        save_path: The path to save the plot
    """
    # Calculate distances and similarities
    target_vector = latent_vectors[target_class]
    distances = calculate_distances(latent_vectors, target_vector)
    similarities = calculate_cosine_similarity(latent_vectors, target_vector)

    # Plot the predictions and distances
    _, axs = plt.subplots(len(counters), 3, figsize=(12, 20))
    for i, (alpha, counter) in enumerate(counters.items()):
        plot_predictions(
            axs[i, 0], counter, name_mapping_fn=name_mapping_fn, title=f"Alpha={alpha}"
        )
        plot_distances(
            axs[i, 1],
            counter,
            similarities,
            name_mapping_fn=name_mapping_fn,
            title="Cosine similarity",
        )
        plot_distances(
            axs[i, 2],
            counter,
            distances,
            name_mapping_fn=name_mapping_fn,
            title="L2 norm",
        )

    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name}.png")

    plt.show()

    # Visualize the images if a dataset is provided
    if class_dataset:
        plot_images(class_dataset, len(class_dataset), save_path=save_path)


def plot_images(
    dataset: Dataset | Subset,
    number_of_images: int = 10,
    save_path: str = None,
) -> None:
    """Plot the images in the dataset.

    Args:
        dataset: The dataset to visualize
        number_of_images: The number of images to visualize
        save_path: The path to save the images
    """
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

        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{save_path}/images_{plot_index}.png")

        fig.show()


def plot_predictions(
    subplot: plt.Axes,
    counter: Counter,
    name_mapping_fn: Callable = None,
    title: str = None,
) -> None:
    """Plot the predictions.

    Args:
        subplot: The subplot to plot the predictions on
        counter: The counter of the predictions
        name_mapping_fn: The function to map the labels to names
        title: The title of the plot
    """
    labels, values = zip(*reversed(counter.items()))

    if name_mapping_fn:
        labels = [name_mapping_fn(label) for label in labels]

    subplot.barh(range(len(labels)), values)
    subplot.set_yticks(range(len(labels)))
    subplot.set_yticklabels(labels)

    if title:
        subplot.set_title(title)


def plot_distances(
    subplot: plt.Axes,
    counter: Counter,
    distances_to_target: dict[int, float],
    name_mapping_fn: Callable = None,
    title: str = None,
    draw_names: bool = False,
) -> None:
    """Plot the distances to the target class.

    Args:
        subplot: The subplot to plot the distances on
        counter: The counter of the predictions
        distances_to_target: The distances to the target class
        name_mapping_fn: The function to map the labels to names
        title: The title of the plot
        draw_names: Whether to draw the names of the classes
    """
    for label in counter:
        counter[label] = distances_to_target.get(label, 0)

    labels, values = zip(*reversed(counter.items()))

    if name_mapping_fn:
        labels = [name_mapping_fn(label) for label in labels]

    subplot.barh(labels, values)
    subplot.set_yticks(range(len(labels)))
    subplot.set_yticklabels(labels if draw_names else [])
    subplot.set_xlim(0, 1)

    if title:
        subplot.set_title(title)


def plot_classes_tsne(
    class_latent_vectors: dict[int, torch.Tensor],
    save_path: str = None,
    fig_size: tuple[int, int] = (10, 10),
    dimension: int = 2,
    label_dict: dict[int, str] = IMAGENET_CLASSES_DICT,
) -> None:
    """Plot the classes in a t-SNE plot.

    Args:
        class_latent_vectors (dict[int, Tensor]): The latent vectors of the classes
        save_path (Optional[str]): The path to save the plot
        fig_size (tuple[int, int]): The size of the figure
        dimension (int): The dimension of the t-SNE plot
    """
    # Get the number of the latent vectors for each class
    class_latent_vec_count = {
        key: value.shape[0] for key, value in class_latent_vectors.items()
    }

    # Concatenate all the latent vectors
    X = torch.cat([value for value in class_latent_vectors.values()])

    # Perform t-SNE
    X_embedded = TSNE(n_components=dimension).fit_transform(X)

    # Plot the results
    fig, ax = plt.subplots(figsize=fig_size)
    start = 0
    for i, (key, count) in enumerate(class_latent_vec_count.items()):
        end = start + count
        ax.scatter(
            X_embedded[start:end, 0],
            X_embedded[start:end, 1],
            label=label_dict[key].split(",")[0],
        )
        start = end

    ax.legend()

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}/tsne.png")

    plt.show()
