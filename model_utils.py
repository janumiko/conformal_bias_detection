import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from datasets import get_class


class LatentModel(nn.Module):
    """A model that returns the latent vectors of the data."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def get_latent(self, x):
        return self.backbone(x)


def test_model(model: nn.Module, data: DataLoader, device: torch.device) -> float:
    """Test the model on the data.

    Args:
        model: The model to test
        data: The data to test on
        device: The device to use

    Returns:
        The accuracy of the model
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def get_latent_vectors(
    model: LatentModel, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Get the latent vectors of the data.
    
    Args:
        model: The model to get the latent vectors from
        loader: The data loader to get the data from
        device: The device to use
        
    Returns:
        The latent vectors of the data
        """
    latent_vectors = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            latent = model.get_latent(x)
            latent_vectors.append(latent)
    return torch.cat(latent_vectors).cpu()


def get_latent_classes(
    model: LatentModel,
    dataset: Dataset | Subset,
    classes: set[int],
    device: torch.device,
    batch_size: int = 256,
) -> dict[int, torch.Tensor]:
    """Get the latent vectors of the classes.

    Args:
        model: The model to get the latent vectors from
        dataset: The dataset to get the classes from
        classes: The classes to get the latent vectors from
        device: The device to use
        batch_size: The batch size to use

    Returns:
        The latent vectors of the classes
    """
    results = {}
    for class_id in tqdm(classes):
        class_subset = get_class(dataset, class_id)

        if len(class_subset) == 0:
            continue

        class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)
        results[class_id] = get_latent_vectors(model, class_loader, device)
    return results

def get_prediction_vectors(   
    model, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Get the predictions of the data.
    
    Args:
        model: The model to get the latent vectors from
        loader: The data loader to get the data from
        device: The device to use
        
    Returns:
        The prediction vectors of the data
        """
    
    prediction_vectors = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            prediction = torch.argmax(logits, dim=1) 
            prediction_vectors.append(prediction)
    return torch.cat(prediction_vectors).cpu()


def get_prediction_classes(
    model,
    dataset: Dataset | Subset,
    classes: set[int],
    device: torch.device,
    batch_size: int = 256,
) -> dict[int, torch.Tensor]:
    """Get the prediction vectors of the classes.

    Args:
        model: The model to get the latent vectors from
        dataset: The dataset to get the classes from
        classes: The classes to get the prediction vectors from
        device: The device to use
        batch_size: The batch size to use

    Returns:
        The prediction vectors of the classes
    """
    results = {}
    for class_id in tqdm(classes):
        class_subset = get_class(dataset, class_id)

        if len(class_subset) == 0:
            continue

        class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)
        results[class_id] = get_prediction_vectors(model, class_loader, device)
    return results