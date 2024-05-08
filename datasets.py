import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms


def get_imagenet_dataset(
    data_dir: str, transform: transforms.Compose
) -> datasets.ImageNet:
    """Get the ImageNet dataset

    Args:
        batch_size: The batch size to use
        data_dir: The directory to store the data
        transform: The transform to apply to the data

    Returns:
        The ImageNet dataset
    """

    return datasets.ImageNet(data_dir, split="val", transform=transform)


def split_data(data: Dataset | Subset, calibrate_ratio: float) -> tuple[Subset, Subset]:
    """Split the data into test and calibration data

    Args:
        data: The data to split
        calibrate_ratio: The ratio of data to use for calibration

    Returns:
        test_data: The test data
        calibrate_data: The calibration data
    """
    indices = list(range(len(data)))
    test_indices, calibrate_indices = train_test_split(
        indices, test_size=calibrate_ratio, stratify=data.targets
    )
    test_data = Subset(data, test_indices)
    calibrate_data = Subset(data, calibrate_indices)
    return test_data, calibrate_data


def get_class(data: Dataset | Subset, class_idx: int) -> Subset:
    """Get the data for a specific class

    Args:
        data: The data to filter
        class_idx: The index of the class to filter

    Returns:
        The data for the specific class
    """
    if isinstance(data, Subset):
        indices = [
            i
            for i, target in enumerate(data.dataset.targets)
            if target == class_idx and i in data.indices
        ]
        return Subset(data.dataset, indices)
    else:
        indices = [i for i, target in enumerate(data.targets) if target == class_idx]
        return Subset(data, indices)


def test_model(model: torch.nn.Module, data: DataLoader, device: torch.device) -> float:
    """Test the model on the data

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
        for inputs, targets in tqdm.tqdm(data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total
