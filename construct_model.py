from timm import create_model
import torch
from torch import nn


def register_models() -> None:
    """Register all the models in the timm registry by importing the module"""
    import models  # noqa: F401


def construct_model(
    model_name: str, num_classes: int, checkpoint_path: str
) -> nn.Module:
    model: nn.Module = create_model(
        model_name=model_name,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(checkpoint_path))
    return model
