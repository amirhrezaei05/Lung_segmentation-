"""
Model architectures for lung segmentation.
"""

from .base_unet import UNet
from .resunet import ResUNet
from .efficient_unet import EfficientUNet, EfficientNetB4Encoder

__all__ = [
    "UNet",
    "ResUNet", 
    "EfficientUNet",
    "EfficientNetB4Encoder"
]


def get_model(config: dict):
    """
    Factory function to create model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        PyTorch model instance
    """
    model_name = config.get("name", "").lower()
    
    if model_name == "unet":
        return UNet(
            in_channels=config["encoder"]["in_channels"],
            base_channels=config["encoder"]["base_channels"],
            depth=config["encoder"]["depth"],
            out_channels=config["output"]["out_channels"],
            dropout=config["decoder"]["dropout"]
        )
    elif model_name == "resunet":
        return ResUNet(
            encoder_name=config["encoder"]["backbone"],
            encoder_weights="imagenet" if config["encoder"]["pretrained"] else None,
            in_channels=config["encoder"]["in_channels"],
            classes=config["output"]["out_channels"],
            decoder_channels=config["decoder"]["channels"],
            dropout=config["decoder"]["dropout"]
        )
    elif model_name == "efficientunet":
        return EfficientUNet(
            encoder_name=config["encoder"]["backbone"],
            pretrained=config["encoder"]["pretrained"],
            in_channels=config["encoder"]["in_channels"],
            out_channels=config["output"]["out_channels"],
            decoder_channels=config["decoder"]["channels"],
            dropout=config["decoder"]["dropout"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
