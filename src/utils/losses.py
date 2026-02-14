"""
Loss functions for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
        """
        probs = torch.sigmoid(logits)
        targets = (targets > 0.5).float()
        
        dims = (2, 3)  # Spatial dimensions
        intersection = (probs * targets).sum(dims)
        denominator = probs.sum(dims) + targets.sum(dims)
        
        dice_coef = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_coef.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss.
    
    Total = dice_weight * DiceLoss + bce_weight * BCELoss
    """
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_loss_function(config):
    """Factory function to get loss based on config"""
    loss_type = config['loss']['type'].lower()
    
    if loss_type == 'dice':
        return DiceLoss(smooth=config['loss'].get('smooth', 1e-6))
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'combined':
        return CombinedLoss(
            dice_weight=config['loss'].get('dice_weight', 0.7),
            bce_weight=config['loss'].get('bce_weight', 0.3),
            smooth=config['loss'].get('smooth', 1e-6)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
