import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EarthMoversDistanceLoss(nn.Module):
    """
    Earth Mover's Distance (Wasserstein Distance) loss for ordinal regression/classification.
    
    This is particularly useful when your classes have a natural ordering
    (e.g., fatigue levels 0, 5, 10 or KSS scores 1-9) and you want to penalize
    predictions proportionally to how far they are from the true class.
    
    For example:
    - Predicting class 2 when truth is 1: small penalty
    - Predicting class 5 when truth is 1: large penalty
    
    Unlike Cross-Entropy which treats all misclassifications equally.
    """
    
    def __init__(self, num_classes: int, reduction: str = 'mean'):
        """
        Args:
            num_classes: Number of ordinal classes
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw model outputs (logits)
            targets: [B] integer class labels (0 to num_classes-1)
        
        Returns:
            EMD loss value
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(logits, dim=1)  # [B, num_classes]
        
        # Convert targets to one-hot
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()  # [B, num_classes]
        
        # Compute cumulative distributions
        pred_cdf = torch.cumsum(pred_probs, dim=1)  # [B, num_classes]
        target_cdf = torch.cumsum(target_one_hot, dim=1)  # [B, num_classes]
        
        # EMD is the L1 distance between CDFs
        emd = torch.abs(pred_cdf - target_cdf).sum(dim=1)  # [B]

        emd = emd / (self.num_classes - 1) if self.num_classes > 1 else emd

        if self.reduction == 'mean':
            return emd.mean()
        elif self.reduction == 'sum':
            return emd.sum()
        else:  # 'none'
            return emd


class CombinedEMDCrossEntropyLoss(nn.Module):
    """
    Combined EMD and Cross-Entropy loss.
    
    This combines the ordinal awareness of EMD with the classification
    capability of Cross-Entropy. Often works better than either alone.
    
    Loss = alpha * CrossEntropy + (1 - alpha) * EMD
    """
    
    def __init__(self, num_classes: int, alpha: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            num_classes: Number of classes
            alpha: Weight for cross-entropy (0 to 1). Default 0.5 means equal weight
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.emd_loss = EarthMoversDistanceLoss(num_classes, reduction=reduction)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw model outputs
            targets: [B] integer class labels
        
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(logits, targets)
        emd = self.emd_loss(logits, targets)
        
        return self.alpha * ce + (1 - self.alpha) * emd


class WeightedEMDLoss(nn.Module):
    """
    EMD loss with class weights for handling imbalanced datasets.
    """
    
    def __init__(self, num_classes: int, class_weights: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        """
        Args:
            num_classes: Number of classes
            class_weights: [num_classes] tensor of weights for each class
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.reduction = reduction
        self.emd_loss = EarthMoversDistanceLoss(num_classes, reduction='none')
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw model outputs
            targets: [B] integer class labels
        
        Returns:
            Weighted EMD loss value
        """
        emd = self.emd_loss(logits, targets)  # [B]
        
        if self.class_weights is not None:
            # Apply class weights
            weights = self.class_weights[targets]  # [B]
            emd = emd * weights
        
        if self.reduction == 'mean':
            return emd.mean()
        elif self.reduction == 'sum':
            return emd.sum()
        else:  # 'none'
            return emd