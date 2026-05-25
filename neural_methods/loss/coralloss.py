import torch
from typing import List

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss: Aligns second-order statistics (covariance) between source and target.
    Works for inputs of shape [B, C], [B, T, C], [B, H, W, C], etc.
    
    Parameters:
        source (Tensor): source features, shape [B, ..., D]
        target (Tensor): target features, shape [B, ..., D]
    
    Returns:
        Tensor: scalar CORAL loss value
    """
    # Ensure last dimension is feature dimension
    assert source.shape[-1] == target.shape[-1], "Feature dimensions must match"

    # Reshape to [N, D], flatten all but last dimension
    source_flat = source.view(-1, source.shape[-1])
    target_flat = target.view(-1, target.shape[-1])

    # Subtract mean
    source_centered = source_flat - source_flat.mean(dim=0, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)

    # Compute covariance
    cov_source = (source_centered.T @ source_centered) / (source_flat.size(0) - 1)
    cov_target = (target_centered.T @ target_centered) / (target_flat.size(0) - 1)

    # Frobenius norm
    loss = ((cov_source - cov_target) ** 2).sum()
    loss = loss / (4 * source_flat.size(1) ** 2)

    return loss



def coral_loss_pairwise(feature_list: List[torch.Tensor], reduction: str = 'mean') -> torch.Tensor:
    """
    CORAL loss over a list of feature tensors. Computes pairwise CORAL loss between all pairs.

    Parameters:
        feature_list (List[Tensor]): List of feature tensors with shape [..., D]
        reduction (str): 'mean' or 'sum' over all pairwise losses
    
    Returns:
        Tensor: scalar CORAL loss value
    """
    assert len(feature_list) >= 2, "At least two feature tensors are required"
    total_loss = 0.0
    count = 0

    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            source = feature_list[i]
            target = feature_list[j]

            assert source.shape[-1] == target.shape[-1], "Feature dimensions must match"

            # Flatten all but last dimension
            source_flat = source.view(-1, source.shape[-1])
            target_flat = target.view(-1, target.shape[-1])

            # Centering
            source_centered = source_flat - source_flat.mean(dim=0, keepdim=True)
            target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)

            # Covariance
            cov_source = (source_centered.T @ source_centered) / (source_flat.size(0) - 1)
            cov_target = (target_centered.T @ target_centered) / (target_flat.size(0) - 1)

            # CORAL loss
            loss = ((cov_source - cov_target) ** 2).sum()
            loss = loss / (4 * source_flat.size(1) ** 2)

            total_loss += loss
            count += 1

    if reduction == 'mean':
        return total_loss / count
    elif reduction == 'sum':
        return total_loss
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

