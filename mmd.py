import torch
import torch.nn as nn

def gaussian_kernel(x, y, sigma=1.0):
    """Computes the Gaussian kernel between two sets of samples."""
    beta = 1. / (2. * sigma**2)
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-beta * dist_sq)

def mmd_rbf(x, y, sigma=1.0):
    """
    Calculates the Maximum Mean Discrepancy (MMD) using a Radial Basis Function (RBF) kernel.

    Args:
        x (torch.Tensor): Samples from the first distribution (N, D).
        y (torch.Tensor): Samples from the second distribution (M, D).
        sigma (float): Bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: The estimated MMD^2 value.
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    if x.size(0) == 0 or y.size(0) == 0:
        # Handle empty inputs gracefully, perhaps return 0 or raise an error
        return torch.tensor(0.0, device=x.device)


    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    # Formula for unbiased MMD^2 estimate
    m = x.size(0)
    n = y.size(0)

    # Calculate means, excluding diagonal elements for K_xx and K_yy
    mean_K_xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=x.device)
    del K_xx # Free memory
    mean_K_yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1)) if n > 1 else torch.tensor(0.0, device=y.device)
    del K_yy # Free memory
    mean_K_xy = K_xy.mean()
    del K_xy # Free memory

    mmd_sq = mean_K_xx + mean_K_yy - 2 * mean_K_xy
    # Ensure non-negativity due to potential floating point errors
    return torch.relu(mmd_sq)
