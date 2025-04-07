import torch

def generate_3d_sphere_data(N):
    """Generates N points within a 3D sphere of radius 1."""
    data = torch.randn(N, 3)
    radii = torch.norm(data, dim=1)
    data = data[radii < 1]  # Keep points within the sphere
    radii = radii[radii < 1]
    return data, radii

def normalize_data(data):
    """Normalize data to [-1, 1] range."""
    min_vals = torch.min(data, dim=0, keepdim=True).values
    max_vals = torch.max(data, dim=0, keepdim=True).values
    normalized_data = 2 * (data - min_vals) / (max_vals - min_vals) - 1
    return normalized_data, min_vals, max_vals

def denormalize_data(normalized_data, min_vals, max_vals):
    """Denormalize data back to original range."""
    denormalized_data = (normalized_data + 1) * (max_vals - min_vals) / 2 + min_vals
    return denormalized_data