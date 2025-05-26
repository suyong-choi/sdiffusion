# ===================================================================
# Full Flow Matching Script (Optimized for GPU-Resident Data)
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation # Import animation library
from tqdm import tqdm
import pathlib
import torch.cuda.amp as amp
import torch.nn.functional as F # For F.pad
from torchdiffeq import odeint # Import the ODE solver
from mingruinspired import Mingrustack  # Assuming this is your custom module
from utils import generate_3d_sphere_data, DynamicMLP
import ot as pot
import numpy as np

# ===================================================================
# Evaluation Metrics
# ===================================================================

def get_map(x0, x1):
    """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
    minibatch.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch

    Returns
    -------
    p : numpy array, shape (bs, bs)
        represents the OT plan between minibatches
    """
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1) ** 2
    p = pot.emd(a, b, M.detach().cpu().numpy())
    return p

# ===================================================================
# Configuration Class (Adapted for Flow Matching)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters for Flow Matching. """
    def __init__(self, M, nhidden, nlayers, batch_size, learning_rate, epochs, time_embed_dim=64, ode_steps=50, epsilon=1e-5, conditional=False, conditional_dim=0, useOT=False, useAMP=False, fourier_features=0, fourier_max_freq=10.0):
        self.M = M # Data dimensionality
        self.nhidden = nhidden # Hidden layer size
        self.nlayers = nlayers # Number of layers in MLP
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.time_embed_dim = time_embed_dim # Dimension for time embedding MLP
        self.ode_steps = ode_steps # Number of steps for ODE solver during sampling
        self.epsilon = epsilon # Small value to avoid t=0 during training path sampling
        self.conditional = conditional # Flag for conditional training
        self.conditional_dim = conditional_dim # Dimension of conditional variable
        self.useOT = useOT # Flag for using optimal transport plan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available() and useAMP # Use AMP if CUDA is available and requested
        self.fourier_features = fourier_features # Number of Fourier features for Mingrustack
        self.fourier_max_freq = fourier_max_freq # Maximum frequency for Fourier features
        print(f"Using device: {self.device}")
        print(f"Automatic Mixed Precision (AMP) enabled: {self.use_amp}")
        if not torch.cuda.is_available():
             print("WARNING: CUDA not available, running on CPU.")

# ===================================================================
# Model Definitions (Adapted for Flow Matching)
# ===================================================================
class PositionalEmbedding(nn.Module):
    """ Simple sinusoidal positional embedding for time. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: [batch_size]
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Unsqueeze t to [batch_size, 1] for broadcasting
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Handle odd dimension
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1)) # Pad the last dimension
        return embeddings

class VelocityMLP(nn.Module):
    """ MLP model representing the velocity field v(x, t). """
    def __init__(self, M, nhidden, nlayers, time_embed_dim, conditional=False, conditional_dim=0):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.conditional = conditional
        self.conditional_dim = conditional_dim

        # Time embedding layer (can be MLP or sinusoidal)
        # Using sinusoidal for simplicity here
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
         )
        # Using Mingrustack as the core network
        # Input dimension is M (data) + time_embed_dim
        core_input_dim = M + time_embed_dim
        if conditional:
            core_input_dim += conditional_dim
        self.core_model = Mingrustack(nlayers, core_input_dim, nhidden, M) # Output dim is M (velocity)
        #self.core_model = DynamicMLP(core_input_dim, nhidden, nlayers, M) # Output dim is M (velocity)
        print(f"Initialized VelocityMLP with {nlayers} layers, hidden size {nhidden}, time_embed_dim {time_embed_dim}, conditional {conditional}, conditional_dim {conditional_dim}")

    def forward(self, x, t, c=None):
        # x shape: [batch_size, M]
        # t shape: [batch_size]
        t_emb = self.time_embed(t) # Shape: [batch_size, time_embed_dim]
        # Concatenate data and time embedding
        xt_emb = torch.cat([x, t_emb], dim=1) # Shape: [batch_size, M + time_embed_dim]

        # Concatenate conditional variable if present
        if self.conditional:
            if c is None:
                raise ValueError("Conditional variable 'c' must be provided when conditional=True")
            xt_emb = torch.cat([xt_emb, c], dim=1) # Shape: [batch_size, M + time_embed_dim + conditional_dim]

        # Predict velocity
        velocity = self.core_model(xt_emb) # Shape: [batch_size, M]
        return velocity

class VelocityMLP_Conditional(nn.Module):
    def __init__(self, M, nhidden, nlayers, conditional=False, conditional_dim=0, fourier_features=0, fourier_max_freq=10.0):
        super().__init__()
        self.M = M
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.conditional = conditional
        self.conditional_dim = conditional_dim
        core_input_dim = M + 1 # M + 1 for time embedding (t)
        if conditional:
            core_input_dim += conditional_dim
        self.core_model = Mingrustack(
            nlayers, core_input_dim, nhidden, M, dropout=0.0,
            fourier_features=fourier_features, fourier_max_freq=fourier_max_freq, layernorm=True
        ) # Output dim is M (velocity)
        #self.core_model = DynamicMLP(core_input_dim, nhidden, nlayers, M) # Output dim is M (velocity)
    
    def forward(self, x, t, c=None):
        # x shape: [batch_size, M]
        # t shape: [batch_size]
        # c shape: [batch_size, conditional_dim] if conditional=True
        xt = torch.cat([x, t[:,None]], dim=1)
        if self.conditional:
            if c is None:
                raise ValueError("Conditional variable 'c' must be provided when conditional=True")
            # Concatenate data and conditional variable
            xt= torch.cat([xt, c], dim=1)
        
        # Predict velocity
        velocity = self.core_model(xt) # Shape: [batch_size, M]
        return velocity


# ===================================================================
# Utility Functions (Keep as before: model_io.py, plotting_utils.py)
# ===================================================================
def get_config_directory(config:Config):
    """ Creates a directory name based on config parameters. """
    lr_str = f"{config.learning_rate:.0e}" # Format LR concisely
    # Adjusted for Flow Matching config
    fourier_str = f"_FF{config.fourier_features}_FMF{config.fourier_max_freq}" if getattr(config, 'fourier_features', 0) > 0 else ""
    if config.useOT:
        return f"FM_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_st{config.ode_steps}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}_AMP{config.use_amp}{fourier_str}_OT"
    else:
        return f"FM_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_st{config.ode_steps}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}_AMP{config.use_amp}{fourier_str}"

def save_model(model, config, filename="model_fm.pth"): # Changed default filename
    """ Saves the model state dictionary. """
    directory = get_config_directory(config)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    filepath = pathlib.Path(directory) / filename
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, config, filename="model_fm.pth"): # Changed default filename
    """ Loads the model state dictionary. """
    directory = get_config_directory(config)
    filepath = pathlib.Path(directory) / filename
    # Instantiate model first on the correct device
    # Use VelocityMLP or the specific class used for training
    #model = model_class(config.M, config.nhidden, config.nlayers, config.time_embed_dim, config.conditional, config.conditional_dim).to(config.device)
    model = model_class(config.M, config.nhidden, config.nlayers, config.conditional, config.conditional_dim).to(config.device)
    if filepath.exists():
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.device))
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model state_dict from {filepath}: {e}. Starting fresh model.")
    else:
        print(f"Warning: Model file not found at {filepath}. Starting fresh model.")
    return model

def save_plot(fig, config, filename):
    """ Saves a matplotlib figure to the config directory. """
    directory = get_config_directory(config)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    filepath = pathlib.Path(directory) / filename
    try:
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot to {filepath}: {e}")
    plt.close(fig)

def get_config_description(config):
    """ Generates a string description of the configuration. """
    lr_str = f"{config.learning_rate:.1e}"
    # Adjusted for Flow Matching config
    return (f"FM M={config.M}, nh={config.nhidden}, nl={config.nlayers}, T_emb={config.time_embed_dim}, "
            f"BS={config.batch_size}, LR={lr_str}, E={config.epochs}, ODE Steps={config.ode_steps}")


# ===================================================================
# Sampling Function (Flow Matching using ODE Solver)
# ===================================================================
def sample_flow(v_net, config, num_samples=1, conditional_data=None, srcdist=None, returnsrc = False, returntraj=False):
    """Generates samples using the learned velocity field and an ODE solver,
    conditioned on provided conditional data.
"""
    v_net.eval()  # Set model to evaluation mode
    device = config.device
    M = config.M

    if config.conditional:
        if conditional_data is None:
            raise ValueError(
                "Conditional data must be provided when config.conditional is True"
            )
        if conditional_data.shape[0] != num_samples:
            raise ValueError(
                "Number of conditional data points must match num_samples"
            )
        if conditional_data.shape[1] != config.conditional_dim:
            raise ValueError(
                "Dimension of conditional data must match config.conditional_dim"
            )
        conditional_data = conditional_data.to(device)

    # Define the dynamics function for the ODE solver
    # Needs access to the velocity network (v_net)
    def ode_func(t, x):
        # Ensure t is a scalar tensor on the correct device, matching x's batch dim
        t_tensor = torch.full((x.shape[0],), t.item(), device=device, dtype=x.dtype)
        # Predict velocity using the network
        with torch.no_grad(): # Ensure no gradients are computed here
             # Use autocast if AMP was used during training, might improve inference speed
             with amp.autocast(enabled=config.use_amp):
                  # Pass conditional variable to the network if needed
                  if config.conditional:
                        velocity = v_net(x, t_tensor, c=conditional_data)
                  else:
                        velocity = v_net(x, t_tensor)
        return velocity

    with torch.no_grad():  # Overall no_grad context
        # Sample initial points from the prior (standard Gaussian)
        if srcdist is None:
            x0 = torch.randn(num_samples, M, device=device)
        else:
            sampleidx = torch.randperm(num_samples) 
            x0 = srcdist[sampleidx]

        # Define the time steps for integration (from 0 to 1)
        # More steps generally lead to better accuracy but slower sampling
        t_eval = torch.linspace(0.0, 1.0, config.ode_steps, device=device)

        print(f"Starting ODE integration with {config.ode_steps} steps...")
        # Use the ODE solver (e.g., 'dopri5', 'rk4')
        # 'dopri5' is a good adaptive default
        # odeint returns solutions at times specified in t_eval
        # The shape will be [ode_steps, num_samples, M]
        traj = odeint(
            ode_func,
            x0,
            t_eval,
            method='dopri5', # Or 'rk4', 'euler', etc.
            atol=1e-4, # Absolute tolerance
            rtol=1e-4  # Relative tolerance
        )
        print("ODE integration finished.")

        # The final samples are the solutions at t=1 (the last time step)
        samples = traj[-1] # Shape: [num_samples, M]

    v_net.train() # Return model to training mode if needed elsewhere
    if returntraj:
        return traj.cpu()
    elif returnsrc:
        return x0.cpu(), samples.cpu() # Return both source and generated samples on CPU
    else:
        return samples.cpu() # Return samples on CPU


# ===================================================================
# Training Helper Functions
# ===================================================================

def prepare_data_and_loaders(data, config, srcdist=None, shuffle_train=True):
    """Prepares data (splitting features/conditional) and creates DataLoaders."""
    device = config.device
    feature_data = None
    conditional_data_tensor = None # Renamed to avoid conflict

    # --- Data Splitting and Device Check ---
    if config.conditional:
        if data.shape[1] != config.M + config.conditional_dim:
            raise ValueError("Data dimension must be M + conditional_dim when config.conditional is True")
        conditional_data_tensor = data[:, config.M:].to(device)
        feature_data = data[:, :config.M].to(device)
    else:
        feature_data = data.to(device)

    if feature_data.device.type != device.type:
        raise RuntimeError(f"Error: Feature data device type ('{feature_data.device.type}') differs from target device type ('{device.type}').")
    if srcdist is not None and srcdist.device.type != device.type:
         srcdist = srcdist.to(device) # Ensure srcdist is on the correct device
         print(f"Moved srcdist to {device}")

    if srcdist is None:
        print("Generating default source distribution (Gaussian)")
        srcdist = torch.randn_like(feature_data, device=device)

    # --- DataLoader Creation ---
    try:
        if config.conditional:
            dataset = TensorDataset(feature_data, conditional_data_tensor)
        else:
            dataset = TensorDataset(feature_data)

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                num_workers=0, pin_memory=False, drop_last=True)
        print(f"Target DataLoader created (shuffle={shuffle_train})")

        # Use TensorDataset for srcdist to ensure consistent batching if drop_last=True
        src_dataset = TensorDataset(srcdist)
        srcdataloader = DataLoader(src_dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                   num_workers=0, pin_memory=False, drop_last=True)
        print(f"Source DataLoader created (shuffle={shuffle_train})")

    except Exception as e:
        raise RuntimeError(f"Error creating DataLoader: {e}")

    return feature_data, conditional_data_tensor, srcdist, dataloader, srcdataloader


def initialize_training_components(config, v_net=None):
    """Initializes the model, optimizer, criterion, and scaler."""
    device = config.device
    if v_net is None:
        # Defaulting to VelocityMLP_Conditional based on previous structure
        print("Initializing new VelocityMLP_Conditional model.")
        v_net = VelocityMLP_Conditional(
            config.M,
            config.nhidden,
            config.nlayers,
            config.conditional,
            config.conditional_dim,
        ).to(device)
    else:
        print("Using provided v_net model.")
        v_net = v_net.to(device) # Ensure provided model is on correct device

    try:
        optimizer = optim.Adam(v_net.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        scaler = amp.GradScaler(enabled=config.use_amp)
    except Exception as e:
        raise RuntimeError(f"Error initializing optimizer/criterion/scaler: {e}")

    return v_net, optimizer, criterion, scaler


# ===================================================================
# Training Function (Flow Matching) - Refactored
# ===================================================================
def train_flow_matching(data, config, srcdist=None, v_net=None, shuffle_train=True):
    """
    Trains a Flow Matching model (velocity network) using helper functions.
    """
    device = config.device
    epsilon = config.epsilon

    print(f"--- Preparing Data and Loaders ---")
    try:
        feature_data, _, srcdist, dataloader, srcdataloader = prepare_data_and_loaders(
            data, config, srcdist, shuffle_train=shuffle_train
        )
    except Exception as e:
        print(f"Error during data/loader preparation: {e}")
        return None, []

    print(f"--- Initializing Training Components ---")
    try:
        v_net, optimizer, criterion, scaler = initialize_training_components(config, v_net)
    except Exception as e:
        print(f"Error during component initialization: {e}")
        return None, []

    epoch_losses = []
    epoch_grad_norms = []

    print(f"--- Starting Flow Matching Training ({config.epochs} epochs) ---")
    epochs_pbar = tqdm(range(config.epochs), desc=f"Training FM", unit="epoch")

    # --- Training Loop ---
    for epoch in epochs_pbar:
        v_net.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        # Use zip_longest if datasets might have slightly different sizes due to drop_last
        # from itertools import zip_longest
        # for batch, srcbatch_tuple in zip_longest(dataloader, srcdataloader):
        #     if batch is None or srcbatch_tuple is None: continue # Skip incomplete pairs
        #     x_1 = batch[0]
        #     x_0 = srcbatch_tuple[0] # srcdataloader yields tuples
        #     c = batch[1] if config.conditional else None

        # Using zip assumes dataloaders yield same number of batches (due to drop_last=True)
        for batch, srcbatch_tuple in zip(dataloader, srcdataloader):
            x_1 = batch[0] # Target data points (feature data only)
            x_0 = srcbatch_tuple[0] # DataLoader wraps tensors in a tuple
            # 2.5 Use conditional variable from data
            c = batch[1] if config.conditional else None
            current_batch_size = x_1.shape[0]

            # 1. Sample time t ~ U(epsilon, 1)
            t = torch.rand(current_batch_size, device=device) * (1.0 - epsilon) + epsilon

            # 2.5.5 Sample optimal transport plan (x_0, x_1) 
            if config.useOT:
                M = get_map(x_0, x_1) # Compute the OT plan (cost matrix)
                indices = np.nonzero(M)[1] # Get non-zero indices for sampling
                x_0 = x_0[indices] # Sample x_0 based on the OT plan

            # 3. Calculate points on the OT path: x_t = t*x_1 + (1-t)*x_0
            # Need to reshape t for broadcasting: [batch_size, 1]
            t_reshaped = t.view(-1, 1)
            x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0
            # 4. Calculate target velocity: v_target = x_1 - x_0
            v_target = x_1 - x_0

            with amp.autocast(enabled=config.use_amp):
                # Predict velocity using the network
                predicted_velocity = v_net(x_t, t, c=c)
                # Calculate loss between predicted and target velocity
                loss = criterion(predicted_velocity, v_target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # --- Gradient Norm Monitoring ---
            if config.use_amp:
                scaler.unscale_(optimizer)
            total_norm = 0.0
            for p in v_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            # Optionally clip gradients
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), max_norm=1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            num_batches += 1

        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_grad_norm = epoch_grad_norm / num_batches
            epoch_losses.append(avg_epoch_loss)
            epoch_grad_norms.append(avg_epoch_grad_norm)
            epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.6f}", avg_grad_norm=f"{avg_epoch_grad_norm:.4f}")
        else:
            print(f"Warning: No batches processed in epoch {epoch+1}.")
            epoch_losses.append(float("nan"))
            epoch_grad_norms.append(float("nan"))


    print("\nTraining finished.")
    return v_net, epoch_losses, epoch_grad_norms


# ===================================================================
# Plotting Helper Functions
# ===================================================================

def plot_training_loss(epoch_losses, config, training_plotfilename="training_loss_fm.png"):
    """Plots the training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss (Flow Matching)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title(f'Flow Matching Training Loss\n{get_config_description(config)}')
    loss_fig = plt.gcf()
    save_plot(loss_fig, config, training_plotfilename)

def plot_grad_norms(epoch_grad_norms, config, gradnorm_plotfilename="training_gradnorm_fm.png"):
    """Plots the average gradient norm per epoch."""
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_grad_norms, label='Avg Gradient Norm (Flow Matching)')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm (L2)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title(f'Flow Matching Training Gradient Norm\n{get_config_description(config)}')
    gradnorm_fig = plt.gcf()
    save_plot(gradnorm_fig, config, gradnorm_plotfilename)

def plot_radii_histogram(original_radii_np, generated_data, config):
    """Plots histograms of original and generated data radii."""
    generated_radii = torch.norm(generated_data, dim=1).numpy()
    plt.figure(figsize=(10, 6))
    hist_range = (0.0, max(1.5, np.percentile(original_radii_np, 99), np.percentile(generated_radii, 99))) # Dynamic range
    plt.hist(original_radii_np, bins=50, range=hist_range, density=True, alpha=0.6, label='Original Data Radii')
    plt.hist(generated_radii, bins=50, range=hist_range, density=True, alpha=0.6, label='Generated Data Radii (Flow)')
    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Histogram of Data Radii (Flow Matching)\n{get_config_description(config)}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    histogram_fig = plt.gcf()
    save_plot(histogram_fig, config, "radii_histogram_fm.png")

def plot_scatter(data_on_device, generated_data, config):
    """Plots a 2D scatter plot of original and generated data samples."""
    if config.M < 2:
        print("Scatter plot requires M >= 2.")
        return
    plt.figure(figsize=(8, 8))
    num_points_to_plot = min(1000, data_on_device.shape[0], generated_data.shape[0])
    # Ensure data is on CPU for plotting
    orig_data_cpu = data_on_device[:num_points_to_plot].detach().cpu()
    gen_data_cpu = generated_data[:num_points_to_plot].cpu() # Already on CPU

    plt.scatter(orig_data_cpu[:, 0], orig_data_cpu[:, 1], alpha=0.5, s=10, label=f'Original Data ({num_points_to_plot} points)')
    plt.scatter(gen_data_cpu[:, 0], gen_data_cpu[:, 1], alpha=0.5, s=10, label=f'Generated Data ({num_points_to_plot} points)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(f'Data Scatter Plot (2D Projection)\n{get_config_description(config)}')
    plt.axis('equal')
    plt.grid(True, linestyle='--', linewidth=0.5)
    scatter_fig = plt.gcf()
    save_plot(scatter_fig, config, "scatter_plot_2d_fm.png")


# ===================================================================
# Animation Function (Refactored slightly)
# ===================================================================
def create_flow_animation(v_net, config, num_samples, conditional_data=None, srcdist=None, filename="flow_animation.gif"):
    """ Generates an animation of the flow from noise to data using the provided model. """
    if config.M < 2:
        print("Animation requires data dimensionality M >= 2. Skipping animation.")
        return

    print(f"\n--- Generating Animation ({num_samples} samples, {config.ode_steps} steps) ---")
    # Ensure srcdist for animation is on the correct device before passing to sample_flow
    if srcdist is not None:
        srcdist_anim = srcdist[:num_samples].to(config.device) # Take subset and move to device
    else:
        srcdist_anim = None # sample_flow will generate Gaussian noise

    # sample_flow expects srcdist on the *same device* as the model will run on (config.device)
    # but returns the trajectory on CPU.
    traj_cpu = sample_flow(v_net, config, num_samples=num_samples, conditional_data=conditional_data, srcdist=srcdist_anim, returntraj=True).numpy()

    # --- Create Animation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    # Determine fixed plot limits based on the final distribution (+ a buffer)
    final_points = traj_cpu[-1]
    if final_points.shape[0] > 0: # Check if any points were generated
        xlim = (final_points[:, 0].min() - 0.5, final_points[:, 0].max() + 0.5)
        ylim = (final_points[:, 1].min() - 0.5, final_points[:, 1].max() + 0.5)
    else:
        xlim = (-2, 2) # Default limits if no points
        ylim = (-2, 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    scatter = ax.scatter([], [], alpha=0.6, s=10) # Initialize empty scatter plot
    # Only create lines if num_samples is reasonable, otherwise animation can be slow
    lines = []
    if num_samples <= 300: # Limit number of trajectory lines
         lines = [ax.plot([], [], linestyle='--', alpha=0.3, linewidth=0.8)[0] for _ in range(num_samples)]
    title_obj = ax.set_title(f'Flow Animation - Step 0/{config.ode_steps}') # Initial title

    def update(frame):
        points = traj_cpu[frame]
        scatter.set_offsets(points[:, :2]) # Use first two dimensions
        title_obj.set_text(f'Flow Animation - Step {frame+1}/{config.ode_steps}')
        artists = [scatter] # Artists to return for blitting

        # Update lines if they exist
        if lines:
            for i in range(num_samples):
                x_traj = traj_cpu[:frame+1, i, 0]
                y_traj = traj_cpu[:frame+1, i, 1]
                lines[i].set_data(x_traj, y_traj)
            artists.extend(lines)

        return artists

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=config.ode_steps,
                                  interval=50, blit=True) # interval in ms

    # Save the animation
    config_dir = get_config_directory(config)
    pathlib.Path(config_dir).mkdir(parents=True, exist_ok=True)
    filepath = pathlib.Path(config_dir) / filename
    try:
        print(f"Saving animation to {filepath}...")
        ani.save(filepath, writer='pillow', fps=15) # Using pillow for GIF
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`).")
    plt.close(fig) # Close the figure after saving


# ===================================================================
# Main Execution Block (Flow Matching) - Refactored
# ===================================================================
if __name__ == "__main__":
    # 1. Generate Data (on CPU initially)
    M = 3 # Data dimensionality
    num_samples_data = 32000
    print(f"Generating {num_samples_data} target samples...")
    data_cpu, original_radii = generate_3d_sphere_data(num_samples_data)
    data_cpu = data_cpu.float()
    original_radii_np = original_radii.numpy() # Keep numpy version

    # Generate Source distribution (e.g., Gaussian) on CPU
    num_samples_source = num_samples_data # Match size for simplicity
    print(f"Generating {num_samples_source} source samples (Gaussian)...")
    srcdist_cpu = torch.randn(num_samples_source, M)

    # 2. Configure Training
    config = Config(
        M=M,
        nhidden=1024,
        nlayers=2,
        batch_size=128,
        learning_rate=1e-3,
        epochs=100, # Reduced epochs for faster testing
        time_embed_dim=64, # Note: Not used by VelocityMLP_Conditional
        ode_steps=50,
        epsilon=1e-5,
        conditional=False,
        conditional_dim=0, # Set to 0 if conditional=False
        useOT=False # Optimal Transport flag
    )

    # 3. Prepare Data (Move to Device handled within prepare_data_and_loaders)
    # We pass CPU tensors; the function moves them based on config.device
    # The function returns feature_data and srcdist already on the target device.
    data_on_device = data_cpu # Keep original CPU data if needed later
    try:
        # prepare_data_and_loaders returns device tensors for feature_data and srcdist
        feature_data_device, _, srcdist_device, _, _ = prepare_data_and_loaders(
            data_cpu, config, srcdist_cpu, shuffle_train=False # Get device tensors without loaders
        )
    except Exception as e:
         print(f"\n----- ERROR preparing data for device: {e}. Exiting. -----")
         exit()

    # --- Training ---
    train_model_flag = True
    config_dir = get_config_directory(config)
    pathlib.Path(config_dir).mkdir(parents=True, exist_ok=True)
    model_filename = "model_fm.pth"
    trained_model = None

    if train_model_flag:
        print("\n--- Starting Flow Matching Training ---")
        # Pass CPU data to training function, it will handle device placement via helpers
        trained_model, epoch_losses, epoch_grad_norms  = train_flow_matching(
            data_cpu, config, srcdist=srcdist_cpu
        )

        if trained_model:
            save_model(trained_model, config, filename=model_filename)
            if epoch_losses:
                plot_training_loss(epoch_losses, config)
                plot_grad_norms(epoch_grad_norms, config) # Plot gradient norms
        else:
            print("Training failed or was skipped, model not saved.")

    # --- Sampling & Evaluation ---
    model_path = pathlib.Path(config_dir) / model_filename
    if not model_path.exists():
        print(f"\nModel file {model_path} not found. Cannot perform sampling/evaluation.")
    else:
        print("\n--- Loading Model for Sampling & Evaluation ---")
        # Load the model - specify the class used during training
        trained_model = load_model(VelocityMLP_Conditional, config, filename=model_filename)

        # 1. Generate Samples (Forward ODE Integration)
        num_samples_generate = 5000
        print(f"\n--- Generating {num_samples_generate} samples ---")
        with torch.no_grad():
            generated_samples = sample_flow(
                trained_model,
                config,
                num_samples=num_samples_generate,
                srcdist=srcdist_cpu[:num_samples_generate], # Use CPU srcdist for consistency
                returnsrc=True, # Return both source and generated samples
                returntraj=False # No need for trajectory in this context
            )

        # --- Save Generated Samples to File ---
        try:
            np.savetxt(config_dir / "generated_samples_fm.txt", generated_samples.numpy(), delimiter=",")
            print(f"Generated samples saved to {config_dir / 'generated_samples_fm.txt'}")
        except Exception as e:
            print(f"Error saving generated samples: {e}")

        # 2. Evaluate Quality of Generated Samples
        print("\n--- Evaluating Generated Samples ---")
        try:
            # Load original data for comparison
            original_data = data_cpu.numpy() # Ensure original data is on CPU and in numpy format
            # Use the first num_samples_generate points for a fair comparison
            original_data_subset = original_data[:num_samples_generate]

            # Compute Wasserstein distance (Earth Mover's Distance) between original and generated samples
            wasserstein_distance = np.mean([
                np.min(get_map(torch.tensor(orig_point), torch.tensor(generated_samples))) # OT cost for each original point
                for orig_point in tqdm(original_data_subset, desc="Computing Wasserstein Distance", leave=False)
            ])

            print(f"Average Wasserstein Distance (EMD) between original and generated samples: {wasserstein_distance:.6f}")
        except Exception as e:
            print(f"Error during evaluation: {e}")

        # 3. Visualization of Generated Samples (Optional)
        # Here you can add code to visualize the generated samples, e.g., scatter plots, histograms, etc.
        # For high-dimensional data, consider plotting pairwise 2D projections or using dimensionality reduction techniques.

        # Example: 2D scatter plot of the first two dimensions
        try:
            plot_scatter(data_on_device, generated_samples, config)
        except Exception as e:
            print(f"Error during visualization: {e}")

    print("\n--- Flow Matching Pipeline Completed ---")