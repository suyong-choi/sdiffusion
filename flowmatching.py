# ===================================================================
# Full Flow Matching Script (Optimized for GPU-Resident Data)
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.cuda.amp as amp
import torch.nn.functional as F # For F.pad
from torchdiffeq import odeint # Import the ODE solver
from mingruinspired import Mingrustack  # Assuming this is your custom module
from utils import generate_3d_sphere_data, normalize_data, denormalize_data # Assume utils exists

# --- Optional: Clear plots at the start ---
plt.close('all')
# ===================================================================
# Configuration Class (Adapted for Flow Matching)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters for Flow Matching. """
    def __init__(self, M, nhidden, nlayers, batch_size, learning_rate, epochs, time_embed_dim=64, ode_steps=50, epsilon=1e-5, conditional=False, conditional_dim=0):
        self.M = M # Data dimensionality
        self.nhidden = nhidden # Hidden layer size
        self.nlayers = nlayers # Number of layers in MLP
        # Removed: timesteps, noise_schedule
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.time_embed_dim = time_embed_dim # Dimension for time embedding MLP
        self.ode_steps = ode_steps # Number of steps for ODE solver during sampling
        self.epsilon = epsilon # Small value to avoid t=0 during training path sampling
        self.conditional = conditional # Flag for conditional training
        self.conditional_dim = conditional_dim # Dimension of conditional variable

        # --- Device and AMP setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
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

# ===================================================================
# Utility Functions (Keep as before: model_io.py, plotting_utils.py)
# ===================================================================
def get_config_directory(config):
    """ Creates a directory name based on config parameters. """
    lr_str = f"{config.learning_rate:.0e}" # Format LR concisely
    # Adjusted for Flow Matching config
    return f"FM_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_ted{config.time_embed_dim}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}"

def save_model(model, config, filename="model_fm.pth"): # Changed default filename
    """ Saves the model state dictionary. """
    directory = get_config_directory(config)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, config, filename="model_fm.pth"): # Changed default filename
    """ Loads the model state dictionary. """
    directory = get_config_directory(config)
    filepath = os.path.join(directory, filename)
    # Instantiate model first on the correct device
    # Use VelocityMLP or the specific class used for training
    model = model_class(config.M, config.nhidden, config.nlayers, config.time_embed_dim, config.conditional, config.conditional_dim).to(config.device)
    if os.path.exists(filepath):
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
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
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
def sample_flow(v_net, config, num_samples=1, conditional_data=None):
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
        x0 = torch.randn(num_samples, M, device=device)

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
            atol=1e-5, # Absolute tolerance
            rtol=1e-5  # Relative tolerance
        )
        print("ODE integration finished.")

        # The final samples are the solutions at t=1 (the last time step)
        samples = traj[-1] # Shape: [num_samples, M]

    v_net.train() # Return model to training mode if needed elsewhere
    return samples.cpu() # Return samples on CPU


# ===================================================================
# Training Function (Flow Matching)
# ===================================================================
def train_flow_matching(data, config):
    """
    Trains a Flow Matching model (velocity network).
    Assumes 'data' tensor is already on the target device (GPU-resident).
    Displays epoch-level tqdm progress.
    Data is expected to have the shape [num_samples, M + conditional_dim],
    where the first M columns are the features and the remaining
    conditional_dim columns are the conditional variables.
    """
    device = config.device
    use_amp = config.use_amp
    epsilon = config.epsilon # Small value to avoid t=0

    print(f"Training Flow Matching model with data on device: {data.device}")

    if config.conditional:
        if data.shape[1] != config.M + config.conditional_dim:
            raise ValueError(
                "Data dimension must be M + conditional_dim when config.conditional is True"
            )

        # Extract conditional data from the last 'conditional_dim' columns
        conditional_data = data[:, config.M :]
        if conditional_data.device.type != device.type:
            conditional_data = conditional_data.to(device)
        # Extract feature data from the first M columns
        feature_data = data[:, : config.M]
        if feature_data.device != device:
            feature_data = feature_data.to(device)
    else:
        feature_data = data
        conditional_data = None

    if feature_data.device.type != device.type:
        # This catches mismatches like data on CPU when target is GPU, or vice-versa
        print( f"Error: Data device type ('{feature_data.device.type}') differs from target device type ('{device.type}').")
        return None, []  # Return indicating failure
    elif device.type == "cuda":
        # Optional: Add a check if a specific non-zero GPU index was requested in config
        # and the data ended up elsewhere. Usually not necessary if using default device.
        # if device.index is not None and device.index != data_device.index:
        #    print(f"Error: Data device index ({data_device.index}) differs from target device index ({device.index}).")
        #    return None, []
        pass  # Types match (both cuda or both cpu), proceed.

    # Create DataLoader for GPU Tensor
    try:
        if config.conditional:
            # Create DataLoader for feature data only (conditional data is not used in DataLoader)
            dataset = TensorDataset(feature_data, conditional_data)
        else:
            dataset = TensorDataset(feature_data)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
        print( f"DataLoader created with batch size {config.batch_size}, num_workers=0, pin_memory=False")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return None, []

    # Initialize the velocity network model
    v_net = VelocityMLP(
        config.M,
        config.nhidden,
        config.nlayers,
        config.time_embed_dim,
        config.conditional,
        config.conditional_dim,
    ).to(device)

    try:
        optimizer = optim.Adam(v_net.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss() # Mean Squared Error loss
        scaler = amp.GradScaler(enabled=use_amp) # GradScaler for mixed precision
    except Exception as e:
        print(f"Error initializing model/optimizer: {e}")
        return None, []

    epoch_losses = []
    print(f"Starting Flow Matching training on {device} for {config.epochs} epochs...")

    # Wrap the epoch loop with tqdm for epoch-level progress tracking
    epochs_pbar = tqdm(range(config.epochs), desc=f"Training FM (Epochs)", unit="epoch")

    # --- Training Loop ---
    for epoch in epochs_pbar:
        v_net.train() # Ensure model is in training mode
        epoch_loss = 0.0

        # Iterate through batches (data points x_1, already on GPU)
        for batch in dataloader:
            x_1 = batch[0] # Target data points (feature data only)
            current_batch_size = x_1.shape[0]

            # 1. Sample time t ~ U(epsilon, 1)
            t = torch.rand(current_batch_size, device=device) * (1.0 - epsilon) + epsilon

            # 2. Sample prior points x_0 ~ N(0, I)
            x_0 = torch.randn_like(x_1)  # Same shape and device as x_1

            # 2.5 Use conditional variable from data
            if config.conditional:
                # Get the corresponding conditional data batch
                c = batch[1]
            else:
                c = None

            # 3. Calculate points on the OT path: x_t = t*x_1 + (1-t)*x_0
            # Need to reshape t for broadcasting: [batch_size, 1]
            t_reshaped = t.view(-1, 1)
            x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0

            # 4. Calculate target velocity: v_target = x_1 - x_0
            v_target = x_1 - x_0

            # --- Mixed Precision Context (forward pass) ---
            with amp.autocast(enabled=use_amp):
                # Predict velocity using the network
                predicted_velocity = v_net(x_t, t, c=c)
                # Calculate loss between predicted and target velocity
                loss = criterion(predicted_velocity, v_target)

            # --- Backpropagation with GradScaler ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Optional: Gradient Clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(v_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Calculate average loss for the completed epoch
        if len(dataloader) > 0:
             avg_epoch_loss = epoch_loss / len(dataloader)
             epoch_losses.append(avg_epoch_loss)
             epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.6f}")
        else:
             print(f"Warning: DataLoader was empty for epoch {epoch+1}.")
             epoch_losses.append(float("nan"))

    print("\nTraining finished.")
    return v_net, epoch_losses


# ===================================================================
# Main Execution Block (Flow Matching)
# ===================================================================
if __name__ == "__main__":
    # 1. Generate Data (on CPU initially)
    M = 3 # Data dimensionality
    num_samples = 32000
    print(f"Generating {num_samples} samples...")
    data_cpu, original_radii = generate_3d_sphere_data(num_samples)
    data_cpu = data_cpu.float()

    # 2. Configure Training (Flow Matching specific)
    config = Config(
        M=M,
        nhidden=1024,        # Reduced hidden size for faster example
        nlayers=2,          # Number of layers (tune)
        batch_size=128,    # Adjust based on VRAM
        learning_rate=1e-3, # Tune learning rate
        epochs=10000,         # Number of training epochs
        time_embed_dim=64, # Dimension for time embedding
        ode_steps=50,       # Number of steps for sampling ODE solver
        epsilon=1e-5,        # Small offset for time sampling
        conditional=False,   # Enable conditional training
        conditional_dim=16    # Dimension of conditional variable
    )

    # 3. Move Entire Dataset to Target Device (GPU if available)
    data_on_device = None
    if config.device.type == 'cuda':
        print(f"Attempting to move dataset ({data_cpu.nelement() * data_cpu.element_size() / 1024**2:.2f} MB) to {config.device}...")
        try:
            data_on_device = data_cpu.to(config.device)
            print(f"Successfully moved dataset to {data_on_device.device}")
        except RuntimeError as e:
            print(f"\n----- ERROR moving full dataset to GPU: {e}. Exiting. -----")
            exit()
    else:
        print("Running on CPU, keeping data on CPU.")
        data_on_device = data_cpu

    # --- Training ---
    train_model_flag = True
    config_dir = get_config_directory(config)
    os.makedirs(config_dir, exist_ok=True)
    model_filename = "model_fm.pth" # Flow Matching model filename
    epoch_losses = []

    if train_model_flag:
        print("\n--- Starting Flow Matching Training ---")
        # Call the flow matching training function
        trained_model, epoch_losses = train_flow_matching(data_on_device, config)

        if trained_model:
            save_model(trained_model, config, filename=model_filename)
            if epoch_losses:
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_losses, label='Training Loss (Flow Matching)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.yscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.title(f'Flow Matching Training Loss\n{get_config_description(config)}')
                loss_fig = plt.gcf()
                save_plot(loss_fig, config, "training_loss_fm.png")
        else:
            print("Training failed, model not saved.")

    # --- Sampling ---
    model_path = os.path.join(config_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"\nModel file {model_path} not found. Cannot perform sampling.")
    else:
        print("\n--- Loading Model for Flow Matching Sampling ---")
        # Load the velocity network model
        loaded_model = load_model(VelocityMLP, config, filename=model_filename)

        print("--- Generating Samples using ODE Solver ---")
        num_generated_samples = 4000
        # Sampling might be done in one go if VRAM allows, or batched if needed
        # For simplicity, generating all samples at once here
        generated_data = sample_flow(loaded_model, config, num_samples=num_generated_samples)

        if generated_data is not None and generated_data.shape[0] > 0:
             print(f"Generated {generated_data.shape[0]} samples.")

             # --- Analysis and Plotting (same as before) ---
             print("--- Analyzing and Plotting Results ---")
             generated_radii = torch.norm(generated_data, dim=1).numpy()
             original_radii_np = original_radii.cpu().numpy()

             # Plot Radii Histogram
             plt.figure(figsize=(10, 6))
             hist_range = (0.0, 1.5) # Adjust range if needed
             plt.hist(original_radii_np, bins=50, range=hist_range, density=True, alpha=0.6, label='Original Data Radii')
             plt.hist(generated_radii, bins=50, range=hist_range, density=True, alpha=0.6, label='Generated Data Radii (Flow)')
             plt.xlabel('Radius')
             plt.ylabel('Density')
             plt.legend()
             plt.title(f'Histogram of Data Radii (Flow Matching)\n{get_config_description(config)}')
             plt.grid(True, linestyle='--', linewidth=0.5)
             histogram_fig = plt.gcf()
             save_plot(histogram_fig, config, "radii_histogram_fm.png")

             # Plot Scatter (if M >= 2)
             if M >= 2:
                 plt.figure(figsize=(8, 8))
                 num_points_to_plot = min(1000, data_on_device.shape[0], generated_data.shape[0])
                 orig_data_cpu = data_on_device[:num_points_to_plot].cpu()
                 gen_data_cpu = generated_data[:num_points_to_plot].cpu() # Already on CPU from sample_flow
                 plt.scatter(orig_data_cpu[:, 0], orig_data_cpu[:, 1], alpha=0.5, s=10, label='Original Data Sample')
                 plt.scatter(gen_data_cpu[:, 0], gen_data_cpu[:, 1], alpha=0.5, s=10, label='Generated Data Sample (Flow)')
                 plt.xlabel('Feature 1')
                 plt.ylabel('Feature 2')
                 plt.legend()
                 plt.title(f'Data Scatter Plot (Flow Matching)\n{get_config_description(config)}')
                 plt.axis('equal')
                 plt.grid(True, linestyle='--', linewidth=0.5)
                 scatter_fig = plt.gcf()
                 save_plot(scatter_fig, config, "scatter_plot_2d_fm.png")

             print(f"\nResults saved in directory: {config_dir}")
        else:
             print("No data generated or sampling failed.")

    print("\nScript finished.")