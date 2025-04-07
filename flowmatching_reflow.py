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
import matplotlib.animation as animation # Import animation library
from mingruinspired import Mingrustack  # Assuming this is your custom module
from utils import generate_3d_sphere_data, normalize_data, denormalize_data # Assume utils exists

# --- Optional: Clear plots at the start ---
plt.close('all')
# ===================================================================
# Configuration Class (Adapted for Flow Matching)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters for Flow Matching. """
    def __init__(self, M, nhidden, nlayers, batch_size, learning_rate, epochs, epochs_reflow, time_embed_dim=64, ode_steps=50, reflow_ode_steps=10, animation_steps=100, animation_samples=500, epsilon=1e-5):
        self.M = M # Data dimensionality
        self.nhidden = nhidden # Hidden layer size
        self.nlayers = nlayers # Number of layers in MLP
        # Removed: timesteps, noise_schedule
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epochs_reflow = epochs_reflow
        self.time_embed_dim = time_embed_dim # Dimension for time embedding MLP
        self.ode_steps = ode_steps # Number of steps for ODE solver during sampling
        self.reflow_ode_steps = reflow_ode_steps # Number of steps for ODE solver during sampling for reflow
        self.animation_steps = animation_steps # Number of steps (frames) for animation generation
        self.animation_samples = animation_samples # Number of samples to show in animation
        self.epsilon = epsilon # Small value to avoid t=0 during training path sampling

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
    def __init__(self, M, nhidden, nlayers, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim

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
        self.core_model = Mingrustack(nlayers, M + time_embed_dim, nhidden, M) # Output dim is M (velocity)
        print(f"Initialized VelocityMLP with {nlayers} layers, hidden size {nhidden}, time_embed_dim {time_embed_dim}")

    def forward(self, x, t):
        # x shape: [batch_size, M]
        # t shape: [batch_size]
        t_emb = self.time_embed(t) # Shape: [batch_size, time_embed_dim]
        # Concatenate data and time embedding
        xt_emb = torch.cat([x, t_emb], dim=1) # Shape: [batch_size, M + time_embed_dim]
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
    return f"FMR_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_ted{config.time_embed_dim}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}"

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
    model = model_class(config.M, config.nhidden, config.nlayers, config.time_embed_dim).to(config.device)
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
            f"BS={config.batch_size}, LR={lr_str}, E={config.epochs}, E_reflow={config.epochs_reflow}, "
            f"ODE Steps={config.ode_steps}, Reflow ODE={config.reflow_ode_steps}, Anim Steps={config.animation_steps}")


# ===================================================================
# Sampling Function (Flow Matching using ODE Solver)
# ===================================================================
def sample_flow(v_net, config, num_samples=1, isreflow=False):
    """ Generates samples using the learned velocity field and an ODE solver. """
    v_net.eval() # Set model to evaluation mode
    device = config.device
    M = config.M

    # Define the dynamics function for the ODE solver
    # Needs access to the velocity network (v_net)
    def ode_func(t, x):
        # Ensure t is a scalar tensor on the correct device, matching x's batch dim
        t_tensor = torch.full((x.shape[0],), t.item(), device=device, dtype=x.dtype)
        # Predict velocity using the network
        with torch.no_grad(): # Ensure no gradients are computed here
             # Use autocast if AMP was used during training, might improve inference speed
             with amp.autocast(enabled=config.use_amp):
                  velocity = v_net(x, t_tensor)
        return velocity

    with torch.no_grad(): # Overall no_grad context
        # Sample initial points from the prior (standard Gaussian)
        x0 = torch.randn(num_samples, M, device=device)

        # Define the time steps for integration (from 0 to 1)
        # More steps generally lead to better accuracy but slower sampling
        if isreflow:
            odesteps = config.reflow_ode_steps # Use reflow-specific steps if applicable
        else:
            odesteps = config.ode_steps
        t_eval = torch.linspace(0., 1., odesteps, device=device)

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
# Trajectory Generation Function (for Reflow)
# ===================================================================
def generate_trajectories(v_net, config, num_samples=1):
    """ Generates full trajectories using the learned velocity field and ODE solver. """
    v_net.eval() # Set model to evaluation mode
    device = config.device
    M = config.M

    # Define the dynamics function for the ODE solver
    def ode_func(t, x):
        t_tensor = torch.full((x.shape[0],), t.item(), device=device, dtype=x.dtype)
        with torch.no_grad():
             with amp.autocast(enabled=config.use_amp):
                  velocity = v_net(x, t_tensor)
        return velocity

    with torch.no_grad():
        # Sample initial points from the prior (standard Gaussian)
        x0 = torch.randn(num_samples, M, device=device)

        # Define the time steps for integration (from 0 to 1)
        t_eval = torch.linspace(0., 1., config.ode_steps, device=device)

        print(f"Generating {num_samples} trajectories with {config.ode_steps} steps...")
        # Use the ODE solver
        traj = odeint(
            ode_func,
            x0,
            t_eval,
            method='dopri5',
            atol=1e-5,
            rtol=1e-5
        )
        print("Trajectory generation finished.")
        # traj shape: [ode_steps, num_samples, M]

    v_net.train() # Return model to training mode
    # Return the full trajectory tensor, still on the device
    return traj


# ===================================================================
# Training Function (Flow Matching)
# ===================================================================
def train_flow_matching(data, config):
    """
    Trains a Flow Matching model (velocity network).
    Assumes 'data' tensor is already on the target device (GPU-resident).
    Displays epoch-level tqdm progress.
    """
    device = config.device
    use_amp = config.use_amp
    epsilon = config.epsilon # Small value to avoid t=0

    print(f"Training Flow Matching model with data on device: {data.device}")

    if data.device.type != device.type:
        # This catches mismatches like data on CPU when target is GPU, or vice-versa
        print(f"Error: Data device type ('{data.device.type}') differs from target device type ('{device.type}').")
        return None, [] # Return indicating failure
    elif device.type == 'cuda':
        # Optional: Add a check if a specific non-zero GPU index was requested in config
        # and the data ended up elsewhere. Usually not necessary if using default device.
        # if device.index is not None and device.index != data_device.index:
        #    print(f"Error: Data device index ({data_device.index}) differs from target device index ({device.index}).")
        #    return None, []
        pass # Types match (both cuda or both cpu), proceed.

    # Create DataLoader for GPU Tensor
    try:
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
        print(f"DataLoader created with batch size {config.batch_size}, num_workers=0, pin_memory=False")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return None, []

    # Initialize the velocity network model
    v_net = VelocityMLP(config.M, config.nhidden, config.nlayers, config.time_embed_dim).to(device)

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
            x_1 = batch[0] # Target data points
            current_batch_size = x_1.shape[0]

            # 1. Sample time t ~ U(epsilon, 1)
            t = torch.rand(current_batch_size, device=device) * (1.0 - epsilon) + epsilon

            # 2. Sample prior points x_0 ~ N(0, I)
            x_0 = torch.randn_like(x_1) # Same shape and device as x_1

            # 3. Calculate points on the OT path: x_t = t*x_1 + (1-t)*x_0
            # Need to reshape t for broadcasting: [batch_size, 1]
            t_reshaped = t.view(-1, 1)
            x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0

            # 4. Calculate target velocity: v_target = x_1 - x_0
            v_target = x_1 - x_0

            # --- Mixed Precision Context (forward pass) ---
            with amp.autocast(enabled=use_amp):
                # Predict velocity using the network
                predicted_velocity = v_net(x_t, t)
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
             epoch_losses.append(float('nan'))

    print("\nTraining finished.")
    return v_net, epoch_losses


# ===================================================================
# Training Function (Reflow Matching)
# ===================================================================
def train_reflow_matching(x0_data, x1_data, config, initial_model=None):
    """
    Trains a Reflow Matching model using generated (x0, x1) pairs.
    Assumes x0_data and x1_data tensors are already on the target device.
    """
    device = config.device
    use_amp = config.use_amp
    epsilon = config.epsilon

    print(f"Training Reflow Matching model with data on device: {x0_data.device}")

    if x0_data.device.type != device.type or x1_data.device.type != device.type:
        print(f"Error: Reflow data device mismatch (x0: {x0_data.device}, x1: {x1_data.device}, target: {device}).")
        return None, []

    # Create DataLoader for GPU Tensors (x0, x1 pairs)
    try:
        dataset = TensorDataset(x0_data, x1_data)
        # Use a potentially smaller batch size for reflow if memory is tight
        reflow_batch_size = config.batch_size # Or adjust: max(1, config.batch_size // 2)
        dataloader = DataLoader(dataset, batch_size=reflow_batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
        print(f"Reflow DataLoader created with batch size {reflow_batch_size}")
    except Exception as e:
        print(f"Error creating Reflow DataLoader: {e}")
        return None, []

    # Initialize or reuse the velocity network model
    if initial_model:
        print("Initializing Reflow model from the previously trained model.")
        v_net = initial_model # Start from the pre-trained weights
    else:
        print("Initializing Reflow model from scratch.")
        v_net = VelocityMLP(config.M, config.nhidden, config.nlayers, config.time_embed_dim).to(device)

    try:
        # Consider a potentially smaller learning rate for fine-tuning/reflow
        reflow_lr = config.learning_rate # Or adjust: config.learning_rate / 5
        optimizer = optim.Adam(v_net.parameters(), lr=reflow_lr)
        criterion = nn.MSELoss()
        scaler = amp.GradScaler(enabled=use_amp)
    except Exception as e:
        print(f"Error initializing Reflow model/optimizer: {e}")
        return None, []

    epoch_losses = []
    # Use fewer epochs for reflow, or make it configurable
    reflow_epochs = config.epochs_reflow # Or adjust: max(1, config.epochs // 2)
    print(f"Starting Reflow Matching training on {device} for {reflow_epochs} epochs...")

    epochs_pbar = tqdm(range(reflow_epochs), desc=f"Training Reflow (Epochs)", unit="epoch")

    # --- Reflow Training Loop ---
    for epoch in epochs_pbar:
        v_net.train()
        epoch_loss = 0.0

        # Iterate through batches (pairs of x_0, x_1 from generated trajectories)
        for batch in dataloader:
            x_0, x_1 = batch # Unpack the pair
            current_batch_size = x_0.shape[0]

            # 1. Sample time t ~ U(epsilon, 1)
            t = torch.rand(current_batch_size, device=device) * (1.0 - epsilon) + epsilon

            # 2. Calculate points on the *straight* path: x_t = t*x_1 + (1-t)*x_0
            t_reshaped = t.view(-1, 1)
            x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0

            # 3. Calculate target velocity for the straight path: v_target = x_1 - x_0
            v_target = x_1 - x_0

            # --- Mixed Precision Context (forward pass) ---
            with amp.autocast(enabled=use_amp):
                predicted_velocity = v_net(x_t, t)
                loss = criterion(predicted_velocity, v_target)

            # --- Backpropagation ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        if len(dataloader) > 0:
             avg_epoch_loss = epoch_loss / len(dataloader)
             epoch_losses.append(avg_epoch_loss)
             epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.6f}")
        else:
             print(f"Warning: Reflow DataLoader was empty for epoch {epoch+1}.")
             epoch_losses.append(float('nan'))

    print("\nReflow Training finished.")
    return v_net, epoch_losses


# ===================================================================
# Animation Function
# ===================================================================
def create_flow_animation(v_net, config, num_samples, filename="flow_animation.gif"):
    """ Generates an animation of the flow from noise to data using the provided model. """
    if config.M < 2:
        print("Animation requires data dimensionality M >= 2. Skipping animation.")
        return

    print(f"\n--- Generating Animation ({num_samples} samples, {config.animation_steps} steps) ---")
    v_net.eval()
    device = config.device

    # Define the dynamics function for the ODE solver
    def ode_func(t, x):
        t_tensor = torch.full((x.shape[0],), t.item(), device=device, dtype=x.dtype)
        with torch.no_grad():
            with amp.autocast(enabled=config.use_amp):
                velocity = v_net(x, t_tensor)
        return velocity

    with torch.no_grad():
        # Sample initial points from the prior
        x0 = torch.randn(num_samples, config.M, device=device)

        # Define time steps for animation frames
        t_eval = torch.linspace(0., 1., config.animation_steps, device=device)

        # Generate the full trajectory for animation
        print("Solving ODE for animation trajectory...")
        traj = odeint(
            ode_func,
            x0,
            t_eval,
            method='dopri5',
            atol=1e-5,
            rtol=1e-5
        )
        print("ODE solution for animation obtained.")
        # traj shape: [animation_steps, num_samples, M]
        traj_cpu = traj.cpu().numpy() # Move trajectory to CPU for plotting

    # --- Create Animation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Flow Animation (Reflow Model)\n{get_config_description(config)}')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    # Determine fixed plot limits based on the final distribution (+ a buffer)
    final_points = traj_cpu[-1]
    xlim = (final_points[:, 0].min() - 0.5, final_points[:, 0].max() + 0.5)
    ylim = (final_points[:, 1].min() - 0.5, final_points[:, 1].max() + 0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5)

    scatter = ax.scatter([], [], alpha=0.6, s=10) # Initialize empty scatter plot

    def update(frame):
        # Update scatter plot data for the current frame
        points = traj_cpu[frame]
        scatter.set_offsets(points[:, :2]) # Use first two dimensions
        ax.set_title(f'Flow Animation (Reflow Model) - Step {frame+1}/{config.animation_steps}')
        return scatter,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=config.animation_steps,
                                  interval=50, blit=True) # interval in ms

    # Save the animation
    config_dir = get_config_directory(config)
    os.makedirs(config_dir, exist_ok=True)
    filepath = os.path.join(config_dir, filename)
    try:
        print(f"Saving animation to {filepath}...")
        ani.save(filepath, writer='pillow', fps=15) # Using pillow for GIF
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`).")
    plt.close(fig) # Close the figure after saving

    v_net.train() # Return model to training mode


# ===================================================================
# Main Execution Block (Flow Matching + Reflow)
# ===================================================================
if __name__ == "__main__":
    # 1. Generate Data (on CPU initially)
    M = 3 # Data dimensionality
    num_samples = 32000
    print(f"Generating {num_samples} samples...")
    data_cpu, original_radii = generate_3d_sphere_data(num_samples)
    data_cpu = data_cpu.float()

    # 2. Configure Training
    config = Config(
        M=M,
        nhidden=1024,
        nlayers=2,
        batch_size=128,
        learning_rate=1e-3,
        epochs=2000, # Epochs for initial training
        epochs_reflow=50, # Epochs for reflow training
        time_embed_dim=64,
        ode_steps=50,        # Steps for SAMPLING (initial and reflow)
        reflow_ode_steps=10, # Steps for generating TRAJECTORIES for reflow training
        animation_steps=100, # Steps/frames for animation
        animation_samples=500, # Number of points in animation
        epsilon=1e-5
    )
    # Add config for reflow if needed (e.g., reflow_epochs, reflow_lr)
    # config.reflow_epochs = config.epochs // 2 # Example

    # 3. Move Original Dataset to Target Device
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

    # --- Initial Training ---
    train_initial_model_flag = False
    config_dir = get_config_directory(config)
    os.makedirs(config_dir, exist_ok=True)
    initial_model_filename = "model_fm_initial.pth"
    reflow_model_filename = "model_fm_reflow.pth"
    initial_model = None
    epoch_losses_initial = []

    if train_initial_model_flag:
        print("\n--- Starting Initial Flow Matching Training ---")
        initial_model, epoch_losses_initial = train_flow_matching(data_on_device, config)

        if initial_model:
            save_model(initial_model, config, filename=initial_model_filename)
            if epoch_losses_initial:
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_losses_initial, label='Initial Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.yscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.title(f'Initial Flow Matching Training Loss\n{get_config_description(config)}')
                loss_fig = plt.gcf()
                save_plot(loss_fig, config, "training_loss_fm_initial.png")
        else:
            print("Initial training failed, model not saved.")
            initial_model = None # Ensure it's None if training failed

    # --- Load Initial Model if not trained in this run ---
    if initial_model is None:
        initial_model_path = os.path.join(config_dir, initial_model_filename)
        if os.path.exists(initial_model_path):
            print(f"\n--- Loading Initial Model from {initial_model_path} ---")
            initial_model = load_model(VelocityMLP, config, filename=initial_model_filename)
        else:
            print(f"\nInitial model file {initial_model_path} not found and training was skipped/failed. Cannot proceed with reflow.")
            exit() # Exit if no initial model is available

    # --- Sampling and Plotting from Initial Model ---
    if initial_model:
        print("\n--- Generating Samples using Initial ODE Solver ---")
        num_generated_samples_initial = 4000 # Use the same number as reflow for comparison
        generated_data_initial = sample_flow(initial_model, config, num_samples=num_generated_samples_initial)

        if generated_data_initial is not None and generated_data_initial.shape[0] > 0:
             print(f"Generated {generated_data_initial.shape[0]} samples using Initial model.")

             # --- Analysis and Plotting (Initial) ---
             print("--- Analyzing and Plotting Initial Model Results ---")
             generated_radii_initial = torch.norm(generated_data_initial, dim=1).numpy()
             original_radii_np = original_radii.cpu().numpy() # Already computed or recompute if needed

             # Plot Radii Histogram (Initial)
             plt.figure(figsize=(10, 6))
             hist_range = (0.0, 1.5)
             plt.hist(original_radii_np, bins=50, range=hist_range, density=True, alpha=0.6, label='Original Data Radii')
             plt.hist(generated_radii_initial, bins=50, range=hist_range, density=True, alpha=0.6, label='Generated Data Radii (Initial)')
             plt.xlabel('Radius')
             plt.ylabel('Density')
             plt.legend()
             plt.title(f'Histogram of Data Radii (Initial Flow Matching)\n{get_config_description(config)}')
             plt.grid(True, linestyle='--', linewidth=0.5)
             histogram_fig_initial = plt.gcf()
             save_plot(histogram_fig_initial, config, "radii_histogram_fm_initial.png")

             # Plot Scatter (Initial, if M >= 2)
             if M >= 2:
                 plt.figure(figsize=(8, 8))
                 num_points_to_plot = min(1000, data_on_device.shape[0], generated_data_initial.shape[0])
                 orig_data_cpu = data_on_device[:num_points_to_plot].cpu()
                 gen_data_initial_cpu = generated_data_initial[:num_points_to_plot].cpu()
                 plt.scatter(orig_data_cpu[:, 0], orig_data_cpu[:, 1], alpha=0.5, s=10, label='Original Data Sample')
                 plt.scatter(gen_data_initial_cpu[:, 0], gen_data_initial_cpu[:, 1], alpha=0.5, s=10, label='Generated Data Sample (Initial)')
                 plt.xlabel('Feature 1')
                 plt.ylabel('Feature 2')
                 plt.legend()
                 plt.title(f'Data Scatter Plot (Initial Flow Matching)\n{get_config_description(config)}')
                 plt.axis('equal')
                 plt.grid(True, linestyle='--', linewidth=0.5)
                 scatter_fig_initial = plt.gcf()
                 save_plot(scatter_fig_initial, config, "scatter_plot_2d_fm_initial.png")

             print(f"\nInitial model results saved in directory: {config_dir}")
        else:
             print("No data generated from initial model or sampling failed.")
    else:
        # This case should technically not be reached due to the exit() above if loading fails
        print("\nNo initial model available for sampling.")
        exit()


    # --- Generate Trajectories for Reflow ---
    print("\n--- Generating Trajectories for Reflow ---")
    # Use a reasonable number of trajectories for reflow training data
    num_reflow_samples = num_samples # Or adjust based on memory/time
    trajectories = generate_trajectories(initial_model, config, num_samples=num_reflow_samples)
    # trajectories shape: [ode_steps, num_reflow_samples, M]

    # Extract x0 (start) and x1 (end) points for reflow training
    # Ensure they remain on the device
    x0_reflow = trajectories[0]   # Shape: [num_reflow_samples, M]
    x1_reflow = trajectories[-1]  # Shape: [num_reflow_samples, M]
    print(f"Generated {x0_reflow.shape[0]} (x0, x1) pairs for reflow training on device {x0_reflow.device}.")
    del trajectories # Free up memory if trajectories tensor is large

    # --- Reflow Training ---
    train_reflow_model_flag = False
    reflow_model = None
    epoch_losses_reflow = []

    if train_reflow_model_flag:
        print("\n--- Starting Reflow Matching Training ---")
        # Train the reflow model. Can optionally pass initial_model to fine-tune.
        # Here, we train a new model from scratch using the reflow data.
        # To fine-tune, pass: initial_model=initial_model
        reflow_model, epoch_losses_reflow = train_reflow_matching(x0_reflow, x1_reflow, config, initial_model=None)

        if reflow_model:
            save_model(reflow_model, config, filename=reflow_model_filename)
            if epoch_losses_reflow:
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_losses_reflow, label='Reflow Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.yscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.title(f'Reflow Matching Training Loss\n{get_config_description(config)}')
                loss_fig = plt.gcf()
                save_plot(loss_fig, config, "training_loss_fm_reflow.png")
        else:
            print("Reflow training failed, model not saved.")
            reflow_model = None

    # --- Sampling from Reflow Model ---
    if reflow_model is None:
        reflow_model_path = os.path.join(config_dir, reflow_model_filename)
        if os.path.exists(reflow_model_path):
             print(f"\n--- Loading Reflow Model from {reflow_model_path} ---")
             reflow_model = load_model(VelocityMLP, config, filename=reflow_model_filename)
        else:
             print(f"\nReflow model file {reflow_model_path} not found and training was skipped/failed. Cannot perform reflow sampling.")
             # Optionally load and sample from the initial model as a fallback
             # reflow_model = initial_model # Sample from initial if reflow failed
             # print("Sampling from initial model instead.")

    if reflow_model:
        print("\n--- Generating Samples using Reflowed ODE Solver ---")
        num_generated_samples = 4000
        generated_data_reflow = sample_flow(reflow_model, config, num_samples=num_generated_samples, isreflow=True)

        if generated_data_reflow is not None and generated_data_reflow.shape[0] > 0:
             print(f"Generated {generated_data_reflow.shape[0]} samples using Reflow model.")

             # --- Analysis and Plotting (Reflow) ---
             print("--- Analyzing and Plotting Reflow Results ---")
             generated_radii_reflow = torch.norm(generated_data_reflow, dim=1).numpy()
             original_radii_np = original_radii.cpu().numpy() # Already computed or recompute

             # Plot Radii Histogram (Reflow)
             plt.figure(figsize=(10, 6))
             hist_range = (0.0, 1.5)
             plt.hist(original_radii_np, bins=50, range=hist_range, density=True, alpha=0.6, label='Original Data Radii')
             plt.hist(generated_radii_reflow, bins=50, range=hist_range, density=True, alpha=0.6, label='Generated Data Radii (Reflow)')
             plt.xlabel('Radius')
             plt.ylabel('Density')
             plt.legend()
             plt.title(f'Histogram of Data Radii (Reflow Matching)\n{get_config_description(config)}')
             plt.grid(True, linestyle='--', linewidth=0.5)
             histogram_fig = plt.gcf()
             save_plot(histogram_fig, config, "radii_histogram_fm_reflow.png")

             # Plot Scatter (Reflow, if M >= 2)
             if M >= 2:
                 plt.figure(figsize=(8, 8))
                 num_points_to_plot = min(1000, data_on_device.shape[0], generated_data_reflow.shape[0])
                 orig_data_cpu = data_on_device[:num_points_to_plot].cpu()
                 gen_data_reflow_cpu = generated_data_reflow[:num_points_to_plot].cpu()
                 plt.scatter(orig_data_cpu[:, 0], orig_data_cpu[:, 1], alpha=0.5, s=10, label='Original Data Sample')
                 plt.scatter(gen_data_reflow_cpu[:, 0], gen_data_reflow_cpu[:, 1], alpha=0.5, s=10, label='Generated Data Sample (Reflow)')
                 plt.xlabel('Feature 1')
                 plt.ylabel('Feature 2')
                 plt.legend()
                 plt.title(f'Data Scatter Plot (Reflow Matching)\n{get_config_description(config)}')
                 plt.axis('equal')
                 plt.grid(True, linestyle='--', linewidth=0.5)
                 scatter_fig = plt.gcf()
                 save_plot(scatter_fig, config, "scatter_plot_2d_fm_reflow.png")

             # --- Generate Animation (Reflow Model) ---
             create_flow_animation(reflow_model, config, config.animation_samples, filename="flow_animation_reflow.gif")

             print(f"\nReflow results saved in directory: {config_dir}")
        else:
             print("No data generated from reflow model or sampling failed.")
    else:
        print("\nNo reflow model available for sampling.")


    # --- Optional: Compare Initial vs Reflow Samples ---
    # You could load samples from the initial model (if saved or generated earlier)
    # and plot them alongside the reflow samples on the same scatter plot for comparison.

    print("\nScript finished.")