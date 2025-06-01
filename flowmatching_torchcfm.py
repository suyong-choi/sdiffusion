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
from mingruinspired import Mingrustack, MLPStack  # Assuming this is your custom module
from utils import generate_3d_sphere_data, DynamicMLP
import ot as pot
import numpy as np
import random # Import random for selecting dataloaders
import itertools # Import itertools for infinite iterators

# Import torchcfm components
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

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
    # Ensure M is on CPU and is a float64 numpy array for POT
    M_np = M.detach().cpu().double().numpy()
    p = pot.emd(a, b, M_np)
    return p

# ===================================================================
# Configuration Class (Adapted for Flow Matching)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters for Flow Matching. """
    def __init__(self, M, nhidden, nlayers, batch_size, learning_rate, epochs, time_embed_dim=64, ode_steps=50, epsilon=1e-5, conditional=False, conditional_dim=0, conditional_bkg=[], useOT=False, useAMP=False, fourier_features=0, fourier_max_freq=10.0):
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
        # conditional_bkg: List of lists, where each inner list specifies the conditional
        # values for a specific category of data. e.g., [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.conditional_bkg = conditional_bkg
        self.useOT = useOT # Flag for using optimal transport plan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available() and useAMP # Use AMP if CUDA is available and requested
        self.fourier_features = fourier_features # Number of Fourier features for Mingrustack
        self.fourier_max_freq = fourier_max_freq # Maximum frequency for Fourier features
        print(f"Using device: {self.device}")
        print(f"Automatic Mixed Precision (AMP) enabled: {self.use_amp}")
        if not torch.cuda.is_available():
             print("WARNING: CUDA not available, running on CPU.")
        if self.conditional and not self.conditional_bkg:
             print("WARNING: Conditional training is enabled but conditional_bkg is empty. Data will not be split by category.")

# ===================================================================
# Model Definitions (Adapted for Flow Matching)
# ===================================================================

# New wrapper class for Mingrustack to work with torchcfm
class VelocityField(nn.Module):
    def __init__(self, M, nhidden, nlayers, conditional=False, conditional_dim=0, fourier_features=0, fourier_max_freq=10.0):
        super().__init__()
        self.M = M
        self.conditional = conditional
        self.conditional_dim = conditional_dim
        # Input to Mingrustack will be concatenation of x_t, t, and c (if conditional)
        core_input_dim = M + 1 # M + 1 for time (t)
        if conditional:
            core_input_dim += conditional_dim

        self.core_model = Mingrustack(
            nlayers, core_input_dim, nhidden, M, dropout=0.0,
            fourier_features=fourier_features, fourier_max_freq=fourier_max_freq, layernorm=True
        ) # Output dim is M (velocity)

    def forward(self, t, x, c=None):
        # x shape: [batch_size, M] (this is x_t from CFM)
        # t shape: [batch_size] or scalar from odeint
        # c shape: [batch_size, conditional_dim] if conditional=True

        # Ensure t is a batch tensor for concatenation
        if t.dim() == 0:
             t_reshaped = t.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype).view(-1, 1)
        else:
             t_reshaped = t.view(-1, 1)

        # Concatenate x_t, t, and c (if conditional)
        xt = torch.cat([x, t_reshaped], dim=1)
        if self.conditional:
            if c is None:
                # This should ideally not happen if config.conditional is True
                raise ValueError("Conditional variable 'c' must be provided when conditional=True")
            xt = torch.cat([xt, c], dim=1)

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
    # Use the VelocityField class
    model = VelocityField(config.M, config.nhidden, config.nlayers, config.conditional, config.conditional_dim, config.fourier_features, config.fourier_max_freq).to(config.device)
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
        conditional_data = conditional_data.to(device) # Ensure conditional data is on the correct device

    # Define the dynamics function for the ODE solver
    # Needs access to the velocity network (v_net)
    def ode_func(t, x):
        # Ensure t is a tensor on the correct device, matching x's batch dim
        # The model's forward method handles scalar vs batch t
        t_tensor = torch.full((x.shape[0],), t.item(), device=device, dtype=x.dtype)

        # Predict velocity using the network
        with torch.no_grad(): # Ensure no gradients are computed here
             # Use autocast if AMP was used during training, might improve inference speed
             with amp.autocast(enabled=config.use_amp):
                  # Pass conditional variable to the network if needed
                  if config.conditional:
                        # Call the new model with t_tensor, x, and c
                        velocity = v_net(t_tensor, x, c=conditional_data) # Pass conditional_data
                  else:
                        # Call the new model with t_tensor, x
                        velocity = v_net(t_tensor, x)
        return velocity

    with torch.no_grad():  # Overall no_grad context
        # Sample initial points from the prior (standard Gaussian)
        if srcdist is None:
            x0 = torch.randn(num_samples, M, device=device)
        else:
            # Ensure srcdist is on the correct device and has enough samples
            if srcdist.shape[0] < num_samples:
                 raise ValueError(f"Source distribution has {srcdist.shape[0]} samples, but {num_samples} are requested.")
            sampleidx = torch.randperm(srcdist.shape[0])[:num_samples] # Sample indices without replacement
            x0 = srcdist[sampleidx].to(device) # Select samples and move to device

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
        return traj.cpu() # Return trajectory on CPU
    elif returnsrc:
        return x0.cpu(), samples.cpu() # Return both source and generated samples on CPU
    else:
        return samples.cpu() # Return samples on CPU


# ===================================================================
# Training Helper Functions
# ===================================================================

def prepare_data_and_loaders(data, config, srcdist=None, shuffle_train=True):
    """
    Prepares data (splitting features/conditional) and creates DataLoaders.
    Creates separate DataLoaders for each category in config.conditional_bkg.
    """
    device = config.device
    target_dataloaders_by_category = []
    srcdataloader = None

    # --- Data Splitting and Device Check ---
    if data.shape[1] != config.M + config.conditional_dim:
        raise ValueError("Data dimension must be M + conditional_dim")

    # Move data to device
    data_on_device = data.to(device)
    feature_data = data_on_device[:, :config.M]
    conditional_data = data_on_device[:, config.M:]

    # --- Source DataLoader Creation ---
    if srcdist is None:
        print("Generating default source distribution (Gaussian)")
        # Generate srcdist on the correct device
        srcdist_on_device = torch.randn_like(feature_data, device=device)
    else:
         # Ensure srcdist is on the correct device
         srcdist_on_device = srcdist.to(device)
         print(f"Moved srcdist to {device}")

    try:
        src_dataset = TensorDataset(srcdist_on_device)
        # Use drop_last=True to ensure consistent batch sizes when zipping/sampling
        srcdataloader = DataLoader(src_dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                   num_workers=0, pin_memory=False, drop_last=True)
        print(f"Source DataLoader created (shuffle={shuffle_train})")
    except Exception as e:
        raise RuntimeError(f"Error creating Source DataLoader: {e}")


    # --- Target DataLoaders Creation (by Category) ---
    if config.conditional and config.conditional_bkg:
        print(f"Creating target DataLoaders for {len(config.conditional_bkg)} categories...")
        for i, category_values in enumerate(config.conditional_bkg):
            if len(category_values) != config.conditional_dim:
                 print(f"Warning: Category {i} has {len(category_values)} values, but conditional_dim is {config.conditional_dim}. Skipping category.")
                 continue

            # Create a mask for data points matching this category's conditional values
            mask = torch.ones(data_on_device.shape[0], dtype=torch.bool, device=device)
            for j, val in enumerate(category_values):
                # Assuming conditional_data[:, j] corresponds to the j-th value in category_values
                mask &= (conditional_data[:, j] == val)

            category_feature_data = feature_data[mask]
            category_conditional_data = conditional_data[mask]

            if category_feature_data.shape[0] > 0:
                try:
                    category_dataset = TensorDataset(category_feature_data, category_conditional_data)
                    # Use drop_last=True for consistent batch sizes
                    category_dataloader = DataLoader(category_dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                                     num_workers=0, pin_memory=False, drop_last=True)
                    target_dataloaders_by_category.append(category_dataloader)
                    print(f"  - Created DataLoader for category {i} with {category_feature_data.shape[0]} samples.")
                except Exception as e:
                    print(f"Error creating DataLoader for category {i}: {e}")
            else:
                print(f"  - No samples found for category {i} ({category_values}). Skipping DataLoader creation.")

        if not target_dataloaders_by_category:
             print("Warning: No target DataLoaders were created based on conditional_bkg.")
             # Optionally, handle this case - maybe fall back to a single dataloader?
             # For now, the training loop will need to check if the list is empty.

    elif config.conditional and not config.conditional_bkg:
         print("Conditional training enabled, but conditional_bkg is empty. Creating a single DataLoader for all data.")
         # Create a single dataloader for all data if conditional but no categories specified
         try:
             dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                     num_workers=0, pin_memory=False, drop_last=True)
             target_dataloaders_by_category.append(dataloader) # Put the single dataloader in the list
             print(f"Single Target DataLoader created with {feature_data.shape[0]} samples.")
         except Exception as e:
             raise RuntimeError(f"Error creating single DataLoader: {e}")

    else: # Not conditional
        print("Conditional training disabled. Creating a single DataLoader for all data.")
        try:
            dataset = TensorDataset(feature_data)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle_train,
                                    num_workers=0, pin_memory=False, drop_last=True)
            target_dataloaders_by_category.append(dataloader) # Put the single dataloader in the list
            print(f"Single Target DataLoader created with {feature_data.shape[0]} samples.")
        except Exception as e:
            raise RuntimeError(f"Error creating single DataLoader: {e}")


    # Return the list of target dataloaders and the single source dataloader
    return target_dataloaders_by_category, srcdataloader


def initialize_training_components(config, v_net=None):
    """Initializes the model, optimizer, criterion, and scaler."""
    device = config.device
    if v_net is None:
        # Initialize the new MingruVelocityField model
        print("Initializing VelocityField model.")
        v_net = VelocityField(
            config.M,
            config.nhidden,
            config.nlayers,
            config.conditional,
            config.conditional_dim,
            fourier_features=config.fourier_features, # Pass Fourier features config
            fourier_max_freq=config.fourier_max_freq # Pass Fourier max freq config
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
    Samples batches randomly from category-specific dataloaders if conditional_bkg is used.
    """
    device = config.device
    epsilon = config.epsilon

    print(f"--- Preparing Data and Loaders ---")
    try:
        # prepare_data_and_loaders now returns a list of target dataloaders and the src dataloader
        target_dataloaders_by_category, srcdataloader = prepare_data_and_loaders(
            data, config, srcdist, shuffle_train=shuffle_train
        )
    except Exception as e:
        print(f"Error during data/loader preparation: {e}")
        return None, [], [] # Return empty lists for losses and norms

    if not target_dataloaders_by_category:
        print("No target DataLoaders available. Skipping training.")
        return None, [], [] # Return empty lists if no data loaders were created

    print(f"--- Initializing Training Components ---")
    try:
        v_net, optimizer, criterion, scaler = initialize_training_components(config, v_net)
    except Exception as e:
        print(f"Error during component initialization: {e}")
        return None, [], [] # Return empty lists for losses and norms

    # Initialize Conditional Flow Matcher
    if config.useOT:
        print("Using ExactOptimalTransportConditionalFlowMatcher")
        cfm = ExactOptimalTransportConditionalFlowMatcher()
    else:
        print("Using ConditionalFlowMatcher (sigma=1.0)") # Default sigma is 1.0
        cfm = ConditionalFlowMatcher()


    epoch_losses = []
    epoch_grad_norms = []

    print(f"--- Starting Flow Matching Training ({config.epochs} epochs) ---")
    epochs_pbar = tqdm(range(config.epochs), desc=f"Training FM", unit="epoch")

    # Create infinite iterators for sampling batches
    target_iterators = [itertools.cycle(dataloader) for dataloader in target_dataloaders_by_category]
    src_iterator = itertools.cycle(srcdataloader)

    # Determine the number of batches per epoch.
    # A common approach is to process a number of batches equivalent to iterating
    # through the largest dataset once, or simply a fixed large number.
    # Let's use the total number of samples divided by batch size as a rough guide.
    total_samples = sum(len(dl.dataset) for dl in target_dataloaders_by_category)
    num_batches_per_epoch = total_samples // config.batch_size
    if num_batches_per_epoch == 0:
         print("Warning: Total samples less than batch size. Adjusting num_batches_per_epoch to 1.")
         num_batches_per_epoch = 1

    print(f"Processing approximately {num_batches_per_epoch} batches per epoch.")

    # --- Training Loop ---
    for epoch in epochs_pbar:
        v_net.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches_processed_this_epoch = 0

        # Iterate for a fixed number of batches per epoch
        for _ in range(num_batches_per_epoch):
            # Randomly select a target dataloader (and its iterator)
            random_category_idx = random.randint(0, len(target_iterators) - 1)
            current_target_iterator = target_iterators[random_category_idx]

            # Get a batch from the selected target dataloader
            try:
                batch = next(current_target_iterator)
                if config.conditional:
                    x_1, c = batch # Target data points (feature data) and conditional data
                else:
                    x_1 = batch[0] # Target data points (feature data only)
                    c = None
            except StopIteration:
                 # This should not happen with itertools.cycle, but good practice
                 print(f"Warning: StopIteration encountered for target iterator {random_category_idx}. Breaking batch loop.")
                 break

            # Get a batch from the source dataloader
            try:
                srcbatch_tuple = next(src_iterator)
                x_0 = srcbatch_tuple[0] # DataLoader wraps tensors in a tuple
            except StopIteration:
                 # This should not happen with itertools.cycle
                 print("Warning: StopIteration encountered for source iterator. Breaking batch loop.")
                 break

            # Ensure source batch size matches target batch size (important if drop_last=False or last batch is smaller)
            if x_0.shape[0] != x_1.shape[0]:
                 # This should ideally not happen with drop_last=True and itertools.cycle
                 # but as a safeguard, truncate x_0 if necessary.
                 # Note: This might slightly bias the source distribution if batch sizes differ often.
                 # A better approach might be to ensure all dataloaders have drop_last=True
                 # and that the number of batches per epoch is consistent.
                 print(f"Warning: Source batch size ({x_0.shape[0]}) does not match target batch size ({x_1.shape[0]}). Truncating source batch.")
                 x_0 = x_0[:x_1.shape[0]]


            current_batch_size = x_1.shape[0]
            if current_batch_size == 0:
                 print(f"Warning: Empty batch from category {random_category_idx}. Skipping.")
                 continue # Skip if an empty batch somehow occurs

            # 1. Sample time t ~ U(epsilon, 1)
            # t = torch.rand(current_batch_size, device=device) * (1.0 - epsilon) + epsilon
            # CFM handles time sampling internally in sample_location_and_conditional_flow
            # We still need a t tensor for the model input, but CFM provides x_t and v_target

            # Use CFM to sample x_t and get the target velocity v_target
            # The t tensor returned here is the one sampled by CFM
            # sample_location_and_conditional_flow returns t, x_t, and v_target
            # It also returns the conditional variable c if provided
            
            t, x_t, v_target = cfm.sample_location_and_conditional_flow(x_0, x_1)
            
            # 2.5.5 Sample optimal transport plan (x_0, x_1) - Handled by ExactOptimalTransportConditionalFlowMatcher if useOT is True
            # The path x_t = t*x_1 + (1-t)*x_0 is the standard CFM path.
            # ExactOptimalTransportConditionalFlowMatcher modifies the target velocity v_target
            # based on the OT plan, but the path x_t remains the same.
            # The original code's OT sampling part seems incorrect for this context and is removed.

            with amp.autocast(enabled=config.use_amp):
                # Predict velocity using the network
                # Pass t, x_t, and c to the model
                predicted_velocity = v_net(t, x_t, c=c)
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
            num_batches_processed_this_epoch += 1

        if num_batches_processed_this_epoch > 0:
            avg_epoch_loss = epoch_loss / num_batches_processed_this_epoch
            avg_epoch_grad_norm = epoch_grad_norm / num_batches_processed_this_epoch
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
        # Ensure srcdist is a tensor and has enough samples
        if not isinstance(srcdist, torch.Tensor):
             srcdist = torch.tensor(srcdist, dtype=torch.float32) # Convert to tensor if not already
        if srcdist.shape[0] < num_samples:
             raise ValueError(f"Source distribution for animation has {srcdist.shape[0]} samples, but {num_samples} are requested.")
        srcdist_anim = srcdist[:num_samples].to(config.device) # Take subset and move to device
    else:
        srcdist_anim = None # sample_flow will generate Gaussian noise

    # sample_flow expects srcdist on the *same device* as the model will run on (config.device)
    # but returns the trajectory on CPU.
    # sample_flow returns a tuple if returnsrc is True, otherwise a tensor.
    # We need the trajectory, so returntraj=True. sample_flow returns traj.cpu() in this case.
    traj_cpu = sample_flow(v_net, config, num_samples=num_samples, conditional_data=conditional_data, srcdist=srcdist_anim, returntraj=True)

    # --- Create Animation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    # Determine fixed plot limits based on the final distribution (+ a buffer)
    final_points = traj_cpu[-1]
    if final_points.shape[0] > 0: # Check if any points were generated
        xlim = (final_points[:, 0].min().item() - 0.5, final_points[:, 0].max().item() + 0.5)
        ylim = (final_points[:, 1].min().item() - 0.5, final_points[:, 1].max().item() + 0.5)
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
            artists.extend(lines) # Extend with the line objects

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


