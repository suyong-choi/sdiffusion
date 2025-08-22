# ===================================================================
# Full Diffusion Model Script (Optimized for GPU-Resident Data)
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
from mingruinspired import Mingrustack  # Assuming this is your custom module
from utils import generate_3d_sphere_data, normalize_data, denormalize_data # Assume utils exists

import itertools
import random


# --- Optional: Clear plots at the start ---
plt.close('all')
# ===================================================================
# Configuration Class (config.py)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters. """
    def __init__(self, M, nhidden, nlayers, timesteps, timeembed, noise_schedule, batch_size, learning_rate, epochs, model_type='MingruMLP', conditional=False, conditional_dim=0, conditional_bkg=[]):
        self.M = M # Data dimensionality
        self.nhidden = nhidden # Hidden layer size
        self.nlayers = nlayers # Number of layers in MLP
        self.timesteps = timesteps # Number of diffusion steps
        self.timeembed = timeembed # Time embedding dimension   
        self.noise_schedule = noise_schedule # 'linear', 'cosine', 'quadratic'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.conditional = conditional # Flag for conditional training
        self.conditional_dim = conditional_dim # Dimension of conditional variable
        self.conditional_bkg = conditional_bkg # Background conditioning information
        # model type
        self.model_type = model_type # Default model type, can be changed to 'SimpleUNet' or others
        # --- Device and AMP setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = False #torch.cuda.is_available() # Enable AMP only if CUDA is available
        print(f"Using device: {self.device}")
        print(f"Automatic Mixed Precision (AMP) enabled: {self.use_amp}")
        if not torch.cuda.is_available():
             print("WARNING: CUDA not available, running on CPU. Performance will be significantly slower.")

# ===================================================================
# Diffusion Scheduler (diffusion_sched.py)
# ===================================================================
def get_noise_schedule(schedule_type, timesteps):
    """ Calculates the beta noise schedule. """
    if schedule_type == 'linear':
        betas = torch.linspace(1e-4, 0.02, timesteps)
    elif schedule_type == 'quadratic':
        # Quadratic schedule increasing from start^2 to end^2
        start, end = 1e-4, 0.02 # Example range, adjust if needed
        betas = torch.linspace(start**0.5, end**0.5, timesteps) ** 2
    elif schedule_type == 'cosine':
        # Cosine schedule based on Improved DDPM paper
        s = 0.008 # Offset to prevent beta_t = 0
        t_steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((t_steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Normalize to start at 1
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # Calculate betas
        betas = torch.clip(betas, 0.0001, 0.9999).float() # Clip for stability
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")

    # Ensure correct length if schedule logic didn't produce exactly `timesteps`
    if len(betas) != timesteps:
         print(f"Warning: Adjusting beta length from {len(betas)} to {timesteps}")
         if len(betas) > timesteps: betas = betas[:timesteps]
         else: betas = F.pad(betas, (0, timesteps - len(betas))) # Pad if too short
    return betas.float() # Ensure float32


class DiffusionScheduler:
    """
    Handles the calculation and storage of diffusion schedule parameters.
    Precomputes terms needed for both forward and reverse processes.
    """
    def __init__(self, schedule_type, timesteps, device):
        self.timesteps = timesteps
        self.device = device

        # Calculate base schedule and move to device
        self.betas = get_noise_schedule(schedule_type, timesteps).to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # --- Precompute terms for forward diffusion q(x_t | x_0) ---
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        # Clipped version for stability if used in division
        self.sqrt_one_minus_alphas_cumprod_clipped = torch.clamp(self.sqrt_one_minus_alphas_cumprod, min=1e-8)

        # --- Precompute terms for reverse diffusion p(x_{t-1} | x_t, x_0) (DDPM) ---
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)

        # Posterior variance: beta_t * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clamp variance > 0, especially for t=0 where alpha_cumprod_prev = 1
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8)
        # Handle t=0 variance (following DDPM): variance is 0, but noise term uses beta_t
        if len(self.posterior_variance) > 0:
             # The actual variance for q(x_0 | x_1, x_0) is 0.
             # However, the sampling formula uses sqrt(beta_t) * z for t=1 -> t=0.
             # Let's store sqrt(beta_t) separately for clarity if needed,
             # or ensure posterior_variance[0] doesn't cause issues.
             # DDPM paper uses sqrt(tilde{beta}_t) where tilde{beta}_t = posterior_variance
             # Let's set posterior_variance[0] = beta_0 for consistency in indexing if needed,
             # but the sampling logic handles t=0 separately anyway.
             self.posterior_variance[0] = self.betas[0] # Or another small value if beta_0 is 0

        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance).to(device)

        print(f"DiffusionScheduler initialized on {self.device} with T={timesteps} ({schedule_type})")

    def get_forward_terms(self, t):
        """ Helper to get terms needed for forward diffusion at time t. """
        # Use gather for safe indexing across batch dimension for t
        sqrt_alpha_t = self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1)
        return sqrt_alpha_t, sqrt_one_minus_alpha_t

# ===================================================================
# Forward Diffusion Process
# ===================================================================
def forward_diffusion(x_0, t, scheduler):
    """
    Applies forward diffusion using precomputed terms from the scheduler.
    Assumes x_0 is on the correct device. t is a tensor of timesteps.
    """
    # Get precomputed sqrt terms for the batch timesteps t
    sqrt_alpha_t, sqrt_one_minus_alpha_t = scheduler.get_forward_terms(t)

    # Generate noise on the same device as x_0
    noise = torch.randn_like(x_0)

    # Calculate noisy sample x_t
    x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
    return x_t, noise

# ===================================================================
# Model Definitions (model.py)
# ===================================================================
class MingruMLP(nn.Module):
    #def __init__(self, M, nhidden, nlayers, timesteps, tembeddim, conditional=False, cond_dim=0):
    def __init__(self, config):
        """
        A simple MLP model to predict noise given noisy data and timestep.

        Args:
            M (int): Number of features in the data.
            timesteps (int): Number of diffusion steps.
            cond_dim (int): Number of conditional features.
        """
        super(MingruMLP, self).__init__()
        self.M = config.M
        self.time_embed = nn.Embedding(config.timesteps, config.timeembed).to(config.device) # Use Embedding instead of Sinusoidal
        self.conditional = config.conditional
        self.cond_dim = config.conditional_dim
        self.model = Mingrustack(config.nlayers, config.M+config.timeembed+config.conditional_dim, config.nhidden, config.M, layernorm=True)

    def forward(self, x_t, t, cond=None):
        """
        Forward pass of the MLP.

        Args:
            x_t (torch.Tensor): Noisy data at timestep t (shape: [batch_size, M]).
            t (torch.Tensor): Timestep (shape: [batch_size]).
            cond (torch.Tensor or None): Conditional variables (shape: [batch_size, cond_dim])

        Returns:
            torch.Tensor: Predicted noise (shape: [batch_size, M]).
        """
        t_embed = self.time_embed(t) # Use Embedding instead of Sinusoidal
        if cond is not None:
            x = torch.cat([x_t, t_embed, cond], dim=1)
        else:
            x = torch.cat([x_t, t_embed], dim=1)
        return self.model(x)

class SimpleUNet(nn.Module):
    """
    A simple 1D U-Net architecture for DDPM-style vector data.
    Supports optional conditional input and time embedding.
    Uses layer normalization for regularization.
    """
    def __init__(self, config):
        super(SimpleUNet, self).__init__()
        self.M = config.M
        self.conditional = config.conditional
        self.cond_dim = config.conditional_dim

        # Time embedding (learned)
        self.time_embed = nn.Embedding(config.timesteps, config.timeembed)

        # Input channels: data + time embedding + conditional (if any)
        in_channels = self.M + config.timeembed + (self.cond_dim if self.conditional else 0)

        # Encoder
        self.enc1 = nn.Linear(in_channels, config.nhidden)
        self.ln1 = nn.LayerNorm(config.nhidden)
        self.enc2 = nn.Linear(config.nhidden, config.nhidden)
        self.ln2 = nn.LayerNorm(config.nhidden)

        # Bottleneck
        self.bottleneck = nn.Linear(config.nhidden, config.nhidden)
        self.ln_bottleneck = nn.LayerNorm(config.nhidden)

        # Decoder
        self.dec2 = nn.Linear(config.nhidden, config.nhidden)
        self.ln_dec2 = nn.LayerNorm(config.nhidden)
        self.dec1 = nn.Linear(config.nhidden, config.M)
        self.act = nn.SiLU()

    def forward(self, x_t, t, cond=None):
        # Embed time
        t_embed = self.time_embed(t)
        # Concatenate inputs
        if self.conditional and cond is not None:
            x = torch.cat([x_t, t_embed, cond], dim=1)
        else:
            x = torch.cat([x_t, t_embed], dim=1)

        # Encoder
        e1 = self.act(self.ln1(self.enc1(x)))
        e2 = self.act(self.ln2(self.enc2(e1)))
        # Bottleneck
        b = self.act(self.ln_bottleneck(self.bottleneck(e2)))
        # Decoder with skip connections
        d2 = self.act(self.ln_dec2(self.dec2(b + e2)))
        d1 = self.dec1(d2 + e1)
        return d1


# ===================================================================
# Utility Functions (model_io.py, plotting_utils.py)
# ===================================================================
def get_config_directory(config):
    """ Creates a directory name based on config parameters. """
    lr_str = f"{config.learning_rate:.0e}" # Format LR concisely
    if config.model_type == 'MingruMLP':
        retstring = f"DM{config.M}_nh{config.nhidden}_nl{config.nlayers}_T{config.timesteps}_ted{config.timeembed}_NS{config.noise_schedule}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}_MLP"
    else:
        retstring = f"DM{config.M}_nh{config.nhidden}_nl{config.nlayers}_T{config.timesteps}_ted{config.timeembed}_NS{config.noise_schedule}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}_UNet"
    return retstring

def save_model(model, config, filename="model.pth"):
    """ Saves the model state dictionary. """
    directory = get_config_directory(config)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, config, filename="model.pth"):
    """ Loads the model state dictionary. """
    directory = get_config_directory(config)
    filepath = os.path.join(directory, filename)
    # Instantiate model first on the correct device
    model = model_class(config).to(config.device)
    if os.path.exists(filepath):
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.device))
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model state_dict from {filepath}: {e}")
            print("Starting with a fresh model.")
    else:
        print(f"Warning: Model file not found at {filepath}. Starting with a fresh model.")
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
    plt.close(fig) # Close the figure after saving to free memory

def get_config_description(config):
    """ Generates a string description of the configuration. """
    lr_str = f"{config.learning_rate:.1e}"
    return (f"M={config.M}, nh={config.nhidden}, nl={config.nlayers}, T={config.timesteps}, "
            f"NS={config.noise_schedule}, BS={config.batch_size}, LR={lr_str}, E={config.epochs}")


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


# ===================================================================
# Sampling Function (DDPM)
# ===================================================================
def sample(model, config, scheduler, num_samples=1, cond=None):
    """
    Generates samples using the DDPM reverse process and precomputed scheduler terms.
    """
    model.eval() # Set model to evaluation mode
    device = config.device
    M = config.M
    timesteps = config.timesteps

    with torch.no_grad(): # Disable gradient calculation for efficiency
        # Start with random noise (standard Gaussian) on the target device
        x_t = torch.randn(num_samples, M, device=device)


        # Iterate backwards through timesteps
        for t in tqdm(reversed(range(timesteps)), desc="Sampling Progress", total=timesteps, leave=False):
            # Create timestep tensor for the batch on the correct device
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

            # Predict noise using the model (potentially with AMP)
            with amp.autocast(enabled=config.use_amp):
                 predicted_noise = model(x_t, t_tensor, cond=cond)

            # Get precomputed terms from the scheduler for the current timestep t
            sqrt_recip_alpha_t = scheduler.sqrt_recip_alphas[t]
            beta_t = scheduler.betas[t]
            sqrt_one_minus_alpha_cumprod_t_clipped = scheduler.sqrt_one_minus_alphas_cumprod_clipped[t]

            # Calculate the mean of the reverse distribution p(x_{t-1} | x_t)
            # mean = sqrt(1/alpha_t) * (x_t - beta_t / sqrt(1 - alpha_cumprod_t) * predicted_noise)
            mean_term = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t_clipped)

            if t > 0:
                # Add noise for steps t > 0
                noise = torch.randn_like(x_t)
                # Use precomputed sqrt posterior variance
                x_t = mean_term + scheduler.sqrt_posterior_variance[t] * noise
            else:
                # Final step (t=0): The mean is the result, no noise added
                x_t = mean_term

        generated = x_t.cpu()
    model.train() # Return model to training mode
    return generated


def initialize_training_components(config, v_net=None):
    """Initializes the model, optimizer, criterion, and scaler."""
    device = config.device
    if v_net is None:
        print("Initializing VelocityField model.")
        if config.model_type == 'MingruMLP':
            v_net = MingruMLP(config).to(device)
        else:
            v_net = SimpleUNet(config).to(device)
    else:
        print("Using provided v_net model.")
        v_net = v_net.to(device)

    try:
        optimizer = optim.Adam(v_net.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        scaler = amp.GradScaler(enabled=config.use_amp)
    except Exception as e:
        raise RuntimeError(f"Error initializing optimizer/criterion/scaler: {e}")

    return v_net, optimizer, criterion, scaler

def train_ddpm(data, config, srcdist=None, v_net=None, shuffle_train=True):
    """
    Trains a Flow Matching model (velocity network) using helper functions.
    Samples batches randomly from category-specific dataloaders if conditional_bkg is used.
    """
    device = config.device


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

    # 3. Instantiate Diffusion Scheduler ONCE
    print("Initializing Diffusion Scheduler...")
    try:
        scheduler = DiffusionScheduler(config.noise_schedule, config.timesteps, config.device)
    except Exception as e:
        print(f"Error initializing DiffusionScheduler: {e}")
        exit()

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



            current_batch_size = x_1.shape[0]
            if current_batch_size == 0:
                 print(f"Warning: Empty batch from category {random_category_idx}. Skipping.")
                 continue # Skip if an empty batch somehow occurs

            #sample timesteps uniformly for the batch
            # Ensure t is a tensor of the correct shape and device

            t = torch.randint(0, config.timesteps, (current_batch_size,), device=device)


            with amp.autocast(enabled=config.use_amp):
                x_t, noise_target = forward_diffusion(x_1, t, scheduler)
                # Predict noise using the model
                predicted_noise = v_net(x_t, t, cond=c)
                # Calculate loss between predicted noise and actual noise
                loss = criterion(predicted_noise, noise_target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # --- Gradient Norm Monitoring ---
            if config.use_amp:
                scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            
            num_batches_processed_this_epoch += 1

        if num_batches_processed_this_epoch > 0:
            avg_epoch_loss = epoch_loss / num_batches_processed_this_epoch
            
            epoch_losses.append(avg_epoch_loss)
            
            epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.6f}")
        else:
            print(f"Warning: No batches processed in epoch {epoch+1}.")
            epoch_losses.append(float("nan"))
            epoch_grad_norms.append(float("nan"))


    print("\nTraining finished.")
    return v_net, epoch_losses

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

