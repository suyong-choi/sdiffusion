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

# --- Optional: Clear plots at the start ---
plt.close('all')
# ===================================================================
# Configuration Class (config.py)
# ===================================================================
class Config:
    """ Stores model and training configuration parameters. """
    def __init__(self, M, nhidden, nlayers, timesteps, noise_schedule, batch_size, learning_rate, epochs):
        self.M = M # Data dimensionality
        self.nhidden = nhidden # Hidden layer size
        self.nlayers = nlayers # Number of layers in MLP
        self.timesteps = timesteps # Number of diffusion steps
        self.noise_schedule = noise_schedule # 'linear', 'cosine', 'quadratic'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # --- Device and AMP setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available() # Enable AMP only if CUDA is available
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
    def __init__(self, M, nhidden, nlayers, timesteps):
        """
        A simple MLP model to predict noise given noisy data and timestep.

        Args:
            M (int): Number of features in the data.
            timesteps (int): Number of diffusion steps.
        """
        super(MingruMLP, self).__init__()
        self.time_embed = nn.Embedding(timesteps, timesteps) # Use Embedding instead of Sinusoidal
        self.model = Mingrustack(nlayers, M+timesteps, nhidden, M)

    def forward(self, x_t, t):
        """
        Forward pass of the MLP.

        Args:
            x_t (torch.Tensor): Noisy data at timestep t (shape: [batch_size, M]).
            t (torch.Tensor): Timestep (shape: [batch_size]).

        Returns:
            torch.Tensor: Predicted noise (shape: [batch_size, M]).
        """
        t_embed = self.time_embed(t) # Use Embedding instead of Sinusoidal
        x = torch.cat([x_t, t_embed], dim=1)
        return self.model(x)


# ===================================================================
# Utility Functions (model_io.py, plotting_utils.py)
# ===================================================================
def get_config_directory(config):
    """ Creates a directory name based on config parameters. """
    lr_str = f"{config.learning_rate:.0e}" # Format LR concisely
    return f"M{config.M}_nh{config.nhidden}_nl{config.nlayers}_T{config.timesteps}_{config.noise_schedule}_BS{config.batch_size}_LR{lr_str}_E{config.epochs}"

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
    model = model_class(config.M, config.nhidden, config.nlayers, config.timesteps).to(config.device)
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
# Sampling Function (DDPM)
# ===================================================================
def sample(model, config, scheduler, num_samples=1):
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
                 predicted_noise = model(x_t, t_tensor)

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

    model.train() # Return model to training mode
    return x_t.cpu() # Return samples on CPU for easier handling

# ===================================================================
# Training Function (Optimized for GPU-Resident Data)
# ===================================================================
def train_diffusion_model_gpu_resident(data, config, scheduler):
    """
    Trains a diffusion model assuming the entire 'data' tensor is already on the GPU.
    Uses a DiffusionScheduler instance and displays epoch-level tqdm progress.
    """
    device = config.device
    use_amp = config.use_amp

    print(f"Training with data tensor on device: {data.device}")
    #if data.device[:-2] != device:
    #    print(f"Error: Data device ({data.device}) differs from target device ({device}).")
    #    return None, [] # Return indicating failure

    # Create DataLoader for GPU Tensor
    try:
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True) # drop_last can help with stability if last batch is small
        print(f"DataLoader created with batch size {config.batch_size}, num_workers=0, pin_memory=False")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return None, []

    # Initialize model and optimizer
    # model = SimpleMLP(config.M, config.nhidden, config.nlayers, config.timesteps).to(device)
    model = MingruMLP(config.M, config.nhidden, config.nlayers, config.timesteps).to(device) # Using MingruMLP placeholder

    try:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss() # Mean Squared Error for noise prediction
        scaler = amp.GradScaler(enabled=use_amp) # GradScaler for mixed precision
    except Exception as e:
        print(f"Error initializing model/optimizer: {e}")
        return None, []

    epoch_losses = []
    print(f"Starting training on {device} for {config.epochs} epochs...")

    # Wrap the epoch loop with tqdm for epoch-level progress tracking
    epochs_pbar = tqdm(range(config.epochs), desc=f"Training (Epochs)", unit="epoch")

    # --- Training Loop ---
    for epoch in epochs_pbar:
        model.train() # Ensure model is in training mode
        epoch_loss = 0.0

        # Iterate through batches (already on GPU)
        for batch in dataloader:
            x_0 = batch[0] # Access tensor from the tuple

            # Sample timesteps uniformly for the batch
            t = torch.randint(0, config.timesteps, (x_0.shape[0],), device=device)

            # --- Mixed Precision Context (forward pass) ---
            with amp.autocast(enabled=use_amp):
                # Apply forward diffusion to get noisy sample x_t and target noise
                x_t, noise_target = forward_diffusion(x_0, t, scheduler)
                # Predict noise using the model
                predicted_noise = model(x_t, t)
                # Calculate loss between predicted noise and actual noise
                loss = criterion(predicted_noise, noise_target)

            # --- Backpropagation with GradScaler ---
            optimizer.zero_grad(set_to_none=True) # Efficiently zero gradients
            scaler.scale(loss).backward() # Scale loss and compute gradients
            # Optional: Gradient Clipping (uncomment if needed)
            # scaler.unscale_(optimizer) # Unscale gradients before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer) # Update model parameters (unscales gradients internally)
            scaler.update() # Update GradScaler's scale factor

            epoch_loss += loss.item()

        # Calculate average loss for the completed epoch
        if len(dataloader) > 0:
             avg_epoch_loss = epoch_loss / len(dataloader)
             epoch_losses.append(avg_epoch_loss)
             # Update the main tqdm postfix with the epoch's average loss
             epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.6f}")
        else:
             print(f"Warning: DataLoader was empty for epoch {epoch+1}.")
             epoch_losses.append(float('nan')) # Record NaN if no batches processed


    print("\nTraining finished.")
    return model, epoch_losses

# ===================================================================
# Main Execution Block
# ===================================================================
if __name__ == "__main__":
    # 1. Generate Data (on CPU initially)
    M = 3 # Data dimensionality
    num_samples = 32000 # Adjust based on your VRAM (e.g., 16k, 32k, 64k)
    print(f"Generating {num_samples} samples...")
    data_cpu, original_radii = generate_3d_sphere_data(num_samples)
    data_cpu = data_cpu.float() # Ensure float32

    # 2. Configure Training
    config = Config(
        M=M,
        nhidden=1024,        # Hidden layer size (tune)
        nlayers=2,          # Number of layers (tune)
        timesteps=100,      # Diffusion timesteps (e.g., 100, 500, 1000)
        noise_schedule='linear', # 'linear', 'cosine', 'quadratic'
        batch_size=64,    # Adjust based on VRAM after loading data+model
        learning_rate=1e-3, # Tune learning rate
        epochs=20000,         # Number of training epochs (e.g., 100, 500, 1000+)
    )

    # 3. Instantiate Diffusion Scheduler ONCE
    print("Initializing Diffusion Scheduler...")
    try:
        scheduler = DiffusionScheduler(config.noise_schedule, config.timesteps, config.device)
    except Exception as e:
        print(f"Error initializing DiffusionScheduler: {e}")
        exit()

    # 4. Move Entire Dataset to Target Device (GPU if available)
    data_on_device = None
    if config.device.type == 'cuda':
        print(f"Attempting to move dataset ({data_cpu.nelement() * data_cpu.element_size() / 1024**2:.2f} MB) to {config.device}...")
        try:
            # Check available memory (optional but recommended)
            # total_mem = torch.cuda.get_device_properties(0).total_memory
            # reserved_mem = torch.cuda.memory_reserved(0)
            # allocated_mem = torch.cuda.memory_allocated(0)
            # free_mem = total_mem - reserved_mem
            # print(f"GPU Memory Free (approx): {free_mem / 1024**2:.2f} MB")
            # if data_cpu.nbytes > free_mem * 0.8: # Check if data exceeds ~80% of free VRAM
            #    raise RuntimeError("Dataset likely too large for available VRAM.")

            data_on_device = data_cpu.to(config.device)
            print(f"Successfully moved dataset to {data_on_device.device}")
            # Optionally release CPU memory if dataset is very large
            # del data_cpu
            # import gc
            # gc.collect() # Force garbage collection
        except RuntimeError as e:
            print(f"\n----- ERROR moving full dataset to GPU: {e} -----")
            print("----- Ensure you have enough VRAM. Exiting. -----")
            exit()
        except Exception as e:
             print(f"\n----- An unexpected error occurred during data transfer: {e} -----")
             exit()
    else:
        print("Running on CPU, keeping data on CPU.")
        data_on_device = data_cpu # Use the CPU tensor if device is CPU

    # --- Training ---
    train_model_flag = True # Set to False to skip training and only load/sample
    config_dir = get_config_directory(config)
    os.makedirs(config_dir, exist_ok=True)
    model_filename = "model_final.pth" # Consistent filename
    epoch_losses = []

    if train_model_flag:
        print("\n--- Starting Training ---")
        # Pass the data tensor (already on target device) and the scheduler
        trained_model, epoch_losses = train_diffusion_model_gpu_resident(data_on_device, config, scheduler)

        if trained_model: # Check if training was successful
            save_model(trained_model, config, filename=model_filename)
            # Plot and save training loss if available
            if epoch_losses:
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.yscale('log') # Log scale is often better for loss plots
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.title(f'Training Loss per Epoch\n{get_config_description(config)}')
                loss_fig = plt.gcf()
                save_plot(loss_fig, config, "training_loss.png")
        else:
            print("Training failed, model not saved.")

    # --- Sampling ---
    model_path = os.path.join(config_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"\nModel file {model_path} not found. Cannot perform sampling.")
    else:
        print("\n--- Loading Model for Sampling ---")
        # Ensure the correct model class is used for loading
        loaded_model = load_model(MingruMLP, config, filename=model_filename)

        print("--- Generating Samples ---")
        num_generated_samples = 4000
        sampling_batch_size = 1000 # Adjust based on VRAM available for inference
        generated_data_list = []

        num_batches = (num_generated_samples + sampling_batch_size - 1) // sampling_batch_size
        for i in range(num_batches): # No tqdm here, sample function has internal tqdm
            current_batch_size = min(sampling_batch_size, num_generated_samples - i * sampling_batch_size)
            if current_batch_size <= 0: break
            # Pass the loaded model, config, and scheduler to sample function
            generated_batch = sample(loaded_model, config, scheduler, num_samples=current_batch_size)
            generated_data_list.append(generated_batch)

        if generated_data_list:
             generated_data = torch.cat(generated_data_list, dim=0)
             print(f"Generated {generated_data.shape[0]} samples.")

             # --- Analysis and Plotting ---
             print("--- Analyzing and Plotting Results ---")
             generated_radii = torch.norm(generated_data, dim=1).numpy()
             original_radii_np = original_radii.cpu().numpy() # Ensure original radii are numpy

             # Plot Radii Histogram
             plt.figure(figsize=(10, 6))
             hist_range = (0.0, 2.0) # Adjust range dynamically or fix
             plt.hist(original_radii_np, bins=50, range=hist_range, density=True, alpha=0.6, label='Original Data Radii')
             plt.hist(generated_radii, bins=50, range=hist_range, density=True, alpha=0.6, label='Generated Data Radii')
             plt.xlabel('Radius')
             plt.ylabel('Density')
             plt.legend()
             plt.title(f'Histogram of Data Radii\n{get_config_description(config)}')
             plt.grid(True, linestyle='--', linewidth=0.5)
             histogram_fig = plt.gcf()
             save_plot(histogram_fig, config, "radii_histogram.png")

             # Plot Scatter (if M >= 2)
             if M >= 2:
                 plt.figure(figsize=(8, 8))
                 # Plot a subset for clarity if dataset is large
                 num_points_to_plot = min(1000, data_on_device.shape[0], generated_data.shape[0])
                 orig_data_cpu = data_on_device[:num_points_to_plot].cpu() # Move needed points to CPU for plotting
                 gen_data_cpu = generated_data[:num_points_to_plot].cpu()
                 plt.scatter(orig_data_cpu[:, 0], orig_data_cpu[:, 1], alpha=0.5, s=10, label='Original Data Sample')
                 plt.scatter(gen_data_cpu[:, 0], gen_data_cpu[:, 1], alpha=0.5, s=10, label='Generated Data Sample')
                 plt.xlabel('Feature 1')
                 plt.ylabel('Feature 2')
                 plt.legend()
                 plt.title(f'Data Scatter Plot (First 2D)\n{get_config_description(config)}')
                 plt.axis('equal')
                 plt.grid(True, linestyle='--', linewidth=0.5)
                 scatter_fig = plt.gcf()
                 save_plot(scatter_fig, config, "scatter_plot_2d.png")

             print(f"\nResults saved in directory: {config_dir}")
        else:
             print("No data generated for plotting.")

    print("\nScript finished.")