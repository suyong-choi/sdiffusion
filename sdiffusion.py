import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # Add this import for plotting
from tqdm import tqdm  # Add this import for progress bar
from mingruinspired import Mingrustack
from utils import generate_3d_sphere_data, normalize_data, denormalize_data
import os  # Add this import for directory management

# Configuration
class Config:
    def __init__(self, M, nhidden, nlayers, timesteps, noise_schedule, batch_size, learning_rate, epochs):
        """
        Configuration class for the diffusion model.

        Args:
            M (int): Number of features in the data.
            nhidden (int): Number of hidden units in the MLP.
            layers (int): Number of layers in the MLP.
            timesteps (int): Number of diffusion steps.
            noise_schedule (str): Type of noise schedule ('linear', 'quadratic', 'cosine').
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        self.M = M
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Noise Schedule
def get_noise_schedule(schedule_type, timesteps):
    """
    Generates the noise schedule (beta_t) for the diffusion process.

    Args:
        schedule_type (str): Type of noise schedule ('linear', 'quadratic', 'cosine').
        timesteps (int): Number of diffusion steps.

    Returns:
        torch.Tensor: The noise schedule (beta_t).
    """
    if schedule_type == 'linear':
        betas = torch.linspace(1e-4, 0.02, timesteps)
    elif schedule_type == 'quadratic':
        betas = torch.linspace(1e-6, 0.01, timesteps) ** 2
    elif schedule_type == 'cosine':
        s = 8  # Controls the steepness of the cosine curve
        t = torch.arange(timesteps)
        f = torch.cos((t / timesteps + s / timesteps) / (1 + s / timesteps) * torch.pi / 2) ** 2
        betas = 1 - f[1:] / f[:-1]
        betas = torch.clip(betas, 0.00001, 0.999)  # Clip to ensure beta_t is in (0, 1)
        # Pad betas with a 0 at the beginning to make it the same length as timesteps
        betas = torch.cat([torch.tensor([0.00001]), betas]) # Add this line to fix the off-by-one error
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")
    return betas

# Diffusion Process
def forward_diffusion(x_0, t, betas):
    """
    Applies the forward diffusion process to the data.

    Args:
        x_0 (torch.Tensor): The original data (shape: [batch_size, M]).
        t (torch.Tensor): Timestep (shape: [batch_size]).
        betas (torch.Tensor): The noise schedule (shape: [timesteps]).

    Returns:
        torch.Tensor: The noisy data at timestep t (x_t) (shape: [batch_size, M]).
        torch.Tensor: The noise added to the data.
    """
    alpha_t_bar = torch.cumprod(1 - betas, dim=0).to(x_0.device)
    # Get the alpha_t_bar for the given timesteps t.  This is crucial for efficiently
    # calculating the noise and the diffused data at any point in the diffusion process.
    alpha_t_bar_t = alpha_t_bar[t].reshape(-1, 1)  # Shape: [batch_size, 1]

    # Sample noise.  This is the core of the diffusion process.  We add this noise
    # to the original data, scaling it by the cumulative product of the noise schedule.
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_t_bar_t) * x_0 + torch.sqrt(1 - alpha_t_bar_t) * noise
    return x_t, noise

# Model: Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, M, timesteps):
        """
        A simple MLP model to predict noise given noisy data and timestep.

        Args:
            M (int): Number of features in the data.
            timesteps (int): Number of diffusion steps.
        """
        super(SimpleMLP, self).__init__()
        self.time_embed = nn.Embedding(timesteps, timesteps) # Use Embedding instead of Sinusoidal
        self.model = nn.Sequential(
            nn.Linear(M + timesteps, M * 4),  # Input: noisy data + time embedding
            nn.GELU(),
            nn.Linear(M * 4, M * 2),
            nn.GELU(),
            nn.Linear(M * 2, M),  # Output: predicted noise
        )

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

def get_config_directory(config):
    """
    Generates a unique directory name for the configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        str: The directory name.
    """
    return f"config_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_T{config.timesteps}_NS{config.noise_schedule}_BS{config.batch_size}_LR{config.learning_rate}_E{config.epochs}"

def save_model(model, config):
    """
    Saves the trained model to a file in a unique directory.

    Args:
        model (nn.Module): The trained model.
        config (Config): Configuration object.
    """
    directory = get_config_directory(config)
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    filename = os.path.join(directory, "model.pth")
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model_class, config):
    """
    Loads a trained model from a file in the configuration's directory.

    Args:
        model_class (type): The class of the model to be loaded.
        config (Config): Configuration object.

    Returns:
        nn.Module: The loaded model.
    """
    directory = get_config_directory(config)
    filename = os.path.join(directory, "model.pth")
    model = model_class(config.M, config.nhidden, config.nlayers, config.timesteps).to(config.device)
    model.load_state_dict(torch.load(filename, map_location=config.device))
    print(f"Model loaded from {filename}")
    return model

def save_plot(fig, config, filename):
    """
    Saves a plot to a file in the configuration's directory.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        config (Config): Configuration object.
        filename (str): The name of the file to save the plot as.
    """
    directory = get_config_directory(config)
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")

def get_model_filename(config):
    """
    Generates a filename for the model based on the configuration parameters.

    Args:
        config (Config): Configuration object.

    Returns:
        str: The generated filename.
    """
    return f"model_M{config.M}_nh{config.nhidden}_nl{config.nlayers}_T{config.timesteps}_NS{config.noise_schedule}_BS{config.batch_size}_LR{config.learning_rate}_E{config.epochs}.pth"

# Sampling
def sample(model, timesteps, M, betas, device, batch_size=1):
    """
    Generates new data samples using the trained diffusion model.

    Args:
        model (nn.Module): The trained diffusion model.
        timesteps (int): Number of diffusion steps.
        M (int): Number of features in the data.
        betas (torch.Tensor): The noise schedule (shape: [timesteps]).
        device (torch.device): The device to run the sampling on.
        batch_size (int): Number of samples to generate in a batch.

    Returns:
        torch.Tensor: Generated data samples (shape: [batch_size, M]).
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x_t = torch.randn(batch_size, M).to(device)  # Start with random noise
        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t] * batch_size, dtype=torch.long).to(device)
            predicted_noise = model(x_t, t_tensor)
            alpha_t = 1 - betas[t]
            alpha_t_bar = torch.cumprod(1 - betas, dim=0).to(device)[t]
            
            # Numerical stability improvements:
            alpha_t_bar = torch.max(alpha_t_bar, torch.tensor(1e-8).to(device))  # Prevent division by zero
            alpha_t = torch.max(alpha_t, torch.tensor(1e-8).to(device))      # Prevent division by zero
            
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar) * predicted_noise)
            if t > 0:
                z = torch.randn(batch_size, M).to(device)
                x_t = x_t + torch.sqrt(betas[t]) * z
    model.train() # return to train mode
    return x_t

def train_diffusion_model(data, config):
    """
    Trains a diffusion model on the given data.

    Args:
        data (torch.Tensor): The training data (shape: [num_samples, M]).
        config (Config): Configuration object.

    Returns:
        nn.Module: The trained diffusion model.
        list: List of training losses for each epoch.
    """
    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    # Get noise schedule
    betas = get_noise_schedule(config.noise_schedule, config.timesteps).to(config.device)

    # Initialize model and optimizer
    #model = SimpleMLP(config.M, config.timesteps).to(config.device)
    model = MingruMLP(config.M, config.nhidden, config.nlayers, config.timesteps).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())    

    # Training loop
    epoch_losses = []  # List to store losses for each epoch
    for epoch in tqdm(range(config.epochs), desc="Training Progress"):  # Add tqdm for progress bar
        epoch_loss = 0  # Variable to accumulate loss for the epoch
        for i, (batch,) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            # Move batch to device
            batch = batch[0].to(config.device)  # Shape: [batch_size, M]

            # Sample timesteps
            t = torch.randint(0, config.timesteps, (config.batch_size,), device=config.device)  # Shape: [batch_size]

            # Forward diffusion
            x_t, noise = forward_diffusion(batch, t, betas)  # x_t shape: [batch_size, M], noise shape: [batch_size, M]

            # Predict noise
            predicted_noise = model(x_t, t)  # Shape: [batch_size, M]

            # Compute loss
            loss = criterion(predicted_noise, noise)

            # Backpropagate and update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss

        epoch_losses.append(epoch_loss / len(dataloader))  # Store average loss for the epoch

        # Generate and print a sample after each epoch
        #generated_sample = sample(model, config.timesteps, config.M, betas, config.device)
        #print(f"Epoch [{epoch+1}/{config.epochs}], Generated Sample: {generated_sample.cpu().numpy()}")

    return model, epoch_losses

def get_config_description(config):
    """
    Generates a string description of the configuration parameters.

    Args:
        config (Config): Configuration object.

    Returns:
        str: A string describing the configuration.
    """
    return (f"M={config.M}, nh={config.nhidden}, nl={config.nlayers}, "
            f"T={config.timesteps}, NS={config.noise_schedule}, "
            f"BS={config.batch_size}, LR={config.learning_rate}, E={config.epochs}")

if __name__ == "__main__":
    # Example usage:
    # 1. Generate some sample data.  Replace this with your actual data.
    M = 3  # Number of features
    num_samples = 16000
    #data = 0.5*torch.randn(num_samples, M) + torch.tensor([1.0, 0, 0, 0, 0]) # Example data: 1000 samples, 5 features each
    data, original_radii = generate_3d_sphere_data(num_samples)

    # 2. Configure the training.  Adjust these parameters as needed.
    config = Config(
        M=M,
        nhidden=1024,
        nlayers=2,
        timesteps=100,
        noise_schedule='linear',  # 'linear', 'quadratic', or 'cosine'
        batch_size=64,
        learning_rate=1e-3,
        epochs=100000,
    )
    
    train = True

    if train:
        # 3. Train the diffusion model.
        trained_model, epoch_losses = train_diffusion_model(data, config)

        # Save the trained model
        save_model(trained_model, config)

    # Load the trained model for further training or inference
    loaded_model = load_model(MingruMLP, config)
    # ... rest of the sampling code ...

    # 4. Generate new samples.
    num_generated_samples = 4000
    batch_size = 100  # Adjust batch size as needed
    generated_data = []
    for _ in range(num_generated_samples // batch_size):
        generated_batch = sample(loaded_model, config.timesteps, M, get_noise_schedule(config.noise_schedule, config.timesteps).to(config.device), config.device, batch_size)
        generated_data.append(generated_batch)
    generated_data = torch.cat(generated_data, dim=0) # Shape: [num_generated_samples, M]
    #print("Generated Data:", generated_data.cpu().numpy())
    generated_radii = torch.norm(generated_data, dim=1).cpu().detach().numpy()
    original_radii = original_radii.cpu().numpy()

    # Plot histograms of the first feature of data and generated_data
    plt.figure(figsize=(10, 5))
    plt.hist(original_radii, bins=50, range=(0.0, 2.0), density=True, alpha=0.5, label='Original Data')
    plt.hist(generated_radii, bins=50, range=(0.0, 2.0), density=True, alpha=0.5, label='Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Histogram of Feature 1\n{get_config_description(config)}')  # Add config description to the title
    histogram_fig = plt.gcf()  # Get the current figure
    save_plot(histogram_fig, config, "histogram.png")
    plt.show()

    # Plot training loss for each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Loss per Epoch\n{get_config_description(config)}')  # Add config description to the title
    loss_fig = plt.gcf()  # Get the current figure
    save_plot(loss_fig, config, "training_loss.png")
    plt.show()
