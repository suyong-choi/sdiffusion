import torch
import torch.nn as nn

class Mingrubare(nn.Module):
    def __init__(self, input_size:int, hidden_size:int) -> None:
        super(Mingrubare, self).__init__()
        self.h = nn.Linear(input_size, hidden_size)
        self.z = nn.Linear(input_size, hidden_size)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zout = torch.sigmoid(self.z(x))
        htilde = self.h(x)
        hout = torch.mul(zout, htilde)

        return hout
    
class Mingrustack(nn.Module):
    def __init__(
        self,
        nlayers:int,
        input_size:int,
        hidden_size:int,
        output_size:int,
        dropout=0.05,
        fourier_features=0,
        fourier_max_freq=10.0,
        layernorm=False
    ) -> None:
        super(Mingrustack, self).__init__()
        self.fourier_features = fourier_features
        self.fourier_max_freq = fourier_max_freq
        self.input_size = input_size
        self.layernorm = layernorm
        # If using Fourier features, input will be expanded
        if fourier_features > 0:
            self.fourier_dim = input_size * fourier_features * 2
            first_input_dim = self.fourier_dim
        else:
            self.fourier_dim = 0
            first_input_dim = input_size
        self.stacks = nn.ModuleList()
        # First layer
        self.stacks.append(Mingrubare(first_input_dim, hidden_size))
        if self.layernorm:
            self.stacks.append(nn.LayerNorm(hidden_size))
        self.stacks.append(nn.Dropout(dropout))
        # Hidden layers
        for _ in range(nlayers):
            self.stacks.append(Mingrubare(hidden_size, hidden_size))
            if self.layernorm:
                self.stacks.append(nn.LayerNorm(hidden_size))
            self.stacks.append(nn.Dropout(dropout))
        self.stacks.append(nn.Linear(hidden_size, output_size))

    def make_fourier_features(self, x):
        # x: [batch, input_size]
        device = x.device
        freqs = torch.linspace(1.0, self.fourier_max_freq, self.fourier_features, device=device)  # [n_freq]
        x_proj = x.unsqueeze(-1) * freqs  # [batch, input_size, n_freq]
        x_proj = x_proj.reshape(x.shape[0], -1)  # [batch, input_size * n_freq]
        fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)  # [batch, input_size * n_freq * 2]
        return fourier

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # If using Fourier features, apply them to input
        if self.fourier_features > 0:
            x = self.make_fourier_features(x)
        out = x
        for alayer in self.stacks:
            out = alayer(out)
        return out
