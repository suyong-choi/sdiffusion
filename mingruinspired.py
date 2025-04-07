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
    def __init__(self, nlayers:int, input_size:int, hidden_size:int, output_size:int, dropout=0.05 ) -> None:
        super(Mingrustack, self).__init__()

        self.stacks = nn.ModuleList([Mingrubare(input_size, hidden_size)])
        for _ in range(nlayers-1):
            #self.stacks.append(nn.LayerNorm(hidden_size))
            self.stacks.append(Mingrubare(hidden_size, hidden_size))
            #self.stacks.append(nn.GELU())
            #self.stacks.append(nn.Dropout(dropout))

        #self.stacks.append(nn.LayerNorm(hidden_size))
        #self.stacks.append(nn.Dropout(dropout))
        self.stacks.append(nn.Linear(hidden_size, output_size))
        pass

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = x
        for alayer in self.stacks:
            out = alayer(out)
        return out
