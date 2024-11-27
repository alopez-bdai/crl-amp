import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_layer_sizes,
                 device='cpu',):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.device = device

        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.linear.train()

    def forward(self, x):
        d = self.linear(self.trunk(x))
        return d

    def compute_grad_pen(self,
                         state,
                         lambda_=10):
        state.requires_grad = True

        disc = self.linear(self.trunk(state))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=state,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen
