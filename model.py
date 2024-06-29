import torch
import torch.nn as nn
from .block import DiffusionBlock
from .geometry import compute_hks_autoscale, compute_wks_autoscale


class DiffusionNet(nn.Module):
    """
    DiffusionNet: stacked of DiffusionBlock
    """
    def __init__(self, in_channels, out_channels,
                 hidden_channels=128,
                 n_block=4,
                 mlp_hidden_channels=None,
                 dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True
                 ):
        super(DiffusionNet, self).__init__()
        # sanity check
        self.out_channels = out_channels

        # mlp options
        if not mlp_hidden_channels:
            mlp_hidden_channels = [hidden_channels, hidden_channels]

        # setup networks

        # first and last linear layers
        self.first_linear = nn.Linear(in_channels, hidden_channels)
        self.last_linear = nn.Linear(hidden_channels, out_channels)

        # diffusion blocks
        blocks = []
        for _ in range(n_block):
            block = DiffusionBlock(
                in_channels=hidden_channels,
                mlp_hidden_channels=mlp_hidden_channels,
                dropout=dropout,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, coords, mass, evals, evecs, gradX, gradY):
        #x = compute_hks_autoscale(evals, evecs)
        #x = compute_wks_autoscale(evals, evecs, mass, n_descr=128, n_eig=200)

        # Apply the first linear layer
        x = self.first_linear(coords)

        # Apply each of the diffusion block
        for block in self.blocks:
            x = block(x, mass, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x_out = self.last_linear(x)

        return x_out