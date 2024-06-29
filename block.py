import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .geometry import to_basis, from_basis


class LearnedTimeDiffusion(nn.Module):
    """
    Applied diffusion with learned time per-channel.
    In the spectral domain this becomes
        f_out = e ^ (lambda_i * t) * f_in
    """
    def __init__(self, in_channels):
        """
        Parameters:
            in_channels (int): number of input channels.
            method (str): method to perform time diffusion. Default 'spectral'.
        """
        super(LearnedTimeDiffusion, self).__init__()
        self.in_channels = in_channels
        self.diffusion_times = nn.Parameter(torch.Tensor(in_channels))
        
        self.high_linear = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, in_channels),
            nn.ReLU()
        )

        # init as zero
        nn.init.constant_(self.diffusion_times, 0.0)

    def forward(self, feat, mass, evals, evecs):
        """
        Input:
            feat (B,Nv,C)
            L (B,Nv,Nv)
            mass (B,Nv)
            evals (B,K)
            evecs (B,Nv,K)
        Output:
            feat_diffuse (B,Nv,C)
        """
        with torch.no_grad():
            self.diffusion_times.data = torch.clamp(self.diffusion_times, min=1e-8)

        assert feat.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat.shape[-1]}'

        feat_high = feat - from_basis(to_basis(feat, evecs, mass), evecs)  # (B,Nv,C)
        feat_high_diffuse = self.high_linear(feat_high)

        # Transform to spectral
        feat_spec = to_basis(feat, evecs, mass)  # (B,K,C)

        # Diffuse
        diffuse_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_times.unsqueeze(0).unsqueeze(0))  # (B,K,C)
        feat_diffuse_spec = diffuse_coefs * feat_spec  # (B,K,C)

        # Transform back to feature
        feat_low_diffuse = from_basis(feat_diffuse_spec, evecs)  # (B,Nv,C)

        feat_diffuse = feat_low_diffuse + feat_high_diffuse

        return feat_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors.
    Uses a learned complex-linear layer to keep dimension down.
    """
    def __init__(self, in_channels, with_gradient_rotations=True):
        """
        Parameters:
            in_channels (int): number of input channels.
            with_gradient_rotations (bool): whether with gradient rotations. Default True.
        """
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """
        Input:
            feat_in (B,Nv,C,2)
        Output:
            feat_out (B,Nv,C)
        """
        feat_a = feat_in

        if self.with_gradient_rotations:
            feat_real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            feat_img_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            feat_real_b = self.A(feat_in[..., 0])
            feat_img_b = self.A(feat_in[..., 1])

        feat_out = feat_a[..., 0] * feat_real_b + feat_a[..., 1] * feat_img_b

        return torch.tanh(feat_out)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes
    """
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name='miniMLP'):
        """
        Parameters:
            layer_sizes (List): list of layer size.
            dropout (bool): whether use dropout. Default False.
            activation (nn.Module): activation function. Default ReLU.
            name (str): module name. Default 'miniMLP'
        """
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            # Dropout Layer
            if dropout and i > 0:
                self.add_module(
                    name + '_dropout_{:03d}'.format(i),
                    nn.Dropout(p=0.2)
                )

            # Affine Layer
            self.add_module(
                name + '_linear_{:03d}'.format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )

            # Activation Layer
            if not is_last:
                self.add_module(
                    name + '_norm_{:03d}'.format(i),
                    nn.Sequential(
                        Rearrange('b n c -> b c n'),
                        nn.BatchNorm1d(layer_sizes[i+1], track_running_stats=False),
                        Rearrange('b c n -> b n c')
                    )
                )

                self.add_module(
                    name + '_activation_{:03d}'.format(i),
                    activation()
                )


class DiffusionBlock(nn.Module):
    """
    Building Block of DiffusionNet.
    """
    def __init__(self, in_channels, mlp_hidden_channels,
                 dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        """
        Parameters:
            in_channels (int): number of input channels.
            mlp_hidden_channels (List): list of mlp hidden channels.
            dropout (bool, optional): whether use dropout in MLP. Default True.
            with_gradient_features (bool): whether use spatial gradient feature. Default True.
            with_gradient_rotations (bool): whether use spatial gradient rotation. Default True.
        """
        super(DiffusionBlock, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_features = with_gradient_features

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(in_channels)

        # concat of both diffused features and original features
        mlp_in_channels = 2*in_channels

        # Spatial gradient block
        if with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(in_channels, with_gradient_rotations=with_gradient_rotations)

        # MLP block
        self.mlp = MiniMLP([mlp_in_channels] + mlp_hidden_channels + [self.in_channels], dropout=dropout)

    def forward(self, feat_in, mass, evals, evecs, gradX, gradY):
        """
        Input:
            feat_in (B,Nv,C)
            mass (B,Nv)
            L (B,Nv,Nv)
            evals (B,K)
            evecs (B,Nv,K)
            gradX (B,Nv,Nv)
            gradY (B,Nv,Nv)
        Output:
            feat_out (B,Nv,C)
        """
        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat_in.shape[-1]}'

        # Diffusion block
        feat_diffuse = self.diffusion(feat_in, mass, evals, evecs)

        # Compute gradient features
        if self.with_gradient_features:
            # Compute gradient
            feat_grads = []
            for b in range(B):
                # gradient after diffusion
                feat_gradX = torch.mm(gradX[b, ...], feat_diffuse[b, ...])
                feat_gradY = torch.mm(gradY[b, ...], feat_diffuse[b, ...])

                feat_grads.append(torch.stack((feat_gradX, feat_gradY), dim=-1))
                
            feat_grad = torch.stack(feat_grads, dim=0) # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_grad_features), dim=-1)
        else:
            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out