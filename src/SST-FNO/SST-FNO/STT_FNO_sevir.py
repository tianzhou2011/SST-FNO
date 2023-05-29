# coding=utf-8

import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# from scipy import signal
# from scipy import linalg as la
# from scipy import special as ss
# from utils import unroll
# from utils.op import transition
# import pickle
# import pdb
from einops import rearrange, repeat
import opt_einsum as oe
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import pdb

contract = oe.contract

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from .utils import apply_initialization,get_activation,get_norm_layer

class Upsample3DLayer(nn.Module):
    """Upsampling based on nn.UpSampling and Conv3x3.

    If the temporal dimension remains the same:
        x --> interpolation-2d (nearest) --> conv3x3(dim, out_dim)
    Else:
        x --> interpolation-3d (nearest) --> conv3x3x3(dim, out_dim)

    """
    def __init__(self,
                 dim,
                 out_dim,
                 target_size,
                 temporal_upsample=False,
                 kernel_size=3,
                 layout='THWC',
                 conv_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
        out_dim
        target_size
            Size of the output tensor. Will be a tuple/list that contains T_new, H_new, W_new
        temporal_upsample
            Whether the temporal axis will go through upsampling.
        kernel_size
            The kernel size of the Conv2D layer
        layout
            The layout of the inputs
        """
        super(Upsample3DLayer,self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode='nearest')  # 3D upsampling
        else:
            self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode='nearest')  # 2D upsampling
        self.conv = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=(kernel_size, kernel_size),
                              padding=(kernel_size // 2, kernel_size // 2))
        assert layout in ['THWC', 'CTHW']
        self.layout = layout

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C) or (B, C, T, H, W)

        Returns
        -------
        out
            Shape (B, T, H_new, W_out, C_out) or (B, C, T, H_out, W_out)
        """
        if self.layout == 'THWC':
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                return self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B * T, C, H, W)
                x = self.up(x)
                return self.conv(x).permute(0, 2, 3, 1).reshape((B,) + self.target_size + (self.out_dim,))
        elif self.layout == 'CTHW':
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.output_size[0] == T
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                x = x.reshape(B * T, C, H, W)
                return self.conv(self.up(x)).reshape(B, self.target_size[0], self.out_dim, self.target_size[1],
                                                     self.target_size[2]).permute(0, 2, 1, 3, 4)

class PatchMerging3D(nn.Module):
    """ Patch Merging Layer"""
    def __init__(self,
                 dim,
                 out_dim=None,
                 downsample=(1, 2, 2),
                 norm_layer='layer_norm',
                 padding_type='nearest',
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            Number of input channels.
        downsample
            downsample factor
        norm_layer
            The normalization layer
        """
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Linear(downsample[0] * downsample[1] * downsample[2] * dim,
                                   out_dim, bias=False)
        self.norm = get_norm_layer(norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (T + pad_t) // self.downsample[0], (H + pad_h) // self.downsample[1], (W + pad_w) // self.downsample[2],\
               self.out_dim

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Input feature, tensor size (B, T, H, W, C).

        Returns
        -------
        out
            Shape (B, T // downsample[0], H // downsample[1], W // downsample[2], out_dim)
        """
        B, T, H, W, C = x.shape

        # padding
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_h or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = _generalize_padding(x, pad_t, pad_w, pad_h, padding_type=self.padding_type)

        x = x.reshape((B,
                       T // self.downsample[0], self.downsample[0],
                       H // self.downsample[1], self.downsample[1],
                       W // self.downsample[2], self.downsample[2], C)) \
             .permute(0, 1, 3, 5, 2, 4, 6, 7) \
             .reshape(B, T // self.downsample[0], H // self.downsample[1], W // self.downsample[2],
                      self.downsample[0] * self.downsample[1] * self.downsample[2] * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x            
            
            
class DownSampling3D(nn.Module):
    """The 3D down-sampling layer.

    3d_interp_2d:
        x --> conv3d_3X3X3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3
    2d_interp_2d:
        x --> conv2d_3x3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3

    We add an additional conv layer before the

    For any options, if the target_size is the same as the input size, we will skip the bilinear downsampling layer.
    """
    def __init__(self, original_size, target_size, in_channels, out_dim, mid_dim=16, act_type='leaky',
                 arch_type='2d_interp_2d'):
        """

        Parameters
        ----------
        original_size
            The original size of the tensor. It will be a tuple/list that contains T, H, W
        target_size
            Will be a tuple/list that contains T_new, H_new, W_new
        in_channels
            The input channels
        out_dim
            The output dimension of the layer
        mid_dim
            Dimension of the intermediate projection layer
        act_type
            Type of the activation
        arch_type
            Type of the layer.
        """
        super(DownSampling3D, self).__init__()
        self.arch_type = arch_type
        self.original_size = original_size
        self.target_size = target_size
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        if self.arch_type == '3d_interp_2d':
            self.inter_conv = nn.Conv3d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1))
            self.act = get_activation(act_type)
        elif self.arch_type == '2d_interp_2d':
            self.inter_conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3),
                                        padding=(1, 1))
            self.act = get_activation(act_type)
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=(3, 3), padding=(1, 1))
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            apply_initialization(m)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (N, T, H, W, C)

        Returns
        -------
        out
            Shape (N, T_new, H_new, W_new, C_out)
        """
        B, T, H, W, C_in = x.shape
        if self.arch_type == '3d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 4, 1, 2, 3)))  # Shape(B, mid_dim, T, H, W)
            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(B * T, self.mid_dim, H, W), size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x, size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        elif self.arch_type == '2d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 1, 4, 2, 3).reshape(B * T, C_in, H, W)))  # (B * T, mid_dim, H, W)

            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x, size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x.reshape(B, T, C_in, H, W).permute(0, 2, 1, 3, 4), size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        else:
            raise NotImplementedError
        x = self.conv(x)  # Shape (B * T_new, out_dim, H_new, W_new)
        x = x.reshape(B, self.target_size[0], self.out_dim, self.target_size[1], self.target_size[2]) \
            .permute(0, 1, 3, 4,2)
        return x

class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)

class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                                         1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# class HiPPO_LegT(nn.Module):
#     def __init__(self, N, dt=1.0, discretization='bilinear'):
#         """
#         N: the order of the HiPPO projection
#         dt: discretization step size - should be roughly inverse to the length of the sequence
#         """
#         super(HiPPO_LegT,self).__init__()
#         self.N = N
#         A, B = transition('lmu', N)
#         C = np.ones((1, N))
#         D = np.zeros((1,))
#         # dt, discretization options
#         A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

#         B = B.squeeze(-1)

#         self.register_buffer('A', torch.Tensor(A).to(device))
#         self.register_buffer('B', torch.Tensor(B).to(device))
#         vals = np.arange(0.0, 1.0, dt)
#         self.register_buffer('eval_matrix',  torch.Tensor(
#             ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))
#     def forward(self, inputs):  # torch.Size([128, 1, 1]) -
#         """
#         # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
#         for:
#             batch, height*width,channel -> height*weight,batch,channel

#         inputs : (length, ...)
#         output : (length, ..., N) where N is the order of the HiPPO projection
#         """

#         c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)  # torch.Size([1, 256])
#         cs = []
#         for f in inputs.permute([-1, 0, 1]):
#             f = f.unsqueeze(-1)
#             # f: [1,1]
#             new = f @ self.B.unsqueeze(0) # [B, D, H, 256]
#             c = F.linear(c, self.A) + new
#             # c = [1,256] * [256,256] + [1, 256]
#             cs.append(c)
#         return torch.stack(cs, dim=0)

#     def reconstruct(self, c):
#         a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
#         return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


################################################################
class Spectral2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        self.w = nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.unet =  U_net(self.in_dim, self.in_dim, 3, 0)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        x_p = self.forward_fourier(x)

        b = self.backcast_ff(x_p)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        B, I, S1, S2, S3 = x.shape
        
        x_w = self.w(x.contiguous().view(B,I,-1)).view(B,I,S1,S2,S3)
        #x_u = self.unet(x)

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        #pdb.set_trace()
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]
        out_ft[:, :, :, :self.modes_y, :] = torch.einsum("bixyz,ioy->boxyz",x_fty[:, :, :, :self.modes_y, :],torch.view_as_complex(self.fourier_weight[1]))
        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + x_w

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x   


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,seq_len,modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels*out_channels))
        self.modes2 =min(modes1,seq_len//2)
        self.index = list(range(0, self.modes2))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))


    def forward(self, x):
        B, H,E, N = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B,H, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :,:, :self.modes1]
        out_ft[:, :,:, :self.modes1] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        self.fourier_weight_raw = nn.ParameterList([])
        for n_modes in [modes_x, modes_y, modes_z]:
            weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight_raw.append(param)
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        self.w = nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.unet =  U_net(self.in_dim, self.in_dim, 3, 0)

    def forward(self, x,x_raw):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        x_p = self.forward_fourier(x,x_raw)

        b = self.backcast_ff(x_p)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x,x_raw):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        x_raw = rearrange(x_raw, 'b s1 s2 s3 i -> b i s1 s2 s3')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        B, I, S1, S2, S3 = x.shape
        
        x_w = self.w(x.contiguous().view(B,I,-1)).view(B,I,S1,S2,S3)
        #x_u = self.unet(x)

        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        #x_raw_ftz = torch.fft.rfft(x_raw, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        #out_raw_ft = x_raw_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        # out_raw_ft[:, :, :, :, :self.modes_z] = torch.einsum(
        #     "bixyz,ioz->boxyz",
        #     x_raw_ftz[:, :, :, :, :self.modes_z],
        #     torch.view_as_complex(self.fourier_weight_raw[2]))

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
        #xz_raw = torch.fft.irfft(out_raw_ft, n=S3, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        x_raw_fty = torch.fft.rfft(x_raw, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        out_raw_ft = x_raw_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))

        out_raw_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_raw_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight_raw[1]))

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
        xy_raw = torch.fft.irfft(out_raw_ft, n=S2, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        x_raw_ftx = torch.fft.rfft(x_raw, dim=-3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        out_raw_ft = x_raw_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        out_raw_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_raw_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight_raw[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
        xx_raw = torch.fft.irfft(out_raw_ft, n=S1, dim=-3, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz+xx_raw+xy_raw+x_w

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        #pdb.set_trace()
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)
    
class SpatialTemporal2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        self.fourier_weight2 = nn.ParameterList([])
        self.w = nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.unet =  U_net(self.in_dim, self.in_dim, 3, 0)
        # x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        # x3 = self.unet3(x) 
        
        
        for n_modes in [(modes_x,modes_y), (modes_y,modes_z), (modes_x,modes_z)]:
            weight = torch.FloatTensor(in_dim, out_dim, n_modes[0],n_modes[1],2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight2.append(param)
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [(modes_x,modes_y), (modes_y,modes_z), (modes_x,modes_z)]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes[0],n_modes[1],2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        
        
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        B, I, S1, S2, S3 = x.shape
        
        x_w = self.w(x.contiguous().view(B,I,-1)).view(B,I,S1,S2,S3)
        #x_u = self.unet(x)
        
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        

        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfftn(x, dim=(-3,-1))
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1 , S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        #pdb.set_trace()
        out_ft[:, :, :self.modes_x, :, :self.modes_z] = torch.einsum(
            "bixyz,ioxz->boxyz",
            x_ftz[:, :, :self.modes_x, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))
        out_ft[:, :, :self.modes_x, :, -self.modes_z:] = torch.einsum(
            "bixyz,ioxz->boxyz",
            x_ftz[:, :, :self.modes_x, :, -self.modes_z:],
            torch.view_as_complex(self.fourier_weight2[2]))

        xz = torch.fft.irfftn(out_ft, dim=(-3,-1))
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfftn(x, dim=(-2,-1))
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes_y, :self.modes_z] = torch.einsum(
            "bixyz,ioyz->boxyz",
            x_fty[:, :, :, :self.modes_y, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[1]))

        out_ft[:, :, :, :self.modes_y, -self.modes_z:] = torch.einsum(
            "bixyz,ioyz->boxyz",
            x_fty[:, :, :, :self.modes_y, -self.modes_z:],
            torch.view_as_complex(self.fourier_weight2[1]))

        yz = torch.fft.irfftn(out_ft,dim=(-2,-1))
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfftn(x, dim=(-3,-2))
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 , S2// 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :self.modes_y, :] = torch.einsum(
            "bixyz,ioxy->boxyz",
            x_ftx[:, :, :self.modes_x, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[0]))

        out_ft[:, :, :self.modes_x:, -self.modes_y:, :] = torch.einsum(
            "bixyz,ioxy->boxyz",
            x_ftx[:, :, :self.modes_x, -self.modes_y:, :],
            torch.view_as_complex(self.fourier_weight2[0]))

        xy = torch.fft.irfftn(out_ft, dim=(-3,-2))
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xy + yz + xz+x_w

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class FTSNO_w_gn_cord_sevir_Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, n_layers, d_model,total_length,input_length,enc_in,enc_out,modes_x,modes_y,modes_z,factor,n_ff_layers,x_dim,y_dim,z_dim,pre,pre_dim,pad_x,pad_y,pad_z):
        super(FTSNO_w_gn_cord_sevir_Model, self).__init__()
        # self.modes = configs.modes
        # modes_x=20,
        # modes_y=20,
        # modes_z=5,
        # factor=4
        # n_ff_layers=2
        
        self.seq_len = input_length
        self.pred_len = total_length-input_length
        self.pre=pre

        self.enc_in = enc_in
        self.enc_out = enc_out
        self.n_layers = n_layers
        self.in_dim = d_model
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.pad_z = pad_z 
        
        
        self.preblock = nn.ModuleList([])
        self.midblock_down = nn.ModuleList([])
        self.endblock_down = nn.ModuleList([])
        self.endblock_up = nn.ModuleList([])
        self.midblock_up =nn.ModuleList([])
        self.downnorm1 = nn.InstanceNorm3d(2*d_model)
        self.downnorm2 = nn.InstanceNorm3d(4*d_model)
        self.upnorm1 = nn.InstanceNorm3d(4*d_model)
        self.upnorm2 = nn.InstanceNorm3d(3*d_model)
        
        self.preblock_norm = nn.ModuleList([])
        self.midblock_down_norm = nn.ModuleList([])
        self.endblock_down_norm = nn.ModuleList([])
        self.endblock_up_norm = nn.ModuleList([])
        self.midblock_up_norm = nn.ModuleList([])
        
        self.preblock_norm_raw = nn.ModuleList([])
        self.midblock_down_norm_raw = nn.ModuleList([])
        self.endblock_down_norm_raw = nn.ModuleList([])
        self.endblock_up_norm_raw = nn.ModuleList([])
        self.midblock_up_norm_raw = nn.ModuleList([])
        
        self.prefactor_norm = nn.ModuleList([])
        self.prefactor = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.prefactor.append( SpatialTemporal2d(in_dim=self.in_dim,
                                                out_dim=self.in_dim,
                                                modes_x=int(self.x_dim/24),
                                                modes_y=int(self.y_dim/24),
                                                modes_z=int(self.z_dim/2),
                                                forecast_ff=None,
                                                backcast_ff=None,
                                                fourier_weight=None,
                                                factor=4,
                                                ff_weight_norm=True,
                                                n_ff_layers=2,
                                                layer_norm=True,
                                                use_fork=False,
                                                dropout=0.0))
            self.prefactor_norm.append(nn.GroupNorm(16,self.in_dim))
        
        for _ in range(self.n_layers):
            self.preblock.append(SpectralConv2d(in_dim=self.in_dim,
                                              out_dim=self.in_dim,
                                              modes_x = int(self.modes_x/12),
                                              modes_y = int(self.modes_y/12),
                                              modes_z = self.modes_z,
                                              forecast_ff=None,
                                              backcast_ff=None,
                                              fourier_weight=None,
                                              factor=factor,
                                              ff_weight_norm=True,
                                              n_ff_layers=n_ff_layers,
                                              layer_norm=True,
                                              use_fork=False,
                                              dropout=0.0))
            self.preblock_norm.append(nn.GroupNorm(16,d_model))
            #self.preblock_norm_raw.append(nn.GroupNorm(16,d_model))
        
        for _ in range(self.n_layers):
            self.midblock_down.append(SpectralConv2d(in_dim=2*self.in_dim,
                                                     out_dim=2*self.in_dim,
                                                     modes_x = int(self.modes_x/24),
                                                    modes_y = int(self.modes_y/24),
                                                    modes_z = self.modes_z,
                                                     forecast_ff=None,
                                                     backcast_ff=None,
                                                     fourier_weight=None,
                                                     factor=4,
                                                     ff_weight_norm=True,
                                                     n_ff_layers=2,
                                                     layer_norm=True,
                                                     use_fork=False,
                                                     dropout=0.0))
            self.midblock_down_norm.append(nn.GroupNorm(16,2*d_model))
            #self.midblock_down_norm_raw.append(nn.GroupNorm(16,2*d_model))
        for _ in range(self.n_layers):
            self.endblock_down.append(
                SpectralConv2d(in_dim=4*self.in_dim,
                               out_dim=4*self.in_dim,
                               modes_x=int(self.modes_x/48),
                               modes_y=int(self.modes_y/48),
                               modes_z=self.modes_z,
                               forecast_ff=None,
                               backcast_ff=None,
                               fourier_weight=None,
                               factor=4,
                               ff_weight_norm=True,
                               n_ff_layers=2,
                               layer_norm=True,
                               use_fork=False,
                               dropout=0.0)
            )
            self.endblock_down_norm.append(nn.GroupNorm(16,4*d_model))
            #self.endblock_down_norm_raw.append(nn.GroupNorm(16,4*d_model))

        for _ in range(self.n_layers):
            self.endblock_up.append(SpectralConv2d(in_dim=2*self.in_dim,
                                                     out_dim=2*self.in_dim,
                                                     modes_x=int(self.modes_x/24),
                                                     modes_y=int(self.modes_y/24),
                                                     modes_z=self.modes_z,
                                                     forecast_ff=None,
                                                     backcast_ff=None,
                                                     fourier_weight=None,
                                                     factor=4,
                                                     ff_weight_norm=True,
                                                     n_ff_layers=2,
                                                     layer_norm=True,
                                                     use_fork=False,
                                                     dropout=0.0))
            self.endblock_up_norm.append(nn.GroupNorm(16,2*d_model))
            #self.endblock_up_norm_raw.append(nn.GroupNorm(16,2*d_model))

        for _ in range(self.n_layers):
            self.midblock_up.append(SpectralConv2d(in_dim=self.in_dim,
                                                   out_dim=self.in_dim,
                                                   modes_x=int(self.modes_x/12),
                                                   modes_y=int(self.modes_y/12),
                                                   modes_z=self.modes_z,
                                                   forecast_ff=None,
                                                   backcast_ff=None,
                                                   fourier_weight=None,
                                                   factor=4,
                                                   ff_weight_norm=True,
                                                   n_ff_layers=2,
                                                   layer_norm=True,
                                                   use_fork=False,
                                                   dropout=0.0))
            self.midblock_up_norm.append(nn.GroupNorm(16,d_model))
            #self.midblock_up_norm_raw.append(nn.GroupNorm(16,d_model))

#         self.downsample1 = DownSampling3D((10,64,64),(10,32,32),self.in_dim,2*self.in_dim)
#         self.downsample2 = DownSampling3D((10,32,32),(10,16,16),2*self.in_dim,4*self.in_dim)

#         self.upsample1 = Upsample3DLayer(4*self.in_dim,2*self.in_dim,(10,32,32))
#         self.upsample2 = Upsample3DLayer(2*self.in_dim,self.in_dim,(10,64,64))

        self.downsample_top1 = DownSampling3D((z_dim,x_dim,y_dim),(z_dim,int(x_dim/3),int(y_dim/3)),self.enc_in+3,16*self.enc_in)
        self.downsample_top2 = DownSampling3D((z_dim,int(x_dim/3),int(y_dim/3)),(z_dim,int(x_dim/6),int(y_dim/6)),16*self.enc_in,64*self.enc_in)
        self.downsample_top3 = DownSampling3D((z_dim,int(x_dim/6),int(y_dim/6)),(z_dim,int(x_dim/12),int(y_dim/12)),64*self.enc_in,128*self.enc_in)
        
        self.downsample_top1_norm = nn.GroupNorm(16,16*self.enc_in)
        self.downsample_top2_norm = nn.GroupNorm(16,64*self.enc_in)
        #self.downsample_top3_norm = nn.GroupNorm(16,128*self.enc_in)
        
        
        self.upsample_end1 = Upsample3DLayer(self.in_dim,self.in_dim,(self.z_dim,int(self.x_dim/6),int(self.y_dim/6)))
        self.upsample_end2 = Upsample3DLayer(self.in_dim,int(self.in_dim/2),(self.z_dim,int(self.x_dim/3),int(self.y_dim/3)))
        self.upsample_end3 = Upsample3DLayer(int(self.in_dim/2),self.enc_out,(self.z_dim,int(self.x_dim/1),int(self.y_dim/1)))
        
        self.upsample_end1_norm = nn.GroupNorm(16,self.in_dim)
        self.upsample_end2_norm =nn.GroupNorm(16,self.in_dim)
        self.upsample_end3_norm =nn.GroupNorm(16,int(self.in_dim/2))
        
        
        
        self.downsample1 = DownSampling3D((z_dim,int(x_dim/12),int(y_dim/12)),(z_dim,int(x_dim/24),int(y_dim/24)),self.in_dim,2*self.in_dim)
        self.downsample2 = DownSampling3D((z_dim,int(x_dim/12),int(y_dim/12)),(z_dim,int(x_dim/24),int(y_dim/24)),2*self.in_dim,4*self.in_dim)
        #self.downsample1 = PatchMerging3D(dim=self.in_dim)
        #self.downsample2 = PatchMerging3D(dim=2*self.in_dim)

        self.upsample1 = Upsample3DLayer(4*self.in_dim,2*self.in_dim,(self.z_dim,int(self.x_dim/24),int(self.y_dim/24)))
        self.upsample2 = Upsample3DLayer(2*self.in_dim,self.in_dim,(self.z_dim,int(self.x_dim/12),int(self.y_dim/12)))
        
        
        self.in_proj_pre = nn.Linear(128*self.enc_in,self.in_dim)
        
        self.in_proj = nn.Linear(self.in_dim,self.in_dim)
        if not self.pre:
            self.in_proj = nn.Linear(self.in_dim,self.in_dim)
        
        self.out_proj = nn.Linear(self.in_dim,self.enc_out)
        self.out_proj_z = nn.Linear(self.seq_len+self.pad_z,self.pred_len)

    def forward(self, x_enc,mask_tensor=None):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]


        x = x_enc[:,:self.seq_len,:,:,:].contiguous()
        grid = self.get_grid(x.shape, x.device)
        x = x.permute(0,2,3,1,4)
        #B, L, H, W, C = x.shape
        #x = x.permute(0,4,1,2,3)
        #x = F.pad(x, (0,0,0,self.pad_z,0,0), "replicate")
        x = nn.ReplicationPad3d((0,0,self.pad_z,0,0,0))(x)
        x = torch.cat((x, grid.permute(0,2,3,1,4)), dim=-1)
        
        x = self.downsample_top1(x.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x = self.downsample_top1_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        
        x = self.downsample_top2(x.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x = self.downsample_top2_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        
        x = self.downsample_top3(x.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        
        if self.pre:
            x_pre = x
            x_pre = self.in_proj_pre(x)
            for i in range(self.n_layers):
                layer = self.prefactor[i]
                norm = self.prefactor_norm[i]
                ##pdb.set_trace()
                x_pre = norm(x_pre.permute(0,4,1,2,3)).permute(0,2,3,4,1)
                b,_ = layer(x_pre)
                x_pre = x_pre+b
        x_pre = self.in_proj(x_pre)
        x_raw = x_pre.clone()
        #pdb.set_trace()
        for i in range(self.n_layers):
            layer = self.preblock[i]
            norm = self.preblock_norm[i]
            #norm2 = self.preblock_norm_raw[i]
            x_pre = norm(x_pre.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            #x_raw = norm2(x_raw.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b, _ = layer(x_pre,x_raw)
            x_pre = x_pre + b
        #pdb.set_trace()
        x_pre_down = self.downsample1(x_pre.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x_pre_down = self.downnorm1(x_pre_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_pre_down_raw = x_pre_down.clone()
        for i in range(self.n_layers):
            layer = self.midblock_down[i]
            norm = self.midblock_down_norm[i]
            #norm2 = self.midblock_down_norm_raw[i]
            x_pre_down = norm(x_pre_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            #x_pre_down_raw = norm2(x_pre_down_raw.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x_pre_down,x_pre_down_raw)
            x_pre_down = x_pre_down+b
        x_mid_down = self.downsample2(x_pre_down.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x_mid_down = self.downnorm2(x_mid_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_mid_down_raw = x_mid_down.clone()

        for i in range(self.n_layers):
            layer = self.endblock_down[i]
            norm = self.endblock_down_norm[i]
            #norm2 = self.endblock_down_norm_raw[i]
            x_mid_down = norm(x_mid_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            #x_mid_down_raw = norm2(x_mid_down_raw.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x_mid_down,x_mid_down_raw)
            x_mid_down = x_mid_down+b
        x_end_up = self.upsample1(x_mid_down.permute(0,3,1,2,4)).permute(0,2,3,1,4)+x_pre_down
        x_end_up = self.upnorm1(x_end_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_end_up_raw = x_end_up.clone()
        for i in range(self.n_layers):
            layer = self.endblock_up[i]
            norm = self.endblock_up_norm[i]
            #norm2 = self.endblock_up_norm_raw[i]
            x_end_up = norm(x_end_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            #x_end_up_raw = norm2(x_end_up_raw.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x_end_up,x_end_up_raw)
            x_end_up = x_end_up+b
        x_mid_up = self.upsample2(x_end_up.permute(0,3,1,2,4)).permute(0,2,3,1,4)+x_pre
        x_mid_up = self.upnorm2(x_mid_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_mid_up_raw =x_mid_up.clone()
        for i in range(self.n_layers):
            layer = self.midblock_up[i]
            norm = self.midblock_up_norm[i]
            #norm2 = self.midblock_up_norm_raw[i]
            x_mid_up = norm(x_mid_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            #x_mid_up_raw = norm2(x_mid_up_raw.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x_mid_up,x_mid_up_raw)
            x_mid_up = x_mid_up+b
        x = x_mid_up.permute(0,3,1,2,4)
        x = self.upsample_end1_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.upsample_end1(x)
        x = self.upsample_end2_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.upsample_end2(x)
        x = self.upsample_end3_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.upsample_end3(x)
        x = x[:,:self.pred_len,...]
        #print('input shape',x.shape)
        
        #x = x[..., :,:,:-self.pad_z, :]
        #x = x.permute(0,3,1,2,4)
        #x = self.out_proj(x)
        #pdb.set_trace()
        
        #x = self.out_proj_z(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        return x
    
    def get_grid(self, shape,device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        size_x = size_x+ self.pad_z
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)



if __name__ == '__main__':
    class Configs(object):
        ab = 2
        modes1 = 5
        input_length = 10
        total_length = 22
        pred_len = 12
        W = 64
        H = 64
        output_attention = True
        img_channel = 3
        enc_in = 3
        enc_out = 3
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        n_layers = 12
        d_layers = 1
        moving_avg = 25
        c_out = 1
        activation = 'gelu'
        wavelet = 0
        ours = False
        version = 16
        ratio = 1
    # configs = Configs()
    # configs = Configs()
    # model = Model(num_layers=4, num_hidden=16,configs=configs).to(device)
    # B, length, height, width, channel = 2,20,64,64,3
    # enc = torch.randn([B,length,height,width,channel])
    # out,_ = model(enc)
    # print(out.shape)

    
