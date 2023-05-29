# coding=utf-8

import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
import opt_einsum as oe
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import pdb
contract = oe.contract
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.nn.modules.utils import _triple
from timm.models.layers import DropPath, trunc_normal_

from .utils import apply_initialization,get_activation,get_norm_layer

def exists(val):
    return val is not None

# layernorm 3d
class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 1,
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, dimensions):
        device = self.device
        pdb.set_trace()
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.g

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim,bias =False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        nn.init.zeros_(self.to_out.weight.data) # identity with skip connection

    def forward(
        self,
        q,k,v,
        rel_pos_bias = None,
    ):
        q, k, v = self.to_q(q), self.to_k(k),self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        #pdb.set_trace()

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.temporal_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 1)

    def forward(
        self,
        q,k,v,
        enable_time = True
    ):  
        '''
        q: b c t h w
        k: b c t' h w
        v: b c t' h w
                
        output: b c t h w
        '''
        #pdb.set_trace()
        b, c, _, h, w = q.shape
        
        q = rearrange(q, 'b c f h w -> (b h w) f c')
        k = rearrange(k, 'b c f h w -> (b h w) f c')
        v = rearrange(v, 'b c f h w -> (b h w) f c')

        #time_rel_pos_bias = self.temporal_rel_pos_bias([q.shape[1],k.shape[1]])

        #x = self.temporal_attn(q,k,v, rel_pos_bias = time_rel_pos_bias) 
        x = self.temporal_attn(q,k,v) 

        x = rearrange(x, '(b h w) f c -> b c f h w', w = w, h = h)

        return x
    

class PosEmbed(nn.Module):

    def __init__(self, embed_dim, maxT, maxH, maxW, typ='t+h+w'):
        r"""
        Parameters
        ----------
        embed_dim
        maxT
        maxH
        maxW
        typ
            The type of the positional embedding.
            - t+h+w:
                Embed the spatial position to embeddings
            - t+hw:
                Embed the spatial position to embeddings
        """
        super(PosEmbed, self).__init__()
        self.typ = typ

        assert self.typ in ['t+h+w', 't+hw']
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        # spatiotemporal learned positional embedding
        if self.typ == 't+h+w':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.H_embed = nn.Embedding(num_embeddings=maxH, embedding_dim=embed_dim)
            self.W_embed = nn.Embedding(num_embeddings=maxW, embedding_dim=embed_dim)

            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.H_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.W_embed.weight, std=0.02)
        elif self.typ == 't+hw':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.HW_embed = nn.Embedding(num_embeddings=maxH * maxW, embedding_dim=embed_dim)
            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.HW_embed.weight, std=0.02)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Return the x + positional embeddings
        """
        _, T, H, W, _ = x.shape
        t_idx = torch.arange(T, device=x.device)  # (T, C)
        h_idx = torch.arange(H, device=x.device)  # (H, C)
        w_idx = torch.arange(W, device=x.device)  # (W, C)
        if self.typ == 't+h+w':
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)\
                     + self.H_embed(h_idx).reshape(1, H, 1, self.embed_dim)\
                     + self.W_embed(w_idx).reshape(1, 1, W, self.embed_dim)
        elif self.typ == 't+hw':
            spatial_idx = h_idx.unsqueeze(-1) * self.maxW + w_idx
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim) + self.HW_embed(spatial_idx)
        else:
            raise NotImplementedError

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
                #nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                                         1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight,fourier_weight_raw,factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        self.fourier_weight_raw = fourier_weight_raw
        if not self.fourier_weight_raw:
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
        out_ft[:, :, -self.modes_x:, :, :self.modes_z] = torch.einsum(
            "bixyz,ioxz->boxyz",
            x_ftz[:, :, -self.modes_x:, :, :self.modes_z],
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
            x_fty[:, :, :, -self.modes_y:, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[1]))

        out_ft[:, :, :, :self.modes_y, :self.modes_z] = torch.einsum(
            "bixyz,ioyz->boxyz",
            x_fty[:, :, :, -self.modes_y:, :self.modes_z],
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

        out_ft[:, :, -self.modes_x:, :self.modes_y, :] = torch.einsum(
            "bixyz,ioxy->boxyz",
            x_ftx[:, :, -self.modes_x:, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight2[0]))

        xy = torch.fft.irfftn(out_ft, dim=(-3,-2))
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xy + yz + xz+x_w

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class SpatialTemporal2d_1d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z,  kernel_size, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout,stride=1, padding=0,bias=True):
        super(SpatialTemporal2d_1d,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        #self.w_s = nn.Conv3d(self.in_dim, self.in_dim, 1)
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight1 = nn.ParameterList([])
            self.fourier_weight2 = nn.ParameterList([])
            self.fourier_weight3 = nn.ParameterList([])
            for n_modes in [(modes_x,modes_y), (modes_x,modes_y)]:
                weight1 = torch.FloatTensor(in_dim, out_dim, n_modes[0],2)
                weight2 = torch.FloatTensor(in_dim, out_dim, n_modes[1],2)
                param1 = nn.Parameter(weight1)
                param2 = nn.Parameter(weight2)
                nn.init.xavier_normal_(param1)
                nn.init.xavier_normal_(param2)
                self.fourier_weight1.append(param1)
                self.fourier_weight2.append(param2)
                    
            weight = torch.FloatTensor(in_dim, out_dim, modes_z,2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight3.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        self.intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * self.in_dim * self.out_dim)/ \
                            (kernel_size[1]* kernel_size[2] * self.in_dim + kernel_size[0] * self.out_dim)))

        self.bn = nn.BatchNorm3d(self.intermed_channels)
        self.relu = nn.ReLU()

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(self.in_dim, self.intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.temporal_conv = nn.Conv3d(self.intermed_channels, self.out_dim, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)
    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        #pdb.set_trace()
        #x_w = self.conv_st(x)
        x = self.forward_fourier_xy(x)
        x = self.forward_fourier_z(x)
        
        #x = x+x_w
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def conv_st(self,x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        x_w = self.spatial_conv(x.contiguous())
        x_w = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x_w)
        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        return x

    def forward_fourier_xy(self, x):
        #pdb.set_trace()
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        B, I, S1, S2, S3 = x.shape
        #x_u = self.unet(x)
        # # # Dimesion X Y # # #
        x_ftx = torch.fft.rfftn(x, dim=(-3,-2))
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 , S2// 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight1[0]))
        
        out_ft[:, :, -self.modes_x:, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, -self.modes_x:, :, :],
            torch.view_as_complex(self.fourier_weight1[1]))
        
        out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_ftx[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight2[0]))

        xy = torch.fft.irfftn(out_ft, dim=(-3,-2))
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xy 

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]
        return x
    def forward_fourier_z(self,x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        B, I, S1, S2, S3 = x.shape
        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight3[0]))
        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
        x = rearrange(xz, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        return x

class FNOblock(nn.Module):
    '''
    x shape (B,H,W,T,C)
    output shape (B,H,W,T,C)
    '''
    def __init__(self,hidden_features,modes_x,modes_y,modes_z,x_dim,y_dim,z_dim,n_layers=4,factor=4,n_ff_layers=2,dropout=0,kernel_size=1):
        super(FNOblock,self).__init__()
        self.n_layers = n_layers
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList([])
        self.layers_norm = nn.ModuleList([])
        self.PosEmbed = PosEmbed(hidden_features,z_dim,x_dim,y_dim)
        for i in range(n_layers):
            self.layers.append(SpectralConv2d(
                in_dim = hidden_features,
                out_dim = hidden_features,
                modes_x = modes_x,
              modes_y = modes_y,
              modes_z = modes_z,
              forecast_ff=None,
              backcast_ff=None,
              fourier_weight=None,
              fourier_weight_raw=None,
              factor=factor,
              ff_weight_norm=True,
              n_ff_layers=n_ff_layers,
              layer_norm=True,
              use_fork=False,
              dropout=dropout,
                ))
            self.layers_norm.append(nn.GroupNorm(16,hidden_features))
    def forward(self,x):
        x_raw = x.clone()
        x = self.PosEmbed(x.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        for i in range(self.n_layers):
            layer = self.layers[i]
            norm = self.layers_norm[i]
            x = norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x,x_raw)
            x = x +b
        return x

class FNOblock_ST(nn.Module):
    '''
    x shape (B,H,W,T,C)
    output shape (B,H,W,T,C)
    '''
    def __init__(self,hidden_features,modes_x,modes_y,modes_z,x_dim,y_dim,z_dim,n_layers=4,factor=4,n_ff_layers=2,dropout=0,kernel_size=1):
        super(FNOblock_ST,self).__init__()
        self.n_layers = n_layers
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList([])
        self.layers_norm = nn.ModuleList([])
        self.PosEmbed = PosEmbed(hidden_features,z_dim,x_dim,y_dim)
        for _ in range(n_layers):
            self.layers.append(SpatialTemporal2d_1d(
                in_dim = hidden_features,
                out_dim = hidden_features,
                modes_x = modes_x,
              modes_y = modes_y,
              modes_z = modes_z,
              forecast_ff=None,
              backcast_ff=None,
              fourier_weight=None,
              factor=factor,
              ff_weight_norm=True,
              n_ff_layers=n_ff_layers,
              layer_norm=True,
              use_fork=False,
              dropout=dropout,
                kernel_size=kernel_size,
                ))
            if hidden_features % 16 ==0:
                self.layers_norm.append(nn.GroupNorm(16,hidden_features))
            else:
                self.layers_norm.append(nn.GroupNorm(hidden_features,hidden_features))
    def forward(self,x):
        #x_raw = x.clone()
        #pdb.set_trace()
        x = self.PosEmbed(x.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        for i in range(self.n_layers):
            layer = self.layers[i]
            norm = self.layers_norm[i]
            x = norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            b,_ = layer(x)
            x = x +b
        return x

class FTSNO_w_gn_bottom_cord_position_aux_decoder_moving_Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, n_layers, d_model,total_length,input_length,enc_in,enc_out,modes_x,modes_y,modes_z,factor,n_ff_layers,x_dim,y_dim,z_dim,pre,pre_dim,pad_x,pad_y,pad_z,dropout):
        super(FTSNO_w_gn_bottom_cord_position_aux_decoder_moving_Model, self).__init__()        
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
#         self.input_shape = input_shape

#         self.target_shape = target_shape
#         T_in, H_in, W_in, C_in = input_shape
#         T_out, H_out, W_out, C_out = target_shape
#         assert H_in == H_out and W_in == W_out
#         self.auxiliary_channels = auxiliary_channels
        
        
        self.prefactor = FNOblock_ST(hidden_features=pre_dim*self.enc_in,n_layers=n_layers,modes_x=modes_x,modes_y=modes_y,modes_z=modes_z,x_dim=x_dim,y_dim=y_dim,z_dim=z_dim)
        self.preblock = FNOblock(hidden_features=self.in_dim,n_layers=n_layers,modes_x=int(modes_x/1),modes_y=int(modes_y/1),modes_z=modes_z,x_dim=int(x_dim/1),y_dim=int(y_dim/1),z_dim=z_dim)
        self.midblock_down = FNOblock(hidden_features=2*self.in_dim,n_layers=n_layers,modes_x=int(modes_x/2),modes_y=int(modes_y/2),modes_z=modes_z,x_dim=int(x_dim/2),y_dim=int(y_dim/2),z_dim=z_dim)
        self.endblock_down = FNOblock(hidden_features=4*self.in_dim,n_layers=n_layers,modes_x=int(modes_x/4),modes_y=int(modes_y/4),modes_z=modes_z,x_dim=int(x_dim/4),y_dim=int(y_dim/4),z_dim=z_dim)
        self.endblock_up = FNOblock(hidden_features=2*self.in_dim,n_layers=n_layers,modes_x=int(modes_x/2),modes_y=int(modes_y/2),modes_z=modes_z,x_dim=int(x_dim/2),y_dim=int(y_dim/2),z_dim=z_dim)
        self.midblock_up = FNOblock(hidden_features=self.in_dim,n_layers=n_layers,modes_x=int(modes_x/1),modes_y=int(modes_y/1),modes_z=modes_z,x_dim=int(x_dim/1),y_dim=int(y_dim/1),z_dim=z_dim)
        #self.posterfactor = FNOblock_ST(hidden_features=self.in_dim,n_layers=n_layers,modes_x=modes_x,modes_y=modes_y,modes_z=modes_z,x_dim=int(x_dim/1),y_dim=int(y_dim/1),z_dim=z_dim)
        self.outblock = FNOblock(hidden_features=self.in_dim,n_layers=n_layers,modes_x=int(modes_x/1),modes_y=int(modes_y/1),modes_z=int(self.pred_len/2),x_dim=x_dim,y_dim=y_dim,z_dim=self.pred_len)

        self.downnorm1 = nn.InstanceNorm3d(2*d_model)
        self.downnorm2 = nn.InstanceNorm3d(4*d_model)
        self.upnorm1 = nn.InstanceNorm3d(4*d_model)
        self.upnorm2 = nn.InstanceNorm3d(3*d_model)
        
        #self.downsample_top1 = DownSampling3D((z_dim,x_dim,y_dim),(z_dim,int(x_dim/2),int(y_dim/2)),pre_dim*self.enc_in,pre_dim*self.enc_in)
        
        #self.upsample_end1 = Upsample3DLayer(self.in_dim,self.in_dim,(self.z_dim,self.x_dim,self.y_dim))
        
        self.downsample1 = DownSampling3D((z_dim,int(x_dim/1),int(y_dim/1)),(z_dim,int(x_dim/2),int(y_dim/2)),self.in_dim,2*self.in_dim)
        self.downsample2 = DownSampling3D((z_dim,int(x_dim/2),int(y_dim/2)),(z_dim,int(x_dim/4),int(y_dim/4)),2*self.in_dim,4*self.in_dim)

        self.upsample1 = Upsample3DLayer(4*self.in_dim,2*self.in_dim,(self.z_dim,int(self.x_dim/2),int(self.y_dim/2)))
        self.upsample2 = Upsample3DLayer(2*self.in_dim,self.in_dim,(self.z_dim,int(self.x_dim/1),int(self.y_dim/1)))
        
        
        self.in_proj_pre = nn.Linear(self.enc_in+3,pre_dim*self.enc_in)
        self.aux_enc_proj= nn.Linear(3,self.in_dim)
        self.aux_dec_proj= nn.Linear(3,self.in_dim)
        
        self.in_proj = nn.Linear(pre_dim*self.enc_in,self.in_dim)
        if not self.pre:
            self.in_proj = nn.Linear(self.enc_in,self.in_dim)
        
        self.out_proj = nn.Linear(self.in_dim,self.enc_out)
        self.out_proj_z = nn.Linear(self.seq_len+self.pad_z,self.pred_len)
        self.temp_att = TemporalAttention(dim=self.in_dim)
        
        self.posEmbed_enc_aux = PosEmbed(self.in_dim,self.z_dim,self.x_dim,self.y_dim)
        
        self.posEmbed_dec_aux = PosEmbed(self.in_dim,self.pred_len,self.x_dim,self.y_dim)
    
    def _interp_aux(self, aux_input, ref_shape):
        r"""
        Parameters
        ----------
        aux_input:  torch.Tensor
            Shape (B, T_aux, H_aux, W_aux, C_aux)
        ref_shape:  Sequence[int]
            (B, T, H, W, C)
        Returns
        -------
        ret:    torch.Tensor
            Shape (B, T, H, W, C_aux)
        """
        ret = rearrange(aux_input,
                        "b t h w c -> b c t h w")
        ret = F.interpolate(input=ret, size=ref_shape[:-1])
        ret = rearrange(ret,
                        "b c t h w -> b t h w c")
        return ret

    def interp_enc_aux(self, aux_input):
        return self._interp_aux(aux_input=aux_input, ref_shape=self.input_shape)

    def forward(self, x_enc,verbose=False,mask_tensor=None):
        #aux_enc, aux_dec, 
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]        
        x = x_enc[:,:self.seq_len,:,:,:].contiguous()
        grid = self.get_grid(x.shape, x.device)   
        output_shape = [x.shape[0],self.pred_len,x.shape[2],x.shape[3],x.shape[4]]
        output_grid = self.get_grid(output_shape,x.device)
        x = x.permute(0,2,3,1,4)
        #B, L, H, W, C = x.shape
        x = nn.ReplicationPad3d((0,0,self.pad_z,0,0,0))(x)
        x = torch.cat((x, grid.permute(0,2,3,1,4)), dim=-1)
        
        aux_enc=grid.contiguous().permute(0,2,3,1,4)  ### b,h,w,t,c
        aux_dec=output_grid.contiguous().permute(0,2,3,1,4)
        aux_enc = self.aux_enc_proj(aux_enc)
        aux_dec = self.aux_dec_proj(aux_dec)
        
        aux_enc = self.posEmbed_enc_aux(aux_enc.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        aux_dec = self.posEmbed_dec_aux(aux_dec.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        
        x_pre = self.in_proj_pre(x)
        x_pre = self.prefactor(x_pre)
        #x_pre = self.downsample_top1(x_pre.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        
        x_pre = self.in_proj(x_pre)
        x_pre = self.preblock(x_pre)
        #pdb.set_trace()
        x_pre_down = self.downsample1(x_pre.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x_pre_down = self.downnorm1(x_pre_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_pre_down = self.midblock_down(x_pre_down)
    
        x_mid_down = self.downsample2(x_pre_down.permute(0,3,1,2,4)).permute(0,2,3,1,4)
        x_mid_down = self.downnorm2(x_mid_down.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_mid_down = self.endblock_down(x_mid_down)
        x_end_up = self.upsample1(x_mid_down.permute(0,3,1,2,4)).permute(0,2,3,1,4)+x_pre_down
        x_end_up = self.upnorm1(x_end_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_end_up = self.endblock_up(x_end_up)

        x_mid_up = self.upsample2(x_end_up.permute(0,3,1,2,4)).permute(0,2,3,1,4)+x_pre
        x_mid_up = self.upnorm2(x_mid_up.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x_mid_up = self.midblock_up(x_mid_up)
        #x_end = self.posterfactor(x_mid_up)
        #x = x_end
        x = x_mid_up
        #print('input shape',x.shape)
        #x = x[..., :,:,:-self.pad_z, :]
        #x = x[..., :,:,:self.pred_len, :]
        x = x.permute(0,3,1,2,4)
        #x = self.upsample_end1(x)
        #pdb.set_trace()
        x = self.temp_att(aux_dec.permute(0,4,3,1,2),aux_enc.permute(0,4,3,1,2),x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.outblock(x.permute(0,2,3,1,4))
        x = x.permute(0,3,1,2,4)
        x = self.out_proj(x)
        
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

    