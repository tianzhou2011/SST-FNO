import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

def round_to(dat, c):
    return dat + (dat - dat % c) % c

def get_activation(act, inplace=False, **kwargs):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        elif act == 'identity':
            return nn.Identity()
        elif act == 'elu':
            return nn.ELU(inplace=inplace)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softrelu' or act == 'softplus':
            return nn.Softplus()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError('act="{}" is not supported. '
                                      'Try to include it if you can find that in '
                                      'https://pytorch.org/docs/stable/nn.html'.format(act))
    else:
        return act

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """Root Mean Square Layer Normalization proposed in "[NeurIPS2019] Root Mean Square Layer Normalization"

        Parameters
        ----------
        d
            model size
        p
            partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        eps
            epsilon value, default 1e-8
        bias
            whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

def get_norm_layer(normalization: str = 'layer_norm',
                   axis: int = -1,
                   epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    normalization
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(normalization, str):
        if normalization == 'layer_norm':
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        elif normalization == 'rms_norm':
            assert axis == -1
            norm_layer = RMSNorm(d=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError('normalization={} is not supported'.format(normalization))
        return norm_layer
    elif normalization is None:
        return nn.Identity()
    else:
        raise NotImplementedError('The type of normalization must be str')


def _generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """

    Parameters
    ----------
    x
        Shape (B, T, H, W, C)
    pad_t
    pad_h
    pad_w
    padding_type
    t_pad_left

    Returns
    -------
    out
        The result after padding the x. Shape will be (B, T + pad_t, H + pad_h, W + pad_w, C)
    """
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T + pad_t, H + pad_h, W + pad_w)).permute(0, 2, 3, 4, 1)
    else:
        if t_pad_left:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
        else:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def _generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T - pad_t, H - pad_h, W - pad_w)).permute(0, 2, 3, 4, 1)
    else:
        return x[:, :(T - pad_t), :(H - pad_h), :(W - pad_w), :].contiguous()

def apply_initialization(m,
                         linear_mode="0",
                         conv_mode="0",
                         norm_mode="0",
                         embed_mode="0"):
    if isinstance(m, nn.Linear):

        if linear_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_in', nonlinearity="linear")
        elif linear_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0", ):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0", ):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    # # pos_embed already initialized when created
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0", ):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass
