# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES

from mmcv.runner import BaseModule

import mmdet.utils.trt as trt

class MlpImpl(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(MlpImpl, trt.TrtHelper):
    def __init__(self, name: str, *args, **kwargs):
        MlpImpl.__init__(self, *args, **kwargs)
        trt.TrtHelper.__init__(self, name)

    def Forward(self, in0: torch.Tensor):
        out0 = self.forward(in0)
        return out0

        inputs = {'input': in0}
        gt_out = {'output': out0.cpu()}

        trt_out = self.trtinfer(inputs, list(gt_out.keys()))

        for name in trt_out:
            diff = (gt_out[name] - trt_out[name]).pow(2)
            min, mean, median, max = diff.min().item(), diff.mean().item(), \
                    diff.median().item(), diff.max().item()
            print(f'{self.name}.{name}: min={min}, mean={mean}, median={median}, max={max}')

        return trt_out['output'].to(out0.device)

class WindowPartition(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        if not hasattr(self, 'hn'):
            self.hn = H // self.window_size
        if not hasattr(self, 'wn'):
            self.wn = W // self.window_size
        x = x.view(B, self.hn, self.window_size, self.wn, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows


class WindowReverse(nn.Module):
    def __init__(self, window_size, H, W):
        super().__init__()
        self.window_size = window_size
        self.H = H
        self.W = W

    def forward(self, windows):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        if not hasattr(self, 'B'):
            self.B = int(windows.shape[0] / (self.H * self.W / self.window_size / self.window_size))
        if not hasattr(self, 'hn'):
            self.hn = self.H // self.window_size
        if not hasattr(self, 'wn'):
            self.wn = self.W // self.window_size
        x = windows.view(self.B, self.hn, self.wn, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(self.B, self.H, self.W, -1)
        return x


class WindowAttentionImpl(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        slice_q = torch.FloatTensor([1, 0, 0]).reshape(1, 3)
        slice_k = torch.FloatTensor([0, 1, 0]).reshape(1, 3)
        slice_v = torch.FloatTensor([0, 0, 1]).reshape(1, 3)
        self.register_buffer('slice_q', slice_q)
        self.register_buffer('slice_k', slice_k)
        self.register_buffer('slice_v', slice_v)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        if not hasattr(self, "cn"):
            self.cn = C // self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.cn).permute(2, 0, 3, 1, 4)

        if True: # TRT
            qkv = qkv.reshape(3, -1)
            q = torch.matmul(self.slice_q, qkv).reshape(B_, self.num_heads, N, self.cn)
            k = torch.matmul(self.slice_k, qkv).reshape(B_, self.num_heads, N, self.cn)
            v = torch.matmul(self.slice_v, qkv).reshape(B_, self.num_heads, N, self.cn)
        else:
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if not hasattr(self, "bn"):
                self.bn = B_ // nW
            attn = attn.view(self.bn, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention(WindowAttentionImpl, trt.TrtHelper):
    def __init__(self, name: str, *args, **kwargs):
        WindowAttentionImpl.__init__(self, *args, **kwargs)
        trt.TrtHelper.__init__(self, name)

    def Forward(self, x: torch.Tensor, mask: torch.Tensor):
        out0 = self.forward(x, mask)
        return out0

        inputs = {'x': x}
        if mask is not None:
            inputs['mask'] = mask
        gt_out = {'output': out0.cpu()}

        trt_out = self.trtinfer(inputs, list(gt_out.keys()))

        for name in trt_out:
            diff = (gt_out[name] - trt_out[name]).pow(2)
            min, mean, median, max = diff.min().item(), diff.median().item(), \
                    diff.mean().item(), diff.max().item()
            print(f'{self.name}.{name}: min={min:.2e}, median={median:.2e}, mean={mean:.2e}, max={max:.2e}')

        return trt_out['output'].to(out0.device)

class SwinTransformerBlockImpl(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, name, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(f'{name}.attn',
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(f'{name}.mlp', in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.window_partition = WindowPartition(self.window_size)

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        if not hasattr(self, 'pad_r'):
            self.pad_r = (self.window_size - W % self.window_size) % self.window_size
        if not hasattr(self, 'pad_b'):
            self.pad_b = (self.window_size - H % self.window_size) % self.window_size
        if True: # TRT
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (pad_l, self.pad_r, pad_t, self.pad_b))
            x = x.permute(0, 2, 3, 1)
        else:
            x = F.pad(x, (0, 0, pad_l, self.pad_r, pad_t, self.pad_b))
        if not hasattr(self, 'window_reverse'):
            _, Hp, Wp, _ = x.shape
            self.window_reverse = WindowReverse(self.window_size, Hp, Wp)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn.Forward(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp.Forward(self.norm2(x)))

        return x

class SwinTransformerBlock(SwinTransformerBlockImpl, trt.TrtHelper):
    def __init__(self, name: str, *args, **kwargs):
        SwinTransformerBlockImpl.__init__(self, name, *args, **kwargs)
        trt.TrtHelper.__init__(self, name)

    def Forward(self, x: torch.Tensor, mask_matrix: torch.Tensor):
        out0 = self.forward(x, mask_matrix)
        return out0

        inputs = {'x': x, 'mask_matrix': mask_matrix}
        gt_out = {'output': out0.cpu()}

        trt_out = self.trtinfer(inputs, list(gt_out.keys()))

        for name in trt_out:
            diff = torch.abs(gt_out[name] - trt_out[name])
            min, mean, median, max = diff.min().item(), diff.median().item(), \
                    diff.mean().item(), diff.max().item()
            print(f'{self.name}.{name}: min={min:.2e}, median={median:.2e}, mean={mean:.2e}, max={max:.2e}')

        return trt_out['output'].to(out0.device)

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.H = None
        self.W = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        x = x.view(B, self.H, self.W, C)

        # padding
        if not hasattr(self, 'pad_input'):
            self.pad_input = (self.H % 2 == 1) or (self.W % 2 == 1)
            self.pad_h = self.H % 2
            self.pad_w = self.W % 2
            self.half_h = self.H // 2
            self.half_w = self.W // 2
        if self.pad_input:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
            x = x.permute(0, 2, 3, 1)

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # y = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        if not hasattr(self, 'conv_w'):
            conv_w = torch.stack([
                torch.Tensor([[1., 0.], [0., 0.]]),
                torch.Tensor([[0., 0.], [1., 0.]]),
                torch.Tensor([[0., 1.], [0., 0.]]),
                torch.Tensor([[0., 0.], [0., 1.]])
                ]).to(torch.float32)
            conv_w = conv_w.unsqueeze(1).repeat(C, 1, 1, 1)
            conv_w.requires_grad = False
            self.register_buffer('conv_w', conv_w)
            self.conv_w = self.conv_w.to(x.device)

        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = F.conv2d(x, weight=self.conv_w, bias=None, stride=2, padding=0, groups=C)
        x = x.reshape([B, C, 4, self.half_h, self.half_w])
        x = x.permute(0, 3, 4, 2, 1).flatten(3)
        # z = torch.abs(x - y).sum()
        # print(z.item())

        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayerImpl(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 name,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                name = f'{name}.swinblock{i}',
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.H = None
        self.W = None
        self.window_partition = WindowPartition(self.window_size)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        if not hasattr(self, 'Hp'):
            self.Hp = int(np.ceil(self.H / self.window_size)) * self.window_size
        if not hasattr(self, 'Wp'):
            self.Wp = int(np.ceil(self.W / self.window_size)) * self.window_size
        if not hasattr(self, 'img_mask'):
            img_mask = torch.zeros((1, self.Hp, self.Wp, 1))  # 1 Hp Wp 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            self.register_buffer("img_mask", img_mask)
            img_mask.requires_grad = False
            self.img_mask = self.img_mask.to(x.device)

        mask_windows = self.window_partition(self.img_mask)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = self.H, self.W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk.Forward(x, attn_mask)
        if self.downsample is not None:
            self.downsample.H = self.H
            self.downsample.W = self.W
            x_down = self.downsample(x)
            #Wh, Ww = (self.H + 1) // 2, (self.W + 1) // 2
            return x, x_down #, Wh, Ww
        else:
            return x, x #, self.H, self.W


class BasicLayer(BasicLayerImpl, trt.TrtHelper):
    def __init__(self, name: str, *args, **kwargs):
        BasicLayerImpl.__init__(self, name, *args, **kwargs)
        trt.TrtHelper.__init__(self, name)

    def Forward(self, x: torch.Tensor):
        x_out, x_down = self.forward(x)
        return x_out, x_down

        inputs = {'x': x}
        gt_out = {'x_out': x_out.cpu(), 'x_down': x_down.cpu()}

        trt_out = self.trtinfer(inputs, list(gt_out.keys()))

        for name in trt_out:
            diff = torch.abs(gt_out[name] - trt_out[name])
            min, mean, median, max = diff.min().item(), diff.median().item(), \
                    diff.mean().item(), diff.max().item()
            print(f'{self.name}.{name}: min={min:.2e}, median={median:.2e}, mean={mean:.2e}, max={max:.2e}')

        return trt_out['x_out'].to(x_out.device), trt_out['x_down'].to(x_down.device)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if not hasattr(self, 'pad_w'):
            if W % self.patch_size[1] != 0:
                self.pad_w = self.patch_size[1] - W % self.patch_size[1]
            else:
                self.pad_w = 0

        if not hasattr(self, 'pad_h'):
            if H % self.patch_size[0] != 0:
                self.pad_h = self.patch_size[0] - H % self.patch_size[0]
            else:
                self.pad_h = 0

        if self.pad_w != 0 or self.pad_h != 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


@BACKBONES.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 name,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pretrained=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                name=f'{name}.basiclayer{i_layer}',
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def init_weights(self):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
