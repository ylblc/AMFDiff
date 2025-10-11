from abc import abstractmethod

import math
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x ch
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            # q,k, v: (b*heads) x ch x length
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards     # (b*heads) x M x M
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)  # (b*heads) x ch x length
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x length
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            q, k, v = qkv.chunk(3, dim=1)  # b x heads*ch x length
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        cond_lq=True,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, y=None, lq=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if lq is not None:
            assert self.cond_lq
            if lq.shape[2:] != x.shape[2:]:
                lq = F.pixel_unshuffle(lq, 2)
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class UNetModelSwin(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels += base_chn
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                    in_chans=ch,
                    embed_dim=swin_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=swin_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                     ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class ResBlockConv(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModelConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        cond_lq=True,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.dtype = th.float16 if use_fp16 else th.float32

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.cond_lq = cond_lq

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlockConv(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockConv(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if lq is not None:
            assert self.cond_lq
            if lq.shape[2:] != x.shape[2:]:
                lq = F.pixel_unshuffle(lq, 2)
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

from models.pyconv import PyConv3AdaptiveSEResDP, PyConv3AdaptiveSEResDP0, PyConv3AdaptiveSEResDP_lite


class UNetModelSwinPyConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels += base_chn
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                    in_chans=ch,
                    embed_dim=swin_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=swin_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                     ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.skip_connections = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                self.skip_connections.append(
                    PyConv3AdaptiveSEResDP0(ich,ich)
                )
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for ii,module in enumerate(self.output_blocks):
            resH=hs.pop()
            resH = self.skip_connections[ii](resH)
            h = th.cat([h, resH], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class UNetModelSwinPyConv2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=96,
            window_size=8,
            mlp_ratio=2.0,
            patch_norm=False,
            cond_lq=True,
            cond_mask=False,
            lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.skip_connections=nn.ModuleList([])
        self.ag_fusions = nn.ModuleList([])
        self.afpfs = nn.ModuleList([])
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)
        # ch=160
        ch = input_ch = int(channel_mult[0] * model_channels)
        # 6=3+3          base_chn=3
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([
            # in_channels=6, ch=160
                nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
            ]
        )

        input_block_chans = []
        ds = image_size
        for level, mult in enumerate(channel_mult):
            # mult = 1，2，2，4
            if level != 0:
                self.input_blocks.append(nn.ModuleList([]))
            input_block_chans.append([ch])
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        # out_channels=[1,2,2,4] * 160
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                # ch=[1,2,2,4] * 160
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks[level].append(TimestepEmbedSequential(*layers))
                input_block_chans[level].append(ch)
            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks[level].append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                # input_block_chans[level].append(ch)
                ds //= 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([]) for _ in range(len(channel_mult))
        ])
        input_block_chans2 = deepcopy(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # mult = [4,2,2,1]
            self.skip_connections.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans[level].pop()  # 编码器层的输出通道数

                layers = [
                    ResBlock(

                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        )
                ]
                # self.skip_connections[index].append(
                #     PyConv3AdaptiveSEResDP(ich, ich, level + 1, len(channel_mult))
                # )
                ch = int(model_channels * mult)

                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2


                self.output_blocks[index].append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        layer_chs =  [int(model_channels * mult) for mult in channel_mult]
        self.aligns = nn.ModuleList([])
        self.layer_selects = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.afpfs.append(nn.ModuleList([]))
            # self.ag_fusions.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            self.aligns.append(nn.ModuleList([]))
            layer_length = len(channel_mult)
            # self.layer_selects.append(
            #     AdaptiveFeatureSelect(layer_length,layer_length,num=max(1,layer_length-2))
            # )
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans2[level].pop()  # 编码器层的输出通道数
                self.afpfs[index].append(
                    # AFPF2(layer_chs, ich)
                    SCAttentionAFPF(layer_chs,ich)
                )
                # self.ag_fusions[index].append(
                #     # AdaptiveGateFusion(ich)
                #     LightGateFusion(ich)
                #     # AttentionGuidedFusion(ich)
                # )
                self.aligns[index].append(
                    FeatureAlign(layer_chs,ich)
                )
                ch = int(model_channels * mult)

        self.skip_features = []
    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        self.skip_features=[]
        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module_list in enumerate(self.input_blocks):
            hs.append([])
            for jj,module in enumerate(module_list):
                if not (ii == 0 and jj == 0):
                    hs[ii].append(h)
                h = module(h, emb)
        hs[-1].append(h)

        h = self.middle_block(h, emb)
        output_blocks_len = len(self.output_blocks)
        ws =[]
        for ii,module_list in enumerate(self.output_blocks):
            curr_hs_i = output_blocks_len - ii - 1
            hs_row = hs[curr_hs_i]
            # 1、
            # afpf_features,weights = self.layer_selects[curr_hs_i](hs,ii)
            # 2、
            # afpf_features = [(hs[k][-1],k) for k in range(2)]
            # 3、
            afpf_features = [(hs[k][-1],k) for k in range(4)]
            # ws.append(weights)
            for jj,module in enumerate(module_list):
                curr_hs_j = len(hs_row) - jj - 1
                res_h = hs_row[curr_hs_j]
                target_size = res_h.shape[2:]
                afpf_aligned_features = self.aligns[ii][jj](afpf_features,target_size)
                # # 多尺度特征融合
                multi_h = self.afpfs[ii][jj](afpf_aligned_features, res_h)
                # # skip跳越层
                # skip_h = self.skip_connections[ii][jj](multi_h)
                # # 融合层
                res_h = multi_h
                # res_h = self.ag_fusions[ii][jj](multi_h, skip_h)
                # self.skip_features.append((res_h,multi_h,skip_h,fusion_h))
                h = th.cat([h, res_h], dim=1)
                h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        # for k,row in enumerate(ws):
        #     arr1 = [round(x,6) for x in row.tolist()]
        #     print(f"Decoder[{k}] = {arr1}")
        # print("="*50)
        return out

    def get_skip_features(self):
        return self.skip_features
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class UNetModelSwinPyConv0123_aff_no(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=96,
            window_size=8,
            mlp_ratio=2.0,
            patch_norm=False,
            cond_lq=True,
            cond_mask=False,
            lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.skip_connections=nn.ModuleList([])
        self.ag_fusions = nn.ModuleList([])
        self.afpfs = nn.ModuleList([])
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)
        # ch=160
        ch = input_ch = int(channel_mult[0] * model_channels)
        # 6=3+3          base_chn=3
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([
            # in_channels=6, ch=160
                nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
            ]
        )

        input_block_chans = []
        ds = image_size
        for level, mult in enumerate(channel_mult):
            # mult = 1，2，2，4
            if level != 0:
                self.input_blocks.append(nn.ModuleList([]))
            input_block_chans.append([ch])
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        # out_channels=[1,2,2,4] * 160
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                # ch=[1,2,2,4] * 160
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks[level].append(TimestepEmbedSequential(*layers))
                input_block_chans[level].append(ch)
            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks[level].append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                # input_block_chans[level].append(ch)
                ds //= 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([]) for _ in range(len(channel_mult))
        ])
        input_block_chans2 = deepcopy(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # mult = [4,2,2,1]
            self.skip_connections.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans[level].pop()  # 编码器层的输出通道数

                layers = [
                    ResBlock(

                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        )
                ]
                ch = int(model_channels * mult)

                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2


                self.output_blocks[index].append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        layer_chs =  [int(model_channels * mult) for mult in channel_mult]
        self.aligns = nn.ModuleList([])
        self.layer_selects = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.afpfs.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            self.aligns.append(nn.ModuleList([]))
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans2[level].pop()  # 编码器层的输出通道数
                self.afpfs[index].append(
                    SCAttentionAFPF(layer_chs,ich)
                )

                self.aligns[index].append(
                    FeatureAlign(layer_chs,ich)
                )
                ch = int(model_channels * mult)

        self.skip_features = []
    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        self.skip_features=[]
        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module_list in enumerate(self.input_blocks):
            hs.append([])
            for jj,module in enumerate(module_list):
                if not (ii == 0 and jj == 0):
                    hs[ii].append(h)
                h = module(h, emb)
        hs[-1].append(h)

        h = self.middle_block(h, emb)
        output_blocks_len = len(self.output_blocks)
        for ii,module_list in enumerate(self.output_blocks):
            curr_hs_i = output_blocks_len - ii - 1
            hs_row = hs[curr_hs_i]
            # 1、
            # afpf_features,weights = self.layer_selects[curr_hs_i](hs,ii)
            # 2、
            # afpf_features = [(hs[k][-1],k) for k in range(2)]
            # 3、
            afpf_features = [(hs[k][-1],k) for k in range(1)]
            for jj,module in enumerate(module_list):
                curr_hs_j = len(hs_row) - jj - 1
                res_h = hs_row[curr_hs_j]
                target_size = res_h.shape[2:]
                afpf_aligned_features = self.aligns[ii][jj](afpf_features,target_size)
                # # 多尺度特征融合
                multi_h = self.afpfs[ii][jj](afpf_aligned_features, res_h)
                # # 融合层
                res_h = multi_h
                h = th.cat([h, res_h], dim=1)
                h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def get_skip_features(self):
        return self.skip_features
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class UNetModelSwinPyConv3(nn.Module):

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=96,
            window_size=8,
            mlp_ratio=2.0,
            patch_norm=False,
            cond_lq=True,
            cond_mask=False,
            lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.skip_connections=nn.ModuleList([])
        self.ag_fusions = nn.ModuleList([])
        self.afpfs = nn.ModuleList([])
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)
        # ch=160
        ch = input_ch = int(channel_mult[0] * model_channels)
        # 6=3+3          base_chn=3
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([
            # in_channels=6, ch=160
                nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
            ]
        )

        input_block_chans = []
        ds = image_size
        for level, mult in enumerate(channel_mult):
            # mult = 1，2，2，4
            if level != 0:
                self.input_blocks.append(nn.ModuleList([]))
            input_block_chans.append([ch])
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        # out_channels=[1,2,2,4] * 160
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                # ch=[1,2,2,4] * 160
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks[level].append(TimestepEmbedSequential(*layers))
                input_block_chans[level].append(ch)
            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks[level].append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                # input_block_chans[level].append(ch)
                ds //= 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([]) for _ in range(len(channel_mult))
        ])
        input_block_chans2 = deepcopy(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # mult = [4,2,2,1]
            self.skip_connections.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans[level].pop()  # 编码器层的输出通道数

                layers = [
                    ResBlock(

                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        )
                ]
                # self.skip_connections[index].append(
                    # PyConv3AdaptiveSEResDP(ich, ich, level + 1, len(channel_mult))
                # )
                ch = int(model_channels * mult)

                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2


                self.output_blocks[index].append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        layer_chs =  [int(model_channels * mult) for mult in channel_mult]
        self.aligns = nn.ModuleList([])
        layer_length = len(channel_mult)
        # self.layer_selects = nn.ModuleList()
        self.layer_selects = AdaptiveFeatureSelect2(layer_length)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.afpfs.append(nn.ModuleList([]))
            self.ag_fusions.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            self.aligns.append(nn.ModuleList([]))
            # self.layer_selects.append(AdaptiveFeatureSelect2(layer_length))
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans2[level].pop()  # 编码器层的输出通道数
                # self.afpfs[index].append(
                #     SCAttentionAFPF(layer_chs,ich)
                # )
                self.aligns[index].append(
                    FeatureAlign(layer_chs,ich)
                )
                ch = int(model_channels * mult)
        self.dfs_attw = []
    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)


        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module_list in enumerate(self.input_blocks):
            hs.append([])
            for jj,module in enumerate(module_list):
                if not (ii == 0 and jj == 0):
                    hs[ii].append(h)
                h = module(h, emb)
        hs[-1].append(h)

        h = self.middle_block(h, emb)
        self.dfs_attw = []
        output_blocks_len = len(self.output_blocks)
        for ii,module_list in enumerate(self.output_blocks):
            curr_hs_i = output_blocks_len - ii - 1
            hs_row = hs[curr_hs_i]
            # 1、DFS
            afpf_features,weights = self.layer_selects(hs,ii) # weights = [B, num_encoder_layers]
            self.dfs_attw.append(weights)
            # 2、固定
            # afpf_features =[(hs[k][-1],k) for k in range(2)]
            for jj,module in enumerate(module_list):
                curr_hs_j = len(hs_row) - jj - 1
                res_h = hs_row[curr_hs_j]
                target_size = res_h.shape[2:]
                afpf_aligned_features = self.aligns[ii][jj](afpf_features,target_size)
                # 1、多尺度特征融合
                # multi_h = self.afpfs[ii][jj](afpf_aligned_features, res_h)
                # 2、简单融合
                multi_h = 0
                for f in afpf_aligned_features:
                    multi_h += f
                multi_h +=res_h
                # 1、pc增强
                # fusion_h= self.skip_connections[ii][jj](multi_h)
                # 2、 无pc增强
                fusion_h = multi_h
                h = th.cat([h, fusion_h], dim=1)
                h = module(h, emb)
        # for k,row in enumerate(self.dfs_attw):
        #     arr1= [round(x,6) for x in row.tolist()]
        #     print(f'Decoder[{k}] = {arr1}')
        # print("="*50)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

class UNetModelSwinPyConv4(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=96,
            window_size=8,
            mlp_ratio=2.0,
            patch_norm=False,
            cond_lq=True,
            cond_mask=False,
            lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.skip_connections=nn.ModuleList([])
        self.afpfs = nn.ModuleList([])
        self.aligns = nn.ModuleList([])
        self.layer_selects = nn.ModuleList([])
        layer_chs = [int(model_channels * mult) for mult in channel_mult]
        layer_length = len(channel_mult)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)
        # ch=160
        ch = input_ch = int(channel_mult[0] * model_channels)
        # 6=3+3          base_chn=3
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([
            # in_channels=6, ch=160
                nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
            ]
        )

        input_block_chans = []
        ds = image_size
        for level, mult in enumerate(channel_mult):
            # mult = 1，2，2，4
            if level != 0:
                self.input_blocks.append(nn.ModuleList([]))
            input_block_chans.append([ch])
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        # out_channels=[1,2,2,4] * 160
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                # ch=[1,2,2,4] * 160
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks[level].append(TimestepEmbedSequential(*layers))
                input_block_chans[level].append(ch)
            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks[level].append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                # input_block_chans[level].append(ch)
                ds //= 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([]) for _ in range(len(channel_mult))
        ])
        input_block_chans2 = deepcopy(input_block_chans)
        input_block_chans3 = deepcopy(input_block_chans)
        input_block_chans4 = deepcopy(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # mult = [4,2,2,1]
            self.skip_connections.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans[level].pop()  # 编码器层的输出通道数

                layers = [
                    ResBlock(

                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        )
                ]
                self.skip_connections[index].append(
                    PyConv3AdaptiveSEResDP(ich, ich, level + 1, len(channel_mult))
                )
                ch = int(model_channels * mult)

                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2


                self.output_blocks[index].append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        for level, mult in list(enumerate(channel_mult))[::-1]:
            index = len(channel_mult) - level - 1
            layer_length = len(channel_mult)
            self.layer_selects.append(nn.ModuleList())
            ch = int(model_channels * mult)
            self.aligns.append(nn.ModuleList([]))
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans3[level].pop()  # 编码器层的输出通道数
                self.aligns[index].append(
                    FeatureAlign(layer_chs, ich)
                )
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.afpfs.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1

            self.layer_selects.append(nn.ModuleList())
            ch = int(model_channels * mult)
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans2[level].pop()  # 编码器层的输出通道数

                self.afpfs[index].append(
                        SCAttentionAFPF2(layer_chs, ich)
                )
                if i == 0:
                    self.layer_selects[index].append(
                        AdaptiveFeatureSelect3(layer_length,int(model_channels * channel_mult[min(layer_length - 1, level + 1)]))
                    )
                else:
                    self.layer_selects[index].append(
                        AdaptiveFeatureSelect3(layer_length, ch)
                    )


        self.ws=[]
    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)


        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module_list in enumerate(self.input_blocks):
            hs.append([])
            for jj,module in enumerate(module_list):
                if not (ii == 0 and jj == 0):
                    hs[ii].append(h)
                h = module(h, emb)
        hs[-1].append(h)

        h = self.middle_block(h, emb)
        output_blocks_len = len(self.output_blocks)
        self.ws =[]
        for ii,module_list in enumerate(self.output_blocks):
            curr_hs_i = output_blocks_len - ii - 1
            hs_row = hs[curr_hs_i]
            self.ws.append([])
            # afpf_features =[(hs[k][-1],k) for k in range(4)]
            for jj,module in enumerate(module_list):
                curr_hs_j = len(hs_row) - jj - 1
                res_h = hs_row[curr_hs_j]
                target_size = res_h.shape[2:]
                afpf_features, weights= self.layer_selects[ii][jj](hs, ii,h) #  weights = [batchsize,encoder]
                # weights = weights.mean(dim=0) #  weights = [encoder]
                self.ws[ii].append(weights)
                afpf_aligned_features = self.aligns[ii][jj](afpf_features,target_size)
                # # 1、多尺度特征融合
                multi_h = self.afpfs[ii][jj](afpf_aligned_features,res_h)
                # 2、简单加权
                # multi_h =0
                # for afpf_aligned_feature in afpf_aligned_features:
                #     multi_h+=afpf_aligned_feature
                # multi_h += res_h

                # 1、skip跳越层
                skip_h = self.skip_connections[ii][jj](multi_h)
                res_h = skip_h
                h = th.cat([h, res_h], dim=1)
                h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        # ws = self.ws
        # s = ""
        # for i, row in enumerate(ws):
        #     row_s = ""
        #     for j, col in enumerate(row):
        #         arr1 = [f"{k}({round(x, 4)})" for k,x in enumerate(col.mean(dim=0).tolist()) if x > 0.1]
        #         # arr1 = [round(x, 4) for k, x in enumerate(col.mean(dim=0).tolist())]
        #         row_s += f'Decoder({i},{j})={arr1}  '
        #     s += row_s + "\n"
        # print(s)
        # print("="*100)
        return out


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
class UNetModelSwinPyConv5(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=96,
            window_size=8,
            mlp_ratio=2.0,
            patch_norm=False,
            cond_lq=True,
            cond_mask=False,
            lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.skip_connections=nn.ModuleList([])
        self.afpfs = nn.ModuleList([])
        self.aligns = nn.ModuleList([])
        self.layer_selects = nn.ModuleList([])
        layer_chs = [int(model_channels * mult) for mult in channel_mult]
        layer_length = len(channel_mult)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            # nn.Identity() 不做任何处理
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)
        # ch=160
        ch = input_ch = int(channel_mult[0] * model_channels)
        # 6=3+3          base_chn=3
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([
            # in_channels=6, ch=160
                nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
            ]
        )

        input_block_chans = []
        ds = image_size
        for level, mult in enumerate(channel_mult):
            # mult = 1，2，2，4
            if level != 0:
                self.input_blocks.append(nn.ModuleList([]))
            input_block_chans.append([ch])
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        # out_channels=[1,2,2,4] * 160
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                # ch=[1,2,2,4] * 160
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks[level].append(TimestepEmbedSequential(*layers))
                input_block_chans[level].append(ch)
            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks[level].append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                # input_block_chans[level].append(ch)
                ds //= 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([]) for _ in range(len(channel_mult))
        ])
        input_block_chans2 = deepcopy(input_block_chans)
        input_block_chans3 = deepcopy(input_block_chans)
        input_block_chans4 = deepcopy(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # mult = [4,2,2,1]
            self.skip_connections.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans[level].pop()  # 编码器层的输出通道数

                layers = [
                    ResBlock(

                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        )
                ]
                self.skip_connections[index].append(
                    PyConv3AdaptiveSEResDP_lite(ich, ich)
                )
                ch = int(model_channels * mult)

                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2


                self.output_blocks[index].append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        for level, mult in list(enumerate(channel_mult))[::-1]:
            index = len(channel_mult) - level - 1
            layer_length = len(channel_mult)
            self.layer_selects.append(nn.ModuleList())
            ch = int(model_channels * mult)
            self.aligns.append(nn.ModuleList([]))
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans3[level].pop()  # 编码器层的输出通道数
                self.aligns[index].append(
                    FeatureAlign(layer_chs, ich)
                )
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.afpfs.append(nn.ModuleList([]))
            index = len(channel_mult) - level - 1

            self.layer_selects.append(nn.ModuleList())
            ch = int(model_channels * mult)
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans2[level].pop()  # 编码器层的输出通道数

                self.afpfs[index].append(
                        SCAttentionAFPF2(layer_chs, ich)
                )
                if i == 0:
                    self.layer_selects[index].append(
                        AdaptiveFeatureSelect3(layer_length,int(model_channels * channel_mult[min(layer_length - 1, level + 1)]))
                    )
                else:
                    self.layer_selects[index].append(
                        AdaptiveFeatureSelect3(layer_length, ch)
                    )


        self.ws=[]
    def forward(self, x, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality image.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)


        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)


        h = x.type(self.dtype)
        for ii, module_list in enumerate(self.input_blocks):
            hs.append([])
            for jj,module in enumerate(module_list):
                if not (ii == 0 and jj == 0):
                    hs[ii].append(h)
                h = module(h, emb)
        hs[-1].append(h)

        h = self.middle_block(h, emb)
        output_blocks_len = len(self.output_blocks)
        self.ws =[]
        for ii,module_list in enumerate(self.output_blocks):
            curr_hs_i = output_blocks_len - ii - 1
            hs_row = hs[curr_hs_i]
            self.ws.append([])
            for jj,module in enumerate(module_list):
                curr_hs_j = len(hs_row) - jj - 1
                res_h = hs_row[curr_hs_j]
                target_size = res_h.shape[2:]
                afpf_features, weights= self.layer_selects[ii][jj](hs, ii,h) #  weights = [batchsize,encoder]
                self.ws[ii].append(weights)
                afpf_aligned_features = self.aligns[ii][jj](afpf_features,target_size)
                # # 1、多尺度特征融合
                multi_h = self.afpfs[ii][jj](afpf_aligned_features,res_h)
                # 1、skip跳越层
                skip_h = self.skip_connections[ii][jj](multi_h)
                res_h = skip_h
                h = th.cat([h, res_h], dim=1)
                h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


class AdaptiveFeatureSelect(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers=None, num = None):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_encoder_layers
        self.num  = num  if num  is not None else num_encoder_layers
        # 学习每个decoder层级应该关注哪些encoder层级
        self.attention_weights = nn.Parameter(
            th.ones(self.num_decoder_layers, self.num_encoder_layers)
        )

    def forward(self, hs, decoder_level):


        weights = F.softmax(self.attention_weights[decoder_level], dim=0)

        topk_weights, topk_indices = th.topk(weights, k=min(self.num , self.num_encoder_layers))

        selected_features = []
        for idx in topk_indices:
            selected_features.append((hs[idx][-1],idx.item()))
        return selected_features,weights
class AdaptiveFeatureSelect11(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers=None, num = None):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_encoder_layers
        self.num = num if num is not None else num_encoder_layers
        # 学习每个decoder层级应该关注哪些encoder层级
        self.attention_weights = nn.Parameter(
            th.ones(self.num_decoder_layers, self.num_encoder_layers)
        )

    def forward(self, hs, decoder_level):

        # 获取当前decoder层级的注意力权重
        weights = F.softmax(self.attention_weights[decoder_level], dim=0)

        # 选择权重最高的几个特征
        topk_weights, topk_indices = th.topk(weights, k=min(self.num, self.num_encoder_layers))

        selected_features = []
        for idx in topk_indices:
            selected_features.append((hs[idx][-1]*weights[idx],idx.item()))
        return selected_features,weights


class AdaptiveFeatureSelect2(nn.Module):


    def __init__(self, num_encoder_layers, embedding_dim=64):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_encoder_layers

        self.encoder_key_embedding = nn.Embedding(num_encoder_layers, embedding_dim)


        self.query_net = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),  # 输入是层索引（标量）
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )



    def forward(self, encoder_features, decoder_idx):

        batch_size = encoder_features[0][0].size(0)
        device = encoder_features[0][0].device
        decoder_state = torch.tensor([decoder_idx], device=device)

        normalized_state = decoder_state / self.num_encoder_layers
        query = self.query_net(normalized_state.view(1, 1)).view(1, -1)  # [1, embedding_dim]
        query = query.expand(batch_size, -1)  # [B, embedding_dim]

        encoder_indices = torch.arange(self.num_encoder_layers, device=query.device)
        keys = self.encoder_key_embedding(encoder_indices)  # [num_encoder_layers, embedding_dim]
        keys = keys.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_encoder_layers, embedding_dim]

        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # [B, num_encoder_layers]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, num_encoder_layers]

        final_features = []
        for ii, row in enumerate(encoder_features):
            feature = row[-1]
            final_features.append((feature * attn_weights[:, ii].view(batch_size, 1, 1, 1), ii))

        return final_features, attn_weights.mean(dim=0)

class AdaptiveFeatureSelect3(nn.Module):

    def __init__(self, num_encoder_layers, feature_dim, embedding_dim=64, reduction_ratio=16):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.reduction_ratio = reduction_ratio

        self.encoder_key_embedding = nn.Embedding(num_encoder_layers, embedding_dim)

        self.query_net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction_ratio, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，获取全局特征表示
            nn.Flatten(),
            nn.Linear(feature_dim // reduction_ratio, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )


        self.log_temperature = nn.Parameter(torch.tensor(0.0))


        self.layer_embedding = nn.Embedding(num_encoder_layers, embedding_dim)


        self.fusion_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, encoder_features,  layer_idx,query_feat):
        """
        encoder_features: 编码器特征列表，[B, C, H, W]
        query_feat: 解码器当前层的特征，[B, C, H, W]
        layer_idx: 当前解码器层索引
        """
        batch_size = query_feat.size(0)
        device = query_feat.device


        query_from_content = self.query_net(query_feat)  # [B, embedding_dim]

        query_from_layer = self.layer_embedding(
            torch.tensor(layer_idx, device=device).clamp(0, self.num_encoder_layers - 1)
        ).unsqueeze(0).expand(batch_size, -1)  # [B, embedding_dim]


        fused_query = torch.cat([query_from_content, query_from_layer], dim=1)  # [B, embedding_dim * 2]
        query = self.fusion_net(fused_query)  # [B, embedding_dim]
        query = query.unsqueeze(1)  # [B, 1, embedding_dim]


        encoder_indices = torch.arange(self.num_encoder_layers, device=device)
        keys = self.encoder_key_embedding(encoder_indices)  # [num_encoder_layers, embedding_dim]
        keys = keys.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_encoder_layers, embedding_dim]
        keys = keys.transpose(1, 2)  # [B, embedding_dim, num_encoder_layers]

        attn_scores = torch.bmm(query, keys).squeeze(1)  # [B, num_encoder_layers]

        temperature = torch.exp(self.log_temperature)
        attn_weights = F.softmax(attn_scores / temperature, dim=1)  # [B, num_encoder_layers]

        final_features = []
        for i, row in enumerate(encoder_features):
            feature = row[-1] if isinstance(row, (list, tuple)) else row
            weighted_feature = feature * attn_weights[:, i].view(batch_size, 1, 1, 1)
            final_features.append((weighted_feature,i))

        return final_features, attn_weights

class SCAttentionAFPF(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, reduction=8):
        super().__init__()
        self.num_levels = len(in_channels_list)

        # 空间注意力
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3),
                nn.Sigmoid()
            ) for _ in range(self.num_levels)
        ])

        # 通道注意力
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // reduction, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels // reduction, out_channels, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_levels)
        ])




    def forward(self,aligned_feats, base_feature):
        fused = th.zeros_like(base_feature)
        for i,  aligned in enumerate(aligned_feats):

            # 空间注意力
            avg_out = th.mean(aligned, dim=1, keepdim=True)
            max_out, _ = th.max(aligned, dim=1, keepdim=True)
            spatial_attn = self.spatial_attention[i](th.cat([avg_out, max_out], dim=1))

            # 通道注意力
            channel_attn = self.channel_attention[i](aligned)

            attended = aligned *  spatial_attn * channel_attn
            fused += attended

        return fused + base_feature
class SCAttentionAFPF2(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, reduction=8):
        super().__init__()
        self.num_levels = len(in_channels_list)

        # 空间注意力
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3),
                nn.Sigmoid()
            ) for _ in range(self.num_levels)
        ])

        # 通道注意力
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // reduction, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels // reduction, out_channels, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_levels)
        ])

        self.res_alpha = nn.Parameter(torch.tensor(0.5))


    def forward(self,aligned_feats, base_feature):
        fused = th.zeros_like(base_feature)
        for i,  aligned in enumerate(aligned_feats):

            # 空间注意力
            avg_out = th.mean(aligned, dim=1, keepdim=True)
            max_out, _ = th.max(aligned, dim=1, keepdim=True)
            spatial_attn = self.spatial_attention[i](th.cat([avg_out, max_out], dim=1))

            # 通道注意力
            channel_attn = self.channel_attention[i](aligned)

            attended = aligned *  (spatial_attn + channel_attn)
            fused += attended

        return self.res_alpha * fused + base_feature



class FeatureAlign(nn.Module):

    def __init__(self, channels_list, target_channel=256):
        super().__init__()
        self.target_channel = target_channel


        self.aligners = nn.ModuleDict()
        for i, channels in enumerate(channels_list):

            channel_aligner = nn.Conv2d(channels, target_channel, 1)
            self.aligners[f'layer_{i}'] = channel_aligner

    def forward(self, features, target_size):

        aligned_features = []

        for feat,idx in features:

            channel_aligned = self.aligners[f'layer_{idx}'](feat)
            if channel_aligned.shape[2:] != target_size:
                if channel_aligned.size(2) > target_size[0]:  # 需要下采样
                    channel_aligned = F.adaptive_avg_pool2d(channel_aligned, target_size)
                else:  # 需要上采样
                    channel_aligned = F.interpolate(
                        channel_aligned, size=target_size, mode='bilinear', align_corners=False
                    )
            aligned_features.append(channel_aligned)
        return aligned_features

class LightGateFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels

        # 极简门控网络
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2, 1),
            nn.Sigmoid()
        )

        # 残差缩放因子
        self.residual_scale = nn.Parameter(th.tensor(1.0))

    def forward(self, skip_feat, fused_feat):
        # 计算门控权重
        combined = th.cat([skip_feat, fused_feat], dim=1)
        gate_weights = self.gate_net(combined)
        gate_skip, gate_fused = gate_weights.chunk(2, dim=1)

        # 应用门控
        gate_skip = gate_skip.view(-1, 1, 1, 1)
        gate_fused = gate_fused.view(-1, 1, 1, 1)

        combined_resH = skip_feat * gate_skip + fused_feat * gate_fused

        # 残差连接
        final_feat = skip_feat + self.residual_scale * combined_resH

        return final_feat
class AFPF(nn.Module):
    def __init__(self, in_channels_list, out_channels=256,enh_reduction=4,w_reduction=8):

        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels

        # 特征对齐
        self.align_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list
        ])

        # 自适应权重生成网络
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, self.out_channels // w_reduction, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // w_reduction, self.num_levels, 1),
            nn.Softmax(dim=1)
        )

        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // enh_reduction, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // enh_reduction, out_channels, 3, padding=1)
        )

    def forward(self, features, base_feature):

        # 特征对齐
        aligned_feats = []
        for i, (conv, feat) in enumerate(zip(self.align_convs, features)):
            # 统一通道数
            aligned = conv(feat)
            # 对齐到基准特征尺寸
            if aligned.shape[2:] != base_feature.shape[2:]:
                aligned = F.interpolate(
                    aligned,
                    size=base_feature.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            aligned_feats.append(aligned)

        # 生成空间自适应权重
        weights = self.weight_net(base_feature)  # [B, K, 1, 1]
        weights = weights.squeeze(-1).squeeze(-1)  # [B, K]

        # 加权融合
        fused = th.zeros_like(base_feature)
        for i, feat in enumerate(aligned_feats):
            weight_map = weights[:, i].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            fused += weight_map * feat

        # 特征增强 (残差连接)
        return self.enhance(fused) + fused + base_feature

