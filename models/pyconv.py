""" PyConv networks for image recognition as presented in our paper:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import os

from matplotlib import pyplot as plt
from timm.layers import DropPath

# from div.download_from_url import download_from_url

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pretrained')

__all__ = ['PyConvResNet', 'pyconvresnet18', 'pyconvresnet34', 'pyconvresnet50', 'pyconvresnet101', 'pyconvresnet152']


model_urls = {
    'pyconvresnet50': 'https://drive.google.com/uc?export=download&id=128iMzBnHQSPNehgb8nUF5cJyKBIB7do5',
    'pyconvresnet101': 'https://drive.google.com/uc?export=download&id=1fn0eKdtGG7HA30O5SJ1XrmGR_FsQxTb1',
    'pyconvresnet152': 'https://drive.google.com/uc?export=download&id=1zR6HOTaHB0t15n6Nh12adX86AhBMo46m',
}


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): 输入图像中的通道数
        out_channels (list): 卷积生成的每个金字塔层的通道数
        pyconv_kernels (list): 每个金字塔级别的核的空间大小
        pyconv_groups (list): 每个金字塔级别从输入通道到输出通道的阻塞连接数，in_channels和out_channels必须能被groups整除。
        stride (int or tuple, optional): 卷积的步幅。默认值：1
        dilation (int or tuple, optional): 内核元素之间的间距。默认值：1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels=[3,5,7], pyconv_groups=[1,4,8], stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class EfficientPyConv3(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1):
        super().__init__()

        # 使用深度可分离卷积减少计算量
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(inplans, inplans, kernel_size=pyconv_kernels[0],
                      padding=pyconv_kernels[0] // 2, stride=stride, groups=inplans),
            nn.Conv2d(inplans, planes // 2, kernel_size=1)
        )

        # 分解5x5卷积为两个3x3卷积
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(inplans, inplans, kernel_size=3, padding=1, stride=stride, groups=inplans),
            nn.Conv2d(inplans, inplans, kernel_size=3, padding=1, groups=inplans),
            nn.Conv2d(inplans, planes // 4, kernel_size=1)
        )

        # 分解7x7卷积为三个3x3卷积
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(inplans, inplans, kernel_size=3, padding=1, stride=stride, groups=inplans),
            nn.Conv2d(inplans, inplans, kernel_size=3, padding=1, groups=inplans),
            nn.Conv2d(inplans, inplans, kernel_size=3, padding=1, groups=inplans),
            nn.Conv2d(inplans, planes // 4, kernel_size=1)
        )

        self.alpha1 = nn.Parameter(torch.tensor([1.0]))
        self.alpha2 = nn.Parameter(torch.tensor([1.0]))
        self.alpha3 = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x):
        out1 = self.conv2_1(x)
        out2 = self.conv2_2(x)
        out3 = self.conv2_3(x)
        out1 = out1 * self.alpha1
        out2 = out2 * self.alpha2
        out3 = out3 * self.alpha3
        return torch.cat([out1, out2, out3], dim=1)
class PyConv3AdaptiveSEResDP0(nn.Module):
    def __init__(self, inplans, planes, layer_depth=None, max_depth=None, drop_prob=0.1, *args):
        super(PyConv3AdaptiveSEResDP0, self).__init__()
        self.pyconv = PyConv3(inplans, planes, *args)
        self.skip = conv(inplans, planes, kernel_size=1, padding=0) if inplans != planes else nn.Identity()
        self.se = SELayer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        self.drop_prob = drop_prob
        self.drop_path = DropPath(self.drop_prob) if self.drop_prob > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.eps = 1e-6  # 数值安全阈值
    def forward(self, x):
        identity = x
        out = self.pyconv(x)
        out = self.relu(out)
        # 第一种：
        # 数值安全处理
        out_safe = torch.clamp(out, min=self.eps)
        # 计算幂次项
        alpha = torch.pow(out_safe, self.alpha)
        # 增强控制
        out = alpha * self.se(out)

        # 第二种：
        # out = self.alpha * self.se(out)

        identity = self.skip(identity)
        out = self.drop_path(out) + identity
        return out
class PyConv3AdaptiveSEResDP(nn.Module):
    def __init__(self, inplans, planes, layer_depth=None, max_depth=None, drop_prob=0.1, *args):
        super(PyConv3AdaptiveSEResDP, self).__init__()
        self.pyconv = PyConv3(inplans, planes, *args)
        self.skip = conv(inplans, planes, kernel_size=1, padding=0) if inplans != planes else nn.Identity()
        self.se = SELayer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        self.drop_prob = drop_prob
        self.drop_path = DropPath(self.drop_prob) if self.drop_prob > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.eps = 1e-6  # 数值安全阈值
    def forward(self, x):
        identity = x
        out = self.pyconv(x)
        out = self.relu(out)
        out = identity + out
        # 第一种：
        # 数值安全处理
        out_safe = torch.clamp(out, min=self.eps)
        # 计算幂次项
        alpha = torch.pow(out_safe, self.alpha)
        # 增强控制
        out = alpha * self.se(out)

        # 第二种：
        # out = self.alpha * self.se(out)

        identity = self.skip(identity)
        out = self.drop_path(out) + identity
        return out
class PyConv3AdaptiveSEResDP2(nn.Module):
    def __init__(self, inplans, planes,drop_prob=0.1, *args):
        super(PyConv3AdaptiveSEResDP2, self).__init__()
        self.pyconv = EfficientPyConv3(inplans, planes, *args)
        self.skip = conv(inplans, planes, kernel_size=1, padding=0) if inplans != planes else nn.Identity()
        self.se = SELayer2(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop_prob = drop_prob
        self.drop_path = DropPath(self.drop_prob) if self.drop_prob > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        self.eps = 1e-6  # 数值安全阈值
    def forward(self,x ):
        identity = x
        out = self.pyconv(x)
        out = self.relu(out)
        out = identity + out
        # 第一种：
        # 数值安全处理
        out_safe = torch.clamp(out, min=self.eps)
        # 计算幂次项
        alpha_out = torch.pow(out_safe, self.alpha)
        # 增强控制
        out = alpha_out * self.se(out)

        identity = self.skip(identity)
        out = self.drop_path(out) + identity
        return out
# 通道注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class SELayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channel // reduction,channel, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)
class EnhancedSEAttention(nn.Module):
    """增强型通道注意力"""

    def __init__(self, channels, reduction=8, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 动态通道交互
        self.dynamic_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # 局部上下文增强
        self.local_ctx = nn.Conv2d(channels, channels, k_size,
                                   padding=k_size // 2, groups=channels)

        self.scale = nn.Parameter(torch.tensor([0.2]))  # 可学习缩放因子

    def forward(self, x):
        b, c, _, _ = x.size()

        # 全局通道注意力
        avg_out = self.dynamic_fc(self.avg_pool(x).view(b, c))
        max_out = self.dynamic_fc(self.max_pool(x).view(b, c))
        global_attn = (avg_out + max_out).view(b, c, 1, 1)

        # 局部上下文调制
        local_ctx = torch.sigmoid(self.local_ctx(x))

        # 组合并缩放
        return x * (1 + self.scale * global_attn * local_ctx)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)

def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=1,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes * self.expansion)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, dropout_prob0=0.0):
        super(PyConvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3], pyconv_groups=[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PyConvBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, pyconv_kernels=[3], pyconv_groups=[1]):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer,
                            pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)

        return x


def pyconvresnet18(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = PyConvResNet(PyConvBasicBlock1, [2, 2, 2, 2], **kwargs) #params=11.21M GFLOPs 1.55
    model = PyConvResNet(PyConvBasicBlock2, [2, 2, 2, 2], **kwargs)  #params=5.91M GFLOPs 0.88
    if pretrained:
        raise NotImplementedError("Not available the pretrained model yet!")

    return model


def pyconvresnet34(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = PyConvResNet(PyConvBasicBlock1, [3, 4, 6, 3], **kwargs) #params=20.44M GFLOPs 3.09
    model = PyConvResNet(PyConvBasicBlock2, [3, 4, 6, 3], **kwargs)  #params=11.09M GFLOPs 1.75
    if pretrained:
        raise NotImplementedError("Not available the pretrained model yet!")

    return model


def pyconvresnet50(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyConvResNet(PyConvBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     os.makedirs(default_cache_path, exist_ok=True)
    #     model.load_state_dict(torch.load(download_from_url(model_urls['pyconvresnet50'],
    #                                                        root=default_cache_path)))
    return model


def pyconvresnet101(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyConvResNet(PyConvBlock, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     os.makedirs(default_cache_path, exist_ok=True)
    #     model.load_state_dict(torch.load(download_from_url(model_urls['pyconvresnet101'],
    #                                                        root=default_cache_path)))
    return model


def pyconvresnet152(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyConvResNet(PyConvBlock, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     os.makedirs(default_cache_path, exist_ok=True)
    #     model.load_state_dict(torch.load(download_from_url(model_urls['pyconvresnet152'],
    #                                                        root=default_cache_path)))
    return model
