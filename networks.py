from BaseModule import BaseModule
import torch
import torch.nn as nn
from typing import Union, Optional, Dict


# class ConvModule(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  activation='relu',
#                  norm_cfg=None,
#                  order=('conv', 'norm', 'act')):
#         super().__init__()
#         self.order = order
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
#
#         # 如果传入了 norm_cfg，则获取归一化类型
#         if norm_cfg:
#             norm_type = norm_cfg.get('type', 'BatchNorm2d')
#             self.norm = getattr(nn, norm_type)(out_channels)
#         else:
#             self.norm = None
#
#         # 激活函数选择
#         if activation == 'relu':
#             self.act = nn.ReLU(inplace=True)
#         elif activation == 'leaky_relu':
#             self.act = nn.LeakyReLU(0.1, inplace=True)
#         elif activation == 'sigmoid':
#             self.act = nn.Sigmoid()
#         else:
#             self.act = None
#
#     def forward(self, x):
#         # 根据指定顺序逐步应用层
#         for layer in self.order:
#             if layer == 'conv':
#                 x = self.conv(x)
#             elif layer == 'norm' and self.norm:
#                 x = self.norm(x)
#             elif layer == 'act' and self.act:
#                 x = self.act(x)
#         return x
#
#     def init_weights(self):
#         # 如果卷积层没有自己的初始化方法，则进行默认初始化
#         if not hasattr(self.conv, 'init_weights'):
#             # 根据激活函数类型选择不同的非线性形式
#             if isinstance(self.act, nn.LeakyReLU):
#                 nonlinearity = 'leaky_relu'
#                 a = self.act.negative_slope if hasattr(self.act, 'negative_slope') else 0.01
#             else:
#                 nonlinearity = 'relu'
#                 a = 0
#
#             # 使用 Kaiming 初始化权重
#             init.kaiming_normal_(self.conv.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
#             if self.conv.bias is not None:
#                 init.zeros_(self.conv.bias)
#
#         # 如果使用了归一化层，则使用常量初始化
#         if self.norm:
#             init.constant_(self.norm.weight, 1)
#             if hasattr(self.norm, 'bias') and self.norm.bias is not None:
#                 init.zeros_(self.norm.bias)


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution. Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer. Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer. Defaults to dict(type='Swish').
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[Dict] = dict(type='Swish'),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return out + identity
        else:
            return out