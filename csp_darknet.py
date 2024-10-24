import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
from networks import BaseModule, ConvModule, CSPLayerWithTwoConv, SPPFBottleneck
from utils import make_divisible, make_round
class YOLOv8CSPDarknet(BaseModule):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {'P5', 'P6'}.
            Defaults to 'P5'.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict], optional): List of plugins for stages.
        deepen_factor (float): Depth multiplier. Defaults to 1.0.
        widen_factor (float): Width multiplier. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output from which stages. Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Config for normalization layer. Defaults to BN.
        act_cfg (dict): Config for activation layer. Defaults to SiLU.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 6, True, False],
               [1024, None, 3, True, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Optional[List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: Optional[Union[dict, List[dict]]] = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(init_cfg=init_cfg)

        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval

        self.stem = self.build_stem_layer()

        # Build stages based on the arch settings.
        self.stages = nn.ModuleList()
        for idx, setting in enumerate(self.arch_settings[arch]):
            stage = self.build_stage_layer(idx, setting)
            self.stages.append(nn.Sequential(*stage))

        # Freeze stages if needed
        self._freeze_stages()

    def build_stem_layer(self) -> nn.Module:
        """Build the stem layer, which is the first convolutional layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_settings['P5'][0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def _freeze_stages(self):
        """Freeze the specified stages to stop their gradients."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                stage = self.stages[i]
                for param in stage.parameters():
                    param.requires_grad = False
                stage.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the backbone."""
        outputs = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()