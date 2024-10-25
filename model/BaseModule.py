import copy
import logging
import torch.nn as nn
from collections import defaultdict
from typing import Union, List


class BaseModule(nn.Module):
    """Base module for all modules with parameter initialization functionality.

    This is a simplified version of the original `BaseModule` for general PyTorch usage.

    Args:
        init_cfg (dict or List[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`."""
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value

    def init_weights(self):
        """Initialize the weights."""
        is_top_level_module = not hasattr(self, '_params_init_info')

        # If this is the top-level module, initialize `_params_init_info`
        if is_top_level_module:
            self._params_init_info = defaultdict(dict)
            for name, param in self.named_parameters():
                self._params_init_info[param]['init_info'] = (
                    f'The value is the same before and after calling `init_weights` '
                    f'of {self.__class__.__name__}')
                self._params_init_info[param]['tmp_mean_value'] = param.data.mean().item()

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                logging.info(f'Initializing {module_name} with init_cfg {self.init_cfg}')

                # Support for list or dict configurations
                init_cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]
                other_cfgs = [cfg for cfg in init_cfgs if cfg.get('type') != 'Pretrained']
                pretrained_cfgs = [cfg for cfg in init_cfgs if cfg.get('type') == 'Pretrained']

                # Initialize other weights first
                self.apply_initialization(other_cfgs)

                # Initialize submodules
                for m in self.children():
                    if hasattr(m, 'init_weights') and not m.is_init:
                        m.init_weights()

                # Apply pretrained weights if available
                if pretrained_cfgs:
                    self.apply_initialization(pretrained_cfgs)

                self._is_init = True
            else:
                logging.warning(f'No `init_cfg` provided for {module_name}, using default initialization.')
        else:
            logging.warning(f'init_weights of {module_name} has been called more than once.')

        if is_top_level_module:
            self._log_init_info()

    def apply_initialization(self, init_cfgs):
        """Apply initialization based on provided configurations."""
        for init_cfg in init_cfgs:
            # Here, add the logic to apply initialization methods (e.g., 'kaiming', 'xavier', etc.)
            init_type = init_cfg.get('type', 'kaiming')
            if init_type == 'kaiming':
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_normal_(m.weight)
            # Add more initialization methods if needed

    def _log_init_info(self):
        """Log initialization information."""
        for name, param in self.named_parameters():
            logging.info(f'{name} - Shape: {param.shape}, '
                         f'Init Info: {self._params_init_info[param]["init_info"]}')

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s
