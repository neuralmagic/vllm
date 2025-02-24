# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


#always symmetric for now
@dataclass
class GroupedMMLinearLayerConfig:
    is_per_act_token: bool
    is_per_out_ch: bool
    is_static_input_scheme: bool


class GroupedMMLinearKernel(ABC):

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(
            cls, c: GroupedMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def __init__(self, c: GroupedMMLinearLayerConfig, w_q_param_name: str,
                 w_s_param_name: str, i_s_param_name: str) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        self.i_s_name = i_s_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _get_weight_params(
            self, layer: torch.nn.Module) -> Tuple[
                torch.Tensor,  # weight
                torch.Tensor,  # weight_scale
                Optional[torch.Tensor],  # input_scale, 
            ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.i_s_name),
        )
