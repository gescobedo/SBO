import numpy as np
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss,L1Loss

from src.config_classes.adv_config import AdvConfig
from src.config_classes.atk_config import AtkConfig

loss_fn_lookup = {
    "mse": MSELoss(),
    "mae": L1Loss(),
    "ce": CrossEntropyLoss()
}


        

class AdvLosses(nn.Module):
    """
    Loss function that evaluates the performance of adversarial networks.
    """

    def __init__(self, adv_config: List[Union[AdvConfig, AtkConfig]]):
        super().__init__()
        self.adv_config = adv_config

    def _calc_loss_for_adv_group(self, config: Union[AdvConfig, AtkConfig], inputs: List[torch.Tensor],
                                 targets: torch.Tensor):
        individual_results = [self._calc_loss_for_single_adv(config, inp, targets) for inp in inputs]
        return sum(individual_results) / len(individual_results)

    @staticmethod
    def _calc_loss_for_single_adv(config: Union[AdvConfig, AtkConfig], inputs: torch.Tensor, targets: torch.Tensor):
        if config.loss_fn in ["mse","mae"]:
            targets = torch.reshape(targets,(-1,1))
        if config.loss_fn == "ce":
            weights = torch.Tensor(config.loss_class_weights).to(inputs.get_device())
            loss_fn =CrossEntropyLoss(weight=weights) if config.loss_class_weights else CrossEntropyLoss()     
            return loss_fn(inputs, targets)
        return loss_fn_lookup[config.loss_fn](inputs, targets)

    def forward(self, inputs: List[List[torch.Tensor]], targets: List[torch.Tensor]):
        """
        Calculates the losses for the different adversaries in the different adversarial groups.

        :param inputs: A list of lists of tensors, where the tensors are the results of the individual adversaries
                       and the sublists are the results for a group of adversaries.
        :param targets: A list of tensors, where each tensor is the target value an adversary should achieve
        """
        # calculate individual losses
        # already re-weight losses here, as we also want to report the new loss, rather than the original one
        losses = {cfg.group_name: self._calc_loss_for_adv_group(
            cfg, inp, tar.to(dtype=torch.int64 if cfg.type == "categorical" else torch.float32)) * cfg.loss_weight
                  for cfg, inp, tar in zip(self.adv_config, inputs, targets)}

        return sum(losses.values()), losses

