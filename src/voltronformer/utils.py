import math
import os
from typing import Optional, Set, Type

import torch

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def set_activation_checkpointing(
        model: nn.Module, auto_wrap_policy: Optional[Set[Type[nn.Module]]] = None, **kwargs
) -> None:
    """Utility to setup activation checkpointing and wrap the model for checkpointing.

    Args:
        model (nn.Module): Model to setup activation checkpointing.
        auto_wrap_policy (Optional[Set[nn.Module]]): Policy to wrap module.
        **kwargs: additional arguments to pass to torch.distributed activation checkpointing.
    """
    wrap_policy = ModuleWrapPolicy(auto_wrap_policy or set())
    apply_activation_checkpointing(model, auto_wrap_policy=wrap_policy, **kwargs)


def device_get_local_rank():
    """
    Returns the local rank of the current device.
    """
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    return local_rank


def device_get_cuda():
    rank = device_get_local_rank()
    device = torch.device(type="cuda", index=rank)
    torch.cuda.set_device(device)
    return device


def get_cosine_schedule_with_min_lr_lambda(
        current_step: int,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float,
):
    # Warm up
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # Cosine learning rate decay
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    scaling = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1 - min_lr_ratio) * scaling + min_lr_ratio
