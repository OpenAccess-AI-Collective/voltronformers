import math
import os

import torch


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
