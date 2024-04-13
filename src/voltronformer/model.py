import functools
from typing import List, Optional, Callable

import torch
from torch import nn
from denseformer import DWAModules
from torch.utils.checkpoint import checkpoint

from .core import Linear
from .mod import MoDBlock


class FeedForward(nn.Module):
    def __init__(self, gate_proj: Linear, down_proj: Linear, up_proj: Linear):
        super().__init__()
        self.gate_proj = gate_proj
        self.down_proj = down_proj
        self.up_proj = up_proj
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads


class RMSNorm(nn.Module):
    """copied from torchtune"""
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = (
                x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale

def mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = Linear(dim, hidden_dim, bias=False)
    down_proj = Linear(hidden_dim, dim, bias=False)
    up_proj = Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


class TransformerDecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config.hidden_size, config.num_attention_heads)
        self.mlp = mlp(config.hidden_size, config.intermediate_size)



class CheckpointingMixin(nn.Module):
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable


class Transformer(CheckpointingMixin):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dwa_modules = DWAModules(config.num_hidden_layers, config.dwa_dilation, config.dwa_period)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.h = nn.ModuleList([
            (
                MoDBlock(config, TransformerDecoderBlock)
                if i % self.config.mod_every == 0
                else TransformerDecoderBlock(config)
            )
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = RMSNorm(config.hidden_size, eps=1e-6)
        self.gradient_checkpointing = False

    def forward(self, x):
        x = self.wte(x)
        self.dwa_modules.init_accumulators(x)
        for i, decoder_layer in enumerate(self.h):
            # gradient checkpointing
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    decoder_layer,
                    x,
                )
            else:
                x = decoder_layer(x)
            x = self.dwa_modules(x, block_idx=i)
        x = self.ln_f(x)
        return x


class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config)
        self.vocab_size = config.vocab_size
        # should this use a BitLinear layer?
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.embed_out.weight

    def forward(self, x):
        x = self.transformer(x)
        logits = self.embed_out(x)

        return logits.float()

    def train(self, mode: bool = True):
        """
        Override the default train() to enable gradient checkpointing.
        """
        if mode:
            self.transformer.gradient_checkpointing_enable()
        return super().train(mode)
