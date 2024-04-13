import functools
from typing import List, Optional, Callable, Tuple

import torch
from bitnet.bit_attention import scaled_dot_product_gqa, BitMGQA
from functorch.einops import rearrange
from torch import nn, Tensor
from denseformer import DWAModules
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

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


class LlamaBitMGQA(BitMGQA):
    def __init__(self, embed_dim, query_heads, *args, max_position_embeddings=2048, rope_theta=10_000, **kwargs):
        self.head_dim = embed_dim // query_heads
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)
        super().__init__(embed_dim, query_heads, *args, **kwargs)

    def forward(
            self,
            x: Tensor,
            need_weights: bool = False,
            # attn_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        output, attn_weights = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            # TODO
            # mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            force_grouped=False,
        )
        output = rearrange(output, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'!  We follow the MAGNETO
        # architecture (https://arxiv.org/pdf/2210.06423.pdf), which applies an extra
        # layer norm before the linear output projection.  The cross-attention layer in
        # the MAGNETO decoder does not include this layer norm, so users have the
        # option to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            output = self.norm(output)
        # Linear projection on attention outputs.
        output = self.out_proj(output)

        return output, attn_weights


class TransformerDecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = LlamaBitMGQA(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, max_position_embeddings=config.max_position_embeddings, rope_theta=config.rope_theta)
        self.mlp = mlp(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        output, _ = x + self.attn(self.input_layernorm(x), position_ids=position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


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
        inputs_embeds = self.wte(x)
        past_seen_tokens = 0
        position_ids = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)

        hidden_states = inputs_embeds
        self.dwa_modules.init_accumulators(hidden_states)
        for i, decoder_layer in enumerate(self.h):
            # gradient checkpointing
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                )
            else:
                hidden_states = decoder_layer(hidden_states, position_ids)
            hidden_states = self.dwa_modules(hidden_states, block_idx=i)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


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
