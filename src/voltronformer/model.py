import functools
from typing import Optional, Callable, Tuple

import torch
from .bitlinear import BitLinear, scaled_dot_product_gqa

from functorch.einops import rearrange
from torch import nn, Tensor
from denseformer import DWAModules
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from .mod import MoDBlock
from .infini_attention import CompressiveMemory as InfiniAttention

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    from .kernels.rms_norm import RMSNorm

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FeedForward(nn.Module):
    def __init__(self, gate_proj: BitLinear, down_proj: BitLinear, up_proj: BitLinear):
        super().__init__()
        self.gate_proj = gate_proj
        self.down_proj = down_proj
        self.up_proj = up_proj
        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # FIXME layernorm???
        x = self.down_proj(x)
        return x


def mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = BitLinear(dim, hidden_dim, bias=False)
    down_proj = BitLinear(hidden_dim, dim, bias=False)
    up_proj = BitLinear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


# copied from https://github.com/kyegomez/BitNet/blob/main/bitnet/bit_attention.py
class LlamaBitMGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

    def __init__(
            self,
            embed_dim: int,
            query_heads: int = 8,
            kv_heads: int = 4,
            dropout: float = 0.1,
            bias: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
            gamma_init: float = 1.0,
            linear_groups: int = 1,
            *args,
            max_position_embeddings=2048,
            rope_theta=10_000,
            **kwargs,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = BitLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            *args,
            **kwargs,  # device=device, dtype=dtype
        )
        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(
            embed_dim,
            kv_embed_dim,
            bias=bias,
            *args,
            **kwargs,  # device=device, dtype=dtype
        )
        self.v_proj = BitLinear(
            embed_dim,
            kv_embed_dim,
            bias=bias,
            *args,
            **kwargs,  # device=device, dtype=dtype
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                kv_embed_dim,
                eps=layer_norm_eps,  # device=device, dtype=dtype
            )
        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = BitLinear(
            embed_dim,
            embed_dim,
            bias=bias,  # device=device, dtype=dtype
        )
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
            self,
            x: Tensor,
            position_ids: Optional[Tensor] = None,
            need_weights: bool = False,
            # attn_mask: Optional[Tensor] = None,
            is_causal: bool = True,
            average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b h n d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.kv_heads)

        # Generate rotary embeddings
        cos, sin = self.rotary_emb(x, position_ids)

        # Reshape cos and sin to match the shape of q and k
        seq_len = q.shape[2]  # Get the sequence length from q
        cos = cos[:, :seq_len, :].unsqueeze(1)
        sin = sin[:, :seq_len, :].unsqueeze(1)

        # Apply rotary position embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Adjust the dimensions of q, k, and v
        q = q.view(-1, *q.shape[-3:])
        k = k.view(-1, *k.shape[-3:])
        v = v.view(-1, *v.shape[-3:])

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

        # Re-assemble all head outputs side-by-side.
        # output = output.transpose(1, 2).contiguous().view(b, n, d)
        output = rearrange(output, "b n h d -> b h (n d)")

        # Linear projection on attention outputs.
        output = self.out_proj(output)

        return output, attn_weights


class TransformerDecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.infini_attention:
            self.attn = InfiniAttention(
                config.hidden_size,
                config.num_key_value_heads,
                config.num_key_value_heads,
                config.num_attention_heads,
                config.ia_segment_len,
                update="delta",
            )
        else:
            self.attn = LlamaBitMGQA(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                bias=False,
                layer_norm=False,
            )

        self.mlp = mlp(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        residual = x
        h = self.input_layernorm(x)
        output = residual + self.attn(h, position_ids=position_ids)[0]
        return residual + self.mlp(self.post_attention_layernorm(output))


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
        if config.dwa:
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
        if self.config.dwa:
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
            if self.config.dwa:
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
