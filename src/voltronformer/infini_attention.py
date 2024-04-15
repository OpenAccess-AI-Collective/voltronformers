from typing import Optional

import torch
from torch import nn

from .core import Linear


# https://github.com/dingo-actual/infini-transformer/blob/main/infini_transformer/compressive_memory.py

class CompressiveMemory(nn.Module):
    """Implements the Compressive Transformer memory module."""
    def __init__(self, dim_input: int, dim_key: int, dim_value: int, num_heads: int, segment_len: int, update: str = "delta"):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            update (str, optional): Type of memory update rule to use ("linear" or "delta"). Defaults to "delta".
        """
        super(CompressiveMemory, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len

        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.update = update

        # Projections for stacked SDP attention
        self.proj_k = Linear(dim_input, num_heads * dim_key, bias=False)
        self.proj_v = Linear(dim_input, num_heads * dim_value, bias=False)
        self.proj_q = Linear(dim_input, num_heads * dim_key, bias=False)

        # Initialize betas for weighted average of dot-product and memory-based attention
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_value))

        # Projection for output
        self.proj_out = Linear(num_heads * dim_value, dim_input, bias=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Scaled Dot-Product Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape

        n_seq, rem = divmod(seq_len, self.segment_len)

        if rem != 0:
            raise ValueError(f"Sequence length must be divisible by segment length. seq_len: {seq_len} segment_len: {self.segment_len}")

        out = []

        # Initialize mem and normalization
        # !!! Initialization was never specified in the paper, so this is an educated guess
        mem = torch.zeros(1, self.num_heads, self.dim_key, self.dim_value).to(device=x.device)
        z = torch.zeros(1, self.num_heads, self.dim_value, 1).repeat(batch_size, 1, 1, 1).to(device=x.device)

        for ix in range(n_seq):
            ix_lo = ix * self.segment_len
            ix_hi = ix_lo + self.segment_len

            # Extract segment from input
            x_seg = x[:, ix_lo:ix_hi, :]

            # Project the input tensor to get the key, value, and query tensors
            k = self.proj_k(x_seg).unsqueeze(1).view((batch_size, self.num_heads, self.segment_len, self.dim_key))
            v = self.proj_v(x_seg).unsqueeze(1).view((batch_size, self.num_heads, self.segment_len, self.dim_value))
            q = self.proj_q(x_seg).unsqueeze(1).view((batch_size, self.num_heads, self.segment_len, self.dim_key))

            # Pre-calculate sigma(q) for updating memory and calculating attention
            sigma_q = (nn.functional.elu(q) + 1.0) # shape: (batch_size, num_heads, segment_len, dim_key)

            # Apply mem update
            if self.update == "linear":
                mem = mem + sigma_q.transpose(-2, -1) @ v
            elif self.update == "delta":
                sigma_k = nn.functional.elu(k) + 1.0
                mem = mem + sigma_q.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))

            # Apply normalization term update
            z = z + (nn.functional.elu(k) + 1.0).sum(dim=-2, keepdim=True)

            # Apply SDP attention
            att_dot = nn.functional.softmax(q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_key)), dim=-1) @ v

            # Calculate normalized linear attention
            att_mem = (sigma_q @ mem) / (sigma_q @ z) # shape: (batch_size, num_heads, segment_len, dim_value)

            # Calculate weighted average of dot-product and memory-based attention
            att = nn.functional.sigmoid(self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot
            att = att.view((batch_size, self.segment_len, self.num_heads * self.dim_value))

            # Append output to buffer
            out.append(self.proj_out(att))

        # Return concatenated full sequence from buffer
        return torch.concat(out, dim=1)
