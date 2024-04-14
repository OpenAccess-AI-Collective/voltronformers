import torch
import triton
import triton.language as tl
from torch import nn


# from https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/#openai-triton-rewriting
@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)

    # first loop over input tensor to compute the root mean of the square
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        # recompute address at each iteration
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    # we keep this reduction operation outside the loop for perf reasons
    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # apply the normalization and multiply by RMS weights
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


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


"""not ready for use yet. 2X Faster, but not accurate"""
class RMSNormTriton(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the grid and block dimensions
        N_SIZE = x.shape[-1]
        BLOCK_N_SIZE = 512  # Adjust this value based on your requirements

        # Allocate output tensor
        output = torch.empty_like(x)

        # Define the strides for input, scale, and output tensors
        stride_x_batch, stride_x_m, stride_x_k = x.stride()
        stride_rms_w = self.scale.stride(0)
        stride_out_batch, stride_out_m, stride_out_k = output.stride()

        # Launch the Triton kernel
        grid = lambda meta: (x.shape[0], x.shape[1])
        rmsnorm_triton[grid](
            x, self.scale, output,
            stride_x_batch, stride_x_m, stride_x_k,
            stride_rms_w,
            stride_out_batch, stride_out_m, stride_out_k,
            N_SIZE, self.eps, BLOCK_N_SIZE
        )

        return output