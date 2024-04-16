import triton
import triton.language as tl


@triton.jit
def silu(x):
    """
    SiLU activation function, also known as Swish-1.
    """
    return x * tl.sigmoid(x)


@triton.jit
def silu_grad(x):
    sigmoid_x = tl.sigmoid(x)
    return sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
