try:
    from .bitlinear import BitLinear as Linear
except ImportError:
    from torch.nn import Linear
