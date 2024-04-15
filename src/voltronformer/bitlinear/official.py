import math
import torch
from torch import nn


def weight_quant(weight, dtype=torch.float16):
    weight = weight.bfloat16()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.to(dtype=dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)


class BitLinear(nn.Linear):

    def __init__(self,
                 *kargs,
                 weight_bits=1,
                 input_bits=8,
                 **kwargs
                 ):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        # Convert the uint8 weights to the input data type
        fp_weight = self.weight.to(input.dtype)

        # seems silly, but this is done for the cuda graph's sake
        quant_weight = fp_weight + (weight_quant(self.weight, dtype=input.dtype) - fp_weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out