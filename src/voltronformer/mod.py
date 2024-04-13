"""
from https://github.com/epfml/llm-baselines/compare/main...mixture_of_depth
"""
import torch
from torch import nn


class MoDBlock(nn.Module):
    def __init__(self, config, block_class):
        super().__init__()
        self.config = config
        self.block = block_class(config)
        self.router = nn.Linear(config.hidden_size, 1, bias=False)
        self.capacity_factor = config.mod_capacity_factor
        self.top_k =int(self.capacity_factor * config.max_position_embeddings)

    def forward(self, x, position_ids, **kwargs):
        # [batch_size, sequence_length, n_embd]
        B, T, C = x.shape
        # inference time optimization: sequence length can
        # be smaller than seq len during training
        top_k = min(self.top_k, int(self.capacity_factor * T))

        """STEP 1: get logits and top_k tokens"""
        # [batch_size, sequence_length, 1]
        router_logits = self.router(x)
        # weights and selected tokens: [batch_size, top_k, 1]
        weights, selected_tokens = torch.topk(router_logits, top_k, dim=1, sorted=False)
        # IMPORTANT: need to sort indices to keep causal order for those tokens that
        # are processed in a block
        selected_tokens, index = torch.sort(selected_tokens, dim=1)
        weights = torch.gather(weights, dim=1, index=index)

        """STEP 2: expand indices to process batches with _reduced_ seqlen"""
        # We need to expand indices' dimensions from
        # [batch_size, top_k, 1] to [batch_size, top_k, n_embd] for gathering
        indices_expanded = selected_tokens.expand(-1, -1, C)
        # [batch_size, top_k, n_embd]
        top_k_tokens = torch.gather(x, 1, indices_expanded)
        top_k_tokens_processed = self.block(top_k_tokens, position_ids, **kwargs)

        """STEP 3: combine results"""
        x = torch.scatter_add(
            x,
            dim=1,
            index=indices_expanded,
            src=top_k_tokens_processed * weights,
        )

        return x