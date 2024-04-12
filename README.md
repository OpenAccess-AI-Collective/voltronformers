# voltronformers
Assembling the best SotA AI techniques into a unified model

```
- 13B parameter BitNet + infini-Attention + DenseFormer + MoD + 
  In Context-Pretraining + 2 stage pretraining 
- upcycle w c-BTX to an 8 expert sparse MoE + MoA 
```

https://twitter.com/winglian/status/1778675583817326842

References

## BitNet
BitNet: Scaling 1-bit Transformers for Large Language Models

- arXiv: https://arxiv.org/abs/2310.11453
- reference implementations:
  - https://github.com/kyegomez/BitNet
  - https://github.com/cg123/bitnet/tree/main

## DenseFormer
DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging

- arXiv: https://arxiv.org/abs/2402.02622
- reference implementations:
  - https://github.com/epfml/DenseFormer/tree/main

## Mixture-of-Depths
Mixture-of-Depths: Dynamically allocating compute in transformer-based language models

- arXiv: https://arxiv.org/abs/2404.02258
- reference implementations:
  - https://github.com/thepowerfuldeez/OLMo/blob/80be1a3ff1d4a80167b37a1d97509cc0b54d821d/olmo/mod.py

## In-Context Pretraining
In-Context Pretraining: Language Modeling Beyond Document Boundaries

- arXiv: https://arxiv.org/abs/2310.10638

## MiniCPM (Two Stage Pre-training Strategy) 
MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

- arXiv: https://arxiv.org/abs/2404.06395

## Cluster-Branch-Train-Merge (c-BTM)
Scaling Expert Language Models with Unsupervised Domain Discovery

- arXiv: https://arxiv.org/abs/2303.14177
- reference implementations:
  - https://github.com/kernelmachine/cbtm

##  Branch-Train-MiX (BTX)
Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM

- arXiv: https://arxiv.org/abs/2403.07816

##  Mixture Of Attention Heads
Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM

- arXiv: https://arxiv.org/abs/2403.07816
- reference implementations:
  - https://github.com/lucidrains/mixture-of-attention
  - JetMoE: https://arxiv.org/pdf/2404.07413.pdf
- errata:
  - https://twitter.com/Yikang_Shen/status/1777029389638647950 
