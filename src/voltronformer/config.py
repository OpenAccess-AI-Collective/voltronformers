from axolotl.utils.dict import DictDefault


def tiny():
    """50M parameters"""
    return DictDefault({
        "hidden_size": 512,
        "intermediate_size": 1408,
        "rope_theta": 10_000,
        "max_position_embeddings": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
    })


def small():
    return DictDefault({
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "max_position_embeddings": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "vocab_size": 32000,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
    })


def medium():
    return DictDefault({
        "hidden_size": 4096,
        "intermediate_size": 11264,
        "max_position_embeddings": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "vocab_size": 32000,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 2,
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
        "rms_norm_eps": 0.000001,
    })