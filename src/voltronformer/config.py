from axolotl.utils.dict import DictDefault

def tiny():
    return DictDefault({
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "max_position_embeddings": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "vocab_size": 100352,
        "dwa_dilation": 4,
        "dwa_period": 5,
        "pad_token_id": 100277, # dbrx <{|pad|}>
        "mod_every": 2,
        "mod_capacity_factor": 0.125,
    })


def medium():
    return DictDefault({
        "hidden_size": 1024,
        "intermediate_size": 2816,

        "max_position_embeddings": 32768,

        "vocab_size": 100352
    })