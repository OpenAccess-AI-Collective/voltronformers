import functools
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.samplers import MultipackBatchSampler
from datasets import Dataset
from torch.utils.data import RandomSampler


def get_dataset_lengths(dataset):
    input_ids = dataset.column("input_ids")
    lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
    return lengths


def wrap_pretraining_dataset(
        dataset,
        tokenizer,
        ds_wrapper_fn,
        max_tokens=2048,
        batch_size=1,
        buffer_size=10_000,
):
    collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=max_tokens,
        multipack_attn=False,
    )
    encode = functools.partial(
        encode_packed_pretraining,
        collate_fn,
        ds_wrapper_fn,
        max_seq_length=max_tokens,
        batch_size=batch_size,
    )

    # remove all the existing columns after mapping since they end up having
    # a different length than the encoded/tokenized column
    # this is empty during streaming/pretraining
    remove_columns = []
    if dataset.features is None:
        for first_row in dataset:
            remove_columns = first_row.keys()
            break
    else:
        remove_columns = dataset.features.keys()

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=buffer_size,
        remove_columns=remove_columns,
    )
    return dataset


def encode_packed_pretraining(
        collate_fn,
        ds_wrapper: Callable,
        examples: Dict[str, List],
        max_seq_length: int = 2048,
        batch_size: int = 4,
) -> Dict[str, List]:
    # pylint: disable=duplicate-code
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = Dataset.from_dict(examples).map(ds_wrapper, batched=True)

    sampler = MultipackBatchSampler(
        RandomSampler(train_dataset),
        batch_size=1,
        drop_last=True,
        batch_max_len=batch_size * max_seq_length,
        lengths=get_dataset_lengths(train_dataset),
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
            if "overflow_to_sample_mapping" in features:
                del features["overflow_to_sample_mapping"]
            if "labels" not in features:
                features["labels"] = features["input_ids"].copy()
            collated_features = collate_fn(features)

            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data
