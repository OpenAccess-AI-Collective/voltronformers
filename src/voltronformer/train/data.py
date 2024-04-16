import functools
from collections import defaultdict
from queue import Queue
from threading import Thread
from typing import Callable, Dict, List

import numpy as np
from datasets import Dataset
from torch.utils.data import RandomSampler, DataLoader

from src.voltronformer.train.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from src.voltronformer.train.samplers import MultipackBatchSampler


def get_dataset_lengths(dataset: Dataset):
    input_ids = dataset.data.column("input_ids")
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


def drop_long_seq(sample, sequence_len=2048):
    return len(sample["input_ids"]) <= sequence_len and len(sample["input_ids"]) > 0


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
    train_dataset = Dataset.from_dict(examples)
    train_dataset = train_dataset.map(
        ds_wrapper,
        batched=True,
        remove_columns = list(train_dataset.features.keys())
    )

    drop_long = functools.partial(drop_long_seq, sequence_len=max_seq_length)
    train_dataset = train_dataset.filter(
        drop_long,
        num_proc=8,
    )

    sampler = MultipackBatchSampler(
        RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        batch_max_len=max_seq_length,
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


class QueuedDataLoader(DataLoader):
    def __init__(self, *args, queue_len=1_000, **kwargs):
        kwargs["persistent_workers"] = True
        super().__init__(*args, **kwargs)
        self.data_queue = Queue(maxsize=queue_len)
        self.prefetch_thread = Thread(target=self.prefetch_data)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def prefetch_data(self):
        for data in super().__iter__():
            self.data_queue.put(data)
        self.data_queue.put(None)

    def __iter__(self):
        return super().__iter__()

    def __next__(self):
        if hasattr(self, 'data_queue'):
            data = self.data_queue.get()
            if data is None:
                raise StopIteration
            return data
        else:
            return self._iterator.__next__()
