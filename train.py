import functools
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import wandb
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from src.voltronformer.config import tiny
from src.voltronformer.model import CausalLM
from src.voltronformer.train.data import wrap_pretraining_dataset
from src.voltronformer.utils import device_get_cuda, device_get_local_rank, get_cosine_schedule_with_min_lr_lambda


@dataclass
class TrainingArguments:
    gradient_accumulation_steps: int = 1
    max_steps_per_epoch: Optional[int] = None
    log_steps: int = 1
    output_dir: Optional[str] = None
    weight_decay: float = 0.0
    warmup_steps: Optional[int] = 1000
    per_gpu_train_batch_size: Optional[int] = 1
    save_steps: Optional[int] = 5_000
    max_sequence_length: Optional[int] = 8192
    learning_rate: float = 5e-5

class Trainer:
    def __init__(self, model, args, dataloader, accelerator):
        self.args = args
        self._model = model
        self.build_optimizer_and_scheduler()
        self._model, self.dataloader, self.optimizer = accelerator.prepare(self._model, dataloader, self.optimizer)

        self.device = device_get_cuda()
        self.global_step = 0
        self.rank = device_get_local_rank()
        wandb.init(project="voltronformer")
        self.accelerator = accelerator

    @property
    def model_num_parameters(self):
        all_param = 0
        for _, param in self._model.named_parameters():
            num_params = param.numel()
            all_param += num_params

        return all_param

    def build_optimizer_and_scheduler(self):
        self.optimizer = AdamWScheduleFree(self._model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, warmup_steps=self.args.weight_decay)
        self.lr_scheduler = None

    def _loss_fn(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss

    def save_checkpoint(self):
        output_dir = self.args.output_dir if self.args.output_dir is not None else "."
        torch.save(
            self._model.state_dict(),
            os.path.join(output_dir, f"model_{self.global_step}.pt"),
        )

    def train(self, dataloader, rank):
        self._model.train()
        try:
            self.optimizer.train()
        except:
            pass
        self.train_loop(dataloader, rank)

    def train_loop(self, dataloader, rank):
        for idx, batch in enumerate(pbar := tqdm(dataloader, disable=not (rank == 0))):
            if (
                    self.args.max_steps_per_epoch is not None
                    and (idx // self.args.gradient_accumulation_steps)
                    == self.args.max_steps_per_epoch
            ):
                break

            input_ids = batch["input_ids"].to(self.device)
            if "labels" in batch.keys():
                labels = batch["labels"].to(self.device)
            else:
                labels = input_ids.clone()

            logits = self._model(input_ids)

            # Shift so that tokens < n predict n
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)

            # Compute loss
            loss = self._loss_fn(logits, labels)

            if (
                self.global_step * self.args.log_steps == 0
                and self.rank == 0
            ):
                pbar.set_description(f"Loss: {loss.item()}")
                wandb.log({"loss": loss.item(), "global_step": self.global_step})

            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            if self.global_step % self.args.save_steps == 0:
                self.save_checkpoint()


def get_ds():
    return load_dataset("togethercomputer/RedPajama-Data-V2",
                name="default",
                partition="head_middle",
                snapshots=["2023-14"],
                languages=["en"],
                split="train",
                streaming=True,
            ), "raw_content"
    # load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

def main():
    state = PartialState()
    config = tiny()

    ds, text_field = get_ds()
    args = TrainingArguments(
        gradient_accumulation_steps=16,
        max_steps_per_epoch=None,
        log_steps=1,
        output_dir="./out",
        weight_decay=0.0,
        warmup_steps=1000,
        per_gpu_train_batch_size=8,
        save_steps=10000,
        max_sequence_length=config.max_position_embeddings,
        learning_rate=5e-5,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    model = CausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-base")

    def tokenize_function(examples, field="text", tokenizer=None):
        outputs = tokenizer(examples[field], truncation=True, max_length=None)
        return outputs

    with state.main_process_first():
        ds_wrapper_partial = functools.partial(
            tokenize_function,
            tokenizer=tokenizer,
            field=text_field,
        )

        train_dataset = wrap_pretraining_dataset(
            ds,
            tokenizer,
            ds_wrapper_partial,
            max_tokens=args.max_sequence_length,
            batch_size=args.per_gpu_train_batch_size,
            buffer_size=10_000,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")

    accelerator = Accelerator()

    dataloader_params = dict(
        sampler=None,
        batch_size=args.per_gpu_train_batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    dataloader = DataLoader(train_dataset, **dataloader_params)

    trainer = Trainer(model, args, dataloader, accelerator)
    print(f"Total number of parameters: {trainer.model_num_parameters:_}")
    trainer.train_loop(dataloader, rank=0)


if __name__ == "__main__":
    main()