import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch
import wandb
from accelerate import Accelerator, PartialState, DistributedDataParallelKwargs
from datasets import load_dataset
from safetensors.torch import save_model
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from src.voltronformer.config import tiny
from src.voltronformer.model import CausalLM, TransformerDecoderBlock
from src.voltronformer.train.data import wrap_pretraining_dataset
from src.voltronformer.utils import device_get_cuda, device_get_local_rank, set_activation_checkpointing


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
    vocab_size: Optional[int] = None


class Trainer:
    def __init__(self, model, args, dataloader, accelerator, activation_checkpointing=True):
        self.args = args
        self._model = model
        if activation_checkpointing:
            set_activation_checkpointing(
                model, auto_wrap_policy={TransformerDecoderBlock}
            )
        self.build_optimizer_and_scheduler()

        self._model, self.dataloader, self.optimizer = accelerator.prepare(self._model, dataloader, self.optimizer)

        self.device = device_get_cuda()
        self.global_step = 0
        self.rank = device_get_local_rank()
        if accelerator.is_main_process:
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
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss

    def save_checkpoint(self):
        output_dir = self.args.output_dir if self.args.output_dir is not None else "."
        save_model(self._model, os.path.join(output_dir, f"model_{self.global_step}.safetensors"))
        torch.save(
            self._model.state_dict(),
            os.path.join(output_dir, f"model_{self.global_step}.pt"),
        )

    def train(self):
        self._model.train()
        try:
            self.optimizer.train()
        except:
            pass
        self.train_loop()

    def train_loop(self):
        for idx, batch in enumerate(pbar := tqdm(self.dataloader, disable=not (self.rank == 0))):
            # if (
            #         self.args.max_steps_per_epoch is not None
            #         and (idx // self.args.gradient_accumulation_steps)
            #         == self.args.max_steps_per_epoch
            # ):
            #     break

            with self.accelerator.accumulate(self._model):
                input_ids = batch["input_ids"].to(self.device)
                if "labels" in batch.keys():
                    labels = batch["labels"].to(self.device)
                else:
                    labels = input_ids.clone()

                logits = self._model(input_ids)

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, self.args.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)

                # Compute loss
                loss = self._loss_fn(shift_logits, shift_labels)

                if (
                    self.global_step % self.args.log_steps == 0
                    and self.rank == 0
                ):
                    pbar.set_description(f"Loss: {loss.item()}")
                    wandb.log({"loss": loss.item(), "global_step": self.global_step})

                self.accelerator.backward(loss)
                self.optimizer.step()
                self._model.zero_grad()
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
        log_steps=10,
        output_dir="./out",
        weight_decay=0.1,
        warmup_steps=1000,
        per_gpu_train_batch_size=24,
        save_steps=1000,
        max_sequence_length=config.max_position_embeddings,
        learning_rate=1e-3,
        vocab_size=config.vocab_size,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    model = CausalLM(config)
    model = model.to(device_get_cuda())
    # tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-base")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples, field="text", tokenizer=None):
        outputs = tokenizer(examples[field], truncation=True, max_length=config.max_position_embeddings)
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

    # ddp kwargs with find_unused_parameters needed for triton rmsnorm
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    dataloader_params = dict(
        batch_size=args.per_gpu_train_batch_size,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=True),
    )
    dataloader = DataLoader(train_dataset, **dataloader_params)

    trainer = Trainer(model, args, dataloader, accelerator, activation_checkpointing=True)
    print(f"Total number of parameters: {trainer.model_num_parameters:_}")
    trainer.train()


if __name__ == "__main__":
    main()