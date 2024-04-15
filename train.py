import functools
import os
import tempfile
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
from transformers.trainer_pt_utils import distributed_concat

from src.voltronformer.config import teeny, tiny, small
from src.voltronformer.model import CausalLM, TransformerDecoderBlock
from src.voltronformer.train.data import wrap_pretraining_dataset, QueuedDataLoader
from src.voltronformer.utils import device_get_cuda, device_get_local_rank, set_activation_checkpointing


state = PartialState()

@dataclass
class TrainingArguments:
    gradient_accumulation_steps: int = 1
    max_steps_per_epoch: Optional[int] = None
    log_steps: int = 1
    adam_epsilon: Optional[float] = 1e-8
    output_dir: Optional[str] = None
    weight_decay: float = 0.0
    warmup_steps: Optional[int] = 1000
    per_gpu_train_batch_size: Optional[int] = 1
    save_steps: Optional[int] = 5_000
    max_sequence_length: Optional[int] = 8192
    learning_rate: float = 5e-5
    vocab_size: Optional[int] = None
    max_grad_norm: Optional[float] = 1.0
    n_gpu: Optional[int] = None
    bf16: Optional[bool] = False
    adam_betas: tuple = (0.9, 0.95)


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
            report_config = self.args.__dict__
            report_config["model_num_parameters"] = self.model_num_parameters

            wandb.init(
                project="voltronformer",
                config=report_config,
            )
        self.accelerator = accelerator

    @property
    def model_num_parameters(self):
        all_param = 0
        for _, param in self._model.named_parameters():
            num_params = param.numel()
            all_param += num_params

        return all_param

    def build_optimizer_and_scheduler(self):
        self.optimizer = AdamWScheduleFree(self._model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, warmup_steps=self.args.weight_decay, eps=self.args.adam_epsilon, betas=self.args.adam_betas)
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
        tr_loss = torch.tensor(0.0).to(self.device)
        total_batched_samples = 0
        for idx, batch in enumerate(pbar := tqdm(self.dataloader, disable=not (self.rank == 0))):
            total_batched_samples += 1
            is_grad_accum_step = total_batched_samples % self.args.gradient_accumulation_steps == 0
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
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                self.accelerator.backward(loss)
                mini_step_loss = loss.detach() / self.args.gradient_accumulation_steps
                tr_loss += mini_step_loss

                if is_grad_accum_step:
                    grad_norm = self.accelerator.clip_grad_norm_(self._model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self._model.zero_grad()

                    if self.accelerator.num_processes > 1:
                        tr_loss_scalar = distributed_concat(tr_loss).mean().item()
                    else:
                        tr_loss_scalar = tr_loss.mean().item()
                    tr_loss -= tr_loss

                    perplexity = torch.exp(tr_loss_scalar)

                    self.global_step += 1

                    if self.global_step % self.args.log_steps == 0:
                        grad_norm = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        if self.rank == 0:
                            pbar.set_description(f"Loss: {tr_loss_scalar} Global Step: {self.global_step} gradient_norm: {grad_norm}")
                            print(f"Loss: {tr_loss_scalar} Global Step: {self.global_step} gradient_norm: {grad_norm}")
                            try:
                                wandb.log({
                                    "training_loss": tr_loss_scalar,
                                    "gradient_norm": grad_norm,
                                    "global_step": self.global_step,
                                    "perplexity": perplexity,
                                }, step=self.global_step)
                            except:
                                pass
                        self.accelerator.log({"training_loss": tr_loss_scalar, "gradient_norm": grad_norm}, step=self.global_step)
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()
                    # TODO Freeze DWA after ~5K-10K steps

        self.accelerator.end_training()


def get_redpajama_v1():
    return load_dataset("togethercomputer/RedPajama-Data-1T", "common_crawl", split="train", streaming=True), "text"

def get_redpajama_v2():
    return load_dataset("togethercomputer/RedPajama-Data-V2",
                        name="default",
                        partition="head_middle",
                        snapshots=["2023-14"],
                        languages=["en"],
                        split="train",
                        trust_remote_code=True,
                        streaming=True,
                        ), "raw_content"


def get_ds(dispatch_batches):
    """
    this is a janky workaround so it doesn't connect to the dataset server unnecessarily
    when using dispatch_batches
    """
    if state.is_main_process or not dispatch_batches:
        return get_redpajama_v2()
    else:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
            f.write("text\n")
            f.write("lorem ipsum dolor sit amet\n")
            # f.writelines(["text", "lorem ipsum dolor sit amet"])
            f.seek(0)
            return load_dataset("csv", data_files={"train": f.name}, split="train"), "text"

    # load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = teeny()
    dispatch_batches = True

    ds, text_field = get_ds(dispatch_batches)

    args = TrainingArguments(
        gradient_accumulation_steps=8,
        max_steps_per_epoch=None,
        log_steps=1,
        adam_epsilon=0.00001,
        output_dir="./out",
        weight_decay=0.1,
        warmup_steps=1000,
        per_gpu_train_batch_size=10,
        save_steps=1000,
        max_sequence_length=config.max_position_embeddings,
        learning_rate=1e-4,
        vocab_size=config.vocab_size,
        n_gpu=state.num_processes,
        bf16=True,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    model = CausalLM(config)
    # model = model.to(device_get_cuda())
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
            buffer_size=40_000,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")

    kwargs_handlers =[]
    # ddp kwargs with find_unused_parameters needed for RMSNormTriton
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # kwargs_handlers.append(ddp_kwargs)

    accelerator_kwargs = {}
    if args.bf16:
        accelerator_kwargs["mixed_precision"] = "bf16"
    accelerator = Accelerator(
        log_with=["wandb", "tensorboard"],
        project_dir="./runs",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dispatch_batches=dispatch_batches,
        kwargs_handlers=kwargs_handlers,
        **accelerator_kwargs,
    )

    dataloader_params = dict(
        batch_size=args.per_gpu_train_batch_size,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2_000,
        drop_last=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=True),
    )
    dataloader = DataLoader(train_dataset, **dataloader_params)

    ### float32 casting for improved accuracy
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
        for name, module in model.named_modules():
            if "layernorm" in name or name == "ln_f":
                module.to(torch.float32)
            elif any(m in name for m in ["wte", "embed_out"]):
                if hasattr(module, "weight"):
                    module.to(torch.float32)
            elif "_proj" in name:
                # module.to(torch.uint8)
                # module.weight.to(torch.float8_e4m3fn)
                pass

    trainer = Trainer(model, args, dataloader, accelerator, activation_checkpointing=True)
    if state.is_main_process:
        print(f"Total number of parameters: {trainer.model_num_parameters:_}")
    trainer.train()


if __name__ == "__main__":
    main()