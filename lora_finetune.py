import argparse
import gc
import os
import random
import threading

import yaml
import psutil
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

from peft import get_peft_model, LoraConfig, TaskType

import logging

from utils.cogvlmsft import ConversationDataset
from utils.roco import ROCO_Dataset
from utils.roco2 import ROCOv2_Dataset
from utils.slake import SLAKE_Dataset
from utils.omnimedqkv import OmniMedVQA_Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def b2mb(x):
    return int(x / 2 ** 20)


class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


def main():
    parser = argparse.ArgumentParser(description="Finetune a CogVLM model with LoRA")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--torch_type", type=str, default="torch.bfloat16", help="Torch type")
    parser.add_argument("--save_step", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--dataset_rate", type=float, default=0.5, help="Proportion of dataset to use")
    parser.add_argument("--train_dataset_rate", type=float, default=0.99, help="Proportion of dataset to use for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank parameter for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_target", type=eval, default=["vision_expert_query_key_value","language_expert_query_key_value"
                                                             ], help="Finetune Target for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=128, help="Maximum output length")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the conversation dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the finetuned model")
    parser.add_argument("--ds_config", type=str, default="ds_config.yaml", help="DeepSpeed configuration file path")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume checkpoint from")
    args = parser.parse_args()
    args.torch_type = eval(args.torch_type)

    with open(args.ds_config) as f:
        ds_config = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_config)

    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=args.torch_type, trust_remote_code=True)

    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    dataset_name = os.path.basename(args.dataset_path.rstrip("/"))
    if dataset_name.lower() == "omnimedvqa":
        dataset = OmniMedVQA_Dataset(
                root_dir=args.dataset_path,
                tokenizer=tokenizer,
                model=model,
                torch_type=args.torch_type,
                input_length=args.max_input_len,
                output_length=args.max_output_len
            )
    elif dataset_name.lower() == "roco":
        dataset = ROCO_Dataset(
                root_dir=args.dataset_path,
                tokenizer=tokenizer,
                model=model,
                torch_type=args.torch_type,
                input_length=args.max_input_len,
                output_length=args.max_output_len
            )
    elif dataset_name.lower() == "rocov2":
        dataset = ROCOv2_Dataset(
                root_dir=args.dataset_path,
                tokenizer=tokenizer,
                model=model,
                torch_type=args.torch_type,
                input_length=args.max_input_len,
                output_length=args.max_output_len
            )
    elif dataset_name.lower() == "slake":
        dataset = SLAKE_Dataset(
                root_dir=args.dataset_path,
                tokenizer=tokenizer,
                model=model,
                torch_type=args.torch_type,
                input_length=args.max_input_len,
                output_length=args.max_output_len
            )
    else:
        dataset = ConversationDataset(
                root_dir=args.dataset_path,
                tokenizer=tokenizer,
                model=model,
                torch_type=args.torch_type,
                input_length=args.max_input_len,
                output_length=args.max_output_len
            )

    total_size = len(dataset)
    subset_size = int(args.dataset_rate * total_size)
    subset_indices = list(range(total_size))
    random.seed(42)
    random.shuffle(subset_indices)

    subset = torch.utils.data.Subset(dataset, subset_indices[:subset_size])

    train_size = int(args.train_dataset_rate * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(subset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        target_modules=args.lora_target,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        accelerator.print(f"Resumed from {args.resume_from}")

    if args.resume_from is not None:
        resume_name = os.path.basename(args.resume_from)
        epoch_str = resume_name.split("_")[2]
        step_str = resume_name.split("_")[4]
        resume_epoch = int(epoch_str)
        resume_step = int(step_str)
    else:
        resume_epoch = 0
        resume_step = 0


    logger.info("Preparation done. Starting training...")
    writer = SummaryWriter(log_dir=args.save_path)

    for epoch in range(args.num_epochs):
        if epoch < resume_epoch:
            continue

        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            if epoch <= resume_epoch and step < resume_step:
                continue
            outputs = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % args.save_step == 0:
                print(f"Epoch {epoch}, Step {step + 1}, Loss {loss.item()}")
                checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}_step_{step + 1}')
                model.save_pretrained(checkpoint_path, safe_serialization=True)
                accelerator.save_state(checkpoint_path)
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_dataloader) + step)

        total_loss = accelerator.gather(total_loss)
        avg_loss = total_loss.mean().item() / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(avg_loss))
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Perplexity', train_ppl, epoch)
        accelerator.print(f"Epoch {epoch}: Average Loss {avg_loss:.4f}, Perplexity {train_ppl:.4f}")

        model.eval()
        eval_loss = 0.0

        for _, batch in enumerate(tqdm(eval_dataloader)):
            inputs = {
                'input_ids': batch['input_ids'],
                'token_type_ids': batch['token_type_ids'],
                'attention_mask': batch['attention_mask'],
                'images': batch['images']
            }
            labels = batch['labels'].to(accelerator.device)

            with torch.no_grad():
                outputs = accelerator.unwrap_model(model)(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'],
                    labels=labels
                )
                loss = outputs.loss
                eval_loss += loss.detach().float()

        eval_loss = accelerator.gather(eval_loss)
        avg_eval_loss = eval_loss.mean().item() / len(eval_dataloader) # 平均 loss
        writer.add_scalar('Eval/Perplexity', torch.exp(torch.tensor(avg_eval_loss)), epoch)
        writer.add_scalar('Eval/Epoch_Loss', avg_eval_loss, epoch)

        final_ckpt_path = os.path.join(args.save_path, 'final_model')
        model.save_pretrained(final_ckpt_path, safe_serialization=True)
        accelerator.save_state(final_ckpt_path)

if __name__ == "__main__":
    main()
