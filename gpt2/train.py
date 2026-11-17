#!/usr/bin/env python
"""
Train GPT-2 (nanoGPT, from scratch) on a causal language-modeling dataset with the dual Banach regularizer.

Derived from the HuggingFace example ``run_clm_no_trainer.py`` (Apache 2.0): the dataset is tokenized with the
GPT-2 tokenizer and concatenated into blocks of ``--block_size`` tokens; the model is trained with
AdamWScheduleFree and evaluated (perplexity) after every epoch.

    python train.py --reg 1.0 --seed 0 --output_dir ../../experiments/gpt2_wikitext103_reg1_seed0
"""

import argparse
import json
import logging
import math
import os
import random
from itertools import chain

import numpy as np
import torch
from datasets import load_dataset
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, default_data_collator

from model import GPT, GPTConfig

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 causal language modeling with the dual Banach regularizer")
    # data
    p.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    p.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--validation_split_percentage", type=int, default=5,
                   help="used only if the dataset has no validation split")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--block_size", type=int, default=1024, help="sequence length")
    p.add_argument("--preprocessing_num_workers", type=int, default=None)
    p.add_argument("--overwrite_cache", action="store_true")
    # model
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--word_dropout", type=float, default=0.0,
                   help="probability of replacing an input token by <|endoftext|>")
    # regularizer
    p.add_argument("--reg", type=float, default=0.0, help="weight lambda of L_DB (0 = baseline)")
    p.add_argument("--kappa_mode", type=str, default="full", choices=["full", "one", "detach"],
                   help="ablation of the amplification factor kappa: full | one (kappa = 1) | detach (stop-gradient)")
    # optimization
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_warmup_steps", type=int, default=2000)
    p.add_argument("--num_train_epochs", type=int, default=20)
    p.add_argument("--max_train_steps", type=int, default=None, help="if given, overrides --num_train_epochs")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--checkpointing_steps", type=str, default=None,
                   help="'epoch' saves a checkpoint per epoch; an integer N saves every N steps")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="checkpoint file to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(args, tokenizer):
    raw = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in raw:
        raw["validation"] = load_dataset(args.dataset_name, args.dataset_config_name,
                                         split=f"train[:{args.validation_split_percentage}%]")
        raw["train"] = load_dataset(args.dataset_name, args.dataset_config_name,
                                    split=f"train[{args.validation_split_percentage}%:]")
    column_names = raw["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]
    tokenized = raw.map(lambda ex: tokenizer(ex[text_column]), batched=True, num_proc=args.preprocessing_num_workers,
                        remove_columns=column_names, load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on dataset")
    block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm = tokenized.map(group_texts, batched=True, num_proc=args.preprocessing_num_workers,
                       load_from_cache_file=not args.overwrite_cache, desc=f"Grouping texts in chunks of {block_size}")
    return lm["train"], lm["validation"], block_size


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, loss, _ = model(batch["input_ids"], batch["labels"])
        n = batch["labels"].ne(-100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    eval_loss = total_loss / total_tokens
    return eval_loss, math.exp(eval_loss)


def save_checkpoint(path, model, optimizer, args, epoch, step, perplexity):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(model.config), "args": vars(args), "epoch": epoch, "completed_steps": step,
                "perplexity": perplexity}, path)


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_dataset, eval_dataset, block_size = build_datasets(args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                              batch_size=args.per_device_train_batch_size)
    eval_loader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    config = GPTConfig(block_size=block_size, vocab_size=50304, n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, dropout=args.dropout, bias=True, word_dropout=args.word_dropout)
    device = torch.device(args.device)
    model = GPT(config, reg=args.reg > 0, kappa_mode=args.kappa_mode).to(device)
    optimizer = AdamWScheduleFree(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                  warmup_steps=args.num_warmup_steps)
    step_fn = torch.compile(model) if args.compile else model

    steps_per_epoch = len(train_loader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    completed_steps, starting_epoch = 0, 0
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        starting_epoch, completed_steps = ckpt["epoch"] + 1, ckpt["completed_steps"]
        logger.info(f"resumed from {args.resume_from_checkpoint} (epoch {ckpt['epoch']}, step {completed_steps})")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}  Num epochs = {args.num_train_epochs}  "
                f"batch size = {args.per_device_train_batch_size}  total steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), initial=completed_steps)
    log_path = os.path.join(args.output_dir, "training_log.txt")
    best_ppl, final_ppl = float("inf"), None

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        optimizer.train()
        ce_sum, total_sum, n_steps = 0.0, 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, ce, reg = step_fn(batch["input_ids"], batch["labels"])
            loss = ce + args.reg * reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ce_sum += ce.detach().float()
            total_sum += loss.detach().float()
            progress_bar.update(1)
            completed_steps += 1
            n_steps += 1
            if args.checkpointing_steps is not None and args.checkpointing_steps.isdigit() \
                    and completed_steps % int(args.checkpointing_steps) == 0:
                save_checkpoint(os.path.join(args.output_dir, f"step_{completed_steps}.pt"), model, optimizer, args,
                                epoch, completed_steps, None)
            if completed_steps >= args.max_train_steps:
                break

        optimizer.eval()
        eval_loss, perplexity = evaluate(model, eval_loader, device)
        best_ppl, final_ppl = min(best_ppl, perplexity), perplexity
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        with open(log_path, "a") as f:
            f.write(f"epoch: {epoch}, step: {completed_steps}, train_loss: {ce_sum / n_steps:.6f}, "
                    f"total_train_loss: {total_sum / n_steps:.6f}, eval_loss: {eval_loss:.6f}, "
                    f"perplexity: {perplexity:.6f}\n")
        save_checkpoint(os.path.join(args.output_dir, "pytorch_model.bin"), model, optimizer, args, epoch,
                        completed_steps, perplexity)
        if args.checkpointing_steps == "epoch":
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch}.pt"), model, optimizer, args, epoch,
                            completed_steps, perplexity)
        if completed_steps >= args.max_train_steps:
            break

    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump({"best_perplexity": best_ppl, "final_perplexity": final_ppl}, f, indent=2)
    logger.info(f"best perplexity {best_ppl:.4f}  final perplexity {final_ppl:.4f}  (saved to {args.output_dir})")


if __name__ == "__main__":
    main()
