"""Supervised fine-tuning (SFT) of a causal language model on the GSM8K math-word-problem dataset.

The script purposely avoids the 🤗 `Trainer` API and instead implements a minimal but
fully-featured PyTorch training loop with gradient accumulation, validation, checkpointing, and
learning-rate scheduling.

Example usage (single-GPU):

```bash
python sft.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./sft_gsm8k_qwen \
    --epochs 3 \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 5e-5
```

You can resume training from a checkpoint by pointing `--model_name_or_path` to the checkpoint
directory.  If `--output_dir` already contains a best model, the new best will overwrite it.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def format_example(example: Dict[str, str]) -> tuple[str, str]:
    """Turn a GSM8K example into an instruction-style prompt.

    We prepend the answer with a direct cue so that the model learns to respond with the
    correct solution when given the question.
    """

    question: str = example["question"].strip()
    answer: str = example["answer"].strip()

    prompt = f"Question: {question}\nAnswer:"
    return prompt, answer


def preprocess_function(
    example: Dict[str, str], tokenizer: AutoTokenizer, max_length: int
) -> Dict[str, List[int]]:
    """Tokenise the prompt-answer pair and build loss masks.

    Tokens corresponding to the prompt (i.e. everything up to *and including* the \nAnswer: cue)
    receive a label of -100 so they do not contribute to the cross-entropy loss.  Tokens for the
    answer retain their IDs so the model is supervised to generate them.
    """

    prompt, answer = format_example(example)
    full_text = f"{prompt} {answer}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


@dataclass
class DataCollatorForCausalLM:
    """Pads a batch for causal LM training."""

    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"][: self.max_length] for f in features]
        labels = [f["labels"][ : self.max_length] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "labels": labels},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        return batch


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:  # noqa: C901 – keep this flat for readability
    parser = argparse.ArgumentParser(description="Supervised fine-tuning on GSM8K without Trainer")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./sft_gsm8k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Model & tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    # ---------------- Dataset ----------------
    raw_datasets = load_dataset("openai/gsm8k", "main")

    preprocess = lambda ex: preprocess_function(ex, tokenizer, args.max_len)  # noqa: E731

    tokenized_train = raw_datasets["train"].map(
        preprocess,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=os.cpu_count() or 1,
    )
    tokenized_val = raw_datasets["test"].map(
        preprocess,
        remove_columns=raw_datasets["test"].column_names,
        num_proc=os.cpu_count() or 1,
    )

    collator = DataCollatorForCausalLM(tokenizer, args.max_len)

    train_loader = DataLoader(
        tokenized_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        tokenized_val, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    # ---------------- Optimiser & scheduler ----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    # ---------------- Training loop ----------------
    global_step = 0
    best_val_loss = float("inf")

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0

        optimizer.zero_grad()
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.accumulation_steps
            loss.backward()

            running_loss += loss.item()

            if step % args.accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % args.logging_steps == 0:
                    pbar.set_postfix({"train_loss": running_loss / args.logging_steps})
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

        # ---------------- Validation ----------------
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: validation loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"New best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()