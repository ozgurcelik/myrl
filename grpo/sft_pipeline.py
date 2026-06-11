"""
SFT pipeline.

Here, we prepare the dataset and train our model with supervised fine-tuning.

For SFT, we use the Asap7772/cog_behav_all_strategies dataset. It is a
prompt-completion dataset with two columns:
  - query:      the conversation prompt up to "Assistant: Let me solve this step by step."
  - completion: the assistant's reasoning (in <think> tags) and final <answer>.

Example query:
    A conversation between User and Assistant. The user asks a question, and the
    Assistant solves it. The assistant first thinks about the reasoning process in
    the mind and then provides the user with the answer.
    User: Using the numbers [95, 36, 32], create an equation that equals 91. ...
    Assistant: Let me solve this step by step.

As the model, we use the Qwen 2.5 0.5B model.

General parameters of the training:
- Epochs: 3
- Learning Rate: 2e-5
- Optimizer: AdamW
- Scheduler: Linear schedule with warmup

We use the trl library to train the model and wandb for logging.

Config is hard-coded for a single RTX 4090 (24 GB).
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Data / model.
DATASET_NAME = "Asap7772/cog_behav_all_strategies"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./sft-qwen2.5-0.5b"

# Training hyperparameters (from the spec).
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_LENGTH = 1024

# (effective batch = 8 * 4 = 32).
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
SEED = 42

# Weights & Biases.
WANDB_PROJECT = "grpo-countdown"
WANDB_ENTITY = "dirtem1998"
RUN_NAME = "sft-qwen2.5-0.5b"


def build_dataset():
    """Load the dataset and normalize it into TRL's prompt-completion format.

    TRL expects the columns to be named "prompt" and "completion"; the raw
    dataset names the prompt column "query".
    """
    dataset = load_dataset(DATASET_NAME)

    if "query" in dataset["train"].column_names:
        dataset = dataset.rename_column("query", "prompt")

    return dataset


def main():
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    dataset = build_dataset()
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        # AdamW optimizer with a linear schedule + warmup.
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=True,
        seed=SEED,
        report_to="wandb",
        run_name=RUN_NAME,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
