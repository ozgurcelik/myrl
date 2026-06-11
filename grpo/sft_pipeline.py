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

For testing, we will use the ozgur-celik/countdown_cl dataset.
Specifically, we will take first 1000 examples of the test split.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from countdown_task import build_prompt
from reward import reward_function

# Data / model.
DATASET_NAME = "Asap7772/cog_behav_all_strategies"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./sft-qwen2.5-0.5b"

# Test set: first 1000 examples of the countdown_cl test split. Each epoch we
# generate completions for these prompts and measure the correctness score
# (reward.py -> correctness_reward).
TEST_DATASET_NAME = "ozgur-celik/countdown_cl"
NUM_TEST_EXAMPLES = 1000
EVAL_BATCH_SIZE = 64
EVAL_MAX_NEW_TOKENS = 512

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

    The raw dataset has a `query` (prompt) and a `completion` whose first token
    is the opening `<think>` tag. We move that `<think>` into the prompt so the
    training prompt matches build_prompt() byte-for-byte (the format used for
    evaluation and GRPO). This is exactness-preserving: prompt + completion is
    identical to the original query + completion.
    """
    dataset = load_dataset(DATASET_NAME)

    def to_prompt_completion(example):
        completion = example["completion"]
        if completion.startswith("<think>"):
            completion = completion[len("<think>") :]
        return {"prompt": example["query"] + "<think>", "completion": completion}

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset["train"].column_names,
    )
    return dataset


def build_test_examples():
    """Build the countdown test prompts (first NUM_TEST_EXAMPLES of the test split).

    Uses the shared build_prompt() so the evaluation prompt is byte-for-byte
    identical to the SFT training prompt (and the upcoming GRPO prompt). We keep
    the raw numbers/target around so we can score correctness after generation.
    """
    data = load_dataset(TEST_DATASET_NAME)["test"]
    data = data.select(range(min(NUM_TEST_EXAMPLES, len(data))))

    examples = []
    for item in data:
        prompt = build_prompt(item["nums"], item["target"])
        examples.append(
            {"prompt": prompt, "numbers": item["nums"], "target": item["target"]}
        )
    return examples


class CorrectnessEvalCallback(TrainerCallback):
    """At the end of each epoch, generate on the test set and log the mean
    correctness score (fraction of prompts whose answer evaluates to the target)
    to Weights & Biases."""

    def __init__(self, tokenizer, examples):
        self.tokenizer = tokenizer
        self.examples = examples

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        was_training = model.training
        model.eval()
        # Generation needs the KV cache, which is disabled under gradient
        # checkpointing during training.
        prev_use_cache = model.config.use_cache
        model.config.use_cache = True
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        scores = []
        for start in range(0, len(self.examples), EVAL_BATCH_SIZE):
            batch = self.examples[start : start + EVAL_BATCH_SIZE]
            prompts = [b["prompt"] for b in batch]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            responses = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

            for response, b in zip(responses, batch):
                info = reward_function(
                    response, numbers=b["numbers"], target=b["target"]
                )["reward_info"]
                scores.append(info["correctness_reward"])

        score = sum(scores) / len(scores) if scores else 0.0

        self.tokenizer.padding_side = prev_padding_side
        model.config.use_cache = prev_use_cache
        if was_training:
            model.train()

        print(f"[epoch {state.epoch:.0f}] countdown correctness score = {score:.4f}")
        if state.is_world_process_zero:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {"eval/correctness_score": score, "epoch": state.epoch},
                    step=state.global_step,
                )


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

    test_examples = build_test_examples()

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
        callbacks=[CorrectnessEvalCallback(tokenizer, test_examples)],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
