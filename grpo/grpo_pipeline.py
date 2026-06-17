"""GRPO training for the Countdown task using TRL's GRPOTrainer with vLLM rollout.

vLLM generates the rollouts; TRL does the backprop. Two modes (VLLM_MODE):
  - "colocate" (default): vLLM runs inline in this process, sharing one GPU.
        Single command:  python grpo_pipeline.py
  - "server": vLLM runs as a separate process on its own GPU(s). Start it first:
        # GPU 0 — rollout server
        HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
            --model Qwen/Qwen2.5-1.5B-Instruct --enforce_eager
        # GPU 1 — trainer
        CUDA_VISIBLE_DEVICES=1 VLLM_MODE=server python grpo_pipeline.py

Reuses the existing task/reward code unchanged:
  - countdown_task.build_prompt  -> plain-text prompt prefix (ends with <think>)
  - reward.reward_function       -> 0.1*format + 0.1*number_usage + 0.8*correctness

All knobs are env-overridable (see CONFIG below).
"""

import os

# Pod-specific fixes (safe defaults; export your own to override).
# The hf-xet download backend hangs here, so force plain HTTPS.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# /dev/shm is mounted noexec; never let JIT-compiled kernels land there.
if os.environ.get("TMPDIR", "").startswith("/dev/shm"):
    os.environ["TMPDIR"] = "/tmp"

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from countdown_task import build_prompt
from reward import reward_function


# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET = os.environ.get("DATASET", "ozgur-celik/countdown_cl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/grpo-countdown")
VLLM_MODE = os.environ.get("VLLM_MODE", "colocate")  # "colocate" | "server"


def vram_defaults() -> dict:
    """Pick batch / generation sizes from the training GPU's VRAM.

    In server mode this process only sees the training GPU(s); in colocate mode
    it shares the GPU with vLLM, so we stay conservative there.
    """
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    colocate = VLLM_MODE == "colocate"
    if gb < 24 or colocate:
        return dict(num_generations=4, per_device_train_batch_size=4,
                    gradient_accumulation_steps=4, gradient_checkpointing=True,
                    max_completion_length=512)
    if gb < 48:
        return dict(num_generations=8, per_device_train_batch_size=8,
                    gradient_accumulation_steps=2, gradient_checkpointing=True,
                    max_completion_length=640)
    return dict(num_generations=8, per_device_train_batch_size=8,
                gradient_accumulation_steps=2, gradient_checkpointing=False,
                max_completion_length=768)


def env_int(key, default):
    return int(os.environ.get(key, default))


def env_float(key, default):
    return float(os.environ.get(key, default))


# --------------------------------------------------------------------------- #
# Reward (TRL signature). TRL forwards extra dataset columns (nums, target) as
# keyword args and expects a list[float], one reward per completion.
# --------------------------------------------------------------------------- #
def countdown_reward(completions, nums, target, **kwargs):
    return [
        reward_function(completion, numbers=n, target=t)["reward"]
        for completion, n, t in zip(completions, nums, target)
    ]


def build_train_dataset():
    ds = load_dataset(DATASET, split="train")
    # Add the prompt column; keep nums/target so the reward fn receives them.
    return ds.map(lambda ex: {"prompt": build_prompt(ex["nums"], ex["target"])})


def build_config() -> GRPOConfig:
    d = vram_defaults()

    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        # optimization
        learning_rate=env_float("LR", 1e-6),
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        max_grad_norm=0.2,
        max_steps=env_int("MAX_STEPS", 500),
        per_device_train_batch_size=env_int("TRAIN_BS", d["per_device_train_batch_size"]),
        gradient_accumulation_steps=env_int("GRAD_ACCUM", d["gradient_accumulation_steps"]),
        gradient_checkpointing=os.environ.get("GRAD_CKPT", str(d["gradient_checkpointing"])).lower() == "true",
        bf16=True,
        # GRPO specifics
        num_generations=env_int("NUM_GEN", d["num_generations"]),
        max_completion_length=env_int("MAX_COMPLETION", d["max_completion_length"]),
        temperature=env_float("TEMPERATURE", 1.0),
        top_p=env_float("TOP_P", 1.0),
        beta=env_float("KL_BETA", 0.0),          # KL penalty; 0.0 = pure GRPO
        scale_rewards=True,
        # vLLM rollout
        use_vllm=True,
        vllm_mode=VLLM_MODE,
        # logging / checkpoints
        logging_steps=1,
        log_completions=True,
        save_steps=env_int("SAVE_STEPS", 100),
        report_to=os.environ.get("REPORT_TO", "none"),  # set "wandb" to log
    )

    if VLLM_MODE == "server":
        config_kwargs.update(
            vllm_server_host=os.environ.get("VLLM_HOST", "127.0.0.1"),
            vllm_server_port=env_int("VLLM_PORT", 8000),
            vllm_server_timeout=env_float("VLLM_TIMEOUT", 600.0),
        )
    else:  # colocate: vLLM shares the training GPU, so cap its memory share
        config_kwargs.update(
            vllm_gpu_memory_utilization=env_float("VLLM_GPU_UTIL", 0.3),
        )

    return GRPOConfig(**config_kwargs)


def main():
    args = build_config()

    print(f"[grpo] model={MODEL_ID} mode={VLLM_MODE} "
          f"num_gen={args.num_generations} bs={args.per_device_train_batch_size} "
          f"grad_accum={args.gradient_accumulation_steps} "
          f"max_completion={args.max_completion_length} bf16={args.bf16}")

    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=args,
        train_dataset=build_train_dataset(),
        reward_funcs=countdown_reward,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"[grpo] done. model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
