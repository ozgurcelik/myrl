# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from countdown_task import *
from reward import *
from grpo_helpers import *

import wandb
import os

# %%
def train(
        model_id: str,
        dataset: Dataset,
        train_batch_size: int = 4,
        test_batch_size: int = 8,
        num_epochs: int = 100,
        batch_per_epoch: int = 10,
        update_freq: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        num_return_sequences: int = 4,
        epsilon: float = 0.2,
        learning_rate: float = 1e-5,
        test_freq: int = 5,
        use_wandb: bool = True,
        wandb_entity: str = "erencetin",
        wandb_project: str = "grpo-countdown",
        wandb_run_name: Optional[str] = None,
        use_curriculum: bool = True,
        curriculum_phase1_end: int = 50,
        curriculum_phase2_end: int = 200,
        ):
    """
    Main GRPO (Group Relative Policy Optimization) training loop.
    
    GRPO is a reinforcement learning algorithm for fine-tuning language models.
    It's similar to PPO but uses a clever trick: instead of training a value network
    to estimate baselines, it generates multiple samples per prompt and uses the
    mean reward within each group as the baseline. This simplifies training while
    maintaining the benefits of variance reduction.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        dataset: Dataset class (not instance) that will be instantiated for train/test.
        train_batch_size: Number of prompts per batch for training.
        test_batch_size: Number of prompts per batch for testing.
        num_epochs: Total number of training epochs.
        batch_per_epoch: Number of batches to process per epoch.
        update_freq: Number of gradient updates per rollout (reuses same samples).
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature for generation (higher = more random).
        num_return_sequences: Number of completions per prompt (group size for GRPO).
        epsilon: PPO clipping parameter (0.2 means clip ratio to [0.8, 1.2]).
        learning_rate: Learning rate for AdamW optimizer.
        test_freq: Evaluate on test set every N epochs.
        use_wandb: Whether to log metrics to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (optional).
        use_curriculum: Whether to use curriculum learning based on difficulty.
        curriculum_phase1_end: Global step at which phase 1 ends (only difficulty 2).
        curriculum_phase2_end: Global step at which phase 2 ends (difficulty 2 and 3).
    
    Returns:
        tuple of:
            - model: The trained model
            - avg_losses: List of average losses per epoch
            - avg_rewards: List of average rewards per epoch
            - eval_rewards: List of (epoch, test_reward) tuples from evaluation
    
    Training Loop Structure:
    
    For each epoch:
        1. **Evaluation** (every test_freq epochs):
           - Generate greedy completions (temperature ≈ 0)
           - Compute rewards to track model improvement
        
        2. **Training** (batch_per_epoch times):
           a. **Rollout Generation**:
              - Sample batch of prompts
              - Generate num_return_sequences completions per prompt
              - Compute rewards and GRPO advantages (group normalization)
              - Cache log probs under current policy and reference model
           
           b. **Policy Updates** (update_freq times):
              - Recompute log probs under CURRENT policy (with gradients)
              - Compute GRPO loss (clipped surrogate + KL penalty)
              - Backpropagate and update model parameters
              - Apply gradient clipping for stability
    
    Curriculum Learning (when use_curriculum=True):
        - Phase 1 (steps 0 to phase1_end-1): Only difficulty 2 samples
        - Phase 2 (steps phase1_end to phase2_end-1): Difficulty 2 and 3 samples
        - Phase 3 (steps phase2_end+): All difficulties (2, 3, 4)
        - Samples used in earlier phases are not reused until full dataset loop
    
    Key GRPO Components:
        - **Group-based advantages**: No value network needed
        - **Clipped surrogate objective**: Stable policy updates (from PPO)
        - **KL penalty**: Prevents drift from reference model
        - **Multiple updates per rollout**: Better sample efficiency
    
    Memory Management:
        - Reference model is frozen (no gradients stored)
        - MPS cache cleared periodically on Apple Silicon
        - Gradient clipping prevents exploding gradients
    
    Example:
        >>> model, losses, rewards, eval_rewards = train(
        ...     model_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...     dataset=CountdownTaskDataset,
        ...     batch_size=4,
        ...     num_epochs=50,
        ...     num_return_sequences=4,
        ... )
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")

    # Use bfloat16 for balance between stability and memory (float16 on MPS can cause NaN issues)
    # dtype = torch.bfloat16 if device.type not in ("mps", "cuda") else torch.float32
    dtype = torch.bfloat16
    # For MPS, we need to be careful with memory - use smaller batch or gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
    ref_model.eval()  # Reference model should always be in eval mode
    for param in ref_model.parameters():
        param.requires_grad = False  # Freeze reference model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    run = None
    if use_wandb:
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_id": model_id,
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                "num_epochs": num_epochs,
                "batch_per_epoch": batch_per_epoch,
                "update_freq": update_freq,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "num_return_sequences": num_return_sequences,
                "epsilon": epsilon,
                "learning_rate": learning_rate,
                "test_freq": test_freq,
                "device": str(device),
                "dtype": str(dtype),
                "use_curriculum": use_curriculum,
                "curriculum_phase1_end": curriculum_phase1_end,
                "curriculum_phase2_end": curriculum_phase2_end,
            },
        )

    ds_train = dataset(
        tokenizer=tokenizer,
        split="train",
    )
    ds_test = dataset(
        tokenizer=tokenizer,
        split="test",
    )

    # Create curriculum sampler if enabled
    curriculum_sampler = None
    if use_curriculum:
        curriculum_sampler = CurriculumSampler(
            dataset=ds_train,
            batch_size=train_batch_size,
            phase1_end=curriculum_phase1_end,
            phase2_end=curriculum_phase2_end,
        )
        dl_train = DataLoader(
            ds_train,
            batch_sampler=curriculum_sampler,
            collate_fn=ds_train.collate_fn,
        )
    else:
        dl_train = DataLoader(
            ds_train,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=ds_train.collate_fn,
        )
    
    dl_test = DataLoader(
        ds_test,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=ds_test.collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    avg_losses = []
    avg_rewards = []
    eval_rewards = []
    global_step = 0
    for epoch in range(num_epochs):

        # Evaluate on test set every 5 epochs
        if (epoch + 1) % test_freq == 0 or epoch == num_epochs - 1 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(dl_test))
                test_completion_output = generate_responses(
                    model,
                    tokenizer,
                    test_batch,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,  # Greedy, so only 1 sequence
                    temperature=1e-8,  # Near-zero for greedy (avoid division by zero)
                    device=device,
                )
                test_generated_answers = tokenizer.batch_decode(
                    test_completion_output["sequences"][:, -test_completion_output["completion_len"]:],
                    skip_special_tokens=True
                )
                test_reward_result = calculate_rewards(
                    test_generated_answers,
                    num_return_sequences=1,
                    numbers=test_batch.numbers,
                    targets=test_batch.target,
                    end_token=tokenizer.eos_token,
                )
                test_rewards = test_reward_result["rewards"]
                test_reward_info = test_reward_result["reward_info"]
            avg_test_reward = test_rewards.mean().item()
            eval_rewards.append((epoch + 1, avg_test_reward))
            print(f"  [EVAL] Epoch {epoch+1}: Test Reward (greedy): {avg_test_reward:.4f}")

            if use_wandb:
                eval_log = {
                    "eval/test_reward_greedy": avg_test_reward,
                    "epoch": epoch + 1,
                }
                # Log detailed reward breakdown for evaluation
                for key, values in test_reward_info.items():
                    eval_log[f"eval/{key}_mean"] = values.mean().item()
                wandb.log(eval_log, step=global_step)


        model.train()
        total_loss = 0.0
        total_reward = 0.0
        for _ in range(batch_per_epoch):
            # Clear MPS cache to prevent memory buildup
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

            # Update curriculum sampler with current global step before fetching batch
            if curriculum_sampler is not None:
                curriculum_sampler.set_global_step(global_step)
                current_phase = "Phase 1 (diff=2)" if global_step < curriculum_phase1_end else \
                               "Phase 2 (diff=2,3)" if global_step < curriculum_phase2_end else \
                               "Phase 3 (all)"
                if global_step % 50 == 0:  # Log curriculum phase periodically
                    print(f"  [CURRICULUM] Step {global_step}: {current_phase}")

            batch = next(iter(dl_train))
            print("Batch[0]:", batch.numbers[0])
            rollout_output = generate_rollouts(
                model,
                ref_model,
                tokenizer,
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                device=device,
                debug=(epoch == 0),  # Debug first batch
            )
            rewards = rollout_output["rewards"]
            avg_reward_per_batch = rewards.mean().item()
            total_reward += avg_reward_per_batch

            if use_wandb:  # first batch of epoch
                # Build a small table with detailed reward breakdown
                # Attach correct numbers/targets per expanded sample
                # (because you repeat each prompt num_return_sequences times)
                expanded_nums = []
                expanded_tgts = []
                for j in range(len(batch.numbers)):
                    for __ in range(num_return_sequences):
                        expanded_nums.append(batch.numbers[j])
                        expanded_tgts.append(batch.target[j])

                table = wandb.Table(columns=[
                    "numbers", "target", "response", "reward",
                    "format_reward", "answer_reward", "number_usage_reward", "correctness_reward"
                ])
                # Log up to 8 samples
                k = min(8, len(rollout_output["generated_answers"]))
                reward_info = rollout_output["reward_info"]
                for i in range(k):
                    table.add_data(
                        str(expanded_nums[i]),
                        int(expanded_tgts[i]),
                        rollout_output["generated_answers"][i],
                        float(rollout_output["rewards"][i].item()),
                        float(reward_info["format_reward"][i].item()),
                        float(reward_info["answer_reward"][i].item()),
                        float(reward_info["number_usage_reward"][i].item()),
                        float(reward_info["correctness_reward"][i].item()),
                    )

                wandb.log({"samples/train_generations": table, "train/epoch": epoch + 1}, step=global_step)

            if use_wandb:
                # Log curriculum phase
                if curriculum_sampler is not None:
                    curriculum_phase = 1 if global_step < curriculum_phase1_end else \
                                      2 if global_step < curriculum_phase2_end else 3
                    wandb.log(
                        {"train/curriculum_phase": curriculum_phase},
                        step=global_step,
                    )
                # Log total rewards and detailed breakdown
                train_log = {
                    "train/rewards": rewards.tolist(),
                    "train/reward_mean": avg_reward_per_batch,
                    "train/epoch": epoch + 1,
                }
                # Log detailed reward breakdown
                reward_info = rollout_output["reward_info"]
                for key, values in reward_info.items():
                    train_log[f"train/{key}_mean"] = values.mean().item()
                    train_log[f"train/{key}_list"] = values.tolist()
                wandb.log(train_log, step=global_step)

            for update_idx in range(update_freq):
                completion_mask = rollout_output["attention_mask"][:, -rollout_output["completion_len"]:]
                model.train()
                new_token_log_probs = compute_log_probs(
                    model,
                    rollout_output["sequences"],
                    rollout_output["attention_mask"],
                    rollout_output["completion_len"],
                    temperature=temperature,
                    device=device,
                    training=True,
                )
                loss, kl_loss = grpo_loss(
                    new_token_log_probs,
                    rollout_output["old_token_log_probs"],
                    rollout_output["ref_token_log_probs"],
                    rollout_output["advantages"],
                    completion_mask,
                    epsilon=epsilon,
                    device=device,
                )
                
                # Skip update if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [WARNING] Skipping update due to NaN/Inf loss")
                    continue
                    
                optimizer.zero_grad()
                loss.backward()
                global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
                # print(f"  [DEBUG] gradient norm: {global_norm:.6f}")
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/kl_loss": float(kl_loss),
                            "train/grad_norm_preclip": float(global_norm),
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                global_step += 1

        avg_loss = total_loss / (batch_per_epoch * update_freq)
        avg_reward = total_reward / batch_per_epoch
        avg_losses.append(avg_loss)
        avg_rewards.append(avg_reward)
        print(f"Epoch {epoch+1}/{num_epochs}\n \tLoss: {avg_loss:.4f}\n \tReward: {avg_reward:.4f}")

    return model, avg_losses, avg_rewards, eval_rewards

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    trained_model, losses, rewards, eval_rewards = train(
        model_id=model_id,
        dataset=CountdownTaskDataset,
        train_batch_size=2,  # Reduced for memory
        test_batch_size=8,  # Reduced for memory
        num_epochs=100,
        batch_per_epoch=1,
        update_freq=2,
        max_new_tokens=256,  # Reduced for memory
        temperature=0.6,
        num_return_sequences=2,  # Reduced for memory
        epsilon=0.2,
        learning_rate=5e-6,
        test_freq=5,
        use_wandb=True,
        wandb_entity="dirtem1998",
        wandb_project="metre_tests",
        wandb_run_name="grpo-test-run",
    )
