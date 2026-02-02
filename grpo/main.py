# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
import numpy as np
from typing import Optional
from countdown_task import *

import wandb
import os

# %%
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: MiniBatch,
    max_new_tokens: int = 512,
    num_return_sequences: int = 3,
    temperature: float = 0.6,
    device: torch.device = torch.device("cpu"),
    ):
    """
    Generate multiple completion sequences for each input prompt in the batch.
    
    This function performs autoregressive sampling from the model to generate
    completions. For each input prompt, it generates `num_return_sequences` 
    different completions by sampling from the model's probability distribution.
    
    Args:
        model: The causal language model to generate from.
        tokenizer: Tokenizer matching the model, used for pad/eos token IDs.
        batch: MiniBatch containing input_ids, attention_mask, numbers, and targets.
        max_new_tokens: Maximum number of tokens to generate per sequence.
        num_return_sequences: Number of completions to generate per input prompt.
        temperature: Sampling temperature. Higher = more random, lower = more deterministic.
        device: Device to run generation on (cpu, cuda, mps).
    
    Returns:
        dict containing:
            - sequences: [B * num_return_sequences, L_total] - Full sequences (prompt + completion)
            - completion_len: int - Length of the generated completion part
            - total_len: int - Total sequence length (prompt + completion)
            - attention_mask: [B * num_return_sequences, L_total] - Mask with 1s for real tokens
    
    How it works:
        1. Calls model.generate() with sampling enabled (do_sample=True)
        2. Creates a completion_mask that marks valid completion tokens (up to and including EOS)
        3. Expands the input attention mask to match the repeated sequences
        4. Concatenates input mask with completion mask for full attention mask
    """
    model.to(device)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    # Generate completions from the model
    # Note: batch.input_ids are LEFT-PADDED (done in collate_fn with padding_side="left")
    # Left-padding ensures all prompts END at the same position, so generation
    # starts from a consistent point. Example:
    #   Prompt 1 (short): [PAD][PAD][PAD] "What is 2+2?"
    #   Prompt 2 (long):  "Calculate the sum of two and two"
    #                                                       ↑ Generation starts here for both
    outputs = model.generate(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    
    input_ids = batch.input_ids # [B_in, L_input]
    input_attention_mask = batch.attention_mask

    total_len = outputs.sequences.shape[1]
    input_len = input_ids.shape[1]
    completion_len = total_len - input_len

    # Build completion_mask: marks valid completion tokens (RIGHT-PADDED)
    # Different sequences finish at different times (some hit EOS early, others generate max_new_tokens)
    # Real content comes first, padding (mask=0) comes at the end.
    # Example:
    #   Sequence 1: [generated...][EOS][PAD][PAD][PAD]  -> mask: [1,1,1,1,0,0,0]
    #   Sequence 2: [generated tokens...........][EOS]  -> mask: [1,1,1,1,1,1,1]
    completion_mask = torch.zeros(outputs.sequences.shape[0], completion_len, dtype=torch.int64) # [B, L_completion]
    for i in range(len(outputs.sequences)):
        temp = outputs.sequences[i, input_len:]
        eos_pos = (temp == eos_token_id).nonzero()
        if len(eos_pos) == 0:
            # No EOS found - all tokens are valid (may have hit max_new_tokens)
            completion_mask[i, :] = 1
        else:
            # Mark tokens up to and including EOS as valid
            completion_mask[i, :eos_pos[0]+1] = 1
    
    # Expand input attention mask to match num_return_sequences
    input_attention_mask = torch.repeat_interleave(input_attention_mask, num_return_sequences, dim=0) # [B, L_input]
    
    # Concatenate to form full attention mask:
    # [LEFT-PADDED input mask | RIGHT-PADDED completion mask]
    # Result: [0, 0, 1, 1, 1, 1, 1, 1 | 1, 1, 1, 1, 0, 0, 0]
    #          └── left padding ──┘   └── real ──┘ └─ right padding ─┘
    attention_mask = torch.cat([input_attention_mask, completion_mask], dim=1) # [B, L]

    return {
        "sequences": outputs.sequences,
        "completion_len": completion_len,
        "total_len": total_len,
        "attention_mask": attention_mask
    }

def compute_log_probs(
    model: AutoModelForCausalLM,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    temperature: float = 0.6,
    device: torch.device = torch.device("cpu"),
    training: bool = False,
    ) -> torch.Tensor:
    """
    Compute per-token log probabilities for the completion portion of sequences.
    
    This is a core function for policy gradient methods. It computes how likely
    each token in the completion was under the model's distribution, which is
    needed for computing policy gradients and importance sampling ratios.
    
    Args:
        model: The causal language model to compute probabilities from.
        sequences: [B, L_total] - Full sequences (prompt + completion tokens).
        attention_mask: [B, L_total] - Attention mask for the sequences.
        completion_len: Length of the completion portion to compute log probs for.
        temperature: Temperature to apply to logits before softmax.
        device: Device to run computation on.
        training: If True, keeps gradients for backprop. If False, uses no_grad.
    
    Returns:
        torch.Tensor: [B, L_completion] - Log probability of each completion token,
                      masked by the completion_mask (0 for padding tokens).
    
    How it works:
        1. Forward pass through model to get logits [B, L, V]
        2. Extract completion logits (shifted by 1 since logits[i] predicts token[i+1])
        3. Apply temperature scaling to logits
        4. Compute log_softmax to get log probabilities over vocabulary
        5. Gather the log prob of the actual token at each position
        6. Clamp extremely negative values to prevent NaN in later computations
        7. Apply completion mask to zero out padding positions
    
    Note:
        The logit shift is crucial: model outputs logits where logits[:, i, :] 
        predicts the probability of token[:, i+1]. So we take logits[:, :-1, :]
        and match against tokens[:, 1:] (or in our case, the completion tokens).
    """
    model.to(device)
    
    if training:
        logits = model(
            sequences.to(device),
            attention_mask=attention_mask.to(device)
        ).logits       # [B, L, V]
    else:
        with torch.no_grad():
            logits = model(
                sequences.to(device),
                attention_mask=attention_mask.to(device)
            ).logits       # [B, L, V]

    # Extract logits for the completion part only
    # The logits are shifted by 1 position (logits[i] predicts token[i+1])
    # Then we extract the logits for the completion part
    completion_logits = logits[:, :-1, :] # [B, L-1, V]
    completion_logits = completion_logits[:, -completion_len:, :]  # [B, L_completion, V]
    completion_logits = completion_logits / max(temperature, 1e-8) # apply temperature
    completion_tokens = sequences[:, -completion_len:].to(device) # [B, L_completion]
    completion_mask = attention_mask[:, -completion_len:].to(device) # [B, L_completion]

    log_probs = torch.log_softmax(completion_logits, dim=-1) # [B, L_completion, V]
    token_log_probs = log_probs.gather(
        -1, completion_tokens.unsqueeze(-1)
    ).squeeze(-1)                              # [B, L_completion]
    
    # Replace -inf with a large negative number to avoid NaN in computations
    token_log_probs = torch.clamp(token_log_probs, min=-100.0)
    
    token_log_probs = token_log_probs * completion_mask
    return token_log_probs if training else token_log_probs.detach().cpu()

def calculate_rewards(
    generated_answers: List[str],
    num_return_sequences: int,
    numbers: List[List[int]],
    targets: List[int],
    end_token: Optional[str] = None,
    ) -> torch.Tensor:
    """
    Calculate reward scores for each generated answer using the task-specific reward function.
    
    In GRPO/RLHF, rewards provide the learning signal. This function evaluates how
    well each generated response solves the countdown task (reaching the target number
    using the given numbers with arithmetic operations).
    
    Args:
        generated_answers: List of decoded string responses from the model.
        num_return_sequences: Number of completions generated per original prompt.
                              Used to correctly expand numbers/targets to match answers.
        numbers: List of available numbers for each original prompt in the batch.
        targets: List of target numbers for each original prompt in the batch.
        end_token: Optional end token to strip from responses before evaluation.
    
    Returns:
        torch.Tensor: [B * num_return_sequences] - Reward score for each generated answer.
    
    How it works:
        1. Expand numbers and targets to match the shape of generated_answers
           (each prompt has num_return_sequences corresponding answers)
        2. For each (response, numbers, target) triple, call the reward_function
           from countdown_task.py which evaluates correctness
        3. Collect all rewards into a tensor
    
    Note:
        The reward_function typically returns higher scores for:
        - Correct final answers that reach the target
        - Valid arithmetic expressions using only available numbers
        - Proper formatting following the expected structure
    """
    rewards = []
    # repeat numbers and targets for each generated answer
    expanded_numbers = []
    expanded_targets = []
    for i in range(len(numbers)):
        for _ in range(num_return_sequences):
            expanded_numbers.append(numbers[i])
            expanded_targets.append(targets[i])
    for response, nums, tgt in zip(generated_answers, expanded_numbers, expanded_targets):
        reward_dict = reward_function(response, nums, tgt, end_token)
        rewards.append(reward_dict['reward'])
    return torch.tensor(rewards, dtype=torch.float32)

def calculate_advantages(
        rewards: torch.Tensor,
        num_return_sequences: int,
    ) -> torch.Tensor:
    """
    Normalize rewards within each group to compute advantages (GRPO-style baseline).
    
    GRPO (Group Relative Policy Optimization) uses a clever trick: instead of 
    training a separate value network as a baseline (like PPO), it uses the mean
    reward within each group of samples from the same prompt as the baseline.
    This reduces variance while avoiding the complexity of value function estimation.
    
    Args:
        rewards: [B * num_return_sequences] - Raw rewards for all generated samples.
        num_return_sequences: Number of samples generated per prompt (group size).
    
    Returns:
        torch.Tensor: [B * num_return_sequences] - Normalized advantages where each
                      group has mean ≈ 0 and std ≈ 1.
    
    How it works:
        1. Reshape rewards into [batch_size, num_return_sequences] groups
        2. For each group (samples from same prompt):
           - Compute mean and std of rewards within the group
           - Normalize: advantage = (reward - mean) / (std + epsilon)
        3. Flatten back to [B * num_return_sequences]
    
    Why this works:
        - Samples that perform better than siblings get positive advantages
        - Samples that perform worse than siblings get negative advantages
        - This relative comparison within groups provides a natural baseline
        - No need for a learned value function, reducing training complexity
    
    Edge case:
        When num_return_sequences=1, there's no group to normalize against,
        so raw rewards are returned as-is.
    """
    if rewards.numel() % num_return_sequences != 0:
        raise ValueError("The number of rewards must be divisible by num_return_sequences.")
    
    grouped_rewards = rewards.view(-1, num_return_sequences)  # Shape: (batch_size, num_return_sequences)
    
    # Handle the case when num_return_sequences=1 (std would be NaN)
    if num_return_sequences == 1:
        # When there's only one sample per group, we can't normalize within the group
        # Instead, just return the rewards as is
        return grouped_rewards.view(-1)
    
    means = grouped_rewards.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1)
    stds = grouped_rewards.std(dim=1, keepdim=True) + 1e-8  # Shape: (batch_size, 1)
    normalized_rewards = (grouped_rewards - means) / stds  # Shape: (batch_size, num_return_sequences)
    return normalized_rewards.view(-1)  # Shape: (batch_size * num_return_sequences,)


def generate_rollouts(model: AutoModelForCausalLM, 
                        ref_model: AutoModelForCausalLM,
                        tokenizer: AutoTokenizer, 
                        batch: MiniBatch,
                        max_new_tokens: int = 512,
                        temperature: float = 0.6,
                        num_return_sequences: int = 3,
                        device: torch.device = torch.device("cpu"),
                        debug: bool = False,
                        ) -> Any:
    """
    Generate complete rollouts for a batch: sequences, rewards, advantages, and log probs.
    
    A "rollout" in RL terminology is a complete trajectory through the environment.
    For language models, this means generating completions and computing all the
    statistics needed for the policy gradient update.
    
    Args:
        model: The policy model being trained (used for generation and old log probs).
        ref_model: Frozen reference model for KL divergence computation.
        tokenizer: Tokenizer for decoding generated sequences.
        batch: MiniBatch with prompts (input_ids, attention_mask, numbers, targets).
        max_new_tokens: Maximum tokens to generate per sequence.
        temperature: Sampling temperature for generation.
        num_return_sequences: Number of completions per prompt (for GRPO grouping).
        device: Device to run on.
        debug: If True, print debug information about rewards and advantages.
    
    Returns:
        dict containing:
            - sequences: [B * G, L] - Generated token sequences (G = num_return_sequences)
            - completion_len: int - Length of completion portion
            - total_len: int - Total sequence length
            - attention_mask: [B * G, L] - Attention mask
            - old_token_log_probs: [B * G, L_comp] - Log probs under current policy (detached)
            - ref_token_log_probs: [B * G, L_comp] - Log probs under reference model
            - generated_answers: List[str] - Decoded text responses
            - rewards: [B * G] - Reward scores
            - advantages: [B * G] - Normalized advantages
    
    How it works:
        1. Generate completions using the current policy (model.eval() mode)
        2. Decode sequences to text for reward computation
        3. Calculate rewards using task-specific reward function
        4. Compute GRPO-style advantages (group normalization)
        5. Compute log probs under both current and reference models
           - old_token_log_probs: for importance sampling ratio in PPO objective
           - ref_token_log_probs: for KL penalty to prevent policy drift
        6. Package everything into a dict for the training step
    
    Note:
        All log prob computations here are done with torch.no_grad() since
        this is the "data collection" phase. Gradients are computed later
        when we recompute log probs with the updated model.
    """
    model.eval()
    ref_model.eval()
    completion_output = generate_responses(
        model,
        tokenizer,
        batch,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        device=device,
    )
    generated_answers = tokenizer.batch_decode(
        completion_output["sequences"][:, -completion_output["completion_len"]:], 
        skip_special_tokens=True
    )
    rewards = calculate_rewards(
        generated_answers,
        num_return_sequences,
        batch.numbers,
        batch.target,
        end_token=tokenizer.eos_token,
    )
    advantages = calculate_advantages(
        rewards,
        num_return_sequences,
    )

    with torch.no_grad():
        old_token_log_probs = compute_log_probs(
            model,
            completion_output["sequences"],
            completion_output["attention_mask"],
            completion_output["completion_len"],
            temperature=temperature,
            device=device,
        )
        ref_token_log_probs = compute_log_probs(
            ref_model,
            completion_output["sequences"],
            completion_output["attention_mask"],
            completion_output["completion_len"],
            temperature=temperature,
            device=device,
        )
    rollout_output = {**completion_output,
                      "old_token_log_probs": old_token_log_probs,
                      "ref_token_log_probs": ref_token_log_probs,
                      "generated_answers": generated_answers,
                      "rewards": rewards,
                      "advantages": advantages
                    }
    # if debug:
    #     print(f"  [DEBUG] rewards: {rewards.tolist()}")
    #     print(f"  [DEBUG] advantages: {advantages.tolist()}")
    return rollout_output

def grpo_loss(
    new_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    ref_token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
    beta_kl: float = 0.02,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the GRPO loss: clipped surrogate objective + KL divergence penalty.
    
    This implements the core GRPO/PPO-style objective. The loss encourages the policy
    to increase probability of high-advantage actions while preventing too-large
    policy updates (via clipping) and drift from the reference model (via KL penalty).
    
    Args:
        new_token_log_probs: [B, L_completion] - Log probs under CURRENT policy (has gradients).
        old_token_log_probs: [B, L_completion] - Log probs under policy AT GENERATION TIME.
        ref_token_log_probs: [B, L_completion] - Log probs under FROZEN reference model.
        advantages: [B] - Normalized advantage scores (positive = good, negative = bad).
        completion_mask: [B, L_completion] - Mask for valid (non-padding) tokens.
        epsilon: Clipping parameter for PPO objective (default 0.2 = clip to [0.8, 1.2]).
        beta_kl: Weight for KL divergence penalty term.
        device: Device to run computation on.
        debug: If True, print debug statistics.
    
    Returns:
        tuple of:
            - loss: Scalar total loss (surrogate + KL penalty)
            - kl_loss: Scalar KL divergence (detached, for logging)
    
    How it works:
    
    1. **Importance Sampling Ratio**: 
       ratio = exp(new_log_prob - old_log_prob) = π_new(a|s) / π_old(a|s)
       This ratio measures how much more/less likely the action is under the
       new policy compared to when it was sampled.
    
    2. **Clipped Surrogate Objective** (PPO-style):
       - surr1 = ratio * advantage  (vanilla policy gradient)
       - surr2 = clip(ratio, 1-ε, 1+ε) * advantage  (clipped version)
       - loss = -min(surr1, surr2)
       
       The clipping prevents the policy from changing too much in a single update:
       - If advantage > 0: we want to increase probability, but clip ratio at 1+ε
       - If advantage < 0: we want to decrease probability, but clip ratio at 1-ε
    
    3. **KL Divergence Penalty**:
       KL(π_ref || π_new) ≈ exp(log π_ref - log π_new) - (log π_ref - log π_new) - 1
       
       This approximation of reverse KL prevents the policy from drifting too far
       from the reference model, maintaining generation quality and preventing
       reward hacking.
    
    4. **Token-level to Sequence-level**:
       - Per-token losses are masked and summed
       - Divided by number of valid tokens per sequence
       - Averaged across the batch
    
    Mathematical form:
        L = -E[min(r*A, clip(r,1-ε,1+ε)*A)] + β * KL(π_ref || π_new)
    
    Tensor shapes:
        new_token_log_probs: [B, L_completion]
        old_token_log_probs: [B, L_completion]
        ref_token_log_probs: [B, L_completion]
        advantages: [B]
        completion_mask: [B, L_completion]
    """
    if debug:
        print(f"  [DEBUG] new_token_log_probs: min={new_token_log_probs.min():.4f}, max={new_token_log_probs.max():.4f}, nan={new_token_log_probs.isnan().sum()}, inf={new_token_log_probs.isinf().sum()}")
        print(f"  [DEBUG] old_token_log_probs: min={old_token_log_probs.min():.4f}, max={old_token_log_probs.max():.4f}, nan={old_token_log_probs.isnan().sum()}, inf={old_token_log_probs.isinf().sum()}")
        print(f"  [DEBUG] ref_token_log_probs: min={ref_token_log_probs.min():.4f}, max={ref_token_log_probs.max():.4f}, nan={ref_token_log_probs.isnan().sum()}, inf={ref_token_log_probs.isinf().sum()}")

    old_token_log_probs = old_token_log_probs.to(device)
    ref_token_log_probs = ref_token_log_probs.to(device)
    advantages = advantages.to(device)
    completion_mask = completion_mask.to(device)

    log_ratio = new_token_log_probs - old_token_log_probs  # [B, L_completion]
    # Clamp log_ratio to prevent exploding ratios
    log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
    ratio = torch.exp(log_ratio)  # [B, L_completion]
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)  # [B, L_completion]
    surr1 = ratio * advantages.unsqueeze(-1)  # [B, L_completion]
    surr2 = clipped_ratio * advantages.unsqueeze(-1)  # [B, L_completion]
    loss_per_token = -torch.min(surr1, surr2) * completion_mask  # [B, L_completion]
    # Average loss over non-masked tokens
    denom = completion_mask.sum(dim=1).clamp(min=1.0)  # [B]
    loss_sequence = loss_per_token.sum(dim=1) / denom  # [B]
    loss = loss_sequence.mean()  # scalar

    # KL divergence penalty
    delta = (ref_token_log_probs - new_token_log_probs)  # [B, L_completion]
    delta = torch.clamp(delta, min=-10.0, max=10.0)   # prevents exp overflow & crazy KL
    per_token_kl = torch.exp(delta) - delta - 1.0  # [B, L_completion]
    kl_seq = (per_token_kl * completion_mask).sum(dim=1) / denom  # [B]
    kl_loss = kl_seq.mean()  # scalar
    loss += beta_kl * kl_loss

    if debug:
        print(f"  [DEBUG] log_ratio: min={log_ratio.min():.4f}, max={log_ratio.max():.4f}")
        print(f"  [DEBUG] ratio: min={ratio.min():.4f}, max={ratio.max():.4f}")
        print(f"  [DEBUG] kl_loss: {kl_loss:.4f}")
        print(f"  [DEBUG] total loss: {loss:.4f}")

    return loss, kl_loss.detach()

def grpo(
        model_id: str,
        dataset: Dataset,
        batch_size: int = 4,
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
        batch_size: Number of prompts per batch.
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
        >>> model, losses, rewards, eval_rewards = grpo(
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
    dtype = torch.bfloat16 if device.type not in ("mps", "cuda") else torch.float32
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
                "batch_size": batch_size,
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
        test_size=1000,
    )
    ds_test = dataset(
        tokenizer=tokenizer,
        split="test",
        test_size=100,
    )

    # Create curriculum sampler if enabled
    curriculum_sampler = None
    if use_curriculum:
        curriculum_sampler = CurriculumSampler(
            dataset=ds_train,
            batch_size=batch_size,
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
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ds_train.collate_fn,
        )
    
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size * 2,
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
            test_total_reward = 0.0
            test_batches = 0
            with torch.no_grad():
                for test_batch_idx, test_batch in enumerate(dl_test):
                    if test_batch_idx >= 1:
                        break
                    # if device.type == "mps":
                    #     torch.mps.empty_cache()
                    # elif device.type == "cuda":
                    #     torch.cuda.empty_cache()
                    # Generate with temperature=0 (greedy)
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
                    test_rewards = calculate_rewards(
                        test_generated_answers,
                        num_return_sequences=1,
                        numbers=test_batch.numbers,
                        targets=test_batch.target,
                        end_token=tokenizer.eos_token,
                    )
                    test_total_reward += test_rewards.mean().item()
                    test_batches += 1
            avg_test_reward = test_total_reward / max(test_batches, 1)
            eval_rewards.append((epoch + 1, avg_test_reward))
            print(f"  [EVAL] Epoch {epoch+1}: Test Reward (greedy): {avg_test_reward:.4f}")

            if use_wandb:
                wandb.log(
                    {"eval/test_reward_greedy": avg_test_reward, "epoch": epoch + 1},
                    step=global_step,
                )


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
                # Build a small table (don’t log huge text every step)
                table = wandb.Table(columns=["numbers", "target", "response", "reward"])
                # Log up to 8 samples
                k = min(8, len(rollout_output["generated_answers"]))
                for i in range(k):
                    table.add_data(
                        str(rollout_output["generated_answers"][i]),
                        None,  # placeholder (see below)
                        rollout_output["generated_answers"][i],
                        float(rollout_output["rewards"][i].item()),
                    )
                # Better: attach correct numbers/targets per expanded sample
                # (because you repeat each prompt num_return_sequences times)
                expanded_nums = []
                expanded_tgts = []
                for j in range(len(batch.numbers)):
                    for __ in range(num_return_sequences):
                        expanded_nums.append(batch.numbers[j])
                        expanded_tgts.append(batch.target[j])

                table = wandb.Table(columns=["numbers", "target", "response", "reward"])
                for i in range(k):
                    table.add_data(
                        str(expanded_nums[i]),
                        int(expanded_tgts[i]),
                        rollout_output["generated_answers"][i],
                        float(rollout_output["rewards"][i].item()),
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
                wandb.log(
                    {
                        "train/rewards": rewards.tolist(),
                        "train/reward_mean": avg_reward_per_batch,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

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
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    trained_model, losses, rewards, eval_rewards = grpo(
        model_id=model_id,
        dataset=CountdownTaskDataset,
        batch_size=4,  # Reduced for memory
        num_epochs=100,
        batch_per_epoch=1,
        update_freq=2,
        max_new_tokens=256,  # Reduced for memory
        temperature=0.6,
        num_return_sequences=4,  # Reduced for memory
        epsilon=0.2,
        learning_rate=5e-6,
        test_freq=5,
        use_wandb=True,
        wandb_entity="dirtem1998",
        wandb_project="metre_tests",
        wandb_run_name="grpo-test-run",
    )
