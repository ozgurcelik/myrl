# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
import numpy as np
from typing import Optional
from countdown_task import *

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
    model.to(device)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

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

    completion_mask = torch.zeros(outputs.sequences.shape[0], completion_len, dtype=torch.int64) # [B, L_completion]
    for i in range(len(outputs.sequences)):
        temp = outputs.sequences[i, input_len:]
        eos_pos = (temp == eos_token_id).nonzero()
        if len(eos_pos) == 0:
            completion_mask[i, :] = 1
        else:
            completion_mask[i, :eos_pos[0]+1] = 1
    input_attention_mask = torch.repeat_interleave(input_attention_mask, num_return_sequences, dim=0) # [B, L_input]
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
    ) -> torch.Tensor:
    model.to(device)
    
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
    completion_tokens = sequences[:, -completion_len:] # [B, L_completion]
    completion_mask = attention_mask[:, -completion_len:] # [B, L_completion]

    log_probs = torch.log_softmax(completion_logits, dim=-1) # [B, L_completion, V]
    token_log_probs = log_probs.gather(
        -1, completion_tokens.unsqueeze(-1)
    ).squeeze(-1)                              # [B, L_completion]
    token_log_probs = token_log_probs.to(torch.device("cpu")) * completion_mask
    return token_log_probs

def calculate_rewards(
    generated_answers: List[str],
    num_return_sequences: int,
    numbers: List[List[int]],
    targets: List[int],
    end_token: Optional[str] = None,
    ) -> torch.Tensor:
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
    if rewards.numel() % num_return_sequences != 0:
        raise ValueError("The number of rewards must be divisible by num_return_sequences.")
    
    grouped_rewards = rewards.view(-1, num_return_sequences)  # Shape: (batch_size, num_return_sequences)
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
                        ) -> Any:
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
    return rollout_output

def grpo_loss(
    new_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    ref_token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
    beta_kl: float = 0.02,
    ) -> torch.Tensor:
    """
    Shapes:
        new_token_log_probs: [B, L_completion]
        old_token_log_probs: [B, L_completion]
        advantages: [B]
        completion_mask: [B, L_completion]
    """

    log_ratio = new_token_log_probs - old_token_log_probs  # [B, L_completion]
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
    per_token_kl = torch.exp(ref_token_log_probs - new_token_log_probs) - (ref_token_log_probs - new_token_log_probs) - 1.0  # [B, L_completion]
    kl_seq = (per_token_kl * completion_mask).sum(dim=1) / denom  # [B]
    kl_loss = kl_seq.mean()  # scalar
    loss += beta_kl * kl_loss

    return loss

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
        ):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds_train = dataset(
        tokenizer=tokenizer,
        split="train",
        test_size=100,
    )
    ds_test = dataset(
        tokenizer=tokenizer,
        split="test",
        test_size=100,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ds_train.collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    avg_losses = []
    avg_rewards = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        for _ in range(batch_per_epoch):
            batch = next(iter(dl_train))
            rollout_output = generate_rollouts(
                model,
                ref_model,
                tokenizer,
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                device=device,
            )
            rewards = rollout_output["rewards"]
            avg_reward_per_batch = rewards.mean().item()
            total_reward += avg_reward_per_batch
            for _ in range(update_freq):
                completion_mask = rollout_output["attention_mask"][:, -rollout_output["completion_len"]:]
                new_token_log_probs = compute_log_probs(
                    model,
                    rollout_output["sequences"],
                    rollout_output["attention_mask"],
                    rollout_output["completion_len"],
                    temperature=temperature,
                    device=device,
                )
                loss = grpo_loss(
                    new_token_log_probs,
                    rollout_output["old_token_log_probs"],
                    rollout_output["ref_token_log_probs"],
                    rollout_output["advantages"],
                    completion_mask,
                    epsilon=epsilon,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / batch_per_epoch
        avg_reward = total_reward / batch_per_epoch
        avg_losses.append(avg_loss)
        avg_rewards.append(avg_reward)
        print(f"Epoch {epoch+1}/{num_epochs}\n \tLoss: {avg_loss:.4f}\n \tReward: {avg_reward:.4f}")

    return model, avg_losses, avg_rewards

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    trained_model, losses, rewards = grpo(
        model_id=model_id,
        dataset=CountdownTaskDataset,
        batch_size=4,
        num_epochs=50,
        batch_per_epoch=1,
        update_freq=4,
        max_new_tokens=512,
        temperature=0.6,
        num_return_sequences=4,
        epsilon=0.2,
        learning_rate=1e-5,
    )