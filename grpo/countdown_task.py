# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, Sampler
from typing import Any, Dict, List, Iterator
import random

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
# %%
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    numbers: List[List[int]]
    target: List[int]
    # model-ready tensors (padded)
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
# %%
class CountdownTaskDataset(Dataset):
    """Custom Dataset for Countdown Tasks."""

    def __init__(self, tokenizer: AutoTokenizer, split="train"):
        data = load_dataset("ozgur-celik/countdown_cl")
        self.tokenizer = tokenizer
        self.split = split
        if self.split == "train":
            self.data = data["train"]
        else:
            self.data = data["test"]

    def __len__(self):
        return len(self.data)
    
    def encode_prefix(self, numbers: list[int] , target: int):
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": RESPONSE_PROMPT},
            ],
            tokenize=True,
            continue_final_message=True,
        )
        prefix = self.tokenizer.decode(tokens)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens,
        }
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.encode_prefix(item["nums"], item["target"])
        item.update(encoding)
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]

        input_ids_list = [item["prefix_tokens"] for item in batch]

        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
        )
    
    def get_difficulty_indices(self) -> Dict[int, List[int]]:
        """Return a mapping from difficulty level to list of dataset indices."""
        difficulty_to_indices: Dict[int, List[int]] = {}
        for idx in range(len(self.data)):
            diff = self.data[idx]["difficulty"]
            if diff not in difficulty_to_indices:
                difficulty_to_indices[diff] = []
            difficulty_to_indices[diff].append(idx)
        return difficulty_to_indices


class CurriculumSampler(Sampler[int]):
    """
    A sampler that implements curriculum learning based on difficulty levels.
    
    Curriculum schedule:
    - Phase 1 (global_step 0-99): Only difficulty 2
    - Phase 2 (global_step 100-199): Difficulty 2 and 3
    - Phase 3 (global_step 200+): All difficulties (2, 3, 4)
    
    Ensures samples from earlier phases are not reused until a full loop through
    that difficulty level is completed.
    """
    
    def __init__(
        self,
        dataset: CountdownTaskDataset,
        batch_size: int,
        phase1_end: int = 100,
        phase2_end: int = 200,
    ):
        """
        Args:
            dataset: The CountdownTaskDataset instance
            batch_size: Number of samples per batch
            phase1_end: Global step at which phase 1 ends (exclusive)
            phase2_end: Global step at which phase 2 ends (exclusive)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.global_step = 0
        
        # Get indices grouped by difficulty
        self.difficulty_indices = dataset.get_difficulty_indices()
        
        # Track remaining (unused) indices for each difficulty level
        # These will be shuffled and popped from
        self._remaining: Dict[int, List[int]] = {}
        self._reset_all_pools()
    
    def _reset_all_pools(self):
        """Reset all difficulty pools with shuffled indices."""
        for diff, indices in self.difficulty_indices.items():
            self._remaining[diff] = indices.copy()
            random.shuffle(self._remaining[diff])
    
    def _reset_pool(self, difficulty: int):
        """Reset a single difficulty pool when exhausted."""
        self._remaining[difficulty] = self.difficulty_indices[difficulty].copy()
        random.shuffle(self._remaining[difficulty])
    
    def _get_allowed_difficulties(self) -> List[int]:
        """Determine which difficulties are allowed based on current global_step."""
        if self.global_step < self.phase1_end:
            return [2]
        elif self.global_step < self.phase2_end:
            return [2, 3]
        else:
            return [2, 3, 4]
    
    def set_global_step(self, step: int):
        """Update the global step counter."""
        self.global_step = step
    
    def _sample_batch(self) -> List[int]:
        """Sample a batch of indices from allowed difficulty levels."""
        allowed = self._get_allowed_difficulties()
        batch_indices = []
        
        while len(batch_indices) < self.batch_size:
            # Check which pools still have samples
            non_empty_pools = [d for d in allowed if d in self._remaining and len(self._remaining[d]) > 0]
            
            if not non_empty_pools:
                # All allowed pools are exhausted, reset them
                for d in allowed:
                    if d in self.difficulty_indices:
                        self._reset_pool(d)
                non_empty_pools = [d for d in allowed if d in self._remaining and len(self._remaining[d]) > 0]
                
                if not non_empty_pools:
                    # No samples available at all for these difficulties
                    break
            
            # Randomly pick from one of the non-empty allowed pools
            chosen_diff = random.choice(non_empty_pools)
            idx = self._remaining[chosen_diff].pop()
            batch_indices.append(idx)
        
        return batch_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield a batch of indices based on current curriculum phase."""
        batch_indices = self._sample_batch()
        yield batch_indices
    
    def __len__(self) -> int:
        """Return 1 since we yield one batch per iteration."""
        return 1