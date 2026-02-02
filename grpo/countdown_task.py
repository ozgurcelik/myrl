# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, Sampler
from typing import Any, Dict, List, Iterator
from collections import Counter
import re
from typing import Optional
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

    def __init__(self, tokenizer: AutoTokenizer, split="train", test_size=100):
        data = load_dataset("ozgur-celik/countdown_cl")
        self.tokenizer = tokenizer
        # self.test_size = test_size
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
        difficulty = [item["difficulty"] for item in batch]
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
# %%
def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>\s*$"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    think_response_order_correct = float(is_thinking_after_response(response))
    think_answer_appears_once = float(is_think_answer_appear_once(response))

    # Check for exactly one <think> and one </think>
    reward += 0.05 * (think_answer_appears_once + think_response_order_correct)

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(0)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0
    
    reward = 0.0

    using_every_number_once = 0.0
    if answer_match:
        using_every_number_once = float(is_using_numbers_once(answer_match))

        answer_txt = answer_match.group(0)
        if "=" in answer_txt:
            options = answer_txt.split("=")
            # Longer one is the math expression
            if len(options[0].strip()) >= len(options[1].strip()):
                answer_content = options[0].strip()
                answer_is_correct = float(is_answer_correct(answer_content, target))
            else:
                answer_content = options[1].strip()
                answer_is_correct = float(is_answer_correct(answer_content, target))

            reward += -0.1 + answer_is_correct * 0.2
            
    
    
    # Check if the answer evaluates to the target
    try:
        if is_answer_correct(answer_content, target):
            return 1.0 + using_every_number_once * 0.1
    except Exception as e:
        pass

    return reward + 0.0 + using_every_number_once * 0.1


def is_answer_correct(answer_content: str, target: int) -> bool:
    result = eval(answer_content, {"__builtins__": None}, {})
    if abs(float(result) - float(target)) < 1e-5:
        return True
    return False


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)

    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


def is_thinking_after_response(response: str) -> float:
    """Check if the response, <answer>...</answer> appears before <think>...</think>."""
    think_index = response.find("<think>")
    answer_index = response.find("<answer>")
    think_end_index = response.find("</think>")
    answer_end_index = response.find("</answer>")
    if think_index == -1 or answer_index == -1:
        return False
    return (
        (answer_index > think_index) and 
        (answer_end_index > think_end_index) and 
        (answer_index > think_end_index)
    )

def is_think_answer_appear_once(response: str) -> float:
    """Check if <think>...</think> and <answer>...</answer> appear exactly once."""
    think_open_count = response.count("<think>")
    think_close_count = response.count("</think>")
    answer_open_count = response.count("<answer>")
    answer_close_count = response.count("</answer>")
    if think_open_count == 1 and think_close_count == 1 and answer_open_count == 1 and answer_close_count == 1:
        return True
    return False


def is_using_numbers_once(answer_match: str, numbers: list[int], target: int) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(0)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if Counter(used_numbers) == Counter(numbers):
        return 1.0
        
    if Counter(numbers + [target]) == Counter(used_numbers):
        return 0.8
    
    return 0.0
    

if __name__ == "__main__":
    numbers = [1, 2, 3, 4]
    target = 10
    response = "1 + 2 + 3 + 4 = 10</think>\n<answer>(1 + 2 + 3 + 4)</answer>"
    print(reward_function(response, numbers, target))