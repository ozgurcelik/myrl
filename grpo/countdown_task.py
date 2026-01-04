# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Any, Dict, List
import re
from typing import Optional

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
        data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")["train"]
        self.tokenizer = tokenizer
        self.test_size = test_size
        self.split = split
        if self.split == "train":
            self.data = data.select(range(len(data) - self.test_size))
        else:
            self.data = data.select(range(len(data) - self.test_size, len(data)))

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
# %%
def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    # Check for exactly one <think> and one </think>
    if think_match:
        think_open_count = response.count("<think>")
        think_close_count = response.count("</think>")
        if think_open_count == 1 and think_close_count == 1:
            reward += 0.1

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

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


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