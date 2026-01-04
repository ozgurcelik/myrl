# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List

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
ds = CountdownTaskDataset(
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
    split="train",
    test_size=100,
)
# %%
ds.__len__()
# %%
dl = DataLoader(
    ds,
    batch_size=8,
    shuffle=True,
    collate_fn=ds.collate_fn,  # <-- use it here
)

batch = next(iter(dl))
# %%
batch
# %%
input_ids = batch.input_ids[0]
attention_mask = batch.attention_mask[0]
# %%

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# %%
outputs = model.generate(
    input_ids=input_ids.unsqueeze(0),
    attention_mask=attention_mask.unsqueeze(0),
    max_new_tokens=512,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True,
)
# %%
print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
# %%
keep = attention_mask.bool()
input_ids_keep = input_ids[keep]
attention_mask_keep = attention_mask[keep]
# %%
outputs2 = model.generate(
    input_ids=input_ids_keep.unsqueeze(0),
    attention_mask=attention_mask_keep.unsqueeze(0),
    max_new_tokens=512,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True,
)
# %%
print(tokenizer.decode(outputs2.sequences[0], skip_special_tokens=True))
# %%
outputs3 = model.generate(
    input_ids=input_ids.unsqueeze(0),
    max_new_tokens=512,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True,
)
# %%
print(tokenizer.decode(outputs3.sequences[0], skip_special_tokens=True))
# %%
o2scores = torch.stack(outputs2.scores, dim=1).squeeze(0)
o2scores
# %%
oscores = torch.stack(outputs.scores, dim=1).squeeze(0)
oscores
# %%
# compare the oscores and o2scores to see if they are same
torch.allclose(
    oscores, o2scores, atol=1e-1
)
# %%
import torch
from transformers import AutoModelForCausalLM

# assumes you already ran your code that defines:
# - ds (CountdownTaskDataset)
# - dl (DataLoader with ds.collate_fn)
# - MiniBatch, etc.

# ---- get one batch ----
batch = next(iter(dl))  # MiniBatch

# ---- tokenizer + model (standalone) ----
tok = ds.tokenizer
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- pick one sample from the batch ----
i = 0
input_ids_padded = batch.input_ids[i:i+1].to(device)          # (1, S)
attn_mask_padded = batch.attention_mask[i:i+1].to(device)     # (1, S)

# ---- trim to actual (non-pad) tokens where attention_mask == 1 ----
keep = attn_mask_padded[0].bool()                             # (S,)
input_ids_trim = input_ids_padded[0, keep].unsqueeze(0)       # (1, T)
attn_mask_trim = torch.ones_like(input_ids_trim, device=device)

# ---- forward pass: compare logits on real tokens ----
with torch.no_grad():
    out_padded = model(input_ids=input_ids_padded, attention_mask=attn_mask_padded)
    out_trim   = model(input_ids=input_ids_trim,   attention_mask=attn_mask_trim)

logits_padded = out_padded.logits                              # (1, S, V)
logits_trim   = out_trim.logits                                # (1, T, V)

T = input_ids_trim.shape[1]
logits_padded_aligned = logits_padded[:, -T:, :]               # left pad => real tokens are last T

print("max abs diff (all positions, all vocab):",
      (logits_padded_aligned - logits_trim).abs().max().item())
print("max abs diff (last position, all vocab):",
      (logits_padded_aligned[:, -1, :] - logits_trim[:, -1, :]).abs().max().item())

# ---- generation: compare new tokens correctly ----
gen_kwargs = dict(
    max_new_tokens=64,
    do_sample=False,                 # greedy
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

with torch.no_grad():
    gen_padded = model.generate(
        input_ids=input_ids_padded,
        attention_mask=attn_mask_padded,
        **gen_kwargs,
    )
    gen_trim = model.generate(
        input_ids=input_ids_trim,
        attention_mask=attn_mask_trim,
        **gen_kwargs,
    )

S = input_ids_padded.shape[1]
T = input_ids_trim.shape[1]

new_padded = gen_padded[0, S:]
new_trim   = gen_trim[0, T:]

print("\nExact same generated token ids?", torch.equal(new_padded, new_trim))

print("\nNew text (padded prompt):")
print(tok.decode(new_padded, skip_special_tokens=False))

print("\nNew text (trimmed prompt):")
print(tok.decode(new_trim, skip_special_tokens=False))
# %%
