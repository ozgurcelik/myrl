# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"                       # any causal-LM
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name).eval()

input_texts = [
    "Hello, how are you?",
    "What is the capital of United States of America?"
]
output_texts = [
    "I am fine, thank you!",
    "The capital of United States of America is Washington, D.C."
]
texts = [
    f"{input_text} {output_text}"
    for input_text, output_text in zip(input_texts, output_texts)
]
# %%
# 1) Tokenise WITH PADDING ----------------------------------------------------  ⟵ NEW
batch = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,               # pad to longest in batch
    return_attention_mask=True,  # 1 for real tokens, 0 for padding
    padding_side="left",
)
input_ids      = batch["input_ids"]        # [B, L]
attention_mask = batch["attention_mask"]   # ⟵ NEW same shape
total_lens = attention_mask.sum(dim=-1)
print(input_ids)
print(input_ids.shape)
print(attention_mask.shape)
print(attention_mask)
# Decode the input_ids
print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(input_ids[1]))
print(total_lens)
# %%
# 2) Tokenise the input text --------------------------------------------------
input_batch = tokenizer(
    input_texts,
    return_tensors="pt",
    padding=True,               # pad to longest in batch
    return_attention_mask=True,  # 1 for real tokens, 0 for padding
    #padding_side="left",
)
input_input_ids      = input_batch["input_ids"]        # [B, L_input]
input_attention_mask = input_batch["attention_mask"]   # ⟵ NEW same shape
input_lens = input_attention_mask.sum(dim=-1)
print(input_input_ids)
print(input_input_ids.shape)
print(input_attention_mask.shape)
print(input_attention_mask)
print(input_lens)
# %%
for i in range(len(input_ids)):
    print(tokenizer.decode(input_ids[i][-total_lens[i]:-(total_lens[i]-input_lens[i])]))
    print(tokenizer.decode(input_ids[i][-(total_lens[i]-input_lens[i]):]))
    print("-"*100)
# %%
# 3) Forward pass
with torch.no_grad():
    logits = model(input_ids,
                   attention_mask=attention_mask).logits       # [B, L, V]
print(logits.shape)
# %%
# 3) We now need to shift the logits, input_ids and attention_mask to the right so that we will only focus on the output tokens.
shifted_attention_mask = torch.zeros_like(attention_mask)
shifted_input_ids = torch.zeros_like(input_ids)
for i in range(len(input_ids)):
    shifted_attention_mask[i][-(total_lens[i]-input_lens[i]):] = attention_mask[i][-(total_lens[i]-input_lens[i]):]
    shifted_input_ids[i][-(total_lens[i]-input_lens[i]):] = input_ids[i][-(total_lens[i]-input_lens[i]):]

shifted_logits = logits[:, :-1, :]
shifted_attention_mask = shifted_attention_mask[:, 1:]
shifted_input_ids = shifted_input_ids[:, 1:]

print(shifted_attention_mask)
print(shifted_input_ids)

print(shifted_logits.shape)
print(shifted_attention_mask.shape)
print(shifted_input_ids.shape)
# %%
for i in range(len(shifted_input_ids)):
    print(tokenizer.decode(shifted_input_ids[i]))
    print(tokenizer.decode(shifted_input_ids[i][shifted_attention_mask[i].bool()]))
    print("-"*100)
# %%