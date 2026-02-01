# %%
import torch
from datasets import load_dataset

# %%
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
# %%
ds["train"][0]
# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# fp16 is usually the best speed/memory choice on MPS
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to(device)

prompt = "Write a haiku about GPUs on a MacBook."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    temperature=0.0,
    top_p=0.9,
    use_cache=True,          # important for decode speed
    output_scores=True,
    return_dict_in_generate=True,
)

print(tokenizer.decode(out.sequences[0], skip_special_tokens=True))
# %%
out
# %%
out.scores
# %%
tokenizer.vocab_size
# %%
out.scores[0].shape
# %%
len(tokenizer)
# %%
print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
print(f"len(tokenizer): {len(tokenizer)}")
print(f"model.config.vocab_size: {model.config.vocab_size}")
print(f"model.get_input_embeddings().weight.shape[0]: {model.get_input_embeddings().weight.shape[0]}")
print(f"out.scores[0].shape: {out.scores[0].shape}")
# %%
out.scores[0][0][-(model.config.vocab_size - tokenizer.vocab_size):]
# %%
print(tokenizer.decode(out.sequences[0], skip_special_tokens=True))
# %%
text = ""
for score in out.scores:
    # find the index of the highest score
    top_index = torch.argmax(score, dim=-1)
    text += tokenizer.decode(top_index)
print(text)
# %%