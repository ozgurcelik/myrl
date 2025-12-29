# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"                       # any causal-LM
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name).eval()

system_prompt = "You are a helpful assistant."
user_prompt = "What is the capital of United States of America?"
# %%
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
# %%
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
# %%
tokenizer.decode(inputs['input_ids'][0])
# %%
outputs = model.generate(**inputs,
                         max_new_tokens=100,
                         top_k=None,
                         top_p=None,
                         temperature=0.6,
                         return_dict_in_generate=True,
                         # generate 3 samples
                         num_return_sequences=3,
                         )
# %%
outputs.sequences[0]
# %%
tokenizer.decode(outputs.sequences[2])
# %%
outputs
# %%
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# %%
inputs
# %%
