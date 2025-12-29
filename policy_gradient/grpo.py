# %%
"""Simple GRPO implementation. Butun promptlari ayri ayri tokenize ediyoruz."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_utils import combined_reward, prepare_dataset
# %%
model_name = "Qwen/Qwen2.5-1.5B-Instruct"                       # any causal-LM
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)
BATCH_SIZE = 2
GROUP_SIZE = 2
SYSTEM_PROMPT = """
You are a helpful assistant.
When the user asks you to solve a mathematical question, you must first reason carefully which will be written in <reasoning>...</reasoning> tag, and then provide the final answer which is usually just a number in <answer>...</answer> tag.
So the output should be like this:
<reasoning> {Your reasoning here} </reasoning> <answer> {Your final answer here} </answer>
"""
# %%
def generate_completions(model, tokenizer, prompts: list[str]) -> dict:
    """Generate completions for a given prompt.
    
    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for tokenization.
        prompts: The prompts to use for generation.
        
    Returns:
        completion_output: A dictionary containing the completion output.
        completion_output["sequences"]: The generated sequences.
        completion_output["completion_len"]: The length of the completion.
        completion_output["total_len"]: The total length of the output.
        completion_output["completion_tokens"]: The completion tokens.
        completion_output["attention_mask"]: The attention mask.
    """
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompts.append(chat_prompt)
    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    outputs = model.generate(**inputs,
                             max_new_tokens=400,
                             top_k=None,
                             top_p=None,
                             temperature=0.6,
                             return_dict_in_generate=True,
                             num_return_sequences=GROUP_SIZE,
                             )
    
    attention_mask = inputs.attention_mask # [1, L_input]
    attention_mask = torch.repeat_interleave(attention_mask, GROUP_SIZE, dim=0) # [B, L_input]
    eos_token_id = tokenizer.eos_token_id
    total_len = outputs.sequences.shape[1]
    input_len = inputs['input_ids'].shape[1]
    completion_len = total_len - input_len
    completion_mask = torch.zeros(outputs.sequences.shape[0], completion_len, dtype=torch.int64) # [B, L_completion]
    for i in range(len(outputs.sequences)):
        temp = outputs.sequences[i, input_len:]
        eos_pos = (temp == eos_token_id).nonzero()
        if len(eos_pos) == 0:
            completion_mask[i, :] = 1
        else:
            completion_mask[i, :eos_pos[0]+1] = 1
    attention_mask = torch.cat([attention_mask, completion_mask], dim=1)
    
    completion_output = {
        "sequences": outputs.sequences,
        "completion_len": completion_len,
        "total_len": total_len,
        "attention_mask": attention_mask
    }
    return completion_output

def compute_log_probs(model, sequences, attention_mask, completion_len):
    """Compute the log probabilities of the completion tokens.
    
    Args:
        model: The model to use for computation.
        sequences: The sequences to compute the log probabilities for.
        attention_mask: The attention mask for the sequences.
        completion_len: The length of the completion.
    
    Returns:
        token_log_probs: The log probabilities of the completion tokens.
    """
    logits = model(sequences,
                    attention_mask=attention_mask).logits       # [B, L, V]

    # Extract logits for the completion part only
    # The logits are shifted by 1 position (logits[i] predicts token[i+1])
    # Then we extract the logits for the completion part
    completion_logits = logits[:, :-1, :]
    completion_logits = completion_logits[:, -completion_len:, :]  # [B, L_completion, V]
    completion_tokens = sequences[:, -completion_len:] # [B, L_completion]
    completion_mask = attention_mask[:, -completion_len:] # [B, L_completion]

    log_probs = torch.log_softmax(completion_logits, dim=-1) # [B, L_completion, V]
    token_log_probs = log_probs.gather(
        -1, completion_tokens.unsqueeze(-1)
    ).squeeze(-1)                              # [B, L_completion]
    token_log_probs = token_log_probs * completion_mask
    return token_log_probs

def generate_rollouts(model, ref_model, tokenizer, prompts: list[str]):
    """Generate rollouts for given prompts."""
    completion_outputs = generate_completions(model, tokenizer, prompts)
    generated_answers = tokenizer.batch_decode(completion_outputs["sequences"][:, -completion_outputs["completion_len"]:], 
                                               skip_special_tokens=True)
    with torch.no_grad():
        old_token_log_probs = compute_log_probs(model,
                                                completion_outputs["sequences"],
                                                completion_outputs["attention_mask"],
                                                completion_outputs["completion_len"])
        ref_token_log_probs = compute_log_probs(ref_model,
                                                completion_outputs["sequences"],
                                                completion_outputs["attention_mask"],
                                                completion_outputs["completion_len"])
    rollout_output = {**completion_outputs,
                      "old_token_log_probs": old_token_log_probs,
                      "ref_token_log_probs": ref_token_log_probs,
                      "generated_answers": generated_answers}
    return rollout_output
# %%
data = prepare_dataset()
rollout_output = generate_rollouts(model, model, tokenizer, data["questions"][:2])
# %%
rollout_output["generated_answers"]
# %%
original_answers = data["answers"][:2]
# Repeat the original answers 2 times so [a, b] -> [a, a, b, b]
original_answers = [ans for ans in original_answers for _ in range(2)]
# %%
combined_reward(rollout_output["generated_answers"], original_answers)
# %%







# %%
import torch
# %%
gen_mask = torch.tensor([[True, True, False, False],
                  [True, True, True, True]])
token_count = gen_mask.sum()
token_count
# %%
old_token_log_probs = torch.log(torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5, 0.5]]))
new_token_log_probs = torch.log(torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]]))
adv = torch.tensor([[2., 3.]])
N = old_token_log_probs.shape[0]
# %%
log_ratio = new_token_log_probs - old_token_log_probs
log_ratio
# %%
ratio = torch.exp(log_ratio)
ratio
# %%
adv_tok = adv.view(N, 1).expand_as(ratio)
adv_tok
# %%
surr = ratio * adv_tok
surr
# %%
policy_loss = (surr * gen_mask).sum() / token_count
print(surr * gen_mask)
print(f"{(surr * gen_mask).sum()} / {token_count} = {policy_loss}")
# %%


# %%
log_sequence_importance = (log_ratio * gen_mask).sum(dim=-1) / gen_mask.sum(dim=-1)
log_sequence_importance = log_sequence_importance.unsqueeze(-1)
log_sequence_importance
# %%
sequence_importance = log_sequence_importance.exp()
sequence_importance
# %%
adv_seq = adv.view(N, 1).expand_as(sequence_importance)
adv_seq
# %%
surr_seq = adv_seq * sequence_importance
surr_seq
# %%
(surr_seq).sum() / N
# %%
surr_seq.mean()
# %%
(surr_seq * gen_mask).sum(-1) / gen_mask.sum(-1)
# %%
