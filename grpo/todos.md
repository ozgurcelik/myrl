# GRPO data-parallel + vLLM roadmap

Goal: scale GRPO training across a variable number of GPUs (1 → N) and
eventually move generation onto dedicated GPU(s) with vLLM. We build this
incrementally so failure modes don't compound — each milestone is independently
useful and testable.

Two orthogonal concerns to keep separate in your head:
1. **How training work is split across GPUs** (gradient compute + optimizer step).
2. **Where generation runs and how it stays in sync with the policy.**

For a 1.5B model the right training-parallelism tool is plain **DDP**
(DistributedDataParallel): replicate the full model per GPU, each GPU processes a
different slice of prompts, gradients are all-reduced (averaged) on `backward()`.
The GPU count is owned by the launcher (`torchrun --nproc_per_node=N`), not by
branching logic in the code.

## Milestone 1 — DDP only, HuggingFace generation

Get `torchrun`-launched DDP working with the existing `model.generate`.
Generation stays slow but correct; this forces us to solve device/rank/sampler/
logging once.

- Add a `distributed.py` helper (setup/cleanup/barrier/all-reduce) that is a
  no-op when run single-process, so `python main.py` keeps working.
- Per-rank device (`cuda:{local_rank}`) and per-rank seeding so each GPU draws
  different prompts.
- Wrap the policy in DDP. Keep two handles: the wrapped model for the training
  forward (gradients all-reduce) and the unwrapped `.module` for generation /
  no-grad logprobs (DDP does not expose `.generate()`).
- Gate wandb / prints / eval to rank 0; add a barrier to resync after eval.
- Make collectives safe: global (unanimous) NaN-skip, and cross-rank averaging
  of logged metrics.

Launch: `torchrun --standalone --nproc_per_node=<N> main.py` (N=1 also works).

## Milestone 2 — DDP training + vLLM colocate

Swap the generation step for vLLM running on each rank's GPU (vLLM shares the
training GPU). Big speedup, no separate process, weights reloaded locally after
updates. Watch for memory contention on the training GPUs.

## Milestone 3 — DDP training + vLLM server on a dedicated GPU

The full "generation on a fixed GPU" architecture: vLLM runs as a separate
process on dedicated GPU(s); trainers send prompts over HTTP. Requires a
**weight-sync** step (push updated policy weights to the vLLM server after
updates) and handling the **train/inference mismatch** (importance-sampling
correction). Needs >= 2 GPUs (one for vLLM, the rest for training); falls back to
colocate on a single GPU.
