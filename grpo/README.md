# GRPO (TRL + vLLM) on the Countdown task

One pinned **CUDA 12.8** environment that runs on any RunPod GPU (Ampere / Ada /
Blackwell), plus a single training script that does GRPO with vLLM rollouts.

## Files

| File | Purpose |
| --- | --- |
| `setup.sh` | Build the venv from the lock on a fresh pod, verify the GPU |
| `requirements.lock` | Exact pinned versions — the durable, reproducible artifact |
| `grpo_pipeline.py` | The trainer (reuses `countdown_task.build_prompt` + `reward.reward_function`) |

## 1. Set up the environment (once per pod)

```bash
bash setup.sh
source /root/grpo-venv/bin/activate
```

The venv (~12 GB) lives on the ephemeral root disk and is rebuilt each pod;
`requirements.lock` (in git) is what makes it reproducible. Models are cached on
`/workspace` (persistent, but ~20 GB quota — watch model sizes).

## 2. Run GRPO

**Colocate (simplest — one GPU, one command).** vLLM runs inline with the trainer:

```bash
python grpo_pipeline.py
```

**Server (vLLM on a separate GPU).** Two commands in two terminals:

```bash
# Terminal 1 — rollout server on GPU 0
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --enforce_eager

# Terminal 2 — trainer on GPU 1 (connects to the server, syncs weights)
CUDA_VISIBLE_DEVICES=1 VLLM_MODE=server python grpo_pipeline.py
```

Stop the server with **Ctrl-C** in terminal 1 (cleans up its workers properly).

### Common knobs (env vars)

`MODEL_ID`, `MAX_STEPS`, `NUM_GEN`, `TRAIN_BS`, `GRAD_ACCUM`, `MAX_COMPLETION`,
`LR`, `KL_BETA`, `REPORT_TO` (set `wandb` to log), `OUTPUT_DIR`. Batch/generation
sizes auto-tune to the GPU's VRAM if not set.

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct MAX_STEPS=2 python grpo_pipeline.py   # quick smoke test
REPORT_TO=wandb MAX_STEPS=1000 python grpo_pipeline.py
```

## GPU compatibility

The pinned torch (`2.10.0+cu128`) reports
`sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`, covering all rentable GPUs: Ampere
(native `sm_86`), Ada (runs on the `sm_86` cubin), and Blackwell RTX PRO (native
`sm_120`). When renting, prefer pods with **CUDA ≥ 12.8 / driver ≥ 570**; `setup.sh`
warns otherwise.

## Environment quirks (already handled)

1. **`hf-xet` download backend hangs** — `grpo_pipeline.py` sets
   `HF_HUB_DISABLE_XET=1`; pass it to `trl vllm-serve` too (shown above). If a
   download stalls, clear stale locks: `find $HF_HOME -name '*.lock' -delete`.
2. **`/dev/shm` is `noexec`** — JIT-compiled kernels can't load from there. Keep
   `TMPDIR` on the root disk (`/tmp`); don't point it at `/dev/shm`.
3. **vLLM must be cu128** — vLLM ≥ 0.20 ships CUDA-13 wheels that won't run on a
   12.8 driver, and TRL 1.6 only supports vLLM ≤ 0.19. The lock pins
   `vllm==0.19.0` (the compatible ceiling). Don't bump it without re-checking both.
4. **`trl vllm-serve` needs `--enforce_eager`** — its torch.compile/CUDA-graph
   capture hangs at startup here; eager mode starts reliably.

To change dependencies: install into the venv with `uv pip install ...`, then
re-freeze with `uv pip freeze > requirements.lock`.

## Notes

- The venv is isolated from the RunPod base Python (3.11 + torch 2.4.1+cu124);
  always `activate` it first.
