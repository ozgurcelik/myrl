# GRPO environment setup

Reproducible environment for the GRPO (TRL + vLLM) pipeline on RunPod. One
pinned **CUDA 12.8** stack that runs across every GPU type you might rent
(Ampere, Ada, Blackwell).

## Files

| File | Purpose |
| --- | --- |
| `setup.sh` | Builds the venv from scratch on a fresh pod, verifies the GPU, freezes the lock |
| `requirements.in` | High-level source dependencies (edit this to add/change deps) |
| `requirements.lock` | Exact pinned versions — the durable reproducibility artifact (commit this) |

## Quick start

On any fresh pod:

```bash
bash /root/myrl/grpo/setup.sh
```

Then, in any shell where you want to run the pipeline:

```bash
source /root/grpo-venv/bin/activate
```

That's it. `python`, `vllm`, `trl`, etc. now resolve to the pinned cu128 stack.

- **Cold run** (nothing cached): ~7 min, mostly downloading ~6 GB of wheels.
- **Warm run** (wheels in `/dev/shm` cache from earlier this session): ~90 s.

## What `setup.sh` does

1. Installs `uv` (if missing).
2. Preflight: prints GPU/driver, warns if the host driver is `< 570` (needed for CUDA 12.8 / Blackwell), checks root disk space.
3. Creates an isolated venv with a standalone Python 3.12.
4. Installs the stack with `--torch-backend=cu128`. If `requirements.lock` exists it installs those exact versions; otherwise it resolves `requirements.in` and writes the lock.
5. Verifies this pod's GPU architecture is supported by the installed torch (and runs a CUDA sanity check).
6. Clears the temporary wheel cache from `/dev/shm`.

## Storage layout (important on RunPod)

The container is tight on disk, so the script puts things deliberately:

| What | Where | Notes |
| --- | --- | --- |
| venv (~12 GB) | `/root/grpo-venv` (root disk) | **Ephemeral** — gone on pod restart. Re-run `setup.sh`. |
| wheel cache | `/dev/shm` (tmpfs) | Fast, cleared at the end of setup. |
| HF models / checkpoints | `/workspace/.cache/huggingface` (`HF_HOME`) | **Persistent**, but only a ~20 GB quota — watch model sizes. |

> The venv is intentionally **not** stored on `/workspace`: that volume has a
> hard ~20 GB quota and can't hold both the venv and your models. The durable,
> reproducible artifact is `requirements.lock` (in git), not the venv itself.

## Configuration

Override defaults via environment variables:

```bash
# Re-resolve and re-freeze the lock (e.g. after editing requirements.in)
RELOCK=1 bash setup.sh

# Put the venv somewhere else (e.g. if you enlarged /workspace)
ENV_DIR=/workspace/envs/grpo bash setup.sh

# Pin a different Python or CUDA backend
PY_VERSION=3.11 TORCH_BACKEND=cu126 bash setup.sh
```

## GPU compatibility

The installed torch reports `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`, which
covers all rentable GPUs:

- **Ampere** (A40, A6000, A5000, A4500, A4000) — native `sm_86`
- **Ada** (L4, L40, L40S, RTX 2000/4000/6000 Ada) — runs on the `sm_86` cubin (CUDA binary compat within compute major 8)
- **Blackwell** (RTX PRO 4000/4500/6000) — native `sm_120`

When renting, prefer pods advertised with **CUDA ≥ 12.8** (host driver ≥ 570),
especially for Blackwell cards. `setup.sh` warns if the driver is too old.

## Notes

- This venv is fully isolated from the RunPod base Python (3.11 + torch 2.4.1+cu124). The base is left untouched; just remember to `activate` the venv before running anything, or you'll get the old cu124 torch (which won't run on Blackwell).
- To regenerate the lock after changing dependencies, edit `requirements.in` then run `RELOCK=1 bash setup.sh`.
