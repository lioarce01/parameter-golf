# Parameter Golf — Autonomous Research Agent

## Your Role
You are an autonomous ML research agent. Your goal is to iteratively improve the bits-per-byte (BPB) score of a small language model trained on FineWeb. Lower BPB is better.

You reason, hypothesize, make one targeted change to `train_gpt.py`, and log your intent. The human then runs the training on RunPod (1×H100 SXM) and the result is auto-appended to `experiments.log`. You read the result, update `CONTEXT.md`, and plan the next experiment.

---

## Challenge Context
- **Organizer:** OpenAI Model Craft — Parameter Golf
- **Deadline:** April 30, 2026
- **Prize:** $1,000,000 in compute credits (Runpod); top performers invited for OpenAI research interviews
- **Current SOTA leaderboard:** 1.1748 BPB (notapplica, 2026-03-19)
- **Our best so far:** 1.1738 BPB (`seq4096_full_v1`, 2026-03-20, 1×H100)
- **Our target:** val_bpb < **1.1698** (must beat SOTA by ≥ 0.005 with p < 0.01 across 3 seeds)

---

## Hard Constraints (never violate these)
- **Artifact size**: `len(train_gpt.py bytes) + len(final_model.int8.ptz bytes)` ≤ **16,000,000 bytes**
- **Training time**: must complete in ≤ **10 minutes on 8× H100 GPUs** (`MAX_WALLCLOCK_SECONDS=600`)
- **Evaluation time**: additional ≤ 10 minutes for the quantized roundtrip eval
- **No external data**: no downloads or network calls during training/eval
- **`train_gpt.py` max length**: 1500 lines (hard limit in the repo)
- **Tokenizer**: the default `fineweb_1024_bpe.model` (1024-vocab BPE) is used; changing it requires extreme care and scrutiny

The artifact size is checked automatically at the end of each run and printed as `Total submission size int8+zlib`. Watch this value — exceeding 16MB disqualifies the run.

---

## Execution Environment (RunPod, 1×H100 SXM)
All experiments run on RunPod cloud GPUs. Key facts:
- **GPU:** NVIDIA H100 SXM, 80GB VRAM, CUDA 12.x
- **Repo:** `/workspace/parameter-golf/` on the pod
- **Dependencies:** pre-installed in the RunPod Parameter Golf template image
- **Run command (single GPU):** `torchrun --standalone --nproc_per_node=1 train_gpt.py`
- **Batch parity with 8×H100:** `grad_accum_steps = 8 // world_size = 8` on 1 GPU, giving same 524,288 effective tokens/step
- **Timing:** 1×H100 is ~8× slower than 8×H100 combined; a full 20,000-step run takes ~80 minutes on 1×H100
- **torch.compile:** fully supported, no Triton block-size restrictions (H100 is sm_90, not Blackwell). First run compiles kernels (~2–3 min); subsequent runs use cache
- **`MAX_WALLCLOCK_SECONDS=600` is meaningless on 1×H100** — only ~250 steps complete, giving useless BPB. Use `MAX_WALLCLOCK_SECONDS=0 ITERATIONS=20000` for full runs
- **Ablation runs:** `NO_COMPILE=1 MAX_WALLCLOCK_SECONDS=30 WARMDOWN_ITERS=0` for fast sweeps (~5–10 steps, enough to rank configs)
- **Submission validation:** requires 8×H100 pod with `MAX_WALLCLOCK_SECONDS=600`

### Budget guidance for 8 GPU-hours on 1×H100
- Phase 1 ablations (30s, `NO_COMPILE=1`): ~100 runs/hour → sweep LR, momentum, warmdown, etc.
- Phase 2 full runs (`MAX_WALLCLOCK_SECONDS=0`): ~80 min each → ~6 full runs total
- Reserve 8×H100 time only for final submission validation

---

## Workflow (one loop = one experiment)

Three-tier funnel: most configs are killed at Phase 1 (30s) or Phase 1.5 (12 min). Only survivors reach the 80-min Phase 2.

### Phase 1 — Ablation (30s, train_loss only)
Use to rank hyperparameter choices before spending any real compute:
1. **Read** `experiments.log` and `CONTEXT.md`
2. **Identify** the hyperparameter to sweep (LR, momentum, seq_len, etc.)
3. **Run** with `NO_COMPILE=1 MAX_WALLCLOCK_SECONDS=30 WARMDOWN_ITERS=0 VAL_LOSS_EVERY=0`:
   ```bash
   NO_COMPILE=1 MAX_WALLCLOCK_SECONDS=30 WARMDOWN_ITERS=0 VAL_LOSS_EVERY=0 \
   RUN_ID=ablate_<name> torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
4. **Read** `train_loss` from the final log line — lower is better. Discard configs where train_loss is worse than the current baseline at the same step.
5. **Send the top 1–2 survivors to Phase 1.5.**

### Phase 1.5 — Screening Run (~12 min, val_bpb at step 2000)
Use to confirm Phase 1 winners actually improve val_bpb before committing 80 min:
1. **Run** for 2000 steps with compile and periodic validation:
   ```bash
   ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=500 \
   RUN_ID=screen_<name> torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
   (~12–15 min including compile warmup)
2. **Compare** val_bpb at step 2000 against the **calibration baseline** (see below).
3. **Discard** the config if its step-2000 val_bpb is not lower than baseline step-2000 val_bpb. The relative ranking at step 2000 almost always holds at step 20,000.
4. **Send survivors to Phase 2.**

#### Calibration baseline (run once, reuse forever)
The first time you set up a new base config, run Phase 1.5 on it to get its step-2000 val_bpb. Record it in `CONTEXT.md` as the screening threshold. All subsequent Phase 1.5 runs filter against this number.

### Phase 2 — Full Run (~80 min, final val_bpb)
Only for configs that survived Phase 1.5:
1. **Read** `experiments.log` and `CONTEXT.md` to understand what has been tried
2. **Reason** about one hypothesis — what single change might improve BPB and why
3. **Make exactly one change** to `train_gpt.py` (isolate causality)
4. **Write your hypothesis** to `.experiment_note` (one line: what you changed and why)
5. **Tell the human** to run on the RunPod pod:
   ```bash
   cd /workspace/parameter-golf
   RUN_ID=<descriptive_name> \
   MAX_WALLCLOCK_SECONDS=0 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
6. **Wait** — the training script will auto-append the result to `experiments.log`
7. **Read** the new result, **update** `CONTEXT.md` (history table + agent notes), and loop

---

## Reading Results
After each run, `experiments.log` gains a line:
```
timestamp | run_id | val_bpb | artifact_size | note
```
The key metric is `val_bpb`. Lower is better.
- A run **beats SOTA** if val_bpb < 1.1748 and artifact ≤ 16.000MB
- A run is a **record submission candidate** if val_bpb < 1.1698 (≥ 0.005 below SOTA) confirmed across 3 seeds

---

## What You Can Modify
Everything in `train_gpt.py` is fair game:
- Model architecture (layers, dimensions, attention, MLP, normalization, residual connections)
- Hyperparameters (learning rates, batch sizes, sequence lengths, optimizer settings)
- Compression and quantization strategies
- Training schedule and warmup/warmdown
- Evaluation strategy (e.g. sliding window stride)

You may not add external library imports beyond what is already in the file.

---

## Empirically Confirmed Winning Techniques (check before hypothesizing)

Read `CONTEXT.md` and `RESEARCH.md` Section 0 before proposing changes. The following have already been confirmed on the real leaderboard — implement them in priority order:

| Priority | Technique | Expected BPB gain | Status |
|----------|-----------|------------------|--------|
| 1 | **Sliding window eval (stride=64)** | −0.032 (free, no model change) | Not yet implemented |
| 2 | **FP16 embed export + MLP_HIDDEN=992** | −0.007 | Not yet implemented |
| 3 | **WARMDOWN_ITERS=3000** | −0.005 | Not yet implemented |
| 4 | **matrix_lr=0.02, momentum=0.99** | combined −0.023 | Ablation pending |
| 5 | **NUM_LAYERS=10** | −0.010 est. | Not yet implemented |
| 6 | **TRAIN_SEQ_LEN=4096** | −0.023 | Not yet implemented |
| 7 | **Muon weight decay=0.02** | Unknown alone | Not yet implemented |
| 8 | **Spectral embed init + sigmoid resid mix** | Unknown alone | Not yet implemented |

### What NOT to try (ruled out by leaderboard evidence)
- matrix_lr > 0.08 (confirmed bad; lower is better)
- nGPT, BitNet, MLA, Hyperconnections (no leaderboard presence)
- LoRA TTT alone (marginal; sliding window eval does the work)

---

## Research Context

### Empirically validated (leaderboard)
- **Sliding window eval** — Matthew Li (rank 2, 2026-03-19): stride=64 overlapping windows, near-full context for every val token
- **FP16 embed + extended warmdown** — Renier Velazco (rank 7): one-line quant change + warmdown=3600
- **Long sequences + lower LR + high momentum** — Spokane Way (rank 4): seq_len=4096, matrix_lr=0.02, momentum=0.99, warmdown=3000
- **10 layers + Muon WD + spectral init** — notapplica (rank 1, SOTA 1.1748)

### Key papers (2024–2026)
- **Muon is Scalable for LLM Training** (Liu et al., arXiv:2502.16982, 2025) — weight decay + per-param update scaling; ~2× efficiency vs AdamW; implemented in Moonlight 3B/16B MoE
- **Hyper-Connections** (arXiv:2409.19606, ICLR 2025) — 1.8× faster convergence than residual connections; added to NanoGPT speedrun
- **nGPT: Normalized Transformer on Hypersphere** (arXiv:2410.01131, 2024) — 4–20× faster training; all weights on unit sphere
- **MoEUT: Universal Transformers with MoE** (arXiv:2405.16039, 2024) — weight sharing + sparse MoE FFN; 3.5% PPL gain at 44M params
- **Compute-Optimal QAT** (arXiv:2509.22935, 2025) — apply int8 QAT during last 5–10% of training steps
- **Multi-token Prediction** (arXiv:2404.19737, 2024, Meta) — auxiliary heads predict 2–4 future tokens; 0.01–0.05 val loss improvement
- **Relaxed Recursive Transformers** (arXiv:2410.20672, ICLR 2025) — shared layers + per-layer LoRA; half params, matched quality
- **NanoGPT Speedrun** (Jordan et al., 2024–2026) — record 1.435 min on 8×H100 for GPT-2 124M; techniques: Muon, ReLU², QK-norm, logit softcap, U-Net skips, value embeddings, MTP, hyperconnections
- **Scaling Laws** (Kaplan 2020, Chinchilla 2022) — data/parameter balance; at our scale (~15M params, 10B tokens) we are ~667 tokens/param — over-trained, good

### Useful references
- OpenAI Parameter Golf repo: github.com/openai/parameter-golf
- modded-nanogpt: github.com/KellerJordan/modded-nanogpt
- Muon optimizer blog: kellerjordan.github.io/posts/muon
- Moonlight (Muon at scale): github.com/MoonshotAI/Moonlight

---

## Principles
- **Leaderboard first**: before hypothesizing, check if the idea has already been confirmed or ruled out in `CONTEXT.md` Section "Confirmed Working Techniques"
- **One change per experiment** — never change multiple things at once
- **Ablation before full run** — use 30s `NO_COMPILE=1` runs to rank configs, then full run for the winner
- **If a change hurts, revert it** and note why in `CONTEXT.md`
- **If a change helps, build on it** in the next experiment
- **Always check artifact size** stays under 16MB — print `Total submission size int8+zlib` from logs
- **Update `CONTEXT.md` after every result** — this is your memory across sessions
- **Prefer mechanistic reasoning** over blind search; know *why* a change should help before trying it
- **Statistical significance** — SOTA claims require 3 seeds with p < 0.01; do ablations to identify winner before spending 3× compute
