# Experiment Context

## Current SOTA (Leaderboard as of 2026-03-19)

| Rank | val_bpb | Author | Key Techniques |
|------|---------|--------|----------------|
| 1 | **1.1748** | notapplica | 10 layers + Muon WD=0.02 + FP16 embed + spectral init + sigmoid resid mix init + sliding window eval (stride=64) |
| 2 | 1.1925 | Matthew Li | Sliding window eval ONLY (stride=64, no model change) |
| 3 | 1.1928 | samacqua | LoRA TTT + sliding window eval + doc isolation |
| 4 | 1.2014 | Spokane Way | seq_len=4096 + matrix_lr=0.02 + momentum=0.99 + warmdown=3000 |
| 5 | 1.2060 | Spokane Way | seq_len=2048 |
| 6 | 1.2147 | Nan Liu | 10 layers + mixed int8/int6 |
| 7 | 1.2197 | Renier Velazco | FP16 embed export + matrix_lr=0.06 + warmdown=3600 |
| 8 | 1.2244 | Baseline | 9-layer 512-dim 1024-vocab tied-embed GQA |

**Non-record:** 1.2074 (Will DePue, 4h unlimited compute run)

**Our target:** beat rank 1 by ≥ 0.005 → need val_bpb < 1.1698

---

## Confirmed Working Techniques (from leaderboard evidence)

### Free gains (eval-only, no model change)
- **Sliding window eval (stride=64)**: -0.032 BPB. Score every token with 960+ context tokens instead of mean ~512. Eval time: ~70s (within 10min eval budget). This is the single biggest free gain available.

### Quantization improvements
- **FP16 embed export** (keep tied embedding in fp16, not int8): -0.007 BPB at baseline. Requires reducing MLP hidden from 1024→992 to stay under 16MB (+500KB overhead from fp16 vs int8 embed).
- **Extended warmdown** (3000-3600 steps vs 1200): creates tighter weight distributions, reduces post-quant penalty from 0.014 → 0.005 BPB.

### Architecture/training changes
- **10 layers** (vs 9): meaningful gain, confirmed by rank 1 and rank 6.
- **Muon weight decay = 0.02**: improves generalization + quantization robustness (rank 1).
- **Longer sequences (4096)**: +0.023 BPB improvement vs baseline. Forces 3/4 batch size (393,216 tokens/step) because 4096×96=393,216 fits; 4096×128=524,288 also fits (check).
- **matrix_lr = 0.02** (LOWER than baseline 0.04) + momentum = 0.99 + warmdown = 3000: optimal combo for 4k seq len config.
- **Spectral embedding init** (SVD power-law spectrum, S_k ~ k^{-0.5}): part of rank 1 recipe.
- **Sigmoid-scheduled residual mixing init**: part of rank 1 recipe.

### Techniques tried but marginal (LoRA TTT)
- LoRA TTT itself adds little; document isolation and sliding window do most of the work.

---

## Critical LR Finding (overrides prior hypothesis)

**The optimal LR is LOWER, not higher.** The winning configs:
- Rank 4 (1.2014): matrix_lr = **0.02** (baseline = 0.04), momentum = **0.99**, warmdown = 3000
- Rank 7 (1.2197): matrix_lr = **0.06**, warmdown = 3600

With 4k sequences: 0.02 + high momentum + longer warmdown = best combo.
With standard 1024 seq: 0.06 + longer warmdown = better than 0.04.

**LR ablation sweep range**: 0.01, 0.02, 0.03, 0.04, 0.06, 0.08 (focus on low end)

---

## Experiment History

| Date | run_id | val_bpb | Note |
|------|--------|---------|------|
| 2026-03-19 | LEADERBOARD_rank1 | 1.1748 | notapplica SOTA (reference) |
| 2026-03-20 | sota_baseline_confirm | **1.1752** | SOTA fully reproduced on 1×H100 ✓ |
| 2026-03-20 | seq4096_full_v1 | **1.1738** | SOTA + seq4096 + lr=0.02 + mom=0.99 + warmdown=3000. Δ=−0.0014, not enough alone |
| 2026-03-20 | screen_valemb_mtp | PENDING | Screening: seq4096 + value_emb + MTP(0.1) |

**Current best:** `seq4096_full_v1` at **1.1738 BPB** (Δ=−0.0014 vs SOTA)
**Target:** < 1.1698 BPB (need −0.004 more from current best)

---

## Agent Notes

### Current train_gpt.py defaults (as of 2026-03-20)

All SOTA features baked in as defaults: 10L, seq4096, lr=0.02, mom=0.99, warmdown=3000, batch=393216, MuonWD=0.02, FP16emb, spectral_init, sigmoid_resid, EVAL_STRIDE=64, val_emb, MTP(0.1).

### Next experiments (priority order)

1. **screen_valemb_mtp** (RUNNING) — 2000-step screening of value_emb + MTP on seq4096 base
2. If signal positive → **full run valemb_mtp** (~87 min, target ~1.168)
3. If still insufficient → **MTP weight ablation** (0.05, 0.1, 0.2) via 30s NO_COMPILE runs
4. **QAT during cooldown** — fake-int8 last 600 steps. Risky but could close quant gap.

### What NOT to try
- LRs > 0.08, nGPT, BitNet, MLA, byte-level tokenizers
- LoRA TTT alone (marginal, confirmed by leaderboard)
- Separate val_emb or MTP (test combined first for speed)
