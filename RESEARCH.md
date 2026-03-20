# Deep Research: Winning the OpenAI Parameter Golf Challenge
## Techniques for Sub-16MB, ≤10min/8xH100 Language Models

**Research Date:** March 2026 (leaderboard updated 2026-03-19)
**Challenge Context:** OpenAI Parameter Golf — train the best LM fitting in 16MB (code + compressed weights) in ≤10 min on 8×H100s, evaluated by BPB (bits-per-byte) on the FineWeb validation set.
**Current SOTA:** 1.1748 BPB (notapplica, 2026-03-19 — see leaderboard below)
**Unlimited-compute ceiling:** 1.2074 BPB (Will DePue, 4-hour run)
**Our target:** < 1.1698 BPB (beat SOTA by ≥ 0.005)

---

## 0. LEADERBOARD — EMPIRICALLY CONFIRMED TECHNIQUES (read first)

This section summarizes what has actually worked on the real challenge leaderboard as of 2026-03-19. Treat these as ground truth over any theoretical predictions below.

| Rank | val_bpb | Author | Techniques |
|------|---------|--------|------------|
| 1 | **1.1748** | notapplica | 10 layers + Muon WD=0.02 + FP16 embed + spectral embed init + sigmoid resid mix scheduling + sliding window eval stride=64 |
| 2 | 1.1925 | Matthew Li | **Sliding window eval stride=64 only** (no model change) |
| 3 | 1.1928 | samacqua | LoRA TTT + sliding window + doc isolation (TTT itself is marginal; eval changes do the work) |
| 4 | 1.2014 | Spokane Way | seq_len=4096 + matrix_lr=**0.02** + momentum=0.99 + warmdown=3000 |
| 5 | 1.2060 | Spokane Way | seq_len=2048 |
| 6 | 1.2147 | Nan Liu | 10 layers + mixed int8/int6 quantization |
| 7 | 1.2197 | Renier Velazco | FP16 embed export + matrix_lr=0.06 + warmdown=3600 |
| 8 | 1.2244 | Baseline | 9-layer 512-dim 1024-vocab tied-embed GQA |

### Key empirical findings from leaderboard

**1. Sliding window evaluation** is a free -0.032 BPB improvement:
- Replace non-overlapping 1024-token eval chunks with stride=64 overlapping windows
- Each token scored with ~960+ context tokens instead of mean ~512
- Evaluation time: ~70s on 8×H100 (within 10-minute eval budget)
- Implemented without changing the model — pure eval strategy change
- Accounts for the majority of gains in ranks 2 and 3

**2. FP16 embed export** (not int8) in quantization: -0.007 BPB
- Keep tied embedding matrix in fp16 rather than int8 during export
- Reduces quantization penalty from 0.007 to ~0.0005 BPB on embeddings
- Costs ~500KB extra → must reduce MLP_HIDDEN from 1024 → 992 to stay under 16MB

**3. Extended warmdown** (WARMDOWN_ITERS=3000-3600 vs 1200): reduces post-quant penalty
- Longer decay creates tighter weight distributions → better quantization robustness
- This is a free win — just change the hyperparameter

**4. LR is LOWER, not higher** (critical correction to naive hypothesis):
- Rank 4 winner: matrix_lr = 0.02 (half the baseline), momentum = 0.99
- Rank 7 winner: matrix_lr = 0.06 (1.5× baseline), warmdown = 3600
- Lower LR + higher momentum + longer warmdown = consistent pattern
- DO NOT explore matrix_lr > 0.08 — it hurts

**5. Longer sequences help**: 4096 > 2048 > 1024 for BPB
- 4096 requires slightly smaller batch (393,216 tokens/step vs 524,288)
- Compatible with 10-minute training on 8×H100

**6. 10 layers > 9 layers**: confirmed by ranks 1 and 6

**7. Muon weight decay (WD=0.02)**: in rank 1 recipe, improves generalization

**8. Spectral embedding init + sigmoid residual mix scheduling**: rank 1 specific, not tested independently

### What has NOT worked / not appeared in leaderboard
- High learning rates (> 0.08)
- nGPT, BitNet, MLA, Hyperconnections (no evidence of success in this challenge)
- LoRA TTT itself (marginal gain; eval strategy does the work)
- Byte-level tokenizers

---

## Understanding the Evaluation: BPB vs. Perplexity

`BPB = val_loss_nats / log(2) × (tokens / bytes)`

The baseline uses a 1024-token SentencePiece BPE vocabulary. Because the competition is tokenizer-agnostic (it measures actual bytes compressed), a tokenizer that achieves better bytes-per-token compression directly lowers BPB independent of model quality. The 10-minute constraint and 16MB artifact constraint together create a three-way trade-off between model quality, model size, and tokenizer efficiency.

**Key insight from the codebase:** The existing baseline already uses:
- Post-training int8 quantization + zlib compression to fit in 16MB
- Muon optimizer for matrix params, Adam for embeddings/scalars
- GQA (4 KV heads / 8 query heads)
- RoPE + RMSNorm (QK-norm via `F.rms_norm` on Q and K)
- ReLU² MLP activation (`relu(x).square()`)
- U-Net skip connections (encoder/decoder halves)
- Logit soft-capping (`30.0` default)
- Residual mix (learned interpolation between `x` and `x0` — the initial embedding)

The baseline is already quite modern. The question is what further improvements exist within the 16MB / 10-min budget.

---

## 1. Optimal Architecture for Small LMs: Depth vs. Width

### Findings

**MobileLLM (2024) and depth-preferring results:** Research on the depth-vs-width tradeoff for small transformers consistently shows deeper-and-narrower outperforms wider-and-shallower at fixed parameter count. The "Revisiting the Shape Convention of Transformer Language Models" paper (arXiv:2602.06471, 2026) introduces an hourglass FFN architecture and finds:
- At 113M parameters: hourglass (K=4, L=6) achieves **val PPL 35.10** vs conventional (L=12) at **36.44** — a 3.8% perplexity reduction.
- Optimal `d_model / num_layers` ratio: ~100–250.
- Optimal FFN bottleneck ratio `d_h / d_model ≈ 0.4`.
- **Redistribute parameters from FFN toward attention** rather than the conventional 3:1 FFN dominance.

**Key finding for sub-16MB:** At tiny scales (9 layers, 512 dim), the baseline already sits roughly in the right depth/width zone. More layers with lower dim may help — e.g., 16 layers × 384 dim, or 12 layers × 448 dim might be worth exploring if parameter counts hold.

**Parameter count estimation for 16MB budget:**
- After int8 + zlib compression, the 9×512 model uses ~15.8MB with ~47KB of code.
- Int8 gives ~1 byte/parameter; zlib typically compresses model weights by ~15–25%.
- A rough ceiling: ~12–14M parameters of "raw" weights (allowing ~25% zlib gain → ~10MB weights + 47KB code + quantization scales).
- With tied embeddings (1024 vocab × 512 dim = 0.5M params shared), the baseline has:
  - Embedding: 0.5M params × 512 dim = ~0.5MB at int8
  - 9 blocks: each has Q, K, V, O, FC1, FC2 matrices = 9 × (512² + 512×128 + 512×128 + 512² + 512×1024 + 1024×512) = ~24M params (pre-tied). Wait — at vocab=1024 and dim=512, MLP hidden = 2×512 = 1024, so each block: 512² (attn_q) + 128×512 (attn_k, since 4 KV heads × 64 head_dim) + 128×512 (attn_v) + 512² (attn_proj) + 512×1024 (mlp_fc) + 1024×512 (mlp_proj) ≈ 512²×2 + 512×128×2 + 512×1024×2 ≈ 0.52M + 0.13M + 1.05M ≈ 1.7M per block × 9 ≈ 15.3M matrix params.
  - At int8 + zlib (say 0.8 bytes/param effective), 15.3M params ≈ 12.2MB — which matches the observed ~15.8MB total (adding scales, norms, etc.).

**Practical limit:** With aggressive depth sharing or compression, you could fit more effective compute in 16MB.

### Recommendations
- Explore **12–14 layers × 384–448 dim** configs — likely more depth-efficient.
- Test `mlp_mult=3` or `mlp_mult=4` (wider MLP hidden) with narrower dim.
- The hourglass architecture (narrower FFN in middle layers) could recover parameters for more attention.

---

## 2. Aggressive Parameter Tying

### Embedding Tying (already in baseline)
The baseline already ties input/output embeddings, which is optimal for small vocabulary (1024 tokens). At 1024 vocab × 512 dim = ~0.5MB at int8, this saves ~0.5MB vs untied. The existing code uses a lower LR (`tied_embed_lr=0.05`) for tied embeddings vs regular embeddings (`embed_lr=0.6`) — important for stability.

### Cross-Layer Weight Sharing (Recursive / Universal Transformers)

**MoEUT (arXiv:2405.16039, 2024):** Mixture-of-Experts Universal Transformer combining layer sharing with sparse MoE. Key results:
- At 44M params: MoEUT PPL **18.30** vs standard transformer **18.97** on C4 (3.5% improvement).
- At 1040M params: **10.90** vs **11.15** baseline.
- Uses ~50% fewer MAC operations for equivalent parameter count.
- The improvement is **larger at smaller scales** (3.5% at 44M vs 1.1% at 1040M).
- Compatible constraint analysis: requires MoE routing overhead; adds complexity for a 16MB/10min budget.

**Relaxed Recursive Transformers (arXiv:2410.20672, ICLR 2025):**
- Convert existing models to recursive (weight-tied) form with layer-specific LoRA adapters.
- Recursive Gemma 1B (half the params) outperforms TinyLlama 1.1B and Pythia 1B by "up to 13.5 percentage points" on few-shot accuracy.
- Relaxed recursive (LoRA rank-512) matches original Gemma 2B (58.4% vs 58.6% few-shot) with half the parameters.
- **For pretraining from scratch:** Similar approach — share weights across all layers, add per-layer LoRA matrices of rank `r` to break symmetry. If `r = 16`, this adds `9 × 4 × 2 × (512 × 16 + 16 × 512)` = 9 × 4 × 2 × 16384 ≈ 9.4M extra params, which buys significant quality improvement at relatively small parameter cost vs the shared base.
- **Critical insight:** At 9 layers with a 1-block recursive design (all layers share the same weights), the "base" weight goes from 9×1.7M = 15.3M to just 1.7M params for the block. LoRA at rank 32 adds 9 × (512×32 + 32×512) × (number of linear layers per block) ≈ 9 × 2 × 16384 × 4 ≈ 9.4M. Total: ~11.1M, smaller than baseline, potentially better quality due to depth recurrence.

**Basis Sharing (arXiv:2410.03765, ICLR 2025):**
- SVD-based sharing of basis vectors across layers.
- At 50% compression on LLaMA-7B: PPL **19.99** vs SVD-LLM **23.97** (17% better).
- On GPT-2 (small) at 20% compression: **43.15 PPL** vs dynamic tying **49.37 PPL**.
- Works by horizontally concatenating weight matrices across layers and SVD-decomposing the result, then distributing basis vectors.
- **Verdict for this challenge:** Best applied as a post-training compression step, not pretraining architecture.

**ALBERT-style full layer tying:**
- ALBERT shares *all* transformer layers: effectively a 1-block recursive network run L times.
- Works surprisingly well but requires larger hidden dim to compensate for reduced expressive capacity.
- At sub-16MB: full ALBERT-style tying with 1 block of 512-dim run 9 times saves ~88% of block parameters, freeing budget for more iterations or a larger block.

### Recommendations
- **MoEUT** is the strongest technique here: try full layer sharing across all 9 blocks + top-2 MoE FFN with 8 experts. The 44M baseline result (3.5% PPL improvement) suggests meaningful gains at this scale.
- **Relaxed recursive + LoRA** for pretraining: share block weights, add rank-16 LoRA per layer. ~40% parameter reduction with minimal quality loss.
- Test ALBERT-style (single 512-dim block × 9 loops) vs. baseline — the parameter savings could allow 12–14 loops instead of 9.

---

## 3. Quantization-Aware Training (QAT), BitNets, Int8

### Post-Training Int8 (already in baseline)
The baseline already quantizes to int8 per-row + zlib compression. The baseline achieves 1.2244 BPB at int8 vs 1.1749 BPB pre-quantization (gap of 0.0495 BPB — 4.2% degradation). Reducing this quantization gap is a direct win.

### BitNet b1.58 (Ternary Weights, 1.58 bits)

**BitNet b1.58 Reloaded (arXiv:2407.09527, 2024)** — small model results:
| Hidden Size | Params | FP16 PPL | 1.58-bit PPL |
|-------------|--------|----------|--------------|
| 32 | 6M | 77.8 | 134.4 |
| 64 | 12M | 36.7 | 68.2 |
| 128 | 24M | 21.4 | 36.3 |
| 256 | 48M | 16.7 | 27.1 |

**Critical finding:** At small scales, 1.58-bit models need ~2× larger hidden dim to match 16-bit performance. At our scale (48M+ equivalent), a 1.58-bit model with 2× parameters (96M) might have similar quality to the 48M FP16 baseline — but 1.58 bits × 96M params ≈ 19MB, which exceeds the 16MB limit. However, ternary weights can be stored as 2 bits, so 96M × 2 bits = 24MB uncompressed, but with zlib compression of sparse ternary data (42% zeros in BitNet b1.58), compression ratios of 2–3× are achievable → ~8–12MB.

**BitNet b1.58 2B4T (arXiv:2504.12285, 2025):**
- 2B parameters, trained on 4T tokens.
- Weights quantized to {-1, 0, +1} (ternary).
- Memory: **0.4GB** vs 1.4–4.8GB for FP16 comparable models.
- Matches full-precision 2B models on benchmarks.
- Uses ReLU² activation (same as baseline!), RoPE.
- **Sparse-BitNet (arXiv:2603.05168, 2026):** Combines 1.58-bit quantization with 2:4 semi-structured sparsity. At 50% sparsity, BF16 models degrade >10% while BitNet degrades only 5.7%. Natural sparsity: ~42% zeros in ternary weights.

**pQuant (arXiv:2602.22592, 2026):**
- Decouples linear layers into a 1-bit dominant branch + small high-precision branch (4–5% of params).
- WikiText-2 PPL: **700M pQuant = 21.9** vs BitNet 700M = 27.6 (21% reduction).
- 700M pQuant matches 1.3B BitNet — significant parameter efficiency.
- The 8-bit branch is ~4–5% of params (stored in higher precision).

**Compute-Optimal QAT (arXiv:2509.22935, 2025):**
- For smaller models (86M params), require proportionally more QAT compute.
- **Key finding:** "Cooldown & QAT fusion" — perform learning rate decay jointly with QAT phase. Apply QAT during the last 5–15% of training steps.
- At low compute budgets (our scenario), **4-bit QAT** is optimal; 1–2 bit requires more compute.
- For 10-minute training, 4-bit symmetric QAT during the last 60–90 seconds could significantly reduce the 0.05 BPB quantization gap.

**EfficientQAT (arXiv:2407.11062, ACL 2025):**
- Two-phase: (1) block-wise QAT of all params; (2) end-to-end training of only quantization scales.
- Perplexity improvements of 0.5–0.89 over competing methods at 2–3 bit.

### FP8 Training (baseline already uses BF16)
- FP8 on H100 gives 2× throughput over BF16, 50% memory savings.
- The existing baseline uses `torch.autocast(dtype=torch.bfloat16)` — switching to FP8 training could allow more training iterations in 10 minutes.
- FP8 is essentially lossless at W8A8-FP precision; INT8 shows 1–3% accuracy degradation.

### Recommendations
- **Best near-term win:** Apply QAT during the cooldown phase (last ~600 steps). This could shrink the int8 quantization gap from 0.05 BPB toward ~0.02 BPB.
- **Longer-term:** Full BitNet b1.58 retraining with 2× hidden dim. Ternary weights at 2 bits/param + zlib compression of sparse ternary might allow a significantly larger model within 16MB.
- **pQuant hybrid:** 95% ternary + 5% FP16 for sensitive weights — reduces perplexity by 21% over pure 1-bit at matched parameter count.

---

## 4. Low-Rank Training: GaLore and LoRA Pretraining

### GaLore (arXiv:2403.03507, ICML 2024)
**Memory-efficient LLM training by gradient low-rank projection:**
- Projects optimizer state (not weights) into low-rank subspace.
- 60M model: PPL **34.88** (GaLore) vs **34.06** (full-rank AdamW) — <3% quality loss.
- Reduces optimizer memory by 65.5%; 8-bit GaLore reduces by 82.5%.
- **Not directly useful for this challenge:** The constraint is model size, not training memory. GaLore is useful if you want to train a larger model on limited GPU RAM, but with 8×H100 (80GB each), memory is not the bottleneck for a sub-16MB model.

### GaLore 2 (arXiv:2504.20437, 2025)
- Extended to 500B token training, addresses computational overhead.
- Llama 7B from scratch at 500B tokens — same perplexity as AdamW at 65% memory savings.

### LoRA for Pretraining (BabyLM 2025)
- "Pretraining Language Models with LoRA and Artificial Data" (ACL 2025 BabyLM track).
- Explored LoRA adapters during pretraining on BabyLM corpus.
- Results were mixed — LoRA during pretraining typically underperforms full-rank training unless combined with weight sharing (as in Relaxed Recursive Transformers).

### Recommendations
- **GaLore / LoRA pretraining** is not directly useful for this challenge's size constraint.
- **The relevant use of LoRA** is within a weight-sharing architecture (Relaxed Recursive Transformers): low-rank per-layer adapters break symmetry while keeping most parameters shared. This is a compression technique, not purely a training efficiency technique.

---

## 5. Novel/Compressed Tokenizers

### Vocabulary Size and BPB

The existing 1024-token SentencePiece BPE tokenizer achieves some specific bytes-per-token ratio on FineWeb. The BPB formula is:

`BPB = (cross_entropy_nats / log(2)) × (tokens / bytes)`

A tokenizer producing fewer tokens per byte (worse compression) *increases* the `tokens/bytes` ratio, which *decreases* BPB — but this is counteracted by the model needing to model longer sequences. The optimal vocabulary size for BPB optimization at constrained model size is not obvious.

**Vocabulary size tradeoffs:**
- 1024-vocab: embedding table = 1024 × 512 = ~0.5MB at int8. Tiny! Leaves most of the 16MB budget for transformer weights.
- 4096-vocab: embedding table = 4096 × 512 = ~2MB at int8. Meaningfully larger; reduces budget for transformer blocks.
- 32768-vocab: 32768 × 512 = ~16MB — consumes the entire budget, leaving nothing for transformer blocks.

**Byte-level (256-vocab):** Embedding = 256 × 512 = 128KB — essentially free. But model must handle longer sequences (all English text expands ~3.5× in byte length vs BPE-1024). Sequence length would need to increase, increasing compute.

**SuperBPE (COLM 2025):** Two-pass BPE: first learns standard tokens, then cross-word "superword" tokens. Achieves up to **15% improvement in bytes-per-token** over standard BPE. If applied to the 1024-vocab setting, this would directly reduce the `tokens/bytes` ratio and lower BPB.

**BoundlessBPE (COLM 2025):** Relaxes pre-tokenization boundary constraints. Up to 15% improvement in bytes-per-token.

**Length-MAX Tokenizer (arXiv:2511.20849, 2025):** Alternative tokenization objectives optimizing for maximal compression.

### Byte Latent Transformer (BLT, Meta, arXiv:2412.09871, 2025)
- Byte-level model with dynamic patching based on entropy.
- Matches LLaMA 3 at 8B scale.
- Uses **a small byte-level LM to determine patch boundaries** (costly for a sub-16MB system, as the patcher itself takes parameters).
- At 8B scale, BLT uses 50% fewer inference FLOPs via larger patch sizes (6–8 bytes).
- **For our challenge:** The BLT architecture itself is too complex for 16MB, but the insight about dynamic patching for byte-level models is relevant.

### Optimal Vocabulary Size for 16MB
Given the int8 + zlib constraint and 16MB budget:
- **1024-vocab** (current baseline): ~0.5MB for embeddings (tied). Very space-efficient.
- **2048-vocab**: ~1MB. Might improve tokenizer compression ratio meaningfully for FineWeb English.
- **4096-vocab**: ~2MB. Higher quality tokenization but eats into transformer budget.
- The competition explicitly says "edit the tokenizer" submissions will be examined carefully (preventing BPB gaming via tokenizer manipulation).

**Practical recommendation:** The 1024-vocab BPE is already well-chosen for the size constraint. Switching to SuperBPE or BoundlessBPE to improve bytes-per-token while keeping 1024-vocab could directly improve BPB by ~10–15%.

---

## 6. Test-Time Compute: Recurrent / Looped Transformers

### Looped Transformers (Depth Recurrence at Inference)

**Parallel Loop Transformer (arXiv:2510.24824, 2025):**
- Breaks sequential dependency of looped transformers via Cross-Loop Parallelism (CLP).
- Different loops computed for different tokens simultaneously in a single pass.
- Claims "high accuracy of looped model but with almost no extra latency."
- Shares KV cache from first loop with all subsequent loops.

**Recurrent Depth / Test-Time Scaling (Multiple papers, 2025):**
- OLMo-2 1B retrofitted as 7-4k-5 (7 encoder layers, 4 recursive layers × k iterations, 5 decoder layers).
- "A 3.5B recurrent model with 32 iterations can match or outperform a 50B parameter model."
- Training with many recurrent iterations is significantly slower than fixed-depth training.

**intra-Layer Recurrence (arXiv:2505.01855, 2025):**
- Select layers re-entered independently within a single forward pass.
- Improves perplexity without increasing parameter count.

**LoopFormer:**
- Budget-conditioned inference: conditioning each loop step on internal time t and step size Δt.
- Treats iterative refinement as a trajectory in representation space.

### Implications for Parameter Golf

**Key insight:** The 16MB artifact constraint limits parameters but NOT inference compute. An architecture that loops the same weights multiple times during inference uses more FLOPs per token but has far fewer parameters. This is the core parameter golf opportunity.

**Universal Transformer (ALBERT-style pretraining + looping at inference):**
- 1 shared block (1.7M params at 512-dim) × 9 loops = same depth as baseline but only ~1.7M matrix params.
- Savings: ~13.6M matrix params → ~13.6MB → freed budget can be reallocated.
- If reallocated to a larger block: 1 block × 16.3M params (≈864-dim equivalent) × 9 loops.
- Or: 1 block × 1.7M params but with 16× more loops (144 loops) for much deeper effective reasoning.

**MoEUT** is the strongest published variant: weight-shared blocks + MoE FFN layers. At 44M params, MoEUT achieves 3.5% better PPL than a standard 44M transformer.

**Critical constraint:** Training a recurrent/looped model from scratch in 10 minutes is harder — backpropagation through many loops increases training time. Need to limit training loops to 2–4 and possibly evaluate with more loops (test-time compute scaling). The challenge is whether this generalizes.

### Recommendations
- **Most promising:** ALBERT-style (single block, 9 loops) as the base, but replace the FFN with a small MoE (4–8 experts, top-2 activation) to increase effective capacity. This is exactly MoEUT.
- If training time allows, use 12–18 loops during training; this typically improves quality at inference.
- The 10-minute training limit is tight for deeply looped models; benchmark carefully.

---

## 7. Depth Recurrence / Universal Transformers

### Key Papers

**MoEUT (arXiv:2405.16039, 2024):**
Already covered in §2. The definitive positive result for weight-shared LMs at small scale: consistently outperforms standard transformers from 44M to 1B params on C4/SlimPajama. The key ingredients:
1. Layer grouping (multiple non-shared blocks per "group")
2. Per-group MoE (sparse expert FFN)
3. "Peri-layernorm" (LN only before linear layers preceding sigmoid/softmax)

**SReT (2024):** 13–15M parameter model scaled to 100–1000 recursive applications. Demonstrates feasibility at very small scale.

**Retrofitting Recurrence (arXiv:2511.07384, 2025):**
- Converts pretrained models (TinyLlama 1.1B, OLMo-2 1B, Llama 3.2 1B) to depth-recurrent form.
- **Key advantage:** Increasing loop count does not increase memory consumption or context size during inference.
- "Depth-recurrence has the advantage that increasing compute does not increase memory consumption."

**Relaxed Recursive Transformers (ICLR 2025, arXiv:2410.20672):**
- Pretrain or uptrain with shared layers + per-layer LoRA adapters.
- The LoRA modules allow each "iteration" of the loop to behave differently.
- Recursive Gemma 1B achieves performance close to Gemma 2B (half the parameters).

### Recommendations for Challenge
1. **Start with MoEUT-style architecture**: 1 shared block + MoE FFN, looped 9× during training. Add per-layer scalar gates (like the baseline's `attn_scale` and `mlp_scale`) to differentiate layers.
2. **Add iteration embeddings**: A learnable embedding indexed by loop iteration `t` (like nGPT's adaptation), allowing the network to behave differently at each depth.
3. **Train with fewer loops (e.g., 6)** during training to keep within 10 minutes, then evaluate with 9 loops.

---

## 8. Muon Optimizer and Recent Training Optimizers

### Muon (Already in baseline)
The baseline already uses Muon for 2D matrix params. Key properties:
- Applies Newton-Schulz orthogonalization to gradient updates.
- 1.35× faster training than AdamW on NanoGPT.
- ~2× computational efficiency vs AdamW at compute-optimal scale.
- Only 0.7% FLOP overhead for NanoGPT-scale models.
- Applied to Q, K, V, and projection matrices; NOT to embeddings (Adam used there).

**Muon scalability (arXiv:2502.16982, 2025):**
- Scaled to 3B/16B MoE models (Moonlight).
- Key insights for small models: (1) add weight decay; (2) adjust per-parameter update scale.
- Roughly 52% of AdamW FLOPs needed for equivalent quality.

**NanoGPT speedrun learnings:**
The modded-nanogpt record (1.435 minutes on 8×H100 for 3.28 val loss on GPT-2 124M) provides a gold mine of tested techniques. Key innovations beyond the current baseline:
- **Value embeddings**: Extra embeddings mixed into attention values (from Zhou et al. 2024). Direct improvement to attention expressivity.
- **U-Net skip connections**: Already in the baseline.
- **Multi-token prediction**: Auxiliary heads predicting 2–4 future tokens (added mid-training after embedding untying).
- **QK-norm**: Already in the baseline via `F.rms_norm(q, ...)`.
- **Logit soft-capping**: Already in the baseline.
- **Bigram hash embeddings**: Context-dependent token embeddings using bigram statistics.
- **ReLU²**: Already in the baseline.
- **Flash Attention 3 + sliding window patterns**: Gemma 2-style alternating short/long attention windows.

**nGPT (arXiv:2410.01131, 2024):**
- Fully normalized transformer (all weights, embeddings, hidden states on unit hypersphere).
- **4–20× faster training** depending on context length.
- Reaches same validation loss as GPT with 10% of iterations at 4k context.
- High per-step overhead (~80% at 4k context) but net speedup is 2–5×.
- Compatible with the 10-minute constraint? At 4× speedup, can train 4× more steps → explore 4× more of the loss landscape.
- **Key insight for our challenge:** With a 10-minute hard cap, training efficiency is as important as architecture quality. nGPT's speedup could allow 4× more training steps, likely improving BPB significantly.

**Dynamic Tanh / DyT (arXiv:2503.10622, CVPR 2025):**
- Replaces LayerNorm/RMSNorm with element-wise `DyT(x) = tanh(αx)`.
- Matches or exceeds normalized baselines in all experiments.
- Simpler, no mean/variance computation.
- **Potential benefit:** Faster per-step training (removes normalization compute), matches quality.

**Muon variants:**
- **MuonBP** (OpenReview, 2025): Block-periodic orthogonalization — reduces Newton-Schulz overhead.
- **Turbo-Muon**: Spectral preconditioning of Newton-Schulz step.
- **NorMuon**: Normalized variant for improved stability.

### Recommendations
- The baseline optimizer setup (Muon + Adam) is already state-of-the-art.
- **nGPT normalization scheme** could provide 2–4× training speedup within the 10-minute budget — huge potential win.
- Add **value embeddings** (1–2% parameter overhead, measurable BPB improvement based on NanoGPT speedrun).
- Consider **multi-token prediction** as an auxiliary loss (Facebook Research, 2024) — predicting 2–4 future tokens simultaneously improves sample efficiency and coding-like tasks.

---

## 9. Knowledge Distillation for Small Models

### MiniPLM (arXiv:2410.17215, ICLR 2025)
- Knowledge distillation during pretraining by reweighting training data based on teacher LM's knowledge.
- Uses 1.8B teacher to guide pretraining of 200M, 500M, 1.2B students.
- **Improves downstream task performance on 9 benchmarks**.
- **Reduces data demand by 2.4×** — achieves the same quality with 2.4× fewer tokens.
- Works offline: teacher inference done once, then student trains on reweighted data.
- **Critical for our challenge:** With a 10-minute training limit, data efficiency is crucial. If MiniPLM-style distillation allows 2.4× better token efficiency, that's equivalent to training for 24 minutes on unweighted data — far exceeding what we can do in 10 minutes.

**Practical application:** Pre-compute importance weights for FineWeb training data using a large teacher (e.g., GPT-2 1.5B or similar). The data is fixed (FineWeb), so we can prioritize high-information samples. This is offline precomputation not counted against the 10-minute training budget.

### Pre-training Distillation (ACL 2025)
- Various approaches to distilling larger models during pretraining.
- Chain-of-thought and RL guidance for small student models.

### Distilling Token-Trained → Byte-Level (arXiv:2602.01007, 2026)
- Two-stage curriculum: embedding alignment + byte-level SFT.
- Retains 92%+ of teacher performance with 125B training bytes.
- Not directly applicable to our challenge (we're not distilling to byte-level).

### Recommendations
- **Pre-compute MiniPLM-style data weights** on FineWeb using a freely available large LM (GPT-2 XL, or any open model). Train the small model on the highest-weight subset.
- **Response distillation:** Use a larger teacher's output distribution as the training signal (knowledge distillation loss = KL divergence vs teacher's softmax). This requires a fixed teacher inference pass over FineWeb (offline).
- **Forward KL vs Reverse KL:** Recent research (2024) shows forward KL (standard) is better for student models that need to cover all modes.

---

## 10. MLA (Multi-Head Latent Attention) and GQA Benefits

### Multi-Head Latent Attention (MLA, DeepSeek-V2, 2024)

**Latent MHA for Small LMs (arXiv:2506.09342, 2025):**
- Tested on 17.5M–202.7M parameter models.
- MLA + RoPE (half-rank latent dimension, r = d/2): only **0.3% increase in validation loss** vs standard MHA.
- MLA + RoPE with r = d/4: near-MHA performance with **6× KV-cache reduction**.
- **Critical finding: RoPE is essential for MLA at small scales.** Without RoPE, MLA underperforms MHA by 3–5%. With RoPE, it *surpasses* MHA by 2%.
- GPT-4 evaluation: MLA+RoPE scores 7.4/10 vs MHA 6.2/10 for overall quality.
- Phase transition at r = d/16: below this, severe degradation.

**The baseline already uses GQA (4 KV heads / 8 query heads)**, which is a simpler KV-compression technique. MLA is strictly more expressive than GQA at matched KV-cache overhead (proven theoretically by TransMLA).

**TransMLA (arXiv:2502.07864, 2025):**
- "For the same KV Cache overhead, MLA consistently offers greater expressive power than GQA."
- Converts GQA-based models to MLA post-training.
- MLA+RoPE with r = d/2: 45% KV-cache memory reduction with only 0.3% validation loss increase.

**Grouped Latent Attention (GLA, 2024):**
- Compresses token representations into low-dimensional latent vectors.
- Combines aspects of MLA and linear attention.

### MLA vs GQA for Parameter Golf

The baseline uses GQA with 4 KV heads (vs 8 query heads) — this already saves `(8-4)/8 = 50%` of K and V matrix parameters. At 512-dim with 8 heads (head_dim=64):
- Standard MHA: K matrix = 512×512 = 262K params per block.
- GQA (4 KV): K matrix = 512×256 = 131K params per block.
- MLA (r = d/4 = 128): Compressed KV = 512×128 = 65K params for the down-projection + 2 × (128 × 256) for decompression = 65K + 64K = 129K. About the same as GQA but with higher expressive power.
- MLA (r = d/8 = 64): 32K + 32K = 64K — half the GQA memory, with 2% better quality.

**For the challenge:** Replacing GQA with MLA at r = d/8 would:
1. Reduce K and V parameters by ~50% vs GQA (save ~8M params across 9 layers).
2. Improve quality by ~2% (from the "surpasses vanilla attention" finding).
3. Free ~8MB of the 16MB budget, allowing larger other components.

### Recommendations
- **Switch from GQA to MLA with r = d/8**: saves ~8M params while improving attention quality by 2%. High-value change.
- Use RoPE on both the compressed and decompressed representations.
- The MLA down-projection (C matrix) can be absorbed into the RoPE frame for parameter efficiency.

---

## 11. NanoGPT Speedrunning Results

### Current Record (as of August 2025)
The modded-nanogpt project achieved **1.435 minutes** to reach ≤3.28 val loss on GPT-2 124M (FineWeb, 8×H100s). This is a 97% reduction from the original 45 minutes.

### Key Techniques (In Order of Addition)
| Record | Time | Innovation |
|--------|------|-----------|
| 1 | ~45 min | Baseline llm.c |
| 3 | 24.9 min | Muon optimizer |
| 12 | 5.03 min | FlexAttention (64K context) |
| 20 | 2.992 min | Sub-3-min barrier broken |
| 53 | 1.988 min | Multi-token prediction |
| 77 | 1.435 min | Simplified hyperconnections |

### Techniques Confirmed to Help (NanoGPT 124M scale)
- **Muon optimizer**: ~1.35× speedup over AdamW (already in baseline).
- **ReLU²**: Discovered in NanoGPT context (already in baseline).
- **QK-norm**: Already in baseline.
- **Logit soft-capping**: Already in baseline.
- **U-Net skip connections**: Already in baseline.
- **Value embeddings** (Zhou et al. 2024): Mixes token embeddings into attention values. Small parameter cost, consistent improvement.
- **Multi-token prediction (MTP)**: Auxiliary heads predicting 2–4 future tokens. Added mid-training. Improves by ~0.01–0.05 val loss.
- **Hyperconnections (ICLR 2025)**: Learned alternative to residual connections. DHC×4 configuration: **1.8× faster convergence**, at 1B scale validation loss from 2.811 → 2.781 (-1.1%). "No spikes throughout training."
- **Sliding window attention (short-long pattern)**: Alternating local (128-token) and global (384-token) attention windows reduces FLOPs while maintaining quality.

### Hyper-Connections (arXiv:2409.19606, ICLR 2025)
- At 1B parameters: DHC×4 (expansion rate 4) improves validation loss by ~1.1% and accuracy by 1.3 points.
- Converges **1.8× faster** — critical for the 10-minute training budget.
- "No training spikes" — more stable training, important for short training runs.
- At 7B: validation PPL from 14.316 → 14.023 (2.0% improvement).
- Added to NanoGPT speedrun as the most recent major improvement.
- For OLMoE-1B-7B (MoE): ARC-Challenge +6 points, MMLU +1.2 points.

### Automated LLM Speedrunning Benchmark (arXiv:2506.22419, 2025)
Formal study of which NanoGPT innovations transfer to other settings. Top reproducible improvements:
1. Muon optimizer
2. QK-norm
3. ReLU²
4. Multi-token prediction
5. Value embeddings
6. U-Net skip connections

### Recommendations
- **Hyperconnections** are the most underexplored major gain: 1.8× convergence speedup is huge for a 10-minute budget. Practically equivalent to having 24 minutes of training instead of 10.
- **Multi-token prediction**: As an auxiliary loss during training, improves language modeling quality. Meta's MTP paper shows 12–17% improvement on code tasks; general LM improvement is 0.01–0.05 val loss.
- **Value embeddings**: Simple to add (~0.5% extra parameters for the embedding table), consistent improvement across scales.

---

## 12. BitNet, MatMul-Free LMs, and Extreme Compression

### Scalable MatMul-Free Language Modeling (arXiv:2406.02528, NeurIPS 2024)
- Replaces all matrix multiplications with ternary accumulations.
- Uses MLGRU (MatMul-free Linear Gated Recurrent Unit) instead of attention.
- Models up to 2.7B params; performance "comparable to state-of-the-art Transformers" at 2.7B.
- 61% training memory reduction; 4× neuromorphic hardware throughput.
- **Performance gap:** The gap closes as model size grows. At 2.7B, slightly above Transformer++ on some benchmarks. At smaller scales (our range), likely below full-precision Transformers.
- **For the challenge:** The ~3× expected quality gap at small scale (based on BitNet b1.58 data above) makes this suboptimal compared to QAT-based approaches.

### BitNet (arXiv:2310.11453, 2023 / JMLR 2024)
- 1-bit weights, full-precision activations.
- Competitive with 8-bit baselines at scale, not competitive at small scale.

### BitNet b1.58 (2024)
- Ternary weights {-1, 0, +1} — 1.58 bits/param.
- Matches FP16 at 2B+ parameter scale when trained identically.
- At small scales (6–48M), needs ~2× larger model for equivalent quality.
- **Storage:** Ternary weights pack to 2 bits → 4× compression vs FP32, 2× vs FP16/BF16.
- With zlib: ternary weights (42% zeros) compress ~2–3× further → **effective ~1 bit/param or less**.
- Enables: 96M ternary params ≈ 12MB with good compression (within 16MB budget).

### pQuant (2026) — Hybrid Approach
- 95–96% 1-bit + 4–5% high-precision sensitive weights.
- 700M params matches 1.3B BitNet.
- **Practical for our scale:** A 30M-param pQuant model (our rough budget) with 29M ternary + 1M FP16 might outperform a 15M FP16 model after quantization.

### Sparse-BitNet (arXiv:2603.05168, 2026)
- BitNet b1.58 + N:M sparsification (50% structured sparsity).
- 1.58-bit + 50% sparsity: only 5.7% perplexity degradation (vs 18.8% for BF16 at same sparsity).
- Enables **~4-bit effective precision** through joint quantization + sparsity.
- Custom sparse tensor cores for 1.30× inference speedup.

### Recommendations
- **BitNet b1.58** is the most promising extreme compression approach for this challenge. Key strategy:
  1. Design a 30–40M parameter model (2× baseline params in ternary).
  2. Train with ternary weights from scratch (BitLinear layers).
  3. Compress: 2 bits/ternary param × 35M params = 70Mb = 8.75MB raw. Zlib on sparse ternary: ~5–6MB. Plus 47KB code + quantization scales ≈ 6MB total — well under 16MB.
  4. The 2× larger ternary model should match or exceed the FP16 baseline.
- Alternatively: combine with MoEUT (weight-shared + MoE FFN) for even more parameter efficiency.

---

## 13. Scaling Laws at Tiny Scale and Optimal Hyperparameters

### Chinchilla and Over-Training

**Chinchilla-optimal:** 20 tokens/parameter. For a 15M-param model: optimal training tokens ≈ 300M.

**Over-training for inference efficiency (2024 findings):**
- LLaMA 3 8B: trained on 15T tokens ≈ 1875 tokens/param (94× Chinchilla).
- Phi-3: 870 tokens/param.
- Recent research: loss continues improving up to 10,000 tokens/param.

**For our challenge with a fixed 10-minute training window:**
- 10 minutes × 8×H100s at ~500K tokens/step × ~20K steps ≈ 10B tokens.
- For a 15M-param model: 10B tokens / 15M params ≈ 667 tokens/param — highly over-trained, which is good for quality.
- For a 30M-param model (BitNet strategy): 10B / 30M ≈ 333 tokens/param.
- **Practical finding:** Even at 333 tokens/param, small models are "over-trained" vs Chinchilla, which tends to improve quality.

**The actual baseline:** 20K iterations × 512K tokens/step = ~10.5B tokens on 15M params → 700 tokens/param. Consistent with the above analysis.

### Optimal Hyperparameters for the Baseline Scale

**Learning rate schedule (WSD):**
The baseline uses a warmup-stable-decay (WSD) pattern with 20 warmup steps and 1200 warmdown steps out of 20K total. Research from 2025 confirms:
- WSD outperforms cosine for short training runs.
- Linear LR decay during cooldown works well.
- Skipping warmup doesn't hurt for small models.

**Batch size:**
- Baseline: 512K tokens/step globally (8 GPUs × grad_accum_steps × local_batch).
- Research shows larger batch sizes help efficiency up to ~1M tokens/step for models of this size.

**Gradient clipping:**
- Baseline has `grad_clip_norm=0.0` (disabled). Enabling with ~1.0 might help stability, especially with aggressive learning rates.

**Sequence length:**
- Baseline: 1024 tokens. Longer sequences capture more long-range dependencies but use more memory.
- At 16GB activations, sequence length of 2048 is feasible and might improve BPB on FineWeb's longer documents.

### Scaling Laws Specific to Tiny Models

**Recent 2026 findings (arXiv:2602.06797):**
- "Optimal LR schedules under functional scaling laws: power decay and WSD."
- For compute-optimal tiny models: power decay (LR ∝ t^{-0.5}) slightly outperforms cosine.
- WSD with well-tuned decay fraction (15–20%) matches compute-optimal cosine.

**Compute-optimal QAT timing:**
- For 86M param models: apply QAT during the last 5–10% of training.
- For our model size (~15M): last 1000–2000 steps of 20K.

---

## Summary: Priority Ranking of Techniques

**Updated 2026-03-19 to reflect empirical leaderboard evidence.**

### Tier 1: Empirically Confirmed, Implement Immediately

| Technique | Expected BPB Gain | Evidence | Difficulty |
|-----------|------------------|----------|------------|
| Sliding window eval (stride=64) | **-0.032 BPB** (free, no model change) | Rank 2 and 3 confirmed | Low |
| FP16 embed export + MLP_HIDDEN=992 | **-0.007 BPB** | Rank 7 confirmed | Low |
| Extended warmdown (WARMDOWN_ITERS=3000) | **-0.005 BPB** | Rank 4 and 7 | Low |
| Lower LR (matrix_lr=0.02) + momentum=0.99 | **-0.023 BPB** (combined with 4k) | Rank 4 | Low |
| 10 layers (NUM_LAYERS=10) | **-0.010 BPB est.** | Ranks 1 and 6 | Low |
| 4096 sequence length | **-0.023 BPB** (with LR tuning) | Rank 4 | Low |
| Muon weight decay (WD=0.02) | Unknown alone, part of rank 1 | Rank 1 | Medium |
| Spectral embed init + sigmoid resid mix | Unknown alone, part of rank 1 | Rank 1 | Medium |

### Tier 2: Theoretically Sound, No Leaderboard Evidence Yet

| Technique | Expected BPB Gain | Compatibility | Difficulty |
|-----------|------------------|---------------|------------|
| QAT during cooldown phase | ~0.03–0.04 BPB (reduces quant gap) | ✅ Full | Low |
| Value embeddings | ~0.005 BPB | ✅ Full | Low |
| Multi-token prediction (auxiliary) | ~0.01–0.02 BPB | ✅ Full | Low |
| Hyperconnections (DHC×2-4) | ~1–2% (1.8× convergence) | ✅ Full | Medium |
| MLA (r=d/8) replacing GQA | ~2% quality + param savings | ✅ Full | Medium |

### Tier 3: High-Risk (No Leaderboard Evidence, Not Recommended Yet)

| Technique | Notes |
|-----------|-------|
| nGPT normalization | Theoretically fast but zero leaderboard presence |
| MoEUT (weight sharing + MoE FFN) | Complex, 10-min budget tight for looped models |
| BitNet b1.58 | No entry used it; too complex for this timeframe |
| SuperBPE tokenizer | Scrutinized by organizers; complex to implement correctly |

---

## Concrete BPB Estimates (Updated 2026-03-19)

Current leaderboard SOTA: **1.1748 BPB** (notapplica)
Our target: **< 1.1698 BPB** (must beat by ≥ 0.005)
Baseline: 1.2244 BPB

**Empirically confirmed compounding stack** (starting from baseline):
- Sliding window eval (stride=64): -0.032 → **1.1924 BPB**
- FP16 embed + MLP_HIDDEN=992: -0.007 → **1.1854 BPB**
- Extended warmdown (3000): -0.005 → **1.1804 BPB**
- 10 layers: -0.010 (est.) → **1.1704 BPB**
- LR=0.02 + momentum=0.99: already accounted in rank 4 result
- Muon WD=0.02 + spectral init: unknown gain on top of above

This conservative stack already reaches ~1.1704, which is near but not past SOTA. Adding Muon WD and spectral init (rank 1's remaining ingredients) should push past.

**Aggressive stack** (adding Tier 2 items):
- QAT during cooldown: -0.005–0.015 BPB
- Value embeddings / MTP: -0.005–0.010 BPB
- Hyperconnections: -0.010–0.020 BPB
- **Optimistic ceiling: ~1.13–1.15 BPB**

---

## Architecture Recipe: Recommended Starting Point

Based on the research, here is a concrete architecture that could meaningfully improve over the baseline:

```
Vocab: 1024 (keep existing BPE tokenizer)
Model dim: 512
Layers: 9 (keep depth)
Attention: MLA with latent dim r=64 (d/8) + RoPE
  - Down-projection: 512 → 64 (K and V jointly)
  - Up-projection: 64 → 256 (K) + 64 → 256 (V)
  - Q heads: 8, head_dim: 64
MLP: ReLU² with mult=2 (keep current)
Normalization: RMSNorm (keep) or try DyT for speed
Connections: Hyperconnections (DHC expansion=2) instead of residual
Skip: Keep U-Net skip pattern
Value embeddings: Add embedding-weighted values in attention
Auxiliary loss: 2-token prediction head (shared weights, small overhead)
Optimizer: Muon (matrices) + Adam (embeddings/scalars) [keep current]
QAT: Apply int8 QAT during last 2000 steps (10% of training)
Logit softcap: Keep
```

Parameter count estimate:
- Embedding (tied): 1024 × 512 = 0.5M
- 9 × Block params:
  - Q: 512 × 512 = 0.26M
  - K+V (MLA, r=64): 512×64 (down) + 2×(64×256) (up) = 32K + 32K = 64K (vs 131K for GQA — saves 67K per block × 9 = 0.6M total)
  - O: 512 × 512 = 0.26M
  - MLP: 512×1024 + 1024×512 = 1.05M
  - Norms/scales: ~32K
  - Total per block: ~1.63M
- 9 blocks: 14.7M
- Hyperconnection parameters: ~negligible (small scalar gates)
- Value embedding: 1024 × 512 = 0.5M (can be tied to tok_emb)
- **Total: ~15.7M params** — slightly smaller than baseline due to MLA savings

At int8 + zlib: ~12–14MB + 47KB code + scales ≈ 14MB total — comfortably within 16MB.

---

## References and Links

### Key Papers
- [BitNet b1.58: Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)
- [BitNet b1.58 Reloaded (small models)](https://arxiv.org/html/2407.09527v1)
- [Sparse-BitNet](https://arxiv.org/abs/2603.05168)
- [pQuant](https://arxiv.org/abs/2602.22592)
- [MoEUT: Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2405.16039)
- [Relaxed Recursive Transformers (ICLR 2025)](https://arxiv.org/abs/2410.20672)
- [Hyper-Connections (ICLR 2025)](https://arxiv.org/abs/2409.19606)
- [nGPT: Normalized Transformer](https://arxiv.org/abs/2410.01131)
- [MiniPLM: KD for Pre-training (ICLR 2025)](https://arxiv.org/abs/2410.17215)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [GaLore: Memory-Efficient LLM Training](https://arxiv.org/abs/2403.03507)
- [Latent MHA for Small LMs](https://arxiv.org/html/2506.09342v1)
- [TransMLA: MLA is All You Need](https://arxiv.org/abs/2502.07864)
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871)
- [MatMul-Free Language Modeling](https://arxiv.org/abs/2406.02528)
- [Compute-Optimal QAT](https://arxiv.org/html/2509.22935v1)
- [Revisiting Shape Convention of Transformer LMs](https://arxiv.org/html/2602.06471v1)
- [Basis Sharing (ICLR 2025)](https://arxiv.org/abs/2410.03765)
- [Parallel Loop Transformer](https://arxiv.org/abs/2510.24824)
- [Distilling Token-Trained Models to Byte-Level](https://arxiv.org/abs/2602.01007)
- [Dynamic Tanh (CVPR 2025)](https://arxiv.org/abs/2503.10622)
- [Automated LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419)
- [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Super Tiny Language Models](https://arxiv.org/abs/2405.14159)

### Resources
- [OpenAI Parameter Golf GitHub](https://github.com/openai/parameter-golf)
- [modded-nanogpt: NanoGPT in 2 minutes](https://github.com/KellerJordan/modded-nanogpt)
- [Muon Optimizer Blog Post](https://kellerjordan.github.io/posts/muon/)
- [Moonlight (MoonshotAI, Muon at scale)](https://github.com/MoonshotAI/Moonlight)
- [NanoGPT Speedrun History](https://www.emergentmind.com/topics/nanogpt-speedrun)
- [NanoGPT Speedrun on Prime Intellect](https://app.primeintellect.ai/speedrun/nanogpt)
