# Optuna HP Sweep — Experiment Log

## Task: Square (NutAssemblySquare), bf16 tokens, RAE-DP

---

## Round 1: `v3_square_bf16_hp_sweep`

**Date**: 2026-04-10 to 2026-04-11

**Setup**:
- 15-17 lab PCs (dh2026pc04-pc20), RTX 4080 16GB each
- 300 epochs per trial, cosine LR schedule
- 50-episode eval every 10 epochs (30 eval points per trial)
- Objective: peak success rate (maximize)
- bf16 pre-computed tokens, cache_in_ram
- PostgreSQL Optuna DB on dh2026pc02

**Trials**: 62 completed, 6 pruned, 22 orphaned (from restart)

### Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| lr | [5e-5, 5e-4] | log-uniform |
| lr_adapter | [1e-6, 1e-4] | log-uniform |
| d_model | {256, 384, 512} | categorical |
| n_layers | {6, 8, 10, 12} | discrete |
| n_cond_layers | {2, 4, 6} | discrete |
| p_drop_attn | [0.0, 0.3] | continuous |

### Fixed Settings

| Parameter | Value |
|-----------|-------|
| spatial_pool_size | 7 |
| use_flow_matching | True |
| batch_size | 64 |
| warmup_steps | 1000 |
| T_obs | 2 |
| T_pred | 10 |
| norm_mode | chi |
| no_amp | True (fp32 compute) |

### Top 5 Results (by weighted = 0.5*peak + 0.5*last10_avg)

| Trial | Peak | Last10 | Top5 | Weighted | lr | lr_adapter | d | nl | nc | pdrop |
|-------|------|--------|------|----------|-----|------------|---|----|----|-------|
| #61 | 100% | 95.8% | 98.4% | **0.979** | 1.7e-4 | 3.0e-5 | 384 | 8 | 4 | 0.130 |
| #57 | 100% | 95.4% | 98.8% | 0.977 | 2.8e-4 | 2.8e-5 | 384 | 6 | 4 | 0.132 |
| #52 | 100% | 94.6% | 98.4% | 0.973 | 3.2e-4 | 7.9e-6 | 384 | 6 | 4 | 0.128 |
| #30 | 100% | 94.2% | 98.4% | 0.971 | 2.4e-4 | 1.6e-5 | 384 | 6 | 6 | 0.213 |
| #58 | 100% | 94.0% | 98.4% | 0.970 | 2.9e-4 | 2.7e-5 | 384 | 6 | 4 | 0.048 |

### Key Findings

**Hyperparameter importance** (fANOVA, dashboard):
- lr: 0.36
- lr_adapter: 0.29
- n_layers: 0.17
- p_drop_attn: 0.07
- d_model: 0.06
- n_cond_layers: 0.06

**Architecture**:
- **d_model=384 dominates**: all top 19 trials (100% peak) use d=384. TPE sampled 46/62 trials as d=384.
- **n_layers=6 is optimal**: mean weighted 0.912 (n=34) vs nl=8 at 0.899 (n=14). Wider + shallower beats narrower + deeper.
- **n_cond_layers=4 dominates top-10**: 8/10 use nc=4. But nc barely matters (importance=0.06).

**Training recipe**:
- **lr sweet spot: 1.7e-4 to 3.2e-4**. Higher lr converges faster (r=+0.24).
- **lr_adapter very low (8e-6 to 3e-5)**: the RAE pre-trained adapter should barely be fine-tuned. Supports the hypothesis that RAE pre-training finds the right representation.
- **p_drop_attn ~0.13**: moderate dropout. Very low (<0.03) clearly hurts. Higher (~0.21) also works.

**Diminishing returns**:
- 73% of trials peak by epoch 149, 87% by epoch 199
- Mean SR declines by 0.01 from epoch 149 to 299
- Rankings at epoch 100 have r=0.67 with final rankings; epoch 150 has r=0.80

**Eval noise**: 50-episode evals have ~4% std. Peak SR is a biased estimator (selects for noise). Weighted metric (0.5*peak + 0.5*last10_avg) is the most robust metric (avg Spearman rho=0.834 with all other metrics).

**Toxic combinations**:
- d=256 + low lr (<1.3e-4) → consistently bad
- d=512 + high lr (>3e-4) → training collapse
- d=384 + very low dropout (<0.03) → underperforms

---

## Round 2: `v3_square_bf16_hp_sweep_r2`

**Date**: 2026-04-11 onwards

**Changes from Round 1**:
- Objective: **weighted metric** (0.5*peak + 0.5*last10_avg) instead of peak SR
- 200 epochs (was 300): captures 87% of peaks, saves 33% compute
- 100-episode evals (was 50): less noisy evaluations
- 15 eval envs (was 10)
- Narrowed search space based on R1 findings
- Added T_obs as new search parameter
- Added n_layers=4 (unexplored in R1)
- 3 trials per worker (was 5): 17 PCs × 3 = 51 trials
- New study name to keep R1 data separate

### Search Space

| Parameter | Range | Type | Rationale |
|-----------|-------|------|-----------|
| lr | [1.8e-4, 3.6e-4] | log-uniform | Narrowed to R1 sweet spot |
| lr_adapter | [8e-6, 2.8e-5] | log-uniform | Narrowed to R1 sweet spot |
| d_model | {384} | fixed | All R1 top trials use 384 |
| n_layers | {4, 6, 8} | categorical | Added 4 (unexplored), dropped 10/12 |
| n_cond_layers | 4 | fixed | Dominated R1 top-10 |
| p_drop_attn | [0.05, 0.22] | continuous | Very low (<0.03) hurts, narrowed |
| T_obs | {1, 2, 3} | categorical | **New**: does temporal context matter? |

### Fixed Settings (same as R1 except noted)

| Parameter | Value | Change from R1 |
|-----------|-------|----------------|
| d_model | 384 | was searched |
| n_cond_layers | 4 | was searched |
| num_epochs | 200 | was 300 |
| eval_full_episodes | 100 | was 50 |
| eval_n_envs | 15 | was 10 |

### Questions this round aims to answer
1. Does T_obs=1 (no temporal context) work as well as T_obs=2?
2. Does T_obs=3 (acceleration info) help for the harder Square task?
3. Is n_layers=4 (even shallower) better than 6?
4. With less noisy evals (100 episodes), do the same configs still win?
5. Does the weighted metric change which configs TPE prefers?

### Results

**Final**: 87 completed, 30 pruned (R2 stopped after 135 total trials when TPE plateaued)

#### Top 5 (old metric: 0.5*peak + 0.5*last10_avg)

| Rank | Trial | weighted | peak | last10 | T_obs | nl | lr | lr_adapter | pdrop |
|------|-------|----------|------|--------|-------|----|----|------------|-------|
| 1 | **#100** | **0.9725** | 0.99 | 0.955 | 2 | 6 | 3.25e-4 | 2.27e-5 | 0.188 |
| 2 | #61 | 0.9700 | 0.98 | 0.960 | 2 | 6 | 3.50e-4 | 2.34e-5 | 0.159 |
| 3 | #104 | 0.9700 | 0.98 | 0.960 | 2 | 6 | 2.94e-4 | 2.30e-5 | 0.211 |
| 4 | #26 | 0.9675 | 0.99 | 0.945 | 2 | **4** | 3.37e-4 | 8.63e-6 | 0.181 |
| 5 | #105 | 0.9670 | 0.99 | 0.944 | 2 | 6 | 2.72e-4 | 8.75e-6 | 0.175 |

**R2 champion**: trial #100, beat the R1 champion #61 (0.970 → 0.972). Very marginal improvement with more noisy-eval episodes (100 vs 50 in R1).

#### Key findings

**T_obs breakdown** (completed trials):
| T_obs | n | mean weighted | best | pruning rate |
|-------|---|--------------|------|--------------|
| 1 | 22 | 0.913 | 0.954 | 0% |
| **2** | **58** | **0.942** | **0.972** | 5% |
| 3 | 7 | 0.869 | 0.935 | **79%** (27/34) |

**T_obs=3 is catastrophic**: 79% pruning rate. The extra conditioning tokens (294 vs 196) cause training collapse with nl=6 specifically. Only nl=4 tolerates T_obs=3 (and even then, never wins).

**T_obs=2 wins decisively**: both highest mean (0.942) and best single trial (0.972).

**T_obs=1 is safe but suboptimal**: zero pruning but ceilings lower — the policy needs at least one past frame for velocity information.

**n_layers breakdown (T_obs=2 only)**:
| nl | n | mean | best |
|----|---|------|------|
| 4 | 9 | 0.934 | 0.968 |
| **6** | **35** | **0.944** | **0.972** |
| 8 | 14 | 0.940 | 0.953 |

**nl=6 is the clear winner**. nl=4 is close but never produces a champion. nl=8 is surprisingly worse than nl=6 (possibly overfitting the narrow task).

#### TPE convergence

- Trial 0-5: rapid improvement as TPE builds its model
- Trial 5-30: refinement (best: 0.953 → 0.958)
- Trial 30-100: flat plateau (0.958 → 0.972)
- Trial 100+: **17 trials with zero improvement**, TPE exploiting same local optimum

TPE effectively converged by trial ~100. Additional trials were just micro-perturbations in a narrow region (lr ≈ 2.7-3.5e-4, lr_adapter bimodal at ~8e-6 or ~2.3e-5).

#### Edge exploitation signals (critical for R3 design)

Top 10 R2 trials — parameter positions within the narrow R2 range:
- **lr**: 6/10 trials in upper half; best clustered at 2.8-3.5e-4 (near upper edge 3.6e-4)
- **lr_adapter**: **bimodal** — 6/10 near upper edge (2.0-2.7e-5), 4/10 near lower edge (7-10e-6). No middle ground.
- **p_drop_attn**: mostly in upper half (0.17-0.22, near the 0.22 cap)

**Interpretation**: The narrowed R2 ranges were too tight in lr_adapter particularly — TPE hit both edges, indicating the true optima extend beyond both bounds. The bimodal pattern also confirms there are **two distinct winning regimes**:
- **Low lr_adapter (~1e-5)**: adapter effectively frozen, denoiser does all the work
- **High lr_adapter (~2.5e-5)**: adapter co-adapts with the denoiser

Both achieve equivalent quality (~0.965). Within top-20 R1 trials, Spearman(lr, lr_adapter) ≈ -0.37, suggesting the two regimes may favor slightly different main lr values.

---

## Round 3: `v3_square_bf16_hp_sweep_r3` — Rapid Wide Search

**Date**: 2026-04-12 onwards

**Motivation**: R2 showed TPE is exploiting the narrowed ranges too tightly, hitting edges. R1/R2 ranges were too constrained to find the true optimum. Combined with the finding that 50 epochs is sufficient for rapid convergence (peak typically by epoch 25-50 from multi-seed runs), we can afford to run **many more trials with a much wider search space** in roughly the same wall-clock time.

### Changes from R2

| Aspect | R2 | R3 | Why |
|--------|-----|-----|-----|
| Study name | `..._r2` | `..._r3` | Separate data |
| num_epochs | 200 | **50** | Rapid iteration, TPE ranking converges by epoch ~100 anyway |
| eval_full_every_epoch | 10 | **5** | 10 evals per trial (from 20) |
| n_trials_per_worker | 3 | **10** | More trials per worker for exploration |
| Pruner | MedianPruner (50%) | **PercentilePruner (25%)** | More aggressive pruning for wide search |
| n_warmup_steps | 50 | **19** | Allow pruning after 4 evals in a 50-epoch run |
| Objective | 0.5*peak + 0.5*last10_avg | **0.3*peak + 0.7*overall_avg** | Less peak-sensitive; with 10 evals, peak is noisy |
| T_obs | searched {1, 2, 3} | **fixed at 2** | T_obs=3 is garbage, T_obs=1 is worse on average |
| n_cond_layers | fixed at 4 | fixed at 4 (same) | Already established |
| d_model | fixed at 384 | fixed at 384 (same) | Already established |

### Search Space (wide)

| Parameter | Range | Type | vs R2 |
|-----------|-------|------|-------|
| `lr` | **[1.0e-5, 1.0e-3]** | log-uniform | 2 orders wider (R2: [1.8e-4, 3.6e-4]) |
| `lr_adapter` | **[1.0e-7, 1.0e-4]** | log-uniform | 3 orders wider (R2: [8e-6, 2.8e-5]) |
| `n_layers` | {4, 6, 8} | categorical | same |
| `p_drop_attn` | [0.03, 0.25] | continuous | slightly wider |

### Trial Budget

- **~50 min per trial** (d=384, T_obs=2, bf16 cached tokens, 100-episode evals every 5 epochs)
- 17 workers × 10 trials each = **170 target trials**
- Expected wall-clock: ~9 hours per worker, with strong overlap → **~10-15 hours total**

### Questions R3 aims to answer

1. Where are the **true optima** for lr and lr_adapter when TPE has full freedom?
2. Is the **lr/lr_adapter bimodal pattern** real (two distinct regimes) or an artifact of R2's narrow range?
3. Can **50-epoch rapid training** reliably identify the best HP region? (Sweep efficiency validation.)
4. Does the new metric **0.3*peak + 0.7*avg** produce a stabler ranking than the old weighted?
5. With **PercentilePruner(25%)** instead of MedianPruner(50%), does pruning eliminate more bad trials earlier?

### Expected behavior

Given the evidence that the top R2 configs cluster tightly around (lr ≈ 3e-4, lr_adapter ≈ 1e-5 or 2.5e-5, pdrop ≈ 0.18), R3 TPE should:
- Spend the first ~30 random trials covering the wide space
- Converge toward the same region within 50-60 trials
- Find a slightly better optimum (maybe 0.975 → 0.98) by exploring the true edges of the optimum basin
- Possibly discover that lr just outside R2's range (e.g. 4-5e-4) works even better, or confirm 3e-4 is optimal

### Metric bug found after 24 trials — switching to R3b

After the first 24 R3 trials launched, inspection of their trajectories revealed a structural problem with the `0.3*peak + 0.7*overall_avg` objective. With `warmup_steps=1000` (~2.2 epochs) and `eval_full_every_epoch=5`, the first 2–3 eval points (epochs 4, 9, 14) are dominated by warmup noise, not by HP quality. Averaging them in rewards *fast-converging* HPs rather than *best-converging* HPs.

Concrete example from the running R3 trials:

| Trial | Trajectory | peak | overall_avg | 0.3·peak + 0.7·overall_avg |
|-------|-----------|------|-------------|------------------------------|
| #0 | 0.03, 0.24, 0.36, 0.55, 0.69 (still climbing) | 0.69 | 0.374 | **0.469** |
| #16 | 0.83, 0.95, 0.95, 0.97, 0.95, 0.91 | 0.97 | 0.927 | **0.940** |

Trial #0 may well plateau above 0.85 if it ran to completion, but its epoch-4 value of 0.03 is permanently baked into the overall average — no amount of final-plateau excellence can outweigh the warmup noise. TPE's posterior will systematically prefer HPs that warm up fast, which is orthogonal to what we actually want (final-plateau quality).

**Methodological finding**: for rapid 50-epoch sweeps with dense early evals and nontrivial warmup, use **tail-window averaging**, not overall averaging. The pruner can still report raw step-wise SR (that's a valid step-wise signal — at epoch 14, low SR legitimately means worse-so-far), but the final objective must use the post-convergence window.

R3's 24 in-flight trials were allowed to finish under the old metric and remain in the database as a separate record (`v3_square_bf16_hp_sweep_r3`). TPE for the continued sweep resumes under a new study name (`..._r3b`) with the fixed metric.

---

## Round 3b: `v3_square_bf16_hp_sweep_r3b` — fixed metric

**Date**: 2026-04-12 onwards

**Motivation**: R3's `overall_avg` biased TPE toward fast-converging HPs (see R3 "Metric bug found" note above). R3b restarts TPE cleanly with the corrected metric.

### Only change from R3

| Aspect | R3 | R3b |
|--------|-----|-----|
| Objective | 0.3·peak + 0.7·overall_avg | **0.3·peak + 0.7·last_6_avg** |
| Study name | `..._r3` | `..._r3b` |

Everything else identical: 50 epochs, eval every 5, 100-ep × 15-env evals, PercentilePruner(25%, n_warmup_steps=19), wide search space (`lr` [1e-5, 1e-3] log, `lr_adapter` [1e-7, 1e-4] log, `n_layers` {4,6,8}, `p_drop_attn` [0.03, 0.25]), locked architecture (d=384, T_obs=2, n_cond=4, n_head=6, spatial_pool=7, flow matching, chi norm, rot6d), 17 workers × 10 trials = 170 target.

### Why `last_6_avg` specifically

A 50-epoch run with `eval_full_every_epoch=5` gives 10 eval points (epochs 4, 9, 14, …, 49). The last 6 correspond to epochs 24, 29, 34, 39, 44, 49 — comfortably past warmup (which ends by epoch ~2.2) and past the rapid ramp-up phase seen in R3 trajectories. This leaves the tail window representing the converged plateau, matching the R1/R2 `last_10_avg` philosophy (last-half of training) while still respecting R3's rapid-iteration compute budget.

For pruned trials with fewer than 6 evals, the slice `rates[-6:]` gracefully falls back to the whole trajectory — same behavior as the old metric for pruned trials (which were never the main ranking signal anyway since they're pruned).

### Pruner is unchanged — and should be

Pruning uses raw step-wise SR via `trial.report(sr, epoch)` in `training/train_v3.py`. That's correct for pruning: at a fixed epoch, all trials have had equal training budget, so raw SR is the right comparison. The metric bias only matters for the final objective (what TPE's posterior learns from), which uses the returned `objective_value` from `objective()`.

### Questions R3b aims to answer (unchanged from R3 but now under a valid metric)

1. Where are the **true optima** for lr and lr_adapter with wide search + unbiased ranking?
2. Does the **bimodal lr_adapter pattern** from R2 persist when the range opens to 3 orders of magnitude?
3. Is the 50-epoch + last_6 averaging combination enough to reliably identify top-3 configs?
4. Does `last_6_avg` produce a materially different winning HP region than R1/R2's `last_10_avg` over 200-300 epochs?

### R3b final results (240 trials, 2026-04-13)

**Final state**: 240 trials total — ~95 COMPLETE, ~121 PRUNED, rest marked FAIL from orphaned restarts. TPE converged cleanly: late-sweep trials (#235, #237, #239) land in the same champion basin as early trials (#49, #56, #61), confirming the optimum is real and not a TPE artifact.

#### Final top-10

| Rank | Trial | weighted | peak | last_6 | lr | lr_adapter | nl | pdrop |
|------|-------|----------|------|--------|------|-------------|-----|-------|
| 1 | **#160** | **0.9662** | **1.00** | 0.952 | 4.23e-4 | 2.26e-6 | 8 | 0.240 |
| 2 | #239 | 0.9637 | 0.98 | 0.957 | 4.21e-4 | 2.40e-6 | 8 | 0.233 |
| 3 | #155 | 0.9597 | 0.99 | 0.947 | 4.59e-4 | 3.72e-6 | 8 | 0.243 |
| 4 | #61  | 0.9595 | 0.97 | 0.955 | 2.81e-4 | 1.77e-6 | 8 | 0.216 |
| 5 | #187 | 0.9573 | 0.99 | 0.943 | 2.96e-4 | 1.21e-6 | 8 | 0.194 |
| 6 | #165 | 0.9572 | 0.97 | 0.952 | 2.83e-4 | 1.52e-6 | 8 | 0.175 |
| 7 | #235 | 0.9568 | 1.00 | 0.938 | 3.91e-4 | 2.38e-6 | 8 | 0.239 |
| 8 | #88  | 0.9567 | 0.98 | 0.947 | 3.66e-4 | 2.65e-6 | 8 | 0.187 |
| 9 | #56  | 0.9560 | 0.97 | 0.950 | 2.83e-4 | 1.90e-6 | 8 | 0.218 |
| 10 | #154 | 0.9543 | 0.98 | 0.943 | 4.32e-4 | 3.52e-6 | 8 | 0.242 |

**Trial #160 hit 100% peak SR** (first in any Square sweep), plateau 0.95–0.97 on last_6. Trial #235 also hit 100% peak. Both are in the same basin as other top-10 trials.

#### Aggregated "best HPs" (top-10, log-space medians for lr/lr_adapter)

These are the numbers to cite as the tuned Stage-3 HPs for the paper:

```
lr                = 3.5e-4     (gmedian=3.78e-4, top-10 interquartile [2.83e-4, 4.23e-4])
lr_adapter        = 2.2e-6     (gmedian=2.32e-6, top-10 interquartile [1.77e-6, 2.65e-6])
n_layers          = 8          (100% of top-20)
p_drop_attn       = 0.22       (median=0.226, top-10 interquartile [0.194, 0.240])

# locked from earlier rounds
d_model           = 384
n_head            = 6
n_cond_layers     = 4
T_obs             = 2
spatial_pool_size = 7
use_flow_matching = true
norm_mode         = chi
use_rot6d         = true
```

**Tightness of optimum**: top-10 `lr` interquartile spans only 1.5× (2.83e-4 → 4.23e-4), `lr_adapter` interquartile spans only 1.5× (1.77e-6 → 2.65e-6), `pdrop` interquartile spans [0.19, 0.24]. This is a narrow, well-defined optimum — not a broad flat region.

#### Answers to R3b's questions

1. **True optima**: `lr ≈ 3.5e-4`, `lr_adapter ≈ 2.2e-6` (≈ 10× lower than R2's `2.27e-5`!), `pdrop ≈ 0.22`. See top-10 interquartile above for tight confidence.

2. **Bimodal lr_adapter does NOT replicate**. R2 showed modes at ~8e-6 and ~2.5e-5; R3b has a single mode at ~2e-6. The bimodality was an artifact of R2's narrow range + old metric. The "real" answer is that very-low `lr_adapter` (~2e-6) is best — the RAE-pretrained adapter should barely be fine-tuned, confirming and sharpening the R1/R2 thesis.

3. **50 epochs + last_6_avg reliably identifies champions**. Late-sweep trials (#239, #235) landed in the same basin as early winners (#61, #49). TPE convergence was clean by trial ~100.

4. **New metric finds meaningfully different HPs than R2's `last_10_avg`**:
   - R2 champion: `lr=3.25e-4, lr_a=2.27e-5, nl=6, pdrop=0.188`
   - R3b champion: `lr=4.23e-4, lr_a=2.26e-6, nl=8, pdrop=0.240`
   - Under the new metric, `nl=8` dominates (100% of top-20); R2's `nl=6` winner is gone.
   - Most striking: `lr_adapter` is **10× lower** under the new metric.
   - Plateau quality is similar (R3b ~0.955, R2 ~0.955) — both metrics find similar *quality* optima, but different *parameters*.

#### Surprise finding: `n_layers=8` reversal

R2 found `nl=6` was best; R3b found `nl=8` dominates all top-20. Why?

| nl | n_complete | avg_w | best_w |
|----|-----------|-------|--------|
| 4 | 12 | 0.831 | 0.928 |
| 6 | 10 | 0.884 | 0.934 |
| **8** | **67** | **0.921** | **0.966** |

Three plausible explanations:
1. **New metric favors plateau over peak**: `nl=8`'s higher capacity produces smoother plateaus when regularized (pdrop ~0.22); `nl=6`'s old-metric advantage was driven by peak-detection noise.
2. **Wider lr range unlocks `nl=8`'s advantage**: R2's `lr ∈ [1.8e-4, 3.6e-4]` was tuned for `nl=6`; R3b's `lr ≈ 4.2e-4` needs `nl=8`'s depth to absorb.
3. **50 epochs + aggressive cosine decay**: short training doesn't give `nl=8` enough time to overfit the way it would in 300-epoch R1-style runs. R1 also had `nl=8` competitive (champion #61 used `nl=8`); only R2's longer training + narrower lr favored `nl=6`.

**Caveat — TPE commitment bias**: `nl=8` got 138/186 samples (74%); `nl=6` got only 25. TPE committed to `nl=8` around trial ~60 and never properly explored `nl=6` in the champion `(lr, lr_a, pdrop)` corner. We can't rule out that `nl=6` at `(lr≈4e-4, lr_a≈2e-6, pdrop≈0.22)` would also hit ~0.96 — it just wasn't sampled. This is R3c's job.

#### Lessons recorded

See the **Lessons learned** section below — condensed versions of the findings are in `.claude/projects/-virtual-csc415user/memory/` (`project_hp_sweep.md`, `reference_optuna_gotchas.md`).

### R3 post-mortem analysis (24 completed trials, old metric)

After the 24 R3 trials finished under the old metric, we re-ranked them under the new metric to check how much TPE's posterior would have been corrupted. Spearman ρ(old ranks, new ranks) = **0.97** — highly correlated overall — but the mid-band had specific, interpretable mis-rankings:

- **Top-3 champions largely robust**: under old metric #16, #18, #1; under new metric #18, #16, #6. #1 drops from rank 3 → 6 because its trajectory `[0.78, 0.89, 0.96, 0.9, 0.9, 0.92, 0.92, 0.92, 0.87, 0.88]` has a noisy mid-run peak at 0.96 that doesn't reflect its true plateau (~0.90). Old metric rewards the peak; new metric correctly scores it as a solid top-6 not a champion.
- **Biggest injustice**: trial #2 `[0.01, 0.47, 0.66, 0.8, 0.88, 0.86, 0.88, 0.82, 0.85, 0.8]` was **pruned** at old rank 10 (excluded from TPE's posterior) but is actually new rank 8 (top-10). Its HPs were `lr=3.0e-5, lr_a=1.1e-7, nl=8, pdrop=0.16` — the "very-low-lr + frozen-adapter" corner, consistent with the R1/R2 finding that `lr_adapter` should be essentially frozen.
- **Slow-climbers systematically punished**: trials #0, #7, #17 (all with epoch-4 SR ≤ 0.22) moved up 2-3 positions under the new metric. These represent slow-warmup HPs that end at solid plateaus (0.78-0.91) but are scored down by their warmup noise.

**Takeaway**: the bug is real but bounded. Champions are stable under both metrics; mid-band rankings shift by 1-3 positions, with slow-climbers being the main beneficiaries of the fix. TPE's posterior from R3 would have been slightly mis-calibrated toward fast-convergence but not catastrophically so.

---

## ⚠️ Pruner semantics: Optuna's `PercentilePruner` flips for maximization

After R3b started running, analysis of trial #45 (pruned at step 19 despite having best-so-far = 0.90) revealed a fundamental misunderstanding of `PercentilePruner` semantics that had been baked into all our configs since R3.

### What we thought `PercentilePruner(percentile=25.0)` meant

"Kill bottom 25% at each step" — i.e. prune the 25% worst trials, keep 75%.

### What it actually does for a maximization study

From Optuna's source (`optuna/pruners/_percentile.py`):

```python
def _get_percentile_intermediate_result_over_trials(...):
    ...
    direction = storage.get_study_direction(study_id)
    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile    # <-- flips for max
    return float(numpy.percentile(intermediate_values, percentile))

def prune(self, study, trial):
    ...
    best_intermediate_result = trial.intermediate_values[step]  # best-so-far at this step
    p = self._get_percentile_intermediate_result_over_trials(...)
    if direction == StudyDirection.MAXIMIZE:
        return best_intermediate_result < p   # prune if below cutoff
```

For a **maximization** study, `PercentilePruner(percentile=25.0)` does:
1. Compute the cutoff as the **75th percentile** of other trials' best-so-far at the same step (because 100 - 25 = 75).
2. Prune any trial whose best-so-far < 75th percentile cutoff.

**Translation**: `PercentilePruner(25%)` **keeps only the top 25%**. It is the most aggressive plausible setting, not the least. `PercentilePruner(50%)` = median pruner = keep top 50%. `PercentilePruner(75%)` = keep top 75% (gentle).

### How this was verified on trial #45

Trial #45's trajectory through step 19: `[0.69, 0.90, 0.86, 0.89]`, best-so-far = 0.90.

At the moment #45 reported step 19, the pool of other trials' best-so-far values at step 19 had a 75th percentile of **~0.9050**. Since 0.90 < 0.9050, #45 was pruned — despite being a solid top-15 trial by any reasonable standard. HPs: `lr=2.04e-4, lr_adapter=4.09e-7, n_layers=8, p_drop_attn=0.037` — a legitimate under-explored corner of the search space that got silently excluded from TPE's posterior.

### Consequences for R3 and R3b results

- **R3 pruned 11/24 (46%)** — consistent with "keep top 25%" once 10 startup trials were past.
- **R3b pruned ~24/48 (50%)** by trial #47 — same aggressive pattern.
- Every slow-climber or mid-plateau HP regime is likely under-represented in both posteriors. The bimodal lr_adapter pattern from R2 may be harder to detect because the lower mode (frozen adapter, slow training) warms up slowly and gets killed before it can show its plateau.

### Fix for R3c

Bump `pruning_percentile: 25.0` → `50.0` (= MedianPruner = keep top 50%). This:
- Still kills the obvious losers (step-19 values < median ~0.5)
- Preserves legitimate mid-tier trials like #45 in TPE's posterior
- Doubles the compute cost per sweep round vs R3b's aggressive kill rate — but the sweep is time-bound (9 hours per worker), not compute-bound, so this is essentially free.

---

## Round 3c: `v3_square_bf16_hp_sweep_r3c` — narrow confirmation + `nl=6` validation

**Date**: 2026-04-13 onwards

**Motivation**: R3b found a tight optimum at `(lr≈3.5e-4, lr_a≈2.2e-6, nl=8, pdrop≈0.22)` but left two things open:
1. **`nl=6` under-exploration**: TPE sampled `nl=8` 138 times vs `nl=6` 25 times. We can't actually rule out that `nl=6` at the champion `(lr, lr_a, pdrop)` would match `nl=8`.
2. **`PercentilePruner(25%)` mid-trial bias**: R3b used the "keep top 25%" setting (we only discovered this was backwards mid-sweep), which compounded TPE's commitment to `nl=8` by killing slow-warmup nl=6 trials at step 19.

R3c addresses both via (a) a narrow search space around R3b's champion basin, (b) `n_layers ∈ {6, 8}` only (dropping the clearly-worse `nl=4`), (c) `PercentilePruner(50%)` = MedianPruner for less aggressive pruning, (d) fewer trials (narrow ranges converge faster).

### Changes from R3b

| Aspect | R3b | R3c |
|--------|-----|-----|
| Study name | `..._r3b` | `..._r3c` (fresh TPE posterior) |
| `lr` range | [1e-5, 1e-3] log | **[2e-4, 6e-4]** log (centered on top-10 interquartile, ×2 margin) |
| `lr_adapter` range | [1e-7, 1e-4] log | **[5e-7, 1e-5]** log (centered on 2e-6, ~2 decades wide) |
| `n_layers` | {4, 6, 8} | **{6, 8}** (drop nl=4) |
| `p_drop_attn` range | [0.03, 0.25] | **[0.15, 0.26]** (tighter, centered on top-10) |
| `pruning_percentile` | 25.0 (keep top 25%, aggressive) | **50.0** (MedianPruner = keep top 50%) |
| `n_trials_per_worker` | 10 | **5** (narrow search converges faster) |
| Trial budget | 24 × 10 = 240 | 24 × 5 = **120** |

Everything else identical: same objective (`0.3*peak + 0.7*last_6_avg`), same architecture (d=384, n_head=6, n_cond=4, T_obs=2, spatial_pool=7, flow matching, chi norm, rot6d, bf16 cached), same `n_warmup_steps=19`, same `n_startup_trials=10`.

### Why narrow rather than wide again

R3b's wide search already mapped the landscape. Running another 240 wide-search trials would just re-discover what we already know. The narrow search is about **confirmation + resolution**:

- **Confirmation**: does the `(lr≈3.5e-4, lr_a≈2.2e-6, pdrop≈0.22)` basin hold with MedianPruner semantics? If yes, the R3b champion region is validated and can be quoted in the paper with confidence.
- **`nl=6` resolution**: with 120 trials constrained to `{6, 8}`, TPE will put at least 40–50 trials into `nl=6` (even with its strong `nl=8` prior), directly in the champion `(lr, lr_a, pdrop)` box. If `nl=6` also reaches ~0.96 in that box, we have evidence the `nl` reversal was a spurious TPE commitment. If it clearly underperforms, the `nl=8 > nl=6` finding is solidified.
- **Pruner sanity check**: with the correct MedianPruner, do we find the same optimum? This is a sanity check on R3b's pruner-biased results.

### Why fresh study (not continuing R3b)

Mixing R3c's MedianPruner samples into R3b's aggressive-pruner posterior would confuse TPE's model of "what gets pruned early". Fresh study, fresh TPE, clean comparison.

### Expected behavior

- **~85/120 trials survive** MedianPruner (vs ~50/240 under R3b's 25%)
- TPE posterior converges to the same `(lr, lr_a, pdrop)` region as R3b within ~30 trials (narrow search)
- **`nl=6` gets fair allocation**: TPE will try it with the champion `(lr, lr_a, pdrop)` combo at least 20-30 times, enough to statistically compare vs `nl=8`
- Total wall-clock: ~4–5 hours (24 workers × 5 trials × ~50 min each) + pruning savings

### Questions R3c aims to answer

1. Does `nl=6` at `(lr≈3.5e-4, lr_a≈2.2e-6, pdrop≈0.22)` match `nl=8`, or is `nl=8` genuinely better?
2. Does the R3b champion basin reproduce under correct pruner semantics?
3. Do we find a new, even better optimum inside the narrow region that R3b missed?
4. Does `pdrop` have a cleaner optimum in the narrow range — around 0.22 as R3b suggests, or elsewhere in [0.15, 0.26]?

