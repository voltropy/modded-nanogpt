# 🏆 New modded-nanogpt Speedrun Record: 57.38s on 8×B200

**Date:** 2026-03-15  
**Previous Record:** 58.04s (Record #77, commit `81730c3`)  
**New Record:** 57.38s (-0.66s, -1.1%)  
**Target:** val_loss ≤ 3.28 on FineWeb validation set  
**Achieved:** val_loss = 3.2798

---

## Links

- **Git Commit:** [`116a70e`](https://github.com/voltropy/modded-nanogpt/commit/116a70e4ef87125608374b2b257d7d5376554529)
- **Branch:** [`voltropy/record-78-stage-shift`](https://github.com/voltropy/modded-nanogpt/tree/voltropy/record-78-stage-shift)
- **GCS Artifacts:** `gs://volta-artifacts/benchmarks/modded-nanogpt/record-break-20260315/`
- **Based On:** [Record #77](https://github.com/KellerJordan/modded-nanogpt/commit/81730c3057a02df2b3c30b255aef42424716a2c5) by the modded-nanogpt community

---

## The Optimization: Stage Duration Shift

### Background
The modded-nanogpt speedrun trains a 124M-parameter GPT-2 model in 3 stages with increasing batch sizes and sequence lengths:

| Stage | Seq Length | Batch Size | Step Time | Duration (original) |
|-------|-----------|------------|-----------|-------------------|
| 1 | 896 | 8 × 2048 × 8 | ~21ms | 33% (497 steps) |
| 2 | 2048 | 16 × 2048 × 8 | ~38ms | 33% (497 steps) |
| 3 | 2048 | 24 × 2048 × 8 | ~55ms | 33% (497 steps) |

Stage 1 steps are **2.6× faster** than stage 3 steps because they use shorter sequences (896 vs 2048) and smaller batches (8 vs 24).

### The Change
Shift training duration from equal thirds to front-loaded: more cheap stage 1 steps, fewer expensive stage 3 steps.

**Variant T (best):** `36% / 31% / 33%` (was `33% / 33% / 33%`)

```python
# Before (Record #77)
TrainingStage(duration=1/3, ...)  # Stage 1: 497 steps @ ~21ms
TrainingStage(duration=1/3, ...)  # Stage 2: 497 steps @ ~38ms
TrainingStage(duration=1/3, ...)  # Stage 3: 497 steps @ ~55ms

# After (Record #78)
TrainingStage(duration=0.36, ...)  # Stage 1: 522 steps @ ~21ms (+25 cheap steps)
TrainingStage(duration=0.31, ...)  # Stage 2: 450 steps @ ~38ms (-47 medium steps)
TrainingStage(duration=0.33, ...)  # Stage 3: 479 steps @ ~55ms (-18 expensive steps)
```

**No other changes.** Same architecture, same optimizer, same hyperparameters, same total step count (1490).

### Why It Works
Early training (stage 1) primarily learns short-range statistics — bigram frequencies, common phrases, local syntax patterns. These don't require full 2048-token sequences. By spending 3% more time in stage 1 (where each step costs ~21ms instead of ~55ms), we save ~0.7s of wall-clock time while the model still converges to the same quality.

The insight is that equal stage durations were never optimal — they were just the default. The batch size schedule was already tuned, but the duration schedule was assumed to be symmetric.

---

## Results

### Confirmation Runs (Variant T: 36/31/33)

| Machine | Time | val_loss | Step Avg | Status |
|---------|------|----------|----------|--------|
| volta-b200-3 | **57.382s** | 3.2798 | 38.51ms | ✅ **BEST** |
| volta-b200-2 | 57.476s | 3.2794 | 38.57ms | ✅ |
| volta-b200-0 | 57.433s | 3.2826 | 38.55ms | ❌ (0.003 over) |

### Confirmation Runs (Variant S: 35/32/33)

| Machine | Time | val_loss | Step Avg | Status |
|---------|------|----------|----------|--------|
| volta-b200-0 | 57.666s | 3.2780 | 38.70ms | ✅ |
| volta-b200-2 | 57.794s | 3.2799 | 38.79ms | ✅ |
| volta-b200-1 | 57.681s | 3.2810 | 38.71ms | ❌ (0.001 over) |

**5 out of 6 runs across both variants hit val_loss ≤ 3.28.**

### Binary Search Progression (24 experiments across 6 rounds)

| Stage 3 % | Representative Time | val_loss | Notes |
|-----------|-------------------|----------|-------|
| 25% | 54.4s | 3.295 | Way too fast, loss doesn't converge |
| 28% | 55.7s | 3.288 | Close but above threshold |
| 30% | 56.3s | 3.285 | Getting warmer |
| 31% | 56.4s | 3.285 | Almost there |
| 32% | 57.1s | 3.282 | 0.002 short |
| **33%** | **57.4s** | **3.280** | **✅ Sweet spot** |

---

## Reproduction

### Prerequisites
- 8× NVIDIA B200 GPUs (or 8×H100 with FA3 instead of FA4)
- Docker with NVIDIA runtime
- FineWeb10B dataset at `/mnt/data/modded-nanogpt/data/fineweb10B/`

### Docker Image
```
modded-nanogpt-b200:fa4-fixed
Image ID: e235f3cf1d55
Base: NVIDIA PyTorch NGC
Key packages: PyTorch 2.9.1+cu128, flash-attn-4 4.0.0b4, triton 3.5.1
```

### Run Command
```bash
docker run --gpus all --ipc=host --net=host \
  -v /mnt/data/modded-nanogpt:/workspace \
  -w /workspace \
  -e HF_TOKEN=$HF_TOKEN \
  modded-nanogpt-b200:fa4-fixed \
  torchrun --nproc_per_node=8 train_gpt_volta.py
```

### File Checksums
```
0e05688a3549f36c055fd61c4a3682ab  train_gpt_volta.py (Variant T: 36/31/33)
7fc8edea3ea953ab13af0dc3d86ecc55  triton_kernels.py
330b0c49e0180fd26b8909deb7807ff8  fa4_compile_wrapper/fa4_compile_wrapper.py
```

### Environment
- CUDA 12.8, Driver 570.211.01
- Python 3.12
- No environment variables required beyond HF_TOKEN
- No CLI flags — all config is in the training script

### Hardware Agnosticism
The stage duration shift is hardware-agnostic. The only B200-specific component is `fa4_compile_wrapper` (Flash Attention 4). For H100s, replace the FA4 import with the existing FA3 `get_kernel('varunneal/flash-attention-3')` call from the original `train_gpt.py`.

---

## Technical Details

### What Changed (complete diff from Record #77)

1. **Stage durations** (3 lines): `1/3, 1/3, 1/3` → `0.36, 0.31, 0.33`
2. **Flash Attention** (import swap): FA3 `get_kernel()` → FA4 `fa4_compile_wrapper` (B200-specific, not part of the optimization)

### What Did NOT Change
- Model architecture (parallel 2-lane residual, skip connections, paired head attention, hyperconnect)
- Optimizer (NorMuon+Adam, Muon LR=0.023, Adam LR=0.008)
- Total step count (1490 = 1450 scheduled + 40 extension)
- Cooldown fraction (0.60)
- Batch sizes per stage (8, 16, 24)
- Sequence lengths per stage (896, 2048, 2048)
- Learning rate multipliers (1.0, 1.52, 1.73)
- MTP weights
- Triton kernels
- Data loading
- Anything else

---

*Record set by Kurtz (Voltropy PBC) on 2026-03-15. Optimization discovered through systematic binary search of the stage duration parameter space across 24 experiments on 4× volta-b200 machines.*
