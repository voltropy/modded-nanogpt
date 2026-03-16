# Batch Schedule Optimization: Faster Training via Stage 0 Batch Reduction

**Date:** 2026-03-16
**Authors:** Voltropy PBC (Ted Blackman, Clint)
**Hardware:** 4× NVIDIA B200 (8-GPU nodes), 1× NVIDIA H100 (8-GPU node)

## Summary

We discovered that reducing the batch size in Stage 0 of the multi-stage training schedule from `8×2048×8` to `6×2048×8` saves **~2.6 seconds (~4.5%)** per run on B200 hardware, with only a +0.009 increase in validation loss. By adding extra extension iterations to compensate for the loss degradation, we can achieve both faster training times AND equivalent or better validation loss — a net win.

This finding is being ported to H100 for a potential new world record submission.

## Background

The current modded-nanogpt speedrun uses a 3-stage training schedule:

| Stage | Duration | Seq Len | Batch Size | Window Sizes |
|-------|----------|---------|------------|--------------|
| 0 | 36.0% | 896 | 8×2048×8 | (1, 3) |
| 1 | 31.0% | 2048 | 16×2048×8 | (3, 7) |
| 2 | 33.0% | 2048 | 24×2048×8 | (5, 11) |
| ext | variable | 2048 | 24×2048×8 | (6, 13) |

**Baseline performance (B200, ext=46):** ~57.8s, val_loss ~3.278

## Key Insight

Stage 0 accounts for 36% of all training steps (~522 of 1450 scheduled) and uses the smallest batch size. Reducing this batch from 8× to 6× means 25% fewer tokens per step in Stage 0, which proportionally reduces the cost of each forward/backward pass. Since Stage 0 steps are already the cheapest, this has a large cumulative effect.

The resulting loss degradation (+0.009) is small enough to be recovered by adding extra extension iterations at the end of training. Each extension step costs ~38ms on B200 (~58ms on H100), so we can afford many additional steps and still come out ahead on time.

## Experiment Results

### Phase 1: Architecture/Schedule Sweep (B200-2, B200-3)

Seven experiments tested on B200 hardware:

| Exp | Description | Δ Time | Δ Loss | Verdict |
|-----|-------------|--------|--------|---------|
| A | Baseline (eval-free) | 0 | 0 | No effect on reported metric |
| B | Seq len 512 in stage 0 | +49ms | -0.001 | Neutral |
| **C** | **Batch 6×2048×8 in stage 0** | **-2,645ms** | **+0.009** | **🏆 Best tradeoff** |
| D | 4-stage (0.10/0.26/0.31/0.33) | -773ms | +0.002 | Modest win |
| E | Window sizes (1,1) in stage 0 | -374ms | +0.009 | Negligible speed gain |
| F | 4-stage + batch=6 in stage 1 | -2,276ms | +0.008 | Good but not better than C |
| G | Batch 4×2048×8 in stage 0 | -2,932ms | +0.020 | Too much loss degradation |

### Confirmation Runs (Exp C on both machines)

| Machine | train_time_ms | val_loss |
|---------|--------------|----------|
| B200-2 (original) | 55,646 | 3.2875 |
| B200-2 (confirm) | 55,622 | 3.2885 |
| B200-3 (confirm) | 55,607 | 3.2863 |

**Consistent ~2.6s savings, confirmed across machines.**

### Phase 2: Parameter Space Exploration (B200-0, B200-1)

Five parameter tweaks tested — **none improved on baseline**:

| Exp | Description | Δ Loss | Δ Time | Verdict |
|-----|-------------|--------|--------|---------|
| 1 | Cooldown 0.55 | +0.001 | -24ms | Neutral |
| 2 | Cooldown 0.65 | +0.002 | -27ms | Worse |
| 3 | LR boost (1.58/1.80) | +0.002 | -7ms | Worse |
| 4 | LR reduction (1.45/1.65) | +0.003 | -31ms | Worse |
| 5 | ext=50, sched=1400 | +0.008 | -1,725ms | Much worse loss |

**Conclusion:** The current hyperparameters (cooldown=0.60, lr_mul 1.52/1.73, split 0.36/0.31/0.33) are a well-tuned local optimum. Architectural changes (batch schedule) are more promising than parameter tweaks.

### Phase 3: Batch=6× + Extra Extension (In Progress)

The key hypothesis: combine the 2.6s time savings from batch=6× with additional extension iterations to recover the +0.009 loss. Each extension step costs ~38ms on B200, so even 50 extra extension steps (1.9s) still saves ~700ms net while potentially pushing loss well below 3.28.

Testing ext=56 (10 extra) through ext=70 on B200 to find the sweet spot, then porting the optimal config to H100.

## H100 Baseline Data

Reliability testing on H100 (8×H100, ext=47 and ext=48):

**ext=47 (5 runs):** Mean time 86,983ms, mean loss 3.2807 (above 3.28)
**ext=48 (6 runs):** Mean time 87,093ms, mean loss 3.2800 (barely under 3.28)

Current H100 record: 87,098ms, val_loss 3.2784 (Record #77).

Neither ext=47 nor ext=48 achieves p<0.01 for mean loss ≤ 3.28 — the margin is too thin. The batch=6× approach should create enough time headroom to increase extension and get loss comfortably below 3.28 with statistical significance.

## Next Steps

1. **H100:** Test batch=6× + ext sweep (ext=48→70) to find a record-breaking config
2. **B200:** Continue innovation — MTP weight schedules, gradient clipping, stage 2 batch reduction
3. **Statistical validation:** Once a promising H100 config is found, run 12+ times for one-sided t-test (H0: μ ≥ 3.28, p<0.01)

## Methodology Notes

- All experiments run on identical hardware configurations within each class (B200 or H100)
- Training time measured by the script's internal `train_time_ms` metric (excludes validation overhead)
- Validation loss computed on FineWeb validation set at end of training
- "Hit" threshold: val_loss ≤ 3.28 (the speedrun target)
- Each experiment compared to a same-machine baseline run to control for machine-to-machine variance
