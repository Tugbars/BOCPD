# BOCPD Ultra

**Bayesian Online Change Point Detection — AVX2 Optimized**

A production-grade C implementation of Adams & MacKay (2007) BOCPD, engineered for sub-microsecond latency in high-frequency trading systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C11-green.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![SIMD](https://img.shields.io/badge/SIMD-AVX2-orange.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

---

## Overview

BOCPD maintains a probability distribution over "run lengths" — time since the last changepoint — updated online with each observation. Unlike batch methods, it processes data sequentially with O(active_len) complexity per step.

**Use cases:** Volatility regime detection, trend break identification, adaptive filtering, real-time anomaly detection.

| Method | Latency | Detects | Online |
|--------|---------|---------|--------|
| CUSUM | ~100ns | Mean shift | ✓ |
| **BOCPD** | **~2µs** | **Mean + Variance** | **✓** |
| HMM | ~50µs | Regimes | ✓ |
| Offline CPD | ~10ms | All | ✗ |

---

## Performance

Measured on Intel Core i9-14900K, GCC 12, `-O3 -mavx2 -mfma`:

| Kernel | Throughput | Latency (n=256) | Target |
|--------|------------|-----------------|--------|
| **Generic** | 510K obs/sec | ~2.0 µs | AMD Zen1-4, all Intel |
| **Intel-tuned** | 525K obs/sec | ~1.9 µs | Intel 12th-14th gen |

### Scaling

| Active Length | Throughput | Latency |
|---------------|------------|---------|
| 64 | 766K obs/sec | 1.3 µs |
| 256 | 525K obs/sec | 1.9 µs |
| 1024 | 486K obs/sec | 2.1 µs |
| 4096 | 469K obs/sec | 2.1 µs |

### vs Reference Implementations

| Implementation | Latency (n=200) | Relative |
|----------------|-----------------|----------|
| Python (numpy) | 850 µs | 1× |
| Rust (changepoint) | 45 µs | 19× |
| **BOCPD Ultra** | **~2 µs** | **425×** |

---

## Algorithm

### Bayesian Online Change Point Detection

At each timestep t with observation x_t, BOCPD updates the run-length distribution:

1. **Predict**: Compute P(x_t | run_length = r) via Student-t posterior predictive
2. **Growth**: P(r_t = r) ∝ P(r_{t-1} = r-1) · P(x_t | run) · (1 - H)
3. **Changepoint**: P(r_t = 0) ∝ Σᵣ P(r_{t-1} = r) · P(x_t | run) · H
4. **Normalize**: Ensure Σᵣ P(r_t = r) = 1

Where H = 1/λ is the hazard rate.

### Normal-Gamma Conjugate Model

For Gaussian data with unknown mean μ and variance σ²:

**Prior:** p(μ, τ) = Normal(μ | μ₀, (κ₀τ)⁻¹) · Gamma(τ | α₀, β₀)

**Posterior Predictive (Student-t):**
```
p(x | data) = Student-t(x | μₙ, σ²ₙ, νₙ)

where:
  νₙ = 2αₙ                    (degrees of freedom)
  σ²ₙ = βₙ(κₙ+1)/(αₙκₙ)       (scale)
```

### Detection Signal

The key signal is **not** r[0] directly, but:

1. **MAP drop**: argmax(r) suddenly falls from high to low
2. **Short-run probability**: P(r < k) spikes at changepoints

```
Time:    1   2   3   4   5   6   7   8   9  10  11  12
Data:    ~~~~~~~~~~~~ normal ~~~~~~~~~~~~  | !! SHIFT !!
MAP rl:  1   2   3   4   5   6   7   8   9   2   3   4
P(r<5):  .9  .5  .3  .2  .1  .1  .1  .1  .1  .8  .6  .4
                                            ↑
                                         DETECTED
```

---

## Optimizations

This section details the engineering that achieves 525K obs/sec.

### Memory Layout: Interleaved Blocks

**Problem:** Four parameter arrays (μ, C₁, C₂, 1/σ²ν) accessed together cause 4 cache line fetches per SIMD load.

**Solution:** Interleave into 128-byte blocks (2 cache lines):

```
Traditional layout (4 separate arrays):
  mu[]:     [μ₀ μ₁ μ₂ μ₃ | μ₄ μ₅ μ₆ μ₇ | ...]   ← cache line 1
  C1[]:     [c₀ c₁ c₂ c₃ | c₄ c₅ c₆ c₇ | ...]   ← cache line 2
  C2[]:     [d₀ d₁ d₂ d₃ | d₄ d₅ d₆ d₇ | ...]   ← cache line 3
  inv_ssn[]:[e₀ e₁ e₂ e₃ | e₄ e₅ e₆ e₇ | ...]   ← cache line 4

Interleaved layout (single array):
  Block 0: [μ₀ μ₁ μ₂ μ₃ | c₀ c₁ c₂ c₃ | d₀ d₁ d₂ d₃ | e₀ e₁ e₂ e₃]  ← 2 cache lines
  Block 1: [μ₄ μ₅ μ₆ μ₇ | c₄ c₅ c₆ c₇ | d₄ d₅ d₆ d₇ | e₄ e₅ e₆ e₇]  ← 2 cache lines
```

**Impact:** ~2% improvement at large n, better prefetch behavior.

### Polynomial Approximations

#### log1p via Horner's Method

The Student-t log-pdf requires log(1 + t) where t = z²/σ²ν.

```
log(1+t) ≈ t · (c₁ + t·(c₂ + t·(c₃ + t·(c₄ + t·(c₅ + t·c₆)))))

Coefficients (Taylor series):
  c₁ = 1, c₂ = -1/2, c₃ = 1/3, c₄ = -1/4, c₅ = 1/5, c₆ = -1/6
```

Horner's method: 6 FMAs, fully pipelined, ~15 cycles vs ~100 for libm log().

#### exp via Estrin's Scheme

**Problem:** Horner's method for exp() has 6-deep dependency chain.

**Solution:** Estrin's scheme groups terms to reduce depth:

```
exp(x) = 2^k · 2^f   where k = round(x/ln2), f = frac(x/ln2), |f| ≤ 0.5

2^f ≈ 1 + f·c₁ + f²·c₂ + f³·c₃ + f⁴·c₄ + f⁵·c₅ + f⁶·c₆

Horner (depth 6):
  ((((c₆·f + c₅)·f + c₄)·f + c₃)·f + c₂)·f + c₁)·f + 1

Estrin (depth 4):
  p01 = 1 + f·c₁           }
  p23 = c₂ + f·c₃          } parallel (level 1)
  p45 = c₄ + f·c₅          }
  
  f² = f · f
  
  q0123 = p01 + f²·p23     } parallel (level 2)
  q456  = p45 + f²·c₆      }
  
  f⁴ = f² · f²
  
  result = q0123 + f⁴·q456   (level 3)
```

**Impact:** ~5% improvement on polynomial evaluation.

#### IEEE-754 Exponent Reconstruction

Computing 2^k without libm:

```asm
vcvtpd2dq   xmm0, ymm_k          ; k → int32
vpmovsxdq   ymm0, xmm0           ; sign-extend to int64
vpaddq      ymm0, ymm0, [bias]   ; add exponent bias (1023)
vpsllq      ymm0, ymm0, 52       ; shift to exponent field
; ymm0 now contains bit pattern for 2^k as double
```

Direct bit manipulation: 4 integer ops vs expensive floating-point pow().

### SIMD Strategy: Dual-Block ILP

**Problem:** Single 4-wide AVX2 block leaves FMA units underutilized during memory latency.

**Solution:** Process 8 elements per iteration as two interleaved 4-wide blocks:

```
Iteration structure (Generic kernel):
  Block A: load → z² → t → log1p → exp → update → store
  Block B: load → z² → t → log1p → exp → update → store

Iteration structure (Intel-tuned kernel):
  A: load ─────┬─ z² ─┬─ t ─┬─ log1p ─┬─ exp ─┬─ update
  B:      load ┴─ z² ─┴─ t ─┴─ log1p ─┴─ exp ─┴─ update
              ↑
         Interleaved: B's loads overlap A's compute
```

**Impact:** Intel kernel achieves ~3% better throughput via tighter scheduling.

### Running Index Vectors

**Problem:** MAP tracking needs indices. Broadcasting loop counter is expensive:

```c
// Old: 2 µops, 3-cycle latency per iteration
__m256i idx = _mm256_set1_epi64x(i);
idx = _mm256_add_epi64(idx, offset_vec);
```

**Solution:** Maintain running index vectors, increment by 8 each iteration:

```asm
; Setup (once):
vmovapd  ymm_idx_a, [1.0, 2.0, 3.0, 4.0]
vmovapd  ymm_idx_b, [5.0, 6.0, 7.0, 8.0]

; Per iteration (1 µop, 1-cycle):
vaddpd   ymm_idx_a, ymm_idx_a, [8.0, 8.0, 8.0, 8.0]
vaddpd   ymm_idx_b, ymm_idx_b, [8.0, 8.0, 8.0, 8.0]
```

**Impact:** +1-2% throughput.

### Branchless MAX Tracking

MAP run length requires finding argmax across all growth values:

```asm
vcmppd     ymm_mask, ymm_growth, ymm_max, 14    ; growth > max?
vblendvpd  ymm_max, ymm_max, ymm_growth, ymm_mask
vblendvpd  ymm_idx, ymm_idx, ymm_cur_idx, ymm_mask
```

No branches, no mispredictions. Final horizontal reduction only at loop end.

### Precomputed Student-t Constants

Full Student-t log-pdf:
```
ln p(x) = lgamma((ν+1)/2) - lgamma(ν/2) - ½ln(νπσ²) - ((ν+1)/2)·ln(1 + z²/(νσ²))
```

Optimized (constants precomputed when posterior updates):
```
ln p(x) = C₁ - C₂ · log1p(z² · inv_σ²ν)

C₁ = lgamma(α+½) - lgamma(α) - ½ln(2απσ²)   ← precomputed
C₂ = α + ½                                    ← precomputed
inv_σ²ν = 1/(σ²·2α)                          ← precomputed
```

Hot path: 1 multiply, 1 FMA, 1 log1p. No lgamma, no division.

---

## Assembly Kernel Architecture

Two hand-written AVX2 kernels optimize for different microarchitectures:

### Generic Kernel (`bocpd_kernel_avx2_generic.asm`)

- **Target:** All x86-64 with AVX2+FMA (AMD Zen1-4, Intel Haswell+)
- **Strategy:** Conservative scheduling, sequential A-then-B blocks
- **Throughput:** ~510K obs/sec
- **Use when:** AMD CPUs, unknown target, maximum compatibility

### Intel-Tuned Kernel (`bocpd_kernel_avx2_intel.asm`)

- **Target:** Intel Golden Cove / Raptor Cove (12th-14th gen)
- **Strategy:** Aggressive ILP, interleaved A/B scheduling
- **Throughput:** ~525K obs/sec (+3%)
- **Use when:** Intel Alder Lake, Raptor Lake, or newer

### Register Allocation (AVX2: 16 YMM registers)

```
Preserved (never spilled):
  ymm15 = x (observation, broadcast)
  ymm14 = h (hazard rate)
  ymm13 = 1-h
  ymm12 = threshold
  ymm11 = r0 accumulator
  ymm10 = max_growth_A
  ymm9  = max_growth_B

Scratch (reused per iteration):
  ymm0-8 = computation temporaries

Stack spills:
  idx_vec_A, idx_vec_B (running indices)
  max_idx_A, max_idx_B (argmax tracking)
```

All polynomial constants loaded from L1 cache (fast on modern CPUs, avoids register pressure).

---

## Configuration

### Kernel Selection

```c
// bocpd_config.h

#define BOCPD_KERNEL_GENERIC      0   // Safe default
#define BOCPD_KERNEL_INTEL_PERF   1   // Intel 12th-14th gen

// Set at compile time:
// gcc -DBOCPD_USE_ASM=1 -DBOCPD_KERNEL_VARIANT=1 ...
```

### Prior Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `mu0` | 0 | Prior mean center |
| `kappa0` | 0.001 - 0.1 | Lower = faster adaptation |
| `alpha0` | 0.5 - 5 | Variance prior shape |
| `beta0` | Scale to data | E[σ²] ≈ β₀/(α₀-1) |

**Daily returns (σ ≈ 1-2%):**
```c
bocpd_prior_t prior = {
    .mu0 = 0.0,
    .kappa0 = 0.01,
    .alpha0 = 2.0,
    .beta0 = 0.0002    // E[σ²] ≈ 0.0001
};
```

### Hazard Rate (λ)

| Data Frequency | λ | Expected regime duration |
|----------------|---|--------------------------|
| Tick data | 1000-10000 | Many ticks per regime |
| 1-minute bars | 200-500 | ~1-2 changes per day |
| Daily bars | 50-200 | ~1 change per month |

---

## Integration Examples

### Adaptive Position Sizing

```c
double position_size(double signal, bocpd_t *cpd, double base, double vol) {
    double p_change = bocpd_change_prob(cpd, 5);
    double regime_scale = 1.0 - 0.7 * p_change;  // 30-100%
    double vol_scale = 0.02 / vol;
    return base * signal * regime_scale * vol_scale;
}
```

### Kalman Filter Q Scaling

```c
void process(double x, bocpd_t *cpd, kalman_t *kf) {
    bocpd_step(cpd, x);
    double p = bocpd_change_prob(cpd, 5);
    kalman_set_q_scale(kf, 1.0 + 5.0 * p);  // 1× to 6×
    kalman_update(kf, x);
}
```

### Multi-Instrument

```c
bocpd_t cpd[N_INSTRUMENTS];

void on_tick(int id, double ret) {
    bocpd_step(&cpd[id], ret);
    if (bocpd_change_prob(&cpd[id], 5) > 0.3)
        flag_regime_change(id);
}
```

Each `bocpd_t` instance is independent — trivially parallelizable across threads.

---

## Memory Usage

| Capacity | Memory | Notes |
|----------|--------|-------|
| 128 | ~19 KB | Minimum practical |
| 256 | ~37 KB | Low-latency trading |
| 512 | ~74 KB | General use |
| 1024 | ~148 KB | Long regimes |
| 2048 | ~296 KB | Extended analysis |

---

## Design Philosophy: Why Not Faster?

This implementation achieves ~525K obs/sec. Faster is possible — but not without sacrificing correctness. Every decision below chose **accuracy over speed** where they conflicted.

### What We Refused To Do

| # | Rejected Optimization | Speedup | Why We Refused |
|---|----------------------|---------|----------------|
| 1 | **Low-order polynomials** (4th-order exp, Schraudolph bit-hack, fp32 intermediates) | +7-15% | Tail probabilities span 300+ orders of magnitude. Approximation errors compound across thousands of updates |
| 2 | **Approximate pruning** (top-K, survival decay, half-bin, skip tiny run lengths) | +30-70% | Makes posterior non-normalizable and non-Bayesian. We use only principled threshold truncation |
| 3 | **Block-level truncation** (one threshold check per 8-wide SIMD block) | +1-2% | Mis-handles regime shift boundaries. Per-lane truncation is safer |
| 4 | **float16/float32 compression** (parameters or run-length distribution) | +10-20% | fp32 underflows at ~1e-38. Run-length probabilities routinely hit 1e-200 |
| 5 | **Fused exp(log1p())** (single polynomial for entire expression) | +5-10% | Accuracy dies for large z²/σ². Causes drift and false changepoint detections |
| 6 | **Bit-hack exponentiation** (ML accelerator tricks) | +15-25% | Numerically unsafe for BOCPD input ranges (ln_pp can hit -700) |
| 7 | **Top-1 path tracking** (only track MAP run-length) | +3000-5000% | Not Bayesian. Cannot quantify uncertainty. Useless for VaR/portfolio risk |
| 8 | **Cross-update caching** (reuse partial computations between observations) | +5-10% | Violates independence assumptions. Introduces sequence-specific drift |
| 9 | **Unstable reductions** (fp32 sums, Kahan removal, in-loop horizontal ops) | +2-5% | r0 accumulator sees 256+ additions/step. Errors compound to large bias |
| 10 | **Hard underflow clamp** (`if (pp < 1e-12) pp = 0`) | +1% | Creates discontinuities. We clamp at 1e-300 — effectively zero but continuous |
| 11 | **Approximate lgamma** (Stirling's approximation) | +3-5% | Drifts over long sequences. We use exact recurrence relation |
| 12 | **In-loop horizontal SIMD** (reduce across lanes every iteration) | +0% (actually slower) | We accumulate vertically, reduce once at loop end |

### The Bottom Line

Anyone claiming 2M+ obs/sec for "BOCPD" is likely doing #7 (top-1 path tracking) or #1 (garbage polynomials). That's not Bayesian inference — it's a fast heuristic wearing Bayesian clothing.

**~525K obs/sec is the performance ceiling given these constraints.** For trading systems where wrong signals cost real money, that ceiling is the right place to be.

---

## References

1. **Adams, R. P., & MacKay, D. J. C.** (2007). Bayesian Online Changepoint Detection. *arXiv:0710.3742*. [[PDF]](https://arxiv.org/pdf/0710.3742.pdf)

2. **Murphy, K. P.** (2007). Conjugate Bayesian analysis of the Gaussian distribution. [[PDF]](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)

3. **Intel Intrinsics Guide**. [[Link]](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

4. **Agner Fog's Optimization Manuals**. [[Link]](https://www.agner.org/optimize/)

---

## Future Work

- [ ] AVX-512 kernel (512-bit vectors, 32 YMM registers)
- [ ] ARM NEON port
- [ ] Multi-variate extension
- [ ] Python bindings
- [ ] GPU (CUDA) implementation

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built for speed. Designed for trading.*
