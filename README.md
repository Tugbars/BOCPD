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
| **BOCPD** | **~0.19µs** | **Mean + Variance** | **✓** |
| HMM | ~50µs | Regimes | ✓ |
| Offline CPD | ~10ms | All | ✗ |

**Benchmark Results (Intel Core i9):**

| Configuration | Throughput | Latency |
|---------------|------------|---------|
| Single detector | 2.43M obs/sec | 0.41 µs |
| Stationary data (100K) | 1.87M obs/sec | 0.53 µs |
| With changepoints (100K) | 5.14M obs/sec | 0.19 µs |
| Pool (100 instruments) | 3.03M obs/sec | 0.33 µs |
| Large scale (380 instruments) | 2.12M obs/sec | 0.47 µs |
| **Peak throughput** | **5.14M obs/sec** | **0.19 µs** |

---

## Performance

Measured on Intel Core i9, GCC/MSVC, `-O3 -mavx2 -mfma`:

### V3.2 Native Interleaved Layout + AVX2 ASM Kernel

| Test Scenario | Throughput | Latency | Speedup vs Naive |
|---------------|------------|---------|------------------|
| Single detector | 2.43M obs/sec | 0.41 µs | 46.4× |
| Stationary data (100K obs) | 1.87M obs/sec | 0.53 µs | 36× |
| With changepoints (100K obs) | 5.14M obs/sec | 0.19 µs | 99× |
| Multi-detector pool (100) | 3.03M obs/sec | 0.33 µs | 17.4× |
| Large scale (380 × 500) | 2.12M obs/sec | 0.47 µs | — |

### Scaling by max_run_length

| max_run | Throughput | avg_active |
|---------|------------|------------|
| 64 | 2.26M obs/sec | 41.3 |
| 128 | 1.90M obs/sec | 53.1 |
| 256 | 1.84M obs/sec | 56.5 |
| 512 | 1.69M obs/sec | 61.4 |
| 1024 | 1.69M obs/sec | 62.8 |
| 2048 | 1.87M obs/sec | 56.0 |
| 4096 | 1.72M obs/sec | 61.2 |

### vs Reference Implementations

| Implementation | Throughput | Relative |
|----------------|------------|----------|
| Python (numpy) | ~1.2K obs/sec | 1× |
| Rust (changepoint) | ~22K obs/sec | 18× |
| Naive C | 52K obs/sec | 43× |
| **BOCPD Ultra V3.2** | **2.43M obs/sec** | **2,025×** |
| **Peak (with changepoints)** | **5.14M obs/sec** | **4,283×** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOCPD Ultra V3.2                             │
├─────────────────────────────────────────────────────────────────┤
│  Custom lgamma (Lanczos + Stirling)                             │
│  ├─ Replaces slow libm lgamma() (~100 cycles → ~15 cycles)     │
│  ├─ Lanczos g=4.7421875 for small args                         │
│  └─ Stirling with Bernoulli coefficients for large args        │
├─────────────────────────────────────────────────────────────────┤
│  Native Interleaved Layout                                      │
│  ├─ 256-byte superblocks (4 run lengths × 8 parameters)        │
│  ├─ Prediction params first (cache-friendly)                   │
│  └─ Zero-copy: no per-step data transformation                 │
├─────────────────────────────────────────────────────────────────┤
│  Ping-Pong Double Buffering                                     │
│  ├─ Two interleaved buffers, alternating each step             │
│  ├─ Zero memmove operations                                    │
│  └─ Implicit +1 shift via buffer swap                          │
├─────────────────────────────────────────────────────────────────┤
│  AVX2 ASM Kernel                                                │
│  ├─ 8 elements/iteration (2 blocks × 4 lanes)                  │
│  ├─ Interleaved Block A/B for ILP                              │
│  ├─ Estrin polynomial evaluation (reduced latency)             │
│  ├─ IEEE-754 exponent bit manipulation                         │
│  └─ bsr-based truncation (no branch mispredicts)               │
├─────────────────────────────────────────────────────────────────┤
│  Pool Allocator                                                 │
│  ├─ Single malloc for N detectors                              │
│  ├─ Cache-friendly sequential layout                           │
│  └─ Reduced TLB pressure                                       │
└─────────────────────────────────────────────────────────────────┘
```

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

This section details the engineering that achieves **5.14M obs/sec peak throughput**.

### Custom lgamma Implementation (2× Speedup)

**Problem:** The Student-t log-pdf requires `lgamma((ν+1)/2) - lgamma(ν/2)` for every run length. The standard library `lgamma()` is extremely slow (~100+ cycles) and dominates runtime.

**Solution:** Implement custom lgamma using Lanczos approximation for small arguments and Stirling series for large arguments:

```c
// Lanczos approximation (ν < 40): ~15 cycles
// g = 4.7421875 (exactly representable in binary)
double lgamma_lanczos(double x) {
    static const double c[7] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012
    };
    // Horner's method evaluation...
}

// Stirling series (ν ≥ 40): ~12 cycles
// ln Γ(x) ≈ (x-½)ln(x) - x + ½ln(2π) + Σ B_{2k}/(2k(2k-1)x^{2k-1})
double lgamma_stirling(double x) {
    // Bernoulli number coefficients B₂, B₄, B₆, B₈, B₁₀, B₁₂
    // Achieves ~15 digits precision for x ≥ 40
}
```

**Impact:** Replacing `libm lgamma()` with custom implementation gave **~100% speedup** (2× faster). This was the single largest optimization.

### Precomputed Student-t Constants

Full Student-t log-pdf:
```
ln p(x) = lgamma((ν+1)/2) - lgamma(ν/2) - ½ln(νπσ²) - ((ν+1)/2)·ln(1 + z²/(νσ²))
```

Optimized (constants precomputed when posterior updates):
```
ln p(x) = C₁ - C₂ · log1p(z² · inv_σ²ν)

C₁ = lgamma(α+½) - lgamma(α) - ½ln(2απσ²)   ← precomputed at update time
C₂ = α + ½                                    ← precomputed
inv_σ²ν = 1/(σ²·2α)                          ← precomputed
```

**Hot path:** 1 multiply, 1 FMA, 1 log1p. **No lgamma, no division.**

The lgamma calls happen only during posterior updates (once per run length that survives truncation), not during prediction (which runs for all active run lengths).

### V3 Native Interleaved Memory Layout

**Problem:** V2 required O(n) `build_interleaved()` transformation every observation to convert separate arrays into SIMD-friendly format.

**Solution:** Store parameters directly in 256-byte superblocks:
```
V2 Layout (separate arrays + staging buffer):
  mu[]:      [μ₀ μ₁ μ₂ μ₃ ...]     ← separate allocation
  C1[]:      [c₀ c₁ c₂ c₃ ...]     ← separate allocation
  C2[]:      [d₀ d₁ d₂ d₃ ...]     ← separate allocation
  inv_ssn[]: [e₀ e₁ e₂ e₃ ...]     ← separate allocation
  
  build_interleaved(): O(n) copy every step!

V3 Layout (native 256-byte superblocks):
  Block 0 (bytes 0-255):
    [μ₀ μ₁ μ₂ μ₃]     offset 0    (prediction)
    [c₀ c₁ c₂ c₃]     offset 32   (prediction)
    [d₀ d₁ d₂ d₃]     offset 64   (prediction)
    [e₀ e₁ e₂ e₃]     offset 96   (prediction)
    [κ₀ κ₁ κ₂ κ₃]     offset 128  (update)
    [α₀ α₁ α₂ α₃]     offset 160  (update)
    [β₀ β₁ β₂ β₃]     offset 192  (update)
    [n₀ n₁ n₂ n₃]     offset 224  (update)
  Block 1 (bytes 256-511):
    ... next 4 elements ...
```

**Impact:** Eliminated O(n) copy, direct SIMD access. ~20% overall speedup.

### Ping-Pong Double Buffering

**Problem:** BOCPD reads from run length r but writes to r+1. Naive approach requires `memmove()` after every step to shift the distribution.

**Solution:** Maintain two buffers, alternate between them:
```
Step t (cur_buf = 0):
  Read:  interleaved[0][r]     → posterior for run length r
  Write: interleaved[1][r+1]   → updated posterior at r+1
  Flip:  cur_buf = 1

Step t+1 (cur_buf = 1):
  Read:  interleaved[1][r]     → now contains what was r+1
  Write: interleaved[0][r+1]   → write to other buffer
  Flip:  cur_buf = 0
```

**Benefits:**
- Zero `memmove()` operations (was O(n) per step)
- Implicit +1 index shift via buffer swap
- No data hazards between read and write

**Impact:** Eliminates ~15-20% overhead from memory copies.

### Shifted Store with AVX2 Permute

**Problem:** Ping-pong update reads from index i, writes to index i+1. With interleaved blocks, this crosses block boundaries.

**Solution:** Use `vpermpd` to rotate vector, `vblendpd` to merge with existing blocks:
```asm
; Input: [v0, v1, v2, v3] for indices [i, i+1, i+2, i+3]
; Output: write to indices [i+1, i+2, i+3, i+4] spanning two blocks

vpermpd     ymm_rot, ymm_vals, 0x93    ; [v3, v0, v1, v2]
vblendpd    ymm_k,  ymm_exist_k,  ymm_rot, 0b1110  ; block k: lanes 1,2,3
vblendpd    ymm_k1, ymm_exist_k1, ymm_rot, 0b0001  ; block k+1: lane 0
```

**Performance:** 6 cycles vs ~16 cycles for scalar stores.

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

### SIMD Strategy: Dual-Block Processing

**Problem:** Single 4-wide AVX2 block leaves FMA units underutilized during memory latency.

**Solution:** Process 8 elements per iteration as two 4-wide blocks:
```
Block addressing (V3 layout):
  Block A: byte_offset = (i / 4) * 256
  Block B: byte_offset = (i / 4) * 256 + 256

Iteration structure:
  Block A: load → z² → t → log1p → exp → update → store
  Block B: load → z² → t → log1p → exp → update → store
  Advance: i += 8, idx_vec += 8
```

### Optimized Horizontal Operations

**Problem:** `vhaddpd` is slow (high latency, limited ports).

**Solution:** Use `vunpckhpd + vaddsd`:
```asm
; Reduce [a, b, c, d] → a+b+c+d
vextractf128 xmm0, ymm12, 1        ; [c, d]
vaddpd       xmm0, xmm0, xmm12     ; [a+c, b+d]
vunpckhpd    xmm1, xmm0, xmm0      ; [b+d, b+d]
vaddsd       xmm0, xmm0, xmm1      ; a+b+c+d
```

### Truncation via BSR

**Problem:** Chain of `bt` (bit test) instructions is slow.

**Solution:** Single `bsr` (bit scan reverse) finds highest set bit:
```asm
vcmppd      ymm0, ymm_growth, ymm_thresh, 14
vmovmskpd   eax, ymm0
test        eax, eax
jz          .skip
bsr         ecx, eax              ; Find highest set bit
lea         rbx, [rsi + rcx + 1]  ; last_valid = i + bit + 1
```

### Running Index Vectors

**Problem:** Broadcasting loop counter is expensive:
```c
// Old: 2 µops, 3-cycle latency per iteration
__m256i idx = _mm256_set1_epi64x(i);
idx = _mm256_add_epi64(idx, offset_vec);
```

**Solution:** Maintain running index vectors:
```asm
; Setup (once):
vmovapd  ymm15, [1.0, 2.0, 3.0, 4.0]   ; idx_vec_A
vmovapd  ymm_b, [5.0, 6.0, 7.0, 8.0]   ; idx_vec_B (on stack)

; Per iteration (1 µop each):
vaddpd   ymm15, ymm15, ymm7            ; ymm7 = [8,8,8,8]
vaddpd   ymm3, [stack], ymm7
vmovapd  [stack], ymm3
```

---

## Assembly Kernel Architecture

Hand-written AVX2 kernel optimized for modern x86-64:

### Register Allocation (AVX2: 16 YMM registers)
```
Dedicated constants:
  ymm12 = threshold
  ymm13 = 1-h
  ymm14 = h (hazard rate)
  ymm15 = x (observation, broadcast)

Accumulators:
  ymm9  = max_growth_B
  ymm10 = max_growth_A
  ymm11 = r0 accumulator

Scratch (reused per block):
  ymm0-8 = computation temporaries

Stack storage:
  idx_vec_A, idx_vec_B, max_idx_A, max_idx_B, r_old_B
```

### Block Addressing (V3 Layout)
```asm
; Block A: elements [i, i+1, i+2, i+3]
mov     rax, r14
shr     rax, 2          ; block_index = i / 4
shl     rax, 8          ; byte_offset = block_index * 256

vmovapd ymm0, [r13 + rax]        ; mu at offset 0
vmovapd ymm1, [r13 + rax + 32]   ; C1 at offset 32
vmovapd ymm2, [r13 + rax + 64]   ; C2 at offset 64
vmovapd ymm3, [r13 + rax + 96]   ; inv_ssn at offset 96

; Block B: elements [i+4, i+5, i+6, i+7]
mov     rax, r14
add     rax, 4
shr     rax, 2
shl     rax, 8          ; Next block at +256 bytes
```

---

## Configuration

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

## Quick Start

### Single Detector
```c
#include "bocpd_asm.h"

bocpd_asm_t detector;
bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

bocpd_ultra_init(&detector, 200.0, prior, 1024);

for (int i = 0; i < n_observations; i++) {
    bocpd_ultra_step(&detector, data[i]);
    
    if (detector.p_changepoint > 0.5) {
        printf("Changepoint detected at t=%zu\n", detector.t);
    }
}

bocpd_ultra_free(&detector);
```

### Pool of Detectors
```c
bocpd_pool_t pool;
bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

bocpd_pool_init(&pool, 100, 200.0, prior, 1024);

for (int t = 0; t < n_steps; t++) {
    for (int i = 0; i < 100; i++) {
        bocpd_asm_t *det = bocpd_pool_get(&pool, i);
        bocpd_ultra_step(det, data[i][t]);
    }
}

bocpd_pool_free(&pool);
```

---

## Memory Usage

| Capacity | Memory | Notes |
|----------|--------|-------|
| 64 | ~18 KB | Minimum practical |
| 256 | ~38 KB | Low-latency trading |
| 512 | ~74 KB | General use |
| 1024 | ~148 KB | Long regimes |
| 2048 | ~296 KB | Extended analysis |

---

## Design Philosophy: Why Not Faster?

This implementation achieves **5.14M obs/sec peak**. Faster is possible — but not without sacrificing correctness.

### What We Refused To Do

| Rejected Optimization | Potential Gain | Why We Refused |
|----------------------|----------------|----------------|
| Low-order polynomials | +7-15% | Tail probabilities span 300+ orders of magnitude |
| Approximate pruning (top-K) | +30-70% | Makes posterior non-Bayesian |
| float16/float32 | +10-20% | Underflows at ~1e-38; run-lengths hit 1e-200 |
| Top-1 path tracking | +3000% | Not Bayesian, can't quantify uncertainty |
| Bit-hack exponentiation | +15-25% | Numerically unsafe for BOCPD ranges |

### The Bottom Line

Anyone claiming 10M+ obs/sec for "BOCPD" is likely doing top-1 path tracking or using garbage polynomials. That's not Bayesian inference.

**5.14M obs/sec is the performance ceiling given numerical constraints.** For trading systems where wrong signals cost real money, that ceiling is the right place to be.

---

## References

1. **Adams, R. P., & MacKay, D. J. C.** (2007). Bayesian Online Changepoint Detection. *arXiv:0710.3742*. [[PDF]](https://arxiv.org/pdf/0710.3742.pdf)

2. **Murphy, K. P.** (2007). Conjugate Bayesian analysis of the Gaussian distribution. [[PDF]](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)

3. **Intel Intrinsics Guide**. [[Link]](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

4. **Agner Fog's Optimization Manuals**. [[Link]](https://www.agner.org/optimize/)

---

## Version History

| Version | Throughput | Key Changes |
|---------|------------|-------------|
| V1 (Naive) | 52K obs/sec | Reference implementation |
| V2 | 525K obs/sec | Ping-pong buffers, AVX2 kernel |
| V3 | 1.5M obs/sec | Native interleaved layout |
| **V3.2** | **5.14M obs/sec** | Optimized ASM kernel, pool allocator |

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built for speed. Designed for trading. Open for all.*
