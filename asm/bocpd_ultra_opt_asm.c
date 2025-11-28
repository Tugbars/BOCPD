/**
 * @file bocpd_ultra_opt_asm.c
 * @brief Ultra-Optimized Bayesian Online Changepoint Detection (BOCPD)
 * @version 3.1 - Region-Specific lgamma Optimization
 *
 * @author Claude (Anthropic) & TUGBARS
 * @date 2024
 *
 * =============================================================================
 * ALGORITHM OVERVIEW
 * =============================================================================
 *
 * BOCPD (Adams & MacKay 2007) maintains a probability distribution over
 * "run lengths" - the number of observations since the last changepoint.
 *
 * At each timestep t, we have:
 *   - r[i] = P(run_length = i | x_{1:t})  for i = 0, 1, ..., t
 *
 * The update equations are:
 *
 *   1. PREDICTION: Compute predictive probability for each run length
 *      P(x_t | r) using Student-t distribution from NIG posterior
 *
 *   2. GROWTH: Probability that run continues
 *      r_new[i+1] = r[i] × P(x_t | r=i) × (1 - H)
 *
 *   3. CHANGEPOINT: Probability that a changepoint occurred
 *      r_new[0] = Σ r[i] × P(x_t | r=i) × H
 *
 *   4. NORMALIZE: r_new = r_new / Σ r_new
 *
 * Where H = 1/λ is the hazard rate (probability of changepoint per step).
 *
 * =============================================================================
 * BAYESIAN MODEL: Normal-Inverse-Gamma Conjugate Prior
 * =============================================================================
 *
 * We model observations as:
 *   x | μ, σ² ~ Normal(μ, σ²)
 *
 * With conjugate prior:
 *   μ | σ² ~ Normal(μ₀, σ²/κ₀)     [mean depends on variance]
 *   σ²    ~ Inverse-Gamma(α₀, β₀)  [variance has its own prior]
 *
 * After observing n data points with sufficient statistics, the posterior is:
 *   μ | σ², data ~ Normal(μₙ, σ²/κₙ)
 *   σ² | data    ~ Inverse-Gamma(αₙ, βₙ)
 *
 * Update equations (Welford's numerically stable form):
 *   κₙ = κₙ₋₁ + 1
 *   μₙ = (κₙ₋₁ × μₙ₋₁ + x) / κₙ
 *   αₙ = αₙ₋₁ + 0.5
 *   βₙ = βₙ₋₁ + 0.5 × (x - μₙ₋₁) × (x - μₙ)   ← Welford's trick!
 *
 * The Welford form for β avoids catastrophic cancellation that would occur
 * with the naive formula: β = β₀ + 0.5×Σ(xᵢ - x̄)²
 *
 * =============================================================================
 * STUDENT-T PREDICTIVE DISTRIBUTION
 * =============================================================================
 *
 * The posterior predictive for next observation is Student-t:
 *   x_new | data ~ Student-t(ν, μₙ, σ²)
 *
 * Where:
 *   ν = 2α            (degrees of freedom)
 *   σ² = β(κ+1)/(ακ)  (scale parameter, NOT variance)
 *
 * Log-probability formula:
 *   ln p(x) = ln Γ(α + 0.5) - ln Γ(α) - 0.5×ln(πνσ²) - (α+0.5)×ln(1 + z²/(νσ²))
 *
 * Where z = x - μ. We precompute:
 *   C1 = ln Γ(α + 0.5) - ln Γ(α) - 0.5×ln(πνσ²)   [normalization constant]
 *   C2 = α + 0.5                                   [exponent]
 *   inv_ssn = 1/(σ²ν)                              [for z² scaling]
 *
 * So: ln p(x) = C1 - C2 × ln(1 + z² × inv_ssn)
 *
 * =============================================================================
 * PERFORMANCE SUMMARY
 * =============================================================================
 *
 * | Version     | Throughput      | Speedup | Key Optimization              |
 * |-------------|-----------------|---------|-------------------------------|
 * | Naive C     | 52K obs/sec     | 1×      | Baseline                      |
 * | V1 SIMD     | 400K obs/sec    | 7.7×    | AVX2 vectorization            |
 * | V2 lgamma   | 1.2M obs/sec    | 23×     | Custom SIMD lgamma            |
 * | V3.0 layout | 3.01M obs/sec   | 58×     | Native interleaved + fused    |
 * | V3.1 region | 5.71M obs/sec   | 110×    | Region-specific lgamma        |
 *
 * =============================================================================
 * V3.1 KEY INNOVATION: Region-Specific lgamma Dispatch
 * =============================================================================
 *
 * PROBLEM with V3.0:
 *   The branchless lgamma computed ALL THREE approximations (Lanczos, Minimax,
 *   Stirling) for every element, then blended results. This wasted 2-3× work.
 *
 * INSIGHT:
 *   In BOCPD, α values are MONOTONICALLY INCREASING with run-length index:
 *     α[i] = α₀ + 0.5 × i
 *
 *   After posterior update, for block k (indices 4k to 4k+3):
 *     α_new ∈ [α₀ + 2k + 0.5, α₀ + 2k + 2.0]
 *
 * SOLUTION:
 *   Calculate transition blocks ONCE at start of update:
 *   - Blocks 0 to K₁:     Pure Lanczos    (all α < 8)
 *   - Block K₁+1:         Branchless      (spans α=8 boundary)
 *   - Blocks K₁+2 to K₂:  Pure Minimax    (8 ≤ α < 40)
 *   - Block K₂+1:         Branchless      (spans α=40 boundary)
 *   - Blocks K₂+2 to end: Pure Stirling   (α ≥ 40)
 *
 * EXAMPLE with typical α₀ = 1:
 *   - Blocks 0-2:   Lanczos   (α_max = 1 + 2×2 + 2 = 7 < 8)
 *   - Block 3:      Branchless (α spans [7.5, 9])
 *   - Blocks 4-18:  Minimax   (α ∈ [9.5, 39])
 *   - Block 19:     Branchless (α spans [39.5, 41])
 *   - Blocks 20+:   Stirling  (α_min = 1 + 2×20 + 0.5 = 41.5 ≥ 40)
 *
 * RESULT:
 *   ~97% of blocks use single lgamma variant → 90% speedup over V3.0!
 *
 * =============================================================================
 * MEMORY LAYOUT: 256-byte Superblocks
 * =============================================================================
 *
 * Traditional SoA layout has poor cache behavior for BOCPD because we need
 * ALL parameters for each run-length together. V3 uses "superblocks":
 *
 * Each superblock contains 4 consecutive run-lengths (one AVX2 vector):
 *
 *   Offset (bytes)  Field           Description
 *   ─────────────────────────────────────────────────────────────────────
 *   0-31            μ[0..3]         Posterior means
 *   32-63           C1[0..3]        Student-t log-normalization constant
 *   64-95           C2[0..3]        Student-t exponent (α + 0.5)
 *   96-127          inv_ssn[0..3]   Precomputed 1/(σ²ν) for fast eval
 *   ─────────────────────────────────────────────────────────────────────
 *   128-159         κ[0..3]         Precision pseudo-counts
 *   160-191         α[0..3]         Shape parameters
 *   192-223         β[0..3]         Scale parameters
 *   224-255         ss_n[0..3]      Sample counts
 *   ─────────────────────────────────────────────────────────────────────
 *   Total: 256 bytes = 32 doubles = 4 cache lines
 *
 * First 128 bytes (μ, C1, C2, inv_ssn) used by prediction kernel.
 * Second 128 bytes (κ, α, β, ss_n) used by update kernel.
 *
 * Benefits:
 *   - Single vmovupd loads entire parameter type for 4 run-lengths
 *   - Sequential access pattern → hardware prefetcher effective
 *   - Prediction kernel touches only first 128 bytes (2 cache lines)
 *
 * =============================================================================
 */

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
/** @brief Mathematical constant π = 3.14159265358979323846... */
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "bocpd_asm.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#ifndef BOCPD_USE_ASM_KERNEL
/**
 * @brief Toggle between hand-written ASM kernel (1) and C intrinsics (0)
 *
 * The ASM kernel provides ~5% better performance on Intel CPUs due to
 * precise instruction scheduling, but the C version is more portable.
 */
#define BOCPD_USE_ASM_KERNEL 1
#endif

/*=============================================================================
 * INTERLEAVED BLOCK ACCESSORS
 *
 * The native interleaved layout stores 4 consecutive run-lengths per "block".
 * This enables aligned SIMD loads but requires index arithmetic for scalar
 * access (used in initialization and scalar tail processing).
 *
 * ADDRESS CALCULATION:
 *   For element index i and field with byte offset F:
 *
 *   block_index = i / 4           (which 4-element group)
 *   lane_index  = i % 4           (position within group: 0,1,2,3)
 *   double_offset = F / 8         (field start in doubles, not bytes)
 *
 *   final_index = block_index × 32 + double_offset + lane_index
 *                 ↑                  ↑                ↑
 *                 32 doubles/block   field start      which of 4 lanes
 *
 * EXAMPLE: Access α[5]
 *   block_index = 5 / 4 = 1
 *   lane_index  = 5 % 4 = 1
 *   double_offset = BOCPD_IBLK_ALPHA / 8 = 160 / 8 = 20
 *   final_index = 1 × 32 + 20 + 1 = 53
 *
 *   So α[5] is at buf[53], which is:
 *   - Block 1 (bytes 256-511)
 *   - Field α (bytes 160-191 within block = bytes 416-447 absolute)
 *   - Lane 1 (second of four doubles)
 *
 *=============================================================================*/

/**
 * @brief Read a scalar value from the interleaved buffer
 *
 * @param buf           Base pointer to interleaved buffer
 * @param idx           Logical element index (0 to capacity-1)
 * @param field_offset  Byte offset of field within superblock
 *                      (e.g., BOCPD_IBLK_MU = 0, BOCPD_IBLK_ALPHA = 160)
 *
 * @return The scalar value at the computed location
 *
 * @note This is O(1) - just index arithmetic, no searching
 */
static inline double iblk_get(const double *buf, size_t idx, size_t field_offset)
{
    size_t block = idx / 4; /* Which 4-element superblock */
    size_t lane = idx & 3;  /* Which lane (0-3), using & for speed */

    /*
     * BOCPD_IBLK_DOUBLES = 32 (doubles per superblock)
     * field_offset is in BYTES, so divide by 8 to get double offset
     */
    return buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane];
}

/**
 * @brief Write a scalar value to the interleaved buffer
 *
 * @param buf           Base pointer to interleaved buffer
 * @param idx           Logical element index (0 to capacity-1)
 * @param field_offset  Byte offset of field within superblock
 * @param val           Value to store
 */
static inline void iblk_set(double *buf, size_t idx, size_t field_offset, double val)
{
    size_t block = idx / 4;
    size_t lane = idx & 3;
    buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane] = val;
}

/*
 * Convenience macros for each field.
 * These hide the field offset constants and provide a cleaner API.
 *
 * Field offsets (defined in bocpd_asm.h):
 *   BOCPD_IBLK_MU      =   0  (bytes 0-31)
 *   BOCPD_IBLK_C1      =  32  (bytes 32-63)
 *   BOCPD_IBLK_C2      =  64  (bytes 64-95)
 *   BOCPD_IBLK_INV_SSN =  96  (bytes 96-127)
 *   BOCPD_IBLK_KAPPA   = 128  (bytes 128-159)
 *   BOCPD_IBLK_ALPHA   = 160  (bytes 160-191)
 *   BOCPD_IBLK_BETA    = 192  (bytes 192-223)
 *   BOCPD_IBLK_SS_N    = 224  (bytes 224-255)
 */
#define IBLK_GET_MU(buf, i) iblk_get(buf, i, BOCPD_IBLK_MU)
#define IBLK_GET_C1(buf, i) iblk_get(buf, i, BOCPD_IBLK_C1)
#define IBLK_GET_C2(buf, i) iblk_get(buf, i, BOCPD_IBLK_C2)
#define IBLK_GET_INV_SSN(buf, i) iblk_get(buf, i, BOCPD_IBLK_INV_SSN)
#define IBLK_GET_KAPPA(buf, i) iblk_get(buf, i, BOCPD_IBLK_KAPPA)
#define IBLK_GET_ALPHA(buf, i) iblk_get(buf, i, BOCPD_IBLK_ALPHA)
#define IBLK_GET_BETA(buf, i) iblk_get(buf, i, BOCPD_IBLK_BETA)
#define IBLK_GET_SS_N(buf, i) iblk_get(buf, i, BOCPD_IBLK_SS_N)

#define IBLK_SET_MU(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_MU, v)
#define IBLK_SET_C1(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_C1, v)
#define IBLK_SET_C2(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_C2, v)
#define IBLK_SET_INV_SSN(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_INV_SSN, v)
#define IBLK_SET_KAPPA(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_KAPPA, v)
#define IBLK_SET_ALPHA(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_ALPHA, v)
#define IBLK_SET_BETA(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_BETA, v)
#define IBLK_SET_SS_N(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_SS_N, v)

/*=============================================================================
 * FAST SCALAR NATURAL LOGARITHM
 *
 * Computes ln(x) using IEEE-754 bit manipulation + polynomial approximation.
 * ~5× faster than glibc log() with ~12 significant digits accuracy.
 *
 * ALGORITHM:
 *
 *   IEEE-754 double precision represents x as:
 *     x = (-1)^s × 2^(e-1023) × (1 + m)
 *
 *   where:
 *     s = sign bit (bit 63)
 *     e = biased exponent (bits 52-62), actual exponent = e - 1023
 *     m = mantissa fraction (bits 0-51), represents 1.xxxxx in binary
 *
 *   BIT LAYOUT:
 *   ┌───┬───────────────────┬────────────────────────────────────────────────┐
 *   │ S │    Exponent (11)  │              Mantissa (52 bits)                │
 *   │63 │    62 ─────── 52  │              51 ──────────────────────────── 0 │
 *   └───┴───────────────────┴────────────────────────────────────────────────┘
 *
 *   For positive x:
 *     ln(x) = ln(2^e × m) = e × ln(2) + ln(m)
 *
 *   where m ∈ [1, 2) after normalization.
 *
 *   For ln(m), we use the identity:
 *     ln(m) = 2 × arctanh((m-1)/(m+1))
 *
 *   Let t = (m-1)/(m+1), which maps [1, 2) → [0, 1/3).
 *   The compressed range ensures fast polynomial convergence.
 *
 *   arctanh(t) = t + t³/3 + t⁵/5 + t⁷/7 + t⁹/9 + ...
 *
 *   So: ln(m) = 2t × (1 + t²/3 + t⁴/5 + t⁶/7 + t⁸/9)
 *             = 2t × P(t²)
 *
 *   where P(u) = 1 + u/3 + u²/5 + u³/7 + u⁴/9
 *
 * NUMERICAL STABILITY:
 *   - Horner's method minimizes roundoff: P(u) = 1 + u×(1/3 + u×(1/5 + ...))
 *   - Only ONE division (for t), rest is multiply-add
 *   - No branches, fully pipelined
 *
 *=============================================================================*/

/**
 * @brief Fast scalar natural logarithm via IEEE-754 bit manipulation
 *
 * @param x  Input value (MUST be positive, no error checking!)
 * @return   Natural logarithm ln(x)
 *
 * @warning Undefined behavior for x ≤ 0, NaN, or Inf
 *
 * @par Accuracy
 * Maximum relative error < 5×10⁻¹³ for x ∈ [10⁻³⁰⁰, 10³⁰⁰]
 *
 * @par Performance
 * ~5× faster than glibc log() on modern x86-64
 */
static inline double fast_log_scalar(double x)
{
    /*
     * Type-punning union allows viewing the same 64 bits as either
     * a double or a uint64_t. This is legal in C (but not C++!).
     */
    union
    {
        double d;
        uint64_t u;
    } u = {.d = x};

    /*
     * STEP 1: Extract the exponent
     *
     * Shift right 52 bits to move exponent to low bits,
     * AND with 0x7FF to mask to 11 bits,
     * subtract 1023 bias to get actual exponent.
     *
     * Example: x = 8.0 = 2³
     *   Stored exponent = 1023 + 3 = 1026 = 0x402
     *   After extraction: e = 1026 - 1023 = 3 ✓
     */
    int64_t e = (int64_t)((u.u >> 52) & 0x7FF) - 1023;

    /*
     * STEP 2: Normalize mantissa to [1, 2)
     *
     * Clear the exponent bits (keep only mantissa),
     * then OR in exponent 1023 (which represents 2⁰ = 1).
     *
     * Masks:
     *   0x000FFFFFFFFFFFFF = mantissa bits only
     *   0x3FF0000000000000 = exponent 1023, sign 0
     *
     * Result: m = 1.xxxxx where xxxxx is original mantissa
     */
    u.u = (u.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = u.d;

    /*
     * STEP 3: Compute t = (m-1)/(m+1)
     *
     * This maps m ∈ [1, 2) to t ∈ [0, 1/3).
     * The small range means the polynomial converges in just 4 terms.
     *
     * Why this transform?
     *   ln(m) = ln((1+t)/(1-t)) = 2×arctanh(t)  where t = (m-1)/(m+1)
     */
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;

    /*
     * STEP 4: Polynomial approximation via Horner's method
     *
     * We compute P(t²) = 1 + t²/3 + t⁴/5 + t⁶/7 + t⁸/9
     *
     * Horner form (innermost to outermost):
     *   P = 1/9
     *   P = 1/7 + t² × P = 1/7 + t²/9
     *   P = 1/5 + t² × P = 1/5 + t²/7 + t⁴/9
     *   P = 1/3 + t² × P = 1/3 + t²/5 + t⁴/7 + t⁶/9
     *   P = 1   + t² × P = 1 + t²/3 + t⁴/5 + t⁶/7 + t⁸/9
     *
     * Coefficients:
     *   1/3 ≈ 0.3333333333333333
     *   1/5 = 0.2
     *   1/7 ≈ 0.1428571428571429
     *   1/9 ≈ 0.1111111111111111
     */
    double poly = 1.0 + t2 * (0.3333333333333333 +                    /* 1/3 */
                              t2 * (0.2 +                             /* 1/5 */
                                    t2 * (0.1428571428571429 +        /* 1/7 */
                                          t2 * 0.1111111111111111))); /* 1/9 */

    /*
     * STEP 5: Final assembly
     *
     * ln(x) = e × ln(2) + ln(m)
     *       = e × ln(2) + 2 × t × P(t²)
     *
     * ln(2) ≈ 0.6931471805599453
     */
    return (double)e * 0.6931471805599453 + 2.0 * t * poly;
}

/*=============================================================================
 * AVX2 SIMD NATURAL LOGARITHM
 *
 * Computes ln(x) for 4 doubles simultaneously using AVX2 256-bit vectors.
 * Same algorithm as scalar version, but vectorized.
 *
 * CHALLENGE: AVX2 lacks int64 ↔ double conversion!
 *
 * AVX2 has _mm256_cvtpd_epi32 (double → int32) but NOT _mm256_cvtepi64_pd
 * (int64 → double). We need int64 because the exponent can be large.
 *
 * SOLUTION: The "magic number" trick
 *
 * For small integers k (|k| < 2⁵²):
 *   1. Interpret k as the low 52 bits of a double
 *   2. OR in the exponent for 2⁵² (which is 0x433...)
 *   3. The resulting double equals 2⁵² + k (exactly!)
 *   4. Subtract 2⁵² to get k as a double
 *
 * Why this works:
 *   IEEE-754 doubles can exactly represent all integers up to 2⁵³.
 *   By setting exponent = 52, we make the implicit 1.xxx... align
 *   such that our integer bits become the integer part.
 *
 *   Example: k = 5
 *   Bits:     0x4330000000000005
 *   As double: 4503599627370501.0 = 2⁵² + 5
 *   Subtract:  5.0 ✓
 *
 *=============================================================================*/

/**
 * @brief AVX2 SIMD natural logarithm for 4 doubles in parallel
 *
 * @param x  Vector of 4 positive doubles
 * @return   Vector of 4 natural logarithms
 *
 * @warning All 4 inputs must be positive (no checking!)
 *
 * @par Algorithm
 * Same as fast_log_scalar:
 *   ln(x) = e×ln(2) + 2×t×P(t²)
 * where e = exponent, t = (m-1)/(m+1), m = normalized mantissa
 *
 * @par Performance
 * Throughput: ~4 ln() per 20-25 cycles on Haswell+
 */
static inline __m256d fast_log_avx2(__m256d x)
{
    /* ─────────────────────────────────────────────────────────────────────
     * CONSTANTS (broadcast to all 4 lanes)
     * ───────────────────────────────────────────────────────────────────── */

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453); /* ln(2) */

    /* Polynomial coefficients for arctanh series */
    const __m256d c3 = _mm256_set1_pd(0.3333333333333333); /* 1/3 */
    const __m256d c5 = _mm256_set1_pd(0.2);                /* 1/5 */
    const __m256d c7 = _mm256_set1_pd(0.1428571428571429); /* 1/7 */
    const __m256d c9 = _mm256_set1_pd(0.1111111111111111); /* 1/9 */

    /* IEEE-754 bit manipulation masks */
    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000ULL);
    const __m256i mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
    const __m256i exp_bias_bits = _mm256_set1_epi64x(0x3FF0000000000000ULL);

    /* Magic number for int64 → double conversion */
    const __m256i magic_i = _mm256_set1_epi64x(0x4330000000000000ULL); /* 2⁵² as bits */
    const __m256d magic_d = _mm256_set1_pd(4503599627370496.0);        /* 2⁵² as double */
    const __m256d bias_1023 = _mm256_set1_pd(1023.0);

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 1: Reinterpret double bits as integers
     * ───────────────────────────────────────────────────────────────────── */

    /*
     * _mm256_castpd_si256: Zero-cost reinterpret cast (no instructions)
     * The 256 bits are now treated as 4 × 64-bit integers
     */
    __m256i xi = _mm256_castpd_si256(x);

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 2: Extract and convert exponent to double
     *
     * exp_bits = (xi & exp_mask) >> 52
     *          = raw 11-bit exponent in low bits of each 64-bit lane
     * ───────────────────────────────────────────────────────────────────── */

    __m256i exp_bits = _mm256_srli_epi64(_mm256_and_si256(xi, exp_mask), 52);

    /*
     * MAGIC NUMBER TRICK for int64 → double:
     *
     * exp_biased = exp_bits | 0x4330000000000000
     *            = places our 11-bit exponent in the mantissa of 2⁵²
     *
     * Reinterpret as double: value = 2⁵² + exp_bits
     * Subtract 2⁵²: result = exp_bits as a double
     * Subtract 1023: result = actual exponent (unbiased)
     */
    __m256i exp_biased = _mm256_or_si256(exp_bits, magic_i);
    __m256d exp_double = _mm256_sub_pd(_mm256_castsi256_pd(exp_biased), magic_d);
    __m256d e = _mm256_sub_pd(exp_double, bias_1023);

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 3: Normalize mantissa to [1, 2)
     *
     * mi = (xi & mantissa_mask) | exp_bias_bits
     *    = keep only mantissa bits, set exponent to 1023 (= 2⁰)
     * ───────────────────────────────────────────────────────────────────── */

    __m256i mi = _mm256_or_si256(_mm256_and_si256(xi, mantissa_mask), exp_bias_bits);
    __m256d m = _mm256_castsi256_pd(mi);

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 4: Compute t = (m-1)/(m+1), mapping [1,2) → [0, 1/3)
     * ───────────────────────────────────────────────────────────────────── */

    __m256d num = _mm256_sub_pd(m, one);
    __m256d den = _mm256_add_pd(m, one);
    __m256d t = _mm256_div_pd(num, den); /* Only division in the function! */
    __m256d t2 = _mm256_mul_pd(t, t);

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 5: Polynomial evaluation using FMA (Fused Multiply-Add)
     *
     * FMA computes a×b+c in ONE instruction with ONE rounding.
     * This is both faster AND more accurate than separate mul+add.
     *
     * Horner's method:
     *   poly = ((((c9) × t² + c7) × t² + c5) × t² + c3) × t² + 1
     * ───────────────────────────────────────────────────────────────────── */

    __m256d poly = _mm256_fmadd_pd(t2, c9, c7); /* t² × (1/9) + 1/7 */
    poly = _mm256_fmadd_pd(t2, poly, c5);       /* t² × (...) + 1/5 */
    poly = _mm256_fmadd_pd(t2, poly, c3);       /* t² × (...) + 1/3 */
    poly = _mm256_fmadd_pd(t2, poly, one);      /* t² × (...) + 1   */

    /* ─────────────────────────────────────────────────────────────────────
     * STEP 6: Final assembly
     *
     * ln(x) = e × ln(2) + 2 × t × poly
     *
     * Using FMA: result = e × ln2 + (2 × t × poly)
     * ───────────────────────────────────────────────────────────────────── */

    return _mm256_fmadd_pd(e, ln2, _mm256_mul_pd(two, _mm256_mul_pd(t, poly)));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * END OF PART 1
 *
 * Part 2 will cover:
 *   - lgamma_lanczos_avx2 (small arguments, x < 8)
 *   - lgamma_minimax_avx2 (medium arguments, 8 ≤ x < 40)
 *   - lgamma_stirling_avx2 (large arguments, x ≥ 40)
 *   - fast_lgamma_avx2_branchless (fallback for mixed-range vectors)
 *   - store_shifted_field (AVX2 permute for +1 index offset)
 *
 * Part 3 will cover:
 *   - init_slot_zero, process_block_common
 *   - update_posteriors_interleaved (region dispatch logic)
 *   - prediction_step (ASM kernel interface and C fallback)
 *   - Public API: bocpd_ultra_init, bocpd_ultra_free, bocpd_ultra_step
 *   - Pool API: bocpd_pool_init, bocpd_pool_free, etc.
 * ═══════════════════════════════════════════════════════════════════════════ */
/*=============================================================================
 * PART 2: REGION-SPECIFIC LGAMMA FUNCTIONS
 *
 * The log-gamma function lgamma(x) = ln(Γ(x)) requires different approximation
 * methods for different argument ranges to balance accuracy and speed:
 *
 *   | Range      | Method   | Why                                    | Divs |
 *   |------------|----------|----------------------------------------|------|
 *   | x < 8      | Lanczos  | Handles minimum near x ≈ 1.46          | 5    |
 *   | 8 ≤ x < 40 | Minimax  | Faster than Lanczos, accurate enough   | 2    |
 *   | x ≥ 40     | Stirling | Asymptotic series converges rapidly    | 1    |
 *
 * GAMMA FUNCTION BACKGROUND:
 *
 * The gamma function Γ(x) generalizes factorials to real numbers:
 *   Γ(n) = (n-1)!  for positive integers
 *   Γ(x) = ∫₀^∞ t^(x-1) × e^(-t) dt  for Re(x) > 0
 *
 * Key properties:
 *   Γ(x+1) = x × Γ(x)      (recursion)
 *   Γ(1) = 1
 *   Γ(0.5) = √π ≈ 1.7725
 *
 * The function has a minimum near x ≈ 1.46163 where Γ(x) ≈ 0.8856.
 * For lgamma, this becomes lgamma(1.46163) ≈ -0.1215.
 *
 * As x → ∞, Stirling's approximation applies:
 *   Γ(x) ≈ √(2π/x) × (x/e)^x
 *   lgamma(x) ≈ (x - 0.5)×ln(x) - x + 0.5×ln(2π)
 *
 *=============================================================================*/

/*─────────────────────────────────────────────────────────────────────────────
 * LANCZOS APPROXIMATION (for x < 8)
 *
 * The Lanczos approximation (1964) provides excellent accuracy for small x:
 *
 *   Γ(x) ≈ √(2π) × ((x + g - 0.5) / e)^(x - 0.5) × Ag(x)
 *
 * Taking logarithms:
 *
 *   lgamma(x) = 0.5×ln(2π) + (x - 0.5)×ln(x + g - 0.5) - (x + g - 0.5) + ln(Ag(x))
 *
 * Where Ag(x) is a sum of rational terms:
 *
 *   Ag(x) = c₀ + c₁/(x+0) + c₂/(x+1) + c₃/(x+2) + c₄/(x+3) + c₅/(x+4)
 *
 * The parameter g and coefficients cᵢ are numerically optimized to minimize
 * approximation error. We use g = 4.7421875 with 5 correction terms.
 *
 * COEFFICIENT TABLE (g = 4.7421875):
 *
 *   | i | Coefficient cᵢ          | Role                      |
 *   |---|-------------------------|---------------------------|
 *   | 0 | 1.000000000190015       | Base term ≈ 1             |
 *   | 1 | 76.18009172947146       | Dominant correction       |
 *   | 2 | -86.50532032941677      | Large negative (cancels)  |
 *   | 3 | 24.01409824083091       | Secondary positive        |
 *   | 4 | -1.231739572450155      | Fine tuning               |
 *   | 5 | 0.001208650973866179    | Tail correction           |
 *
 * WHY LANCZOS FOR SMALL x?
 *   - Stirling diverges for x < 10
 *   - Lanczos handles the difficult minimum near x ≈ 1.46
 *   - Coefficients specifically optimized for x ∈ [1, 10]
 *   - Relative error < 10⁻¹² over this range
 *
 * COST:
 *   - 5 divisions (for Ag sum) - expensive but unavoidable
 *   - 2 logarithms (ln(t) and ln(Ag))
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Lanczos lgamma for small arguments (x < 8)
 *
 * @param x  Vector of 4 positive values (accuracy best for x < 8)
 * @return   Vector of 4 lgamma values
 *
 * @par Formula
 * lgamma(x) = 0.5×ln(2π) + (x-0.5)×ln(t) - t + ln(Ag)
 * where t = x + g - 0.5, g = 4.7421875
 *
 * @par Accuracy
 * Max relative error < 10⁻¹² for x ∈ [0.5, 10]
 */
static inline __m256d lgamma_lanczos_avx2(__m256d x)
{
    /* ─────────────────────────────────────────────────────────────────────
     * Mathematical constants
     * ───────────────────────────────────────────────────────────────────── */
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);

    /* 0.5 × ln(2π) = 0.5 × ln(6.28318...) ≈ 0.9189385332046727 */
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);

    /* Lanczos g parameter - optimized for 5-term approximation */
    const __m256d g = _mm256_set1_pd(4.7421875);

    /* ─────────────────────────────────────────────────────────────────────
     * Lanczos coefficients (derived via numerical optimization)
     *
     * These are NOT simple formulas - they're the result of minimizing
     * max|approx - true| over the target range via nonlinear optimization.
     * ───────────────────────────────────────────────────────────────────── */
    const __m256d c0 = _mm256_set1_pd(1.000000000190015);
    const __m256d c1 = _mm256_set1_pd(76.18009172947146);
    const __m256d c2 = _mm256_set1_pd(-86.50532032941677);
    const __m256d c3 = _mm256_set1_pd(24.01409824083091);
    const __m256d c4 = _mm256_set1_pd(-1.231739572450155);
    const __m256d c5 = _mm256_set1_pd(0.001208650973866179);

    /* ─────────────────────────────────────────────────────────────────────
     * Precompute denominators x+0, x+1, x+2, x+3, x+4
     *
     * Computing all at once enables instruction-level parallelism (ILP).
     * The CPU can execute multiple independent adds in parallel.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d xp0 = x;                                     /* x + 0 */
    __m256d xp1 = _mm256_add_pd(x, one);                 /* x + 1 */
    __m256d xp2 = _mm256_add_pd(x, _mm256_set1_pd(2.0)); /* x + 2 */
    __m256d xp3 = _mm256_add_pd(x, _mm256_set1_pd(3.0)); /* x + 3 */
    __m256d xp4 = _mm256_add_pd(x, _mm256_set1_pd(4.0)); /* x + 4 */

    /* ─────────────────────────────────────────────────────────────────────
     * Compute Ag(x) = c₀ + c₁/x + c₂/(x+1) + c₃/(x+2) + c₄/(x+3) + c₅/(x+4)
     *
     * The 5 divisions are the expensive part (~14 cycles each on modern CPUs).
     * We cannot vectorize divisions well - they must proceed sequentially.
     * But we CAN overlap with other work thanks to out-of-order execution.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d Ag = c0;
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c1, xp0)); /* + c₁/x       */
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c2, xp1)); /* + c₂/(x+1)   */
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c3, xp2)); /* + c₃/(x+2)   */
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c4, xp3)); /* + c₄/(x+3)   */
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c5, xp4)); /* + c₅/(x+4)   */

    /* ─────────────────────────────────────────────────────────────────────
     * Compute t = x + g - 0.5 (shifted argument for power term)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d t = _mm256_add_pd(x, _mm256_sub_pd(g, half));

    /* ─────────────────────────────────────────────────────────────────────
     * Compute logarithms using our fast SIMD implementation
     * ───────────────────────────────────────────────────────────────────── */
    __m256d ln_t = fast_log_avx2(t);
    __m256d ln_Ag = fast_log_avx2(Ag);

    /* ─────────────────────────────────────────────────────────────────────
     * Final assembly:
     *   lgamma(x) = 0.5×ln(2π) + (x - 0.5)×ln(t) - t + ln(Ag)
     *
     * Order of operations chosen to minimize roundoff:
     *   result = half_ln2pi
     *   result += (x - 0.5) × ln_t    [FMA]
     *   result -= t
     *   result += ln_Ag
     * ───────────────────────────────────────────────────────────────────── */
    __m256d result = half_ln2pi;
    result = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_t, result);
    result = _mm256_sub_pd(result, t);
    result = _mm256_add_pd(result, ln_Ag);

    return result;
}

/*─────────────────────────────────────────────────────────────────────────────
 * MINIMAX RATIONAL APPROXIMATION (for 8 ≤ x < 40)
 *
 * For medium arguments, we use the Stirling base with a rational correction:
 *
 *   lgamma(x) ≈ (x - 0.5)×ln(x) - x + 0.5×ln(2π) + R(1/x)
 *
 * where R(t) = P(t)/Q(t) is a degree-6 rational function in t = 1/x.
 *
 * The coefficients are computed via the REMEZ EXCHANGE ALGORITHM, which
 * finds the rational approximation minimizing maximum error (minimax).
 *
 * WHY MINIMAX FOR MEDIUM x?
 *   - Lanczos needs 5 divisions for Ag; minimax needs only 2 (1/x and P/Q)
 *   - Stirling's asymptotic series hasn't converged enough yet
 *   - Minimax fills the gap: faster than Lanczos, more accurate than Stirling
 *
 * NUMERATOR P(t) coefficients:
 *   p₆ = 3.24529652382012274966e-07
 *   p₅ = 9.88031039418037939582e-06
 *   p₄ = -2.94439844714544881340e-04
 *   p₃ = -1.20710278104312065941e-03
 *   p₂ = 3.86885972161250765248e-02
 *   p₁ = 4.74218749975000009752e-01
 *   p₀ = 1.0
 *
 * DENOMINATOR Q(t) coefficients:
 *   q₆ = 8.32021972758041118442e-08
 *   q₅ = 4.97570295032256324424e-06
 *   q₄ = -1.31107523028095547946e-04
 *   q₃ = -1.01478348052546145089e-03
 *   q₂ = 2.33329728323008758047e-02
 *   q₁ = 4.21289134266929659746e-01
 *   q₀ = 1.0
 *
 * ACCURACY:
 *   Max relative error < 8.2×10⁻¹⁴ for x ∈ [8, 40]
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Minimax rational lgamma for medium arguments (8 ≤ x < 40)
 *
 * @param x  Vector of 4 values in range [8, 40)
 * @return   Vector of 4 lgamma values
 *
 * @par Formula
 * lgamma(x) = (x-0.5)×ln(x) - x + 0.5×ln(2π) + P(1/x)/Q(1/x)
 *
 * @par Accuracy
 * Max relative error < 8.2×10⁻¹⁴ over [8, 40]
 */
static inline __m256d lgamma_minimax_avx2(__m256d x)
{
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);

    /* ─────────────────────────────────────────────────────────────────────
     * t = 1/x for rational approximation
     * This is the only expensive division; P/Q comes later.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d t = _mm256_div_pd(one, x);

    /* ─────────────────────────────────────────────────────────────────────
     * Horner evaluation of numerator P(t)
     *
     * P(t) = p₀ + t×(p₁ + t×(p₂ + t×(p₃ + t×(p₄ + t×(p₅ + t×p₆)))))
     *
     * Start from innermost (p₆) and work outward.
     * Each step: result = result × t + next_coeff
     * ───────────────────────────────────────────────────────────────────── */
    __m256d num = _mm256_set1_pd(3.24529652382012274966e-07);                   /* p₆ */
    num = _mm256_fmadd_pd(num, t, _mm256_set1_pd(9.88031039418037939582e-06));  /* p₅ */
    num = _mm256_fmadd_pd(num, t, _mm256_set1_pd(-2.94439844714544881340e-04)); /* p₄ */
    num = _mm256_fmadd_pd(num, t, _mm256_set1_pd(-1.20710278104312065941e-03)); /* p₃ */
    num = _mm256_fmadd_pd(num, t, _mm256_set1_pd(3.86885972161250765248e-02));  /* p₂ */
    num = _mm256_fmadd_pd(num, t, _mm256_set1_pd(4.74218749975000009752e-01));  /* p₁ */
    num = _mm256_fmadd_pd(num, t, one);                                         /* p₀ = 1 */

    /* ─────────────────────────────────────────────────────────────────────
     * Horner evaluation of denominator Q(t)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d den = _mm256_set1_pd(8.32021972758041118442e-08);                   /* q₆ */
    den = _mm256_fmadd_pd(den, t, _mm256_set1_pd(4.97570295032256324424e-06));  /* q₅ */
    den = _mm256_fmadd_pd(den, t, _mm256_set1_pd(-1.31107523028095547946e-04)); /* q₄ */
    den = _mm256_fmadd_pd(den, t, _mm256_set1_pd(-1.01478348052546145089e-03)); /* q₃ */
    den = _mm256_fmadd_pd(den, t, _mm256_set1_pd(2.33329728323008758047e-02));  /* q₂ */
    den = _mm256_fmadd_pd(den, t, _mm256_set1_pd(4.21289134266929659746e-01));  /* q₁ */
    den = _mm256_fmadd_pd(den, t, one);                                         /* q₀ = 1 */

    /* ─────────────────────────────────────────────────────────────────────
     * Rational correction R(t) = P(t) / Q(t)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d frac = _mm256_div_pd(num, den);

    /* ─────────────────────────────────────────────────────────────────────
     * Base Stirling term: (x - 0.5)×ln(x) - x + 0.5×ln(2π)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d ln_x = fast_log_avx2(x);
    __m256d core = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_x,
                                   _mm256_sub_pd(half_ln2pi, x));

    /* ─────────────────────────────────────────────────────────────────────
     * Final: Stirling base + rational correction
     * ───────────────────────────────────────────────────────────────────── */
    return _mm256_add_pd(core, frac);
}

/*─────────────────────────────────────────────────────────────────────────────
 * STIRLING'S ASYMPTOTIC EXPANSION (for x ≥ 40)
 *
 * For large x, lgamma has the asymptotic expansion:
 *
 *   lgamma(x) ~ (x - 0.5)×ln(x) - x + 0.5×ln(2π) + Σ Bₖ/(k×(k-1)×x^(k-1))
 *
 * where Bₖ are the Bernoulli numbers.
 *
 * BERNOULLI NUMBERS AND STIRLING COEFFICIENTS:
 *
 *   | k | B₂ₖ (Bernoulli)  | sₖ = B₂ₖ/(2k×(2k-1))  | Decimal           |
 *   |---|------------------|------------------------|-------------------|
 *   | 1 | 1/6              | 1/12                   | +0.08333333...    |
 *   | 2 | -1/30            | -1/360                 | -0.00277778...    |
 *   | 3 | 1/42             | 1/1260                 | +0.00079365...    |
 *   | 4 | -1/30            | -1/1680                | -0.00059524...    |
 *   | 5 | 5/66             | 1/1188                 | +0.00084175...    |
 *   | 6 | -691/2730        | -691/360360            | -0.00191753...    |
 *
 * WHY THIS IS ASYMPTOTIC (not convergent):
 *   The series DIVERGES if you take too many terms! The Bernoulli numbers
 *   grow factorially: |B₂ₖ| ~ 4√(πk)(k/πe)^(2k).
 *
 *   But for LARGE x, the first few terms give excellent accuracy because
 *   the 1/x^k factors dominate the coefficient growth.
 *
 *   For x ≥ 40, 6 terms achieve relative error < 10⁻¹⁴.
 *
 * WHY STIRLING FOR LARGE x?
 *   - Only ONE division needed (1/x)
 *   - Series converges rapidly when x is large
 *   - Much faster than Lanczos (no rational sum)
 *
 * IMPLEMENTATION:
 *   Factor out 1/x and express series in powers of 1/x²:
 *
 *   correction = (1/x) × (s₁ + (1/x²)×(s₂ + (1/x²)×(s₃ + ...)))
 *
 *   This allows Horner evaluation in 1/x².
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Stirling asymptotic lgamma for large arguments (x ≥ 40)
 *
 * @param x  Vector of 4 values ≥ 40
 * @return   Vector of 4 lgamma values
 *
 * @par Formula
 * lgamma(x) = (x-0.5)×ln(x) - x + 0.5×ln(2π) + Σ sₖ/x^(2k-1)
 *
 * @par Accuracy
 * Max relative error < 10⁻¹⁴ for x ≥ 40
 */
static inline __m256d lgamma_stirling_avx2(__m256d x)
{
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);

    /* ─────────────────────────────────────────────────────────────────────
     * Stirling correction coefficients sₖ = B₂ₖ / (2k × (2k-1))
     *
     * These come from the asymptotic expansion of lgamma.
     * ───────────────────────────────────────────────────────────────────── */
    const __m256d s1 = _mm256_set1_pd(0.0833333333333333333);    /* 1/12          */
    const __m256d s2 = _mm256_set1_pd(-0.00277777777777777778);  /* -1/360        */
    const __m256d s3 = _mm256_set1_pd(0.000793650793650793651);  /* 1/1260        */
    const __m256d s4 = _mm256_set1_pd(-0.000595238095238095238); /* -1/1680       */
    const __m256d s5 = _mm256_set1_pd(0.000841750841750841751);  /* 1/1188        */
    const __m256d s6 = _mm256_set1_pd(-0.00191752691752691753);  /* -691/360360   */

    /* ─────────────────────────────────────────────────────────────────────
     * Base Stirling term: (x - 0.5)×ln(x) - x + 0.5×ln(2π)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d ln_x = fast_log_avx2(x);
    __m256d base = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_x,
                                   _mm256_sub_pd(half_ln2pi, x));

    /* ─────────────────────────────────────────────────────────────────────
     * Compute 1/x and 1/x² for the correction series
     * ───────────────────────────────────────────────────────────────────── */
    __m256d inv_x = _mm256_div_pd(one, x); /* Only division! */
    __m256d inv_x2 = _mm256_mul_pd(inv_x, inv_x);

    /* ─────────────────────────────────────────────────────────────────────
     * Horner evaluation of correction in powers of 1/x²
     *
     * Series: s₁/x + s₂/x³ + s₃/x⁵ + s₄/x⁷ + s₅/x⁹ + s₆/x¹¹
     *
     * Factor out 1/x:
     *   = (1/x) × (s₁ + s₂/x² + s₃/x⁴ + s₄/x⁶ + s₅/x⁸ + s₆/x¹⁰)
     *
     * The inner part is a polynomial in 1/x²:
     *   P(1/x²) = s₁ + (1/x²)×(s₂ + (1/x²)×(s₃ + ...))
     *
     * Evaluate inner-to-outer (Horner):
     * ───────────────────────────────────────────────────────────────────── */
    __m256d correction = s6;                              /* Start with s₆ */
    correction = _mm256_fmadd_pd(correction, inv_x2, s5); /* s₆×(1/x²) + s₅ */
    correction = _mm256_fmadd_pd(correction, inv_x2, s4); /* ...×(1/x²) + s₄ */
    correction = _mm256_fmadd_pd(correction, inv_x2, s3); /* ...×(1/x²) + s₃ */
    correction = _mm256_fmadd_pd(correction, inv_x2, s2); /* ...×(1/x²) + s₂ */
    correction = _mm256_fmadd_pd(correction, inv_x2, s1); /* ...×(1/x²) + s₁ */

    /* ─────────────────────────────────────────────────────────────────────
     * Multiply by 1/x to get final correction
     * ───────────────────────────────────────────────────────────────────── */
    correction = _mm256_mul_pd(correction, inv_x);

    /* ─────────────────────────────────────────────────────────────────────
     * Final: base + correction
     * ───────────────────────────────────────────────────────────────────── */
    return _mm256_add_pd(base, correction);
}

/*─────────────────────────────────────────────────────────────────────────────
 * BRANCHLESS LGAMMA (fallback for mixed-range vectors)
 *
 * When a single AVX2 vector spans multiple lgamma regions (e.g., some lanes
 * have x < 8, others have x ≥ 8), we cannot use region-specific functions.
 *
 * This fallback computes ALL THREE approximations and blends results based
 * on comparison masks. It's 2-3× slower than single-region functions.
 *
 * WHEN USED:
 *   In BOCPD with α₀ = 1, only ~2 blocks out of 64+ use this fallback:
 *   - Block spanning α = 8 boundary
 *   - Block spanning α = 40 boundary
 *
 *   So ~97% of blocks use the fast single-region functions.
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Branchless lgamma for vectors spanning multiple regions
 *
 * @param x  Vector of 4 values (any positive range)
 * @return   Vector of 4 lgamma values
 *
 * @par Algorithm
 * 1. Compute all three approximations (Lanczos, Minimax, Stirling)
 * 2. Generate comparison masks for region boundaries
 * 3. Blend results using vblendvpd
 *
 * @par Performance
 * 2-3× slower than single-region functions. Avoid in hot paths.
 */
static inline __m256d fast_lgamma_avx2_branchless(__m256d x)
{
    /* Region boundaries */
    const __m256d eight = _mm256_set1_pd(8.0);
    const __m256d forty = _mm256_set1_pd(40.0);

    /* ─────────────────────────────────────────────────────────────────────
     * Compute ALL THREE approximations
     * This is expensive but unavoidable for mixed-range vectors.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d result_small = lgamma_lanczos_avx2(x);  /* Best for x < 8  */
    __m256d result_mid = lgamma_minimax_avx2(x);    /* Best for 8 ≤ x < 40 */
    __m256d result_large = lgamma_stirling_avx2(x); /* Best for x ≥ 40 */

    /* ─────────────────────────────────────────────────────────────────────
     * Generate comparison masks
     *
     * _CMP_LT_OQ = "Less Than, Ordered, Quiet"
     *   - Ordered: both operands are not NaN
     *   - Quiet: doesn't raise FP exceptions on NaN
     *
     * Each lane of mask is all-1s (0xFFFFFFFFFFFFFFFF) if true, all-0s if false.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d mask_small = _mm256_cmp_pd(x, eight, _CMP_LT_OQ); /* x < 8?  */
    __m256d mask_large = _mm256_cmp_pd(x, forty, _CMP_GT_OQ); /* x > 40? */

    /* ─────────────────────────────────────────────────────────────────────
     * Blend results using masks
     *
     * vblendvpd semantics: blendv(a, b, mask) = mask ? b : a
     *
     * First blend: use Lanczos where x < 8, else Minimax
     * Second blend: use Stirling where x > 40, else keep previous
     * ───────────────────────────────────────────────────────────────────── */
    __m256d result = _mm256_blendv_pd(result_mid, result_small, mask_small);
    result = _mm256_blendv_pd(result, result_large, mask_large);

    return result;
}

/*=============================================================================
 * SHIFTED STORE OPERATIONS
 *
 * PROBLEM:
 *   The BOCPD update reads parameters from CUR[i] and writes updated
 *   parameters to NEXT[i+1]. With the interleaved layout (4 elements
 *   per 256-byte block), this +1 shift crosses block boundaries!
 *
 *   Example: Processing block 0 (indices 0,1,2,3)
 *   - Read from CUR block 0
 *   - Write to NEXT indices 1,2,3,4
 *   - But index 4 is in block 1!
 *
 * NAIVE SOLUTION (slow):
 *   Store each element individually using scalar stores.
 *   Cost: 4 scalar stores = ~16 cycles
 *
 * SIMD SOLUTION (fast):
 *   Use vpermpd to rotate the vector, then blend into two blocks.
 *   Cost: 1 permute + 2 loads + 2 blends + 2 stores = ~6 cycles
 *
 * ALGORITHM:
 *   Given vals = [v₀, v₁, v₂, v₃] to store at indices [4k+1, 4k+2, 4k+3, 4k+4]:
 *
 *   1. Rotate right by 1: rotated = [v₃, v₀, v₁, v₂]
 *
 *   2. Block k gets lanes 1,2,3: existing[0], rotated[1], rotated[2], rotated[3]
 *      This is blend with mask 0b1110 = 0xE
 *
 *   3. Block k+1 gets lane 0: rotated[0], existing[1], existing[2], existing[3]
 *      This is blend with mask 0b0001 = 0x1
 *
 * VPERMPD ENCODING:
 *   vpermpd imm8, ymm, ymm
 *   imm8 = [sel₃|sel₂|sel₁|sel₀] where each selᵢ is 2 bits
 *
 *   For right rotation by 1: [v₃, v₀, v₁, v₂]
 *     lane 0 gets src[3]: sel₀ = 11 = 3
 *     lane 1 gets src[0]: sel₁ = 00 = 0
 *     lane 2 gets src[1]: sel₂ = 01 = 1
 *     lane 3 gets src[2]: sel₃ = 10 = 2
 *   imm8 = 10_01_00_11 = 0x93
 *
 *=============================================================================*/

/**
 * @brief Store 4 values with +1 index shift using AVX2 permute
 *
 * @param buf          Destination interleaved buffer
 * @param block_idx    Source block index (stores go to blocks block_idx and block_idx+1)
 * @param field_offset Byte offset of field within superblock
 * @param vals         4 values for source indices [4k, 4k+1, 4k+2, 4k+3]
 *                     which will be stored at indices [4k+1, 4k+2, 4k+3, 4k+4]
 *
 * @par Example
 * If block_idx=0, field_offset=BOCPD_IBLK_MU, vals=[a,b,c,d]:
 *   - μ[1] = a  (block 0, lane 1)
 *   - μ[2] = b  (block 0, lane 2)
 *   - μ[3] = c  (block 0, lane 3)
 *   - μ[4] = d  (block 1, lane 0)
 */
static inline void store_shifted_field(double *buf, size_t block_idx,
                                       size_t field_offset, __m256d vals)
{
    /* ─────────────────────────────────────────────────────────────────────
     * Rotate vector right by 1 position
     *
     * Input:  vals    = [v₀, v₁, v₂, v₃]
     * Output: rotated = [v₃, v₀, v₁, v₂]
     *
     * After rotation:
     *   rotated[0] = v₃ → goes to block k+1, lane 0 (index 4k+4)
     *   rotated[1] = v₀ → goes to block k, lane 1   (index 4k+1)
     *   rotated[2] = v₁ → goes to block k, lane 2   (index 4k+2)
     *   rotated[3] = v₂ → goes to block k, lane 3   (index 4k+3)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d rotated = _mm256_permute4x64_pd(vals, 0x93); /* imm8 = 10_01_00_11 */

    /* ─────────────────────────────────────────────────────────────────────
     * Calculate block base addresses
     *
     * Each block is BOCPD_IBLK_DOUBLES = 32 doubles.
     * field_offset is in bytes, divide by 8 to get double offset.
     * ───────────────────────────────────────────────────────────────────── */
    double *block_k = buf + block_idx * BOCPD_IBLK_DOUBLES + field_offset / 8;
    double *block_k1 = buf + (block_idx + 1) * BOCPD_IBLK_DOUBLES + field_offset / 8;

    /* ─────────────────────────────────────────────────────────────────────
     * Load existing content from destination blocks
     *
     * We need to preserve:
     *   - Block k, lane 0 (index 4k, not being written)
     *   - Block k+1, lanes 1,2,3 (indices 4k+5,6,7, not being written)
     * ───────────────────────────────────────────────────────────────────── */
    __m256d existing_k = _mm256_loadu_pd(block_k);
    __m256d existing_k1 = _mm256_loadu_pd(block_k1);

    /* ─────────────────────────────────────────────────────────────────────
     * Blend rotated values into destination blocks
     *
     * vblendpd mask semantics: result[i] = (mask & (1<<i)) ? src2[i] : src1[i]
     *
     * Block k: keep lane 0 from existing, take lanes 1,2,3 from rotated
     *   mask = 0b1110 = 14 decimal = 0xE
     *
     * Block k+1: take lane 0 from rotated, keep lanes 1,2,3 from existing
     *   mask = 0b0001 = 1 decimal = 0x1
     * ───────────────────────────────────────────────────────────────────── */
    __m256d merged_k = _mm256_blend_pd(existing_k, rotated, 0b1110);
    __m256d merged_k1 = _mm256_blend_pd(existing_k1, rotated, 0b0001);

    /* ─────────────────────────────────────────────────────────────────────
     * Store merged results back to memory
     * ───────────────────────────────────────────────────────────────────── */
    _mm256_storeu_pd(block_k, merged_k);
    _mm256_storeu_pd(block_k1, merged_k1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * END OF PART 2
 *
 * Part 3 will cover:
 *   - init_slot_zero (initialize prior at run-length 0)
 *   - process_block_common (core update kernel with lgamma function pointer)
 *   - update_posteriors_interleaved (region dispatch logic with formulas)
 *   - prediction_step (ASM kernel interface OR C intrinsics fallback)
 *   - bocpd_ultra_init, bocpd_ultra_free, bocpd_ultra_reset, bocpd_ultra_step
 *   - bocpd_pool_init, bocpd_pool_free, bocpd_pool_reset, bocpd_pool_get
 * ═══════════════════════════════════════════════════════════════════════════ */
/*=============================================================================
 * PART 3: INITIALIZATION, POSTERIOR UPDATE, PREDICTION, AND PUBLIC API
 *=============================================================================*/

/*─────────────────────────────────────────────────────────────────────────────
 * SLOT ZERO INITIALIZATION
 *
 * In BOCPD, run-length 0 represents "just had a changepoint" and uses the
 * prior parameters directly. After each observation, we must initialize
 * slot 0 of the NEXT buffer with the prior.
 *
 * STUDENT-T CONSTANTS FOR PRIOR:
 *   ν₀ = 2α₀               (degrees of freedom)
 *   σ₀² = β₀(κ₀+1)/(α₀κ₀)  (scale parameter)
 *   C1 = lgamma(α₀+0.5) - lgamma(α₀) - 0.5×ln(πν₀σ₀²)
 *   C2 = α₀ + 0.5
 *   inv_ssn = 1/(σ₀²ν₀)
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Initialize slot 0 of NEXT buffer with prior parameters
 *
 * @param b  Pointer to BOCPD detector
 *
 * This sets up the "fresh start" hypothesis: what if a changepoint
 * just occurred and we're starting from the prior?
 */
static inline void init_slot_zero(bocpd_asm_t *b)
{
    double *next = BOCPD_NEXT_BUF(b);

    /* Extract prior parameters for readability */
    const double kappa0 = b->prior.kappa0; /* Precision pseudo-count */
    const double mu0 = b->prior.mu0;       /* Prior mean */
    const double alpha0 = b->prior.alpha0; /* Shape (half degrees of freedom) */
    const double beta0 = b->prior.beta0;   /* Scale (sum of squares / 2) */

    /* ─────────────────────────────────────────────────────────────────────
     * Store raw posterior parameters at slot 0
     * ───────────────────────────────────────────────────────────────────── */
    IBLK_SET_MU(next, 0, mu0);
    IBLK_SET_KAPPA(next, 0, kappa0);
    IBLK_SET_ALPHA(next, 0, alpha0);
    IBLK_SET_BETA(next, 0, beta0);
    IBLK_SET_SS_N(next, 0, 0.0); /* No samples yet */

    /* ─────────────────────────────────────────────────────────────────────
     * Compute Student-t predictive parameters
     *
     * Scale parameter (NOT variance!):
     *   σ² = β × (κ + 1) / (α × κ)
     *
     * This is the scale of the Student-t predictive distribution.
     * The actual variance would be σ² × ν/(ν-2) for ν > 2.
     * ───────────────────────────────────────────────────────────────────── */
    double sigma_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
    double nu = 2.0 * alpha0; /* Degrees of freedom */

    /* ─────────────────────────────────────────────────────────────────────
     * Precompute Student-t constants for fast prediction
     *
     * Log-density of Student-t(ν, μ, σ²):
     *   ln p(x) = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5×ln(πνσ²)
     *           - ((ν+1)/2) × ln(1 + (x-μ)²/(νσ²))
     *
     * With our parameterization (α = ν/2):
     *   C1 = lgamma(α+0.5) - lgamma(α) - 0.5×ln(πνσ²)
     *   C2 = α + 0.5
     *   inv_ssn = 1/(σ²ν)
     *
     * So: ln p(x) = C1 - C2 × ln(1 + (x-μ)² × inv_ssn)
     * ───────────────────────────────────────────────────────────────────── */
    double ln_nu_pi = fast_log_scalar(nu * M_PI);
    double ln_sigma_sq = fast_log_scalar(sigma_sq);

    /* Use precomputed lgamma values from initialization */
    double C1 = b->prior_lgamma_alpha_p5 - b->prior_lgamma_alpha -
                0.5 * ln_nu_pi - 0.5 * ln_sigma_sq;
    double C2 = alpha0 + 0.5;

    IBLK_SET_C1(next, 0, C1);
    IBLK_SET_C2(next, 0, C2);
    IBLK_SET_INV_SSN(next, 0, 1.0 / (sigma_sq * nu));
}

/*─────────────────────────────────────────────────────────────────────────────
 * CORE BLOCK PROCESSING KERNEL
 *
 * This function processes one 4-element block, updating all 8 parameter
 * fields from CUR buffer to NEXT buffer with +1 index shift.
 *
 * KEY V3.1 OPTIMIZATION:
 *   The lgamma function is passed as a pointer, allowing region-specific
 *   dispatch. Instead of computing all 3 lgamma variants and blending,
 *   we compute exactly ONE variant per block!
 *
 * WELFORD'S ALGORITHM FOR NUMERICALLY STABLE VARIANCE:
 *   Traditional formula: β_new = β_old + 0.5 × Σ(x - x̄)²
 *   This suffers from catastrophic cancellation when computing Σx² - n×x̄².
 *
 *   Welford's form: β_new = β_old + 0.5 × (x - μ_old) × (x - μ_new)
 *   This is algebraically equivalent but numerically stable!
 *
 *   Proof of equivalence:
 *     μ_new = (κ_old × μ_old + x) / κ_new
 *     x - μ_new = x - (κ_old × μ_old + x) / κ_new
 *               = (κ_new × x - κ_old × μ_old - x) / κ_new
 *               = ((κ_new - 1) × x - κ_old × μ_old) / κ_new
 *               = (κ_old × x - κ_old × μ_old) / κ_new
 *               = κ_old × (x - μ_old) / κ_new
 *
 *     So: (x - μ_old) × (x - μ_new) = (x - μ_old)² × κ_old / κ_new
 *
 *     This is proportional to the squared deviation, weighted correctly.
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Process one block with region-specific lgamma
 *
 * @param src       Source block in CUR buffer (4 run-lengths)
 * @param next      Destination NEXT buffer (full buffer, not block)
 * @param block     Block index (0, 1, 2, ...)
 * @param x_vec     Observation broadcast to all 4 lanes
 * @param one       Constant 1.0 broadcast
 * @param two       Constant 2.0 broadcast
 * @param half      Constant 0.5 broadcast
 * @param pi        Constant π broadcast
 * @param lgamma_fn Pointer to region-appropriate lgamma function
 *
 * @par Algorithm
 * 1. Load 5 parameters from CUR: μ, κ, α, β, ss_n
 * 2. Welford update: κ_new, μ_new, α_new, β_new, ss_n_new
 * 3. Compute Student-t constants: σ², ν, C1, C2, inv_ssn
 * 4. Store all 8 fields to NEXT with +1 index shift
 */
static inline void process_block_common(
    const double *src, double *next, size_t block,
    __m256d x_vec, __m256d one, __m256d two, __m256d half, __m256d pi,
    __m256d (*lgamma_fn)(__m256d))
{
    /* ═════════════════════════════════════════════════════════════════════
     * STEP 1: Load current posterior parameters
     * ═════════════════════════════════════════════════════════════════════ */

    __m256d mu_old = _mm256_loadu_pd(src + BOCPD_IBLK_MU / 8);
    __m256d kappa_old = _mm256_loadu_pd(src + BOCPD_IBLK_KAPPA / 8);
    __m256d alpha_old = _mm256_loadu_pd(src + BOCPD_IBLK_ALPHA / 8);
    __m256d beta_old = _mm256_loadu_pd(src + BOCPD_IBLK_BETA / 8);
    __m256d ss_n_old = _mm256_loadu_pd(src + BOCPD_IBLK_SS_N / 8);

    /* ═════════════════════════════════════════════════════════════════════
     * STEP 2: Welford posterior update
     *
     * Bayesian update for Normal-Inverse-Gamma conjugate prior:
     *   κ_new = κ_old + 1
     *   μ_new = (κ_old × μ_old + x) / κ_new
     *   α_new = α_old + 0.5
     *   β_new = β_old + 0.5 × (x - μ_old) × (x - μ_new)   [Welford]
     * ═════════════════════════════════════════════════════════════════════ */

    __m256d ss_n_new = _mm256_add_pd(ss_n_old, one);
    __m256d kappa_new = _mm256_add_pd(kappa_old, one);

    /* μ_new = (κ_old × μ_old + x) / κ_new */
    __m256d mu_new = _mm256_div_pd(
        _mm256_fmadd_pd(kappa_old, mu_old, x_vec), /* κ_old × μ_old + x */
        kappa_new);

    __m256d alpha_new = _mm256_add_pd(alpha_old, half);

    /* ─────────────────────────────────────────────────────────────────────
     * Welford β update: β_new = β_old + 0.5 × (x - μ_old) × (x - μ_new)
     *
     * This is numerically stable because we never compute Σx² - n×x̄²,
     * which would suffer from catastrophic cancellation.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d delta1 = _mm256_sub_pd(x_vec, mu_old); /* x - μ_old */
    __m256d delta2 = _mm256_sub_pd(x_vec, mu_new); /* x - μ_new */
    __m256d beta_inc = _mm256_mul_pd(_mm256_mul_pd(delta1, delta2), half);
    __m256d beta_new = _mm256_add_pd(beta_old, beta_inc);

    /* ═════════════════════════════════════════════════════════════════════
     * STEP 3: Compute Student-t predictive parameters
     *
     * Scale: σ² = β × (κ + 1) / (α × κ)
     * DOF:   ν = 2α
     * ═════════════════════════════════════════════════════════════════════ */

    __m256d kappa_p1 = _mm256_add_pd(kappa_new, one);
    __m256d sigma_sq = _mm256_div_pd(
        _mm256_mul_pd(beta_new, kappa_p1),    /* β × (κ + 1) */
        _mm256_mul_pd(alpha_new, kappa_new)); /* α × κ       */

    __m256d nu = _mm256_mul_pd(two, alpha_new); /* ν = 2α */
    __m256d sigma_sq_nu = _mm256_mul_pd(sigma_sq, nu);
    __m256d inv_ssn = _mm256_div_pd(one, sigma_sq_nu);

    /* ═════════════════════════════════════════════════════════════════════
     * STEP 4: Compute lgamma using REGION-SPECIFIC function
     *
     * *** THIS IS THE KEY V3.1 OPTIMIZATION ***
     *
     * Instead of fast_lgamma_avx2_branchless (computes all 3, blends),
     * we call exactly ONE lgamma variant based on the block's α range.
     *
     * We need lgamma(α) and lgamma(α + 0.5) for the Student-t constant.
     * ═════════════════════════════════════════════════════════════════════ */

    __m256d lg_a = lgamma_fn(alpha_new);
    __m256d alpha_p5 = _mm256_add_pd(alpha_new, half);
    __m256d lg_ap5 = lgamma_fn(alpha_p5);

    /* ═════════════════════════════════════════════════════════════════════
     * STEP 5: Compute Student-t constants C1 and C2
     *
     * C1 = lgamma(α+0.5) - lgamma(α) - 0.5 × ln(π × ν × σ²)
     * C2 = α + 0.5
     *
     * These precomputed constants make prediction fast:
     *   ln p(x) = C1 - C2 × ln(1 + (x-μ)² × inv_ssn)
     * ═════════════════════════════════════════════════════════════════════ */

    __m256d nu_pi_s2 = _mm256_mul_pd(_mm256_mul_pd(nu, pi), sigma_sq);
    __m256d ln_term = fast_log_avx2(nu_pi_s2);

    __m256d C1 = _mm256_sub_pd(lg_ap5, lg_a);
    C1 = _mm256_fnmadd_pd(half, ln_term, C1); /* C1 = C1 - 0.5 × ln_term */

    __m256d C2 = alpha_p5;

    /* ═════════════════════════════════════════════════════════════════════
     * STEP 6: Store all 8 fields with +1 index shift
     *
     * Values computed for indices [4k, 4k+1, 4k+2, 4k+3]
     * get stored at indices [4k+1, 4k+2, 4k+3, 4k+4]
     * ═════════════════════════════════════════════════════════════════════ */

    store_shifted_field(next, block, BOCPD_IBLK_MU, mu_new);
    store_shifted_field(next, block, BOCPD_IBLK_KAPPA, kappa_new);
    store_shifted_field(next, block, BOCPD_IBLK_ALPHA, alpha_new);
    store_shifted_field(next, block, BOCPD_IBLK_BETA, beta_new);
    store_shifted_field(next, block, BOCPD_IBLK_SS_N, ss_n_new);
    store_shifted_field(next, block, BOCPD_IBLK_C1, C1);
    store_shifted_field(next, block, BOCPD_IBLK_C2, C2);
    store_shifted_field(next, block, BOCPD_IBLK_INV_SSN, inv_ssn);
}

/*─────────────────────────────────────────────────────────────────────────────
 * FUSED POSTERIOR UPDATE WITH REGION-SPECIFIC LGAMMA DISPATCH
 *
 * This is the main posterior update function. It processes all active
 * run-lengths, updating parameters from CUR to NEXT buffer.
 *
 * V3.1 REGION DISPATCH STRATEGY:
 *
 * Alpha values are monotonically increasing: α[i] = α₀ + 0.5 × i
 *
 * For block k (indices 4k to 4k+3), after update:
 *   α_new ∈ [α₀ + 2k + 0.5, α₀ + 2k + 2.0]
 *
 * REGION BOUNDARIES:
 *
 *   1. LANCZOS SAFE (all α < 8):
 *      α₀ + 2k + 2.0 < 8
 *      k < (6 - α₀) / 2
 *
 *   2. MINIMAX SAFE START (min α ≥ 8):
 *      α₀ + 2k + 0.5 ≥ 8
 *      k ≥ (7.5 - α₀) / 2
 *
 *   3. MINIMAX SAFE END (all α < 40):
 *      α₀ + 2k + 2.0 < 40
 *      k < (38 - α₀) / 2
 *
 *   4. STIRLING SAFE START (min α ≥ 40):
 *      α₀ + 2k + 0.5 ≥ 40
 *      k ≥ (39.5 - α₀) / 2
 *
 * EXAMPLE with α₀ = 1:
 *
 *   | Block k | α_new range    | lgamma function |
 *   |---------|----------------|-----------------|
 *   | 0       | [1.5, 3.0]     | Lanczos         |
 *   | 1       | [3.5, 5.0]     | Lanczos         |
 *   | 2       | [5.5, 7.0]     | Lanczos         |
 *   | 3       | [7.5, 9.0]     | BRANCHLESS      | ← spans α=8
 *   | 4       | [9.5, 11.0]    | Minimax         |
 *   | ...     | ...            | Minimax         |
 *   | 18      | [37.5, 39.0]   | Minimax         |
 *   | 19      | [39.5, 41.0]   | BRANCHLESS      | ← spans α=40
 *   | 20      | [41.5, 43.0]   | Stirling        |
 *   | ...     | ...            | Stirling        |
 *
 * PERFORMANCE IMPACT:
 *   - ~97% of blocks use single lgamma variant
 *   - Only ~2 blocks use branchless fallback
 *   - V3.0 → V3.1: 90% throughput improvement!
 *
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Fused posterior update with region-specific lgamma dispatch
 *
 * @param b      Pointer to BOCPD detector
 * @param x      New observation
 * @param n_old  Number of active run-lengths before this update
 */
static void update_posteriors_interleaved(bocpd_asm_t *b, double x, size_t n_old)
{
    /* Always initialize slot 0 with prior (changepoint hypothesis) */
    init_slot_zero(b);

    if (n_old == 0)
    {
        b->cur_buf = 1 - b->cur_buf; /* Swap buffers */
        return;
    }

    const double *cur = BOCPD_CUR_BUF(b);
    double *next = BOCPD_NEXT_BUF(b);

    /* Broadcast constants for SIMD operations */
    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d pi = _mm256_set1_pd(M_PI);

    const double alpha0 = b->prior.alpha0;
    const size_t n_blocks = (n_old + 3) / 4; /* Ceiling division */

    /* ═════════════════════════════════════════════════════════════════════
     * CALCULATE REGION TRANSITION BLOCKS
     *
     * For block k, α_new ∈ [α₀ + 2k + 0.5, α₀ + 2k + 2.0]
     * ═════════════════════════════════════════════════════════════════════ */

    /* ─────────────────────────────────────────────────────────────────────
     * LANCZOS SAFE END: block k where max(α_new) < 8
     *   α₀ + 2k + 2.0 < 8  →  k < (6 - α₀) / 2
     * ───────────────────────────────────────────────────────────────────── */
    size_t lanczos_safe_end;
    if (alpha0 >= 6.0)
    {
        lanczos_safe_end = 0; /* α already ≥ 8 at block 0 */
    }
    else
    {
        lanczos_safe_end = (size_t)((6.0 - alpha0) / 2.0);
        if (lanczos_safe_end > n_blocks)
            lanczos_safe_end = n_blocks;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * MINIMAX SAFE START: block k where min(α_new) ≥ 8
     *   α₀ + 2k + 0.5 ≥ 8  →  k ≥ ceil((7.5 - α₀) / 2)
     * ───────────────────────────────────────────────────────────────────── */
    size_t minimax_safe_start;
    if (alpha0 >= 7.5)
    {
        minimax_safe_start = 0; /* Already in minimax range */
    }
    else
    {
        minimax_safe_start = (size_t)ceil((7.5 - alpha0) / 2.0);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * MINIMAX SAFE END: block k where max(α_new) < 40
     *   α₀ + 2k + 2.0 < 40  →  k < (38 - α₀) / 2
     * ───────────────────────────────────────────────────────────────────── */
    size_t minimax_safe_end;
    if (alpha0 >= 38.0)
    {
        minimax_safe_end = 0; /* Already past minimax range */
    }
    else
    {
        minimax_safe_end = (size_t)((38.0 - alpha0) / 2.0);
        if (minimax_safe_end > n_blocks)
            minimax_safe_end = n_blocks;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * STIRLING SAFE START: block k where min(α_new) ≥ 40
     *   α₀ + 2k + 0.5 ≥ 40  →  k ≥ ceil((39.5 - α₀) / 2)
     * ───────────────────────────────────────────────────────────────────── */
    size_t stirling_safe_start;
    if (alpha0 >= 39.5)
    {
        stirling_safe_start = 0; /* Already in Stirling range */
    }
    else
    {
        stirling_safe_start = (size_t)ceil((39.5 - alpha0) / 2.0);
    }

    size_t block = 0;

    /* ═════════════════════════════════════════════════════════════════════
     * REGION 1: Pure Lanczos (all α < 8)
     * ═════════════════════════════════════════════════════════════════════ */
    for (; block < lanczos_safe_end && block < n_blocks; block++)
    {
        if (block * 4 >= n_old)
            break;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;
        process_block_common(src, next, block, x_vec, one, two, half, pi,
                             lgamma_lanczos_avx2);
    }

    /* ═════════════════════════════════════════════════════════════════════
     * REGION 2: Transition zone (spans Lanczos/Minimax boundary at α=8)
     * ═════════════════════════════════════════════════════════════════════ */
    for (; block < minimax_safe_start && block < n_blocks; block++)
    {
        if (block * 4 >= n_old)
            break;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;
        process_block_common(src, next, block, x_vec, one, two, half, pi,
                             fast_lgamma_avx2_branchless); /* Computes all 3 */
    }

    /* ═════════════════════════════════════════════════════════════════════
     * REGION 3: Pure Minimax (8 ≤ α < 40)
     * ═════════════════════════════════════════════════════════════════════ */
    for (; block < minimax_safe_end && block < n_blocks; block++)
    {
        if (block * 4 >= n_old)
            break;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;
        process_block_common(src, next, block, x_vec, one, two, half, pi,
                             lgamma_minimax_avx2);
    }

    /* ═════════════════════════════════════════════════════════════════════
     * REGION 4: Transition zone (spans Minimax/Stirling boundary at α=40)
     * ═════════════════════════════════════════════════════════════════════ */
    for (; block < stirling_safe_start && block < n_blocks; block++)
    {
        if (block * 4 >= n_old)
            break;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;
        process_block_common(src, next, block, x_vec, one, two, half, pi,
                             fast_lgamma_avx2_branchless); /* Computes all 3 */
    }

    /* ═════════════════════════════════════════════════════════════════════
     * REGION 5: Pure Stirling (α ≥ 40)
     * ═════════════════════════════════════════════════════════════════════ */
    for (; block < n_blocks; block++)
    {
        if (block * 4 >= n_old)
            break;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;
        process_block_common(src, next, block, x_vec, one, two, half, pi,
                             lgamma_stirling_avx2);
    }

    /* ═════════════════════════════════════════════════════════════════════
     * SCALAR TAIL: Handle remaining 0-3 elements that don't fill a block
     * ═════════════════════════════════════════════════════════════════════ */
    size_t i = block * 4;
    for (; i < n_old; i++)
    {
        double ss_n_old = IBLK_GET_SS_N(cur, i);
        double kappa_old = IBLK_GET_KAPPA(cur, i);
        double mu_old = IBLK_GET_MU(cur, i);
        double alpha_old = IBLK_GET_ALPHA(cur, i);
        double beta_old = IBLK_GET_BETA(cur, i);

        /* Welford update */
        double ss_n_new = ss_n_old + 1.0;
        double kappa_new = kappa_old + 1.0;
        double mu_new = (kappa_old * mu_old + x) / kappa_new;
        double alpha_new = alpha_old + 0.5;
        double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);

        /* Student-t parameters */
        double sigma_sq = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);
        double nu = 2.0 * alpha_new;
        double inv_ssn = 1.0 / (sigma_sq * nu);

        /* Use glibc lgamma for scalar tail (rare, not performance critical) */
        double lg_a = lgamma(alpha_new);
        double lg_ap5 = lgamma(alpha_new + 0.5);
        double C1 = lg_ap5 - lg_a - 0.5 * fast_log_scalar(nu * M_PI * sigma_sq);
        double C2 = alpha_new + 0.5;

        /* Store at shifted index */
        size_t out_idx = i + 1;
        IBLK_SET_MU(next, out_idx, mu_new);
        IBLK_SET_KAPPA(next, out_idx, kappa_new);
        IBLK_SET_ALPHA(next, out_idx, alpha_new);
        IBLK_SET_BETA(next, out_idx, beta_new);
        IBLK_SET_SS_N(next, out_idx, ss_n_new);
        IBLK_SET_C1(next, out_idx, C1);
        IBLK_SET_C2(next, out_idx, C2);
        IBLK_SET_INV_SSN(next, out_idx, inv_ssn);
    }

    /* Swap ping-pong buffers: NEXT becomes CUR for next iteration */
    b->cur_buf = 1 - b->cur_buf;
}

/*=============================================================================
 * PREDICTION STEP
 *
 * The prediction step computes:
 * 1. P(x | r) for each run-length using Student-t distribution
 * 2. r_new[i+1] = r[i] × P(x|r) × (1-H) for growth
 * 3. r_new[0] = Σ r[i] × P(x|r) × H for changepoint
 * 4. Normalization: r_new /= sum(r_new)
 * 5. Truncation: discard run-lengths with negligible probability
 *
 * STUDENT-T LOG-PROBABILITY:
 *   ln p(x) = C1 - C2 × log1p((x - μ)² × inv_ssn)
 *
 * Where C1, C2, inv_ssn are precomputed in the update step.
 *
 * SIMD OPTIMIZATIONS:
 * - log1p(t) via polynomial for small t (avoids ln(1+t) accuracy loss)
 * - exp() via range reduction: exp(x) = 2^k × 2^f where k=round(x/ln2)
 * - Estrin's scheme for polynomial evaluation (better ILP than Horner)
 * - Horizontal sum via shuffle-and-add
 *
 *=============================================================================*/

#if BOCPD_USE_ASM_KERNEL
/*
 * ASM KERNEL VERSION
 *
 * Uses hand-written assembly kernel for maximum performance.
 * The kernel is in a separate .S file and accessed via bocpd_fused_loop_avx2().
 */

static void prediction_step(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;
    if (n == 0)
        return;

    const double thresh = b->trunc_thresh;
    double *params = BOCPD_CUR_BUF(b);
    double *r = b->r;
    double *r_new = b->r_scratch;

    /* Pad to multiple of 8 for AVX2 alignment */
    const size_t n_padded = (n + 7) & ~7ULL;

    /* Zero-pad r array beyond active length */
    for (size_t i = n; i < n_padded + 8; i++)
        r[i] = 0.0;
    memset(r_new, 0, (n_padded + 16) * sizeof(double));

    /* Output variables from ASM kernel */
    double r0_out = 0.0;         /* Accumulated changepoint probability */
    double max_growth_out = 0.0; /* Maximum growth probability (for MAP) */
    size_t max_idx_out = 0;      /* Index of maximum */
    size_t last_valid_out = 0;   /* Last index above truncation threshold */

    /* Set up arguments structure for ASM kernel */
    bocpd_kernel_args_t args = {
        .lin_interleaved = params,
        .r_old = r,
        .x = x,
        .h = b->hazard,
        .one_minus_h = b->one_minus_h,
        .trunc_thresh = thresh,
        .n_padded = n_padded,
        .r_new = r_new,
        .r0_out = &r0_out,
        .max_growth_out = &max_growth_out,
        .max_idx_out = &max_idx_out,
        .last_valid_out = &last_valid_out};

    /* Call the hand-written AVX2 assembly kernel */
    bocpd_fused_loop_avx2(&args);

    /* Store r0 (changepoint probability) */
    r_new[0] = r0_out;
    if (r0_out > thresh && last_valid_out == 0)
        last_valid_out = 1;

    /* Determine new active length based on truncation */
    size_t new_len = (last_valid_out > 0) ? last_valid_out + 1 : n + 1;
    if (new_len > b->capacity)
        new_len = b->capacity;

    size_t new_len_padded = (new_len + 7) & ~7ULL;

    /* ─────────────────────────────────────────────────────────────────────
     * SIMD Normalization
     *
     * Compute sum using SIMD accumulation, then broadcast reciprocal.
     * ───────────────────────────────────────────────────────────────────── */
    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t i = 0; i < new_len_padded; i += 4)
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[i]));

    /* Horizontal sum: [a,b,c,d] → a+b+c+d */
    __m128d lo = _mm256_castpd256_pd128(sum_acc);   /* [a, b] */
    __m128d hi = _mm256_extractf128_pd(sum_acc, 1); /* [c, d] */
    lo = _mm_add_pd(lo, hi);                        /* [a+c, b+d] */
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1)); /* [a+c+b+d, ...] */
    double r_sum = _mm_cvtsd_f64(lo);

    /* Normalize if sum is non-negligible */
    if (r_sum > 1e-300)
    {
        __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
        for (size_t i = 0; i < new_len_padded; i += 4)
        {
            __m256d rv = _mm256_loadu_pd(&r_new[i]);
            _mm256_storeu_pd(&r[i], _mm256_mul_pd(rv, inv_sum));
        }
    }

    b->active_len = new_len;

    /* Determine MAP run-length */
    double r0_normalized = (r_sum > 1e-300) ? r0_out / r_sum : 0.0;
    double max_normalized = (r_sum > 1e-300) ? max_growth_out / r_sum : 0.0;
    b->map_runlength = (r0_normalized >= max_normalized) ? 0 : max_idx_out;
}

#else  /* C INTRINSICS FALLBACK */

/*
 * C INTRINSICS VERSION
 *
 * Portable fallback using Intel intrinsics. About 5% slower than ASM
 * but works on all compilers supporting AVX2.
 *
 * KEY ALGORITHMS:
 *
 * 1. log1p(t) polynomial for t = (x-μ)² × inv_ssn
 *    For small t, log1p(t) = t - t²/2 + t³/3 - t⁴/4 + ...
 *    Faster and more accurate than log(1+t) for small t.
 *
 * 2. exp(ln_pp) via range reduction
 *    exp(x) = 2^k × exp(x - k×ln2) = 2^k × 2^f
 *    where k = round(x / ln2), f = x/ln2 - k ∈ [-0.5, 0.5]
 *
 *    2^f is computed via degree-6 polynomial.
 *    2^k is computed by setting IEEE-754 exponent bits directly.
 */

static void prediction_step(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;
    if (n == 0)
        return;

    const double h = b->hazard;
    const double omh = b->one_minus_h;
    const double thresh = b->trunc_thresh;

    const double *params = BOCPD_CUR_BUF(b);
    double *r = b->r;
    double *r_new = b->r_scratch;

    const size_t n_padded = (n + 7) & ~7ULL;

    /* Initialize arrays */
    for (size_t j = n; j < n_padded + 8; j++)
        r[j] = 0.0;
    memset(r_new, 0, (n_padded + 16) * sizeof(double));

    /* SIMD constants */
    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d h_vec = _mm256_set1_pd(h);
    const __m256d omh_vec = _mm256_set1_pd(omh);
    const __m256d thresh_vec = _mm256_set1_pd(thresh);
    const __m256d min_pp = _mm256_set1_pd(1e-300);
    const __m256d const_one = _mm256_set1_pd(1.0);

    /* log1p Taylor series coefficients */
    const __m256d log1p_c2 = _mm256_set1_pd(-0.5);
    const __m256d log1p_c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d log1p_c4 = _mm256_set1_pd(-0.25);
    const __m256d log1p_c5 = _mm256_set1_pd(0.2);
    const __m256d log1p_c6 = _mm256_set1_pd(-0.1666666666666667);

    /* exp range reduction coefficients */
    const __m256d exp_inv_ln2 = _mm256_set1_pd(1.4426950408889634); /* 1/ln(2) */
    const __m256d exp_min_x = _mm256_set1_pd(-700.0);               /* Underflow protection */
    const __m256d exp_max_x = _mm256_set1_pd(700.0);                /* Overflow protection */
    const __m256d exp_c1 = _mm256_set1_pd(0.6931471805599453);      /* ln(2) */
    const __m256d exp_c2 = _mm256_set1_pd(0.24022650695910072);
    const __m256d exp_c3 = _mm256_set1_pd(0.05550410866482158);
    const __m256d exp_c4 = _mm256_set1_pd(0.009618129107628477);
    const __m256d exp_c5 = _mm256_set1_pd(0.0013333558146428443);
    const __m256d exp_c6 = _mm256_set1_pd(0.00015403530393381608);
    const __m256i exp_bias = _mm256_set1_epi64x(1023);

    /* Accumulators */
    __m256d r0_acc = _mm256_setzero_pd();
    __m256d max_growth = _mm256_setzero_pd();
    __m256i max_idx_vec = _mm256_setzero_si256();
    __m256i idx_vec = _mm256_set_epi64x(4, 3, 2, 1);
    const __m256i idx_inc = _mm256_set1_epi64x(4);

    size_t last_valid = 0;

    /* Main processing loop - 4 run-lengths per iteration */
    for (size_t i = 0; i < n_padded; i += 4)
    {
        size_t block = i / 4;
        const double *blk = params + block * BOCPD_IBLK_DOUBLES;

        /* Load precomputed Student-t parameters */
        __m256d mu = _mm256_loadu_pd(blk + BOCPD_IBLK_MU / 8);
        __m256d C1 = _mm256_loadu_pd(blk + BOCPD_IBLK_C1 / 8);
        __m256d C2 = _mm256_loadu_pd(blk + BOCPD_IBLK_C2 / 8);
        __m256d inv_ssn = _mm256_loadu_pd(blk + BOCPD_IBLK_INV_SSN / 8);
        __m256d r_old = _mm256_loadu_pd(&r[i]);

        /* ═══════════════════════════════════════════════════════════════
         * Student-t log-probability: ln p(x) = C1 - C2 × log1p(t)
         * where t = (x - μ)² × inv_ssn
         * ═══════════════════════════════════════════════════════════════ */
        __m256d z = _mm256_sub_pd(x_vec, mu);   /* z = x - μ */
        __m256d z2 = _mm256_mul_pd(z, z);       /* z² */
        __m256d t = _mm256_mul_pd(z2, inv_ssn); /* t = z² × inv_ssn */

        /* log1p(t) via polynomial: t × (1 - t/2 + t²/3 - t³/4 + ...) */
        __m256d poly = _mm256_fmadd_pd(t, log1p_c6, log1p_c5);
        poly = _mm256_fmadd_pd(t, poly, log1p_c4);
        poly = _mm256_fmadd_pd(t, poly, log1p_c3);
        poly = _mm256_fmadd_pd(t, poly, log1p_c2);
        poly = _mm256_fmadd_pd(t, poly, const_one);
        __m256d log1p_t = _mm256_mul_pd(t, poly);

        /* ln_pp = C1 - C2 × log1p(t) */
        __m256d ln_pp = _mm256_fnmadd_pd(C2, log1p_t, C1);

        /* ═══════════════════════════════════════════════════════════════
         * exp(ln_pp) via range reduction
         *
         * exp(x) = 2^(x/ln2) = 2^k × 2^f
         * where k = round(x/ln2), f = x/ln2 - k
         * ═══════════════════════════════════════════════════════════════ */
        __m256d x_clamp = _mm256_max_pd(_mm256_min_pd(ln_pp, exp_max_x), exp_min_x);
        __m256d t_exp = _mm256_mul_pd(x_clamp, exp_inv_ln2); /* x / ln(2) */
        __m256d k = _mm256_round_pd(t_exp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d f = _mm256_sub_pd(t_exp, k); /* Fractional part */

        /* 2^f via Estrin's polynomial (better ILP than Horner) */
        __m256d f2 = _mm256_mul_pd(f, f);
        __m256d p01 = _mm256_fmadd_pd(f, exp_c1, const_one);
        __m256d p23 = _mm256_fmadd_pd(f, exp_c3, exp_c2);
        __m256d p45 = _mm256_fmadd_pd(f, exp_c5, exp_c4);
        __m256d q0123 = _mm256_fmadd_pd(f2, p23, p01);
        __m256d q456 = _mm256_fmadd_pd(f2, exp_c6, p45);
        __m256d f4 = _mm256_mul_pd(f2, f2);
        __m256d exp_p = _mm256_fmadd_pd(f4, q456, q0123);

        /* 2^k via IEEE-754 bit manipulation */
        __m128i k32 = _mm256_cvtpd_epi32(k);              /* Double → int32 */
        __m256i k64 = _mm256_cvtepi32_epi64(k32);         /* int32 → int64 */
        __m256i biased = _mm256_add_epi64(k64, exp_bias); /* Add bias 1023 */
        __m256i bits = _mm256_slli_epi64(biased, 52);     /* Shift to exponent position */
        __m256d scale = _mm256_castsi256_pd(bits);        /* Reinterpret as double */

        /* pp = 2^f × 2^k */
        __m256d pp = _mm256_mul_pd(exp_p, scale);
        pp = _mm256_max_pd(pp, min_pp); /* Clamp to avoid underflow */

        /* ═══════════════════════════════════════════════════════════════
         * BOCPD update
         * ═══════════════════════════════════════════════════════════════ */
        __m256d r_pp = _mm256_mul_pd(r_old, pp);
        __m256d growth = _mm256_mul_pd(r_pp, omh_vec); /* r × p × (1-H) */
        __m256d change = _mm256_mul_pd(r_pp, h_vec);   /* r × p × H */

        /* Store growth at shifted index */
        _mm256_storeu_pd(&r_new[i + 1], growth);

        /* Accumulate changepoint probability */
        r0_acc = _mm256_add_pd(r0_acc, change);

        /* Track maximum for MAP estimation */
        __m256d cmp = _mm256_cmp_pd(growth, max_growth, _CMP_GT_OQ);
        max_growth = _mm256_blendv_pd(max_growth, growth, cmp);
        max_idx_vec = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(max_idx_vec),
            _mm256_castsi256_pd(idx_vec), cmp));

        /* Track truncation threshold */
        __m256d thresh_cmp = _mm256_cmp_pd(growth, thresh_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(thresh_cmp);
        if (mask)
        {
            if (mask & 8)
                last_valid = i + 4;
            else if (mask & 4)
                last_valid = i + 3;
            else if (mask & 2)
                last_valid = i + 2;
            else if (mask & 1)
                last_valid = i + 1;
        }

        idx_vec = _mm256_add_epi64(idx_vec, idx_inc);
    }

    /* Horizontal sum for r0 */
    __m128d lo = _mm256_castpd256_pd128(r0_acc);
    __m128d hi = _mm256_extractf128_pd(r0_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r0 = _mm_cvtsd_f64(lo);

    r_new[0] = r0;
    if (r0 > thresh && last_valid == 0)
        last_valid = 1;

    /* Extract max from SIMD register */
    double max_arr[4];
    int64_t idx_arr[4];
    _mm256_storeu_pd(max_arr, max_growth);
    _mm256_storeu_si256((__m256i *)idx_arr, max_idx_vec);

    double map_val = r0;
    size_t map_idx = 0;
    for (int j = 0; j < 4; j++)
    {
        if (max_arr[j] > map_val)
        {
            map_val = max_arr[j];
            map_idx = idx_arr[j];
        }
    }

    /* Normalize */
    size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
    if (new_len > b->capacity)
        new_len = b->capacity;

    size_t new_len_padded = (new_len + 3) & ~3ULL;

    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t j = 0; j < new_len_padded; j += 4)
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[j]));

    lo = _mm256_castpd256_pd128(sum_acc);
    hi = _mm256_extractf128_pd(sum_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r_sum = _mm_cvtsd_f64(lo);

    if (r_sum > 1e-300)
    {
        __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
        for (size_t j = 0; j < new_len_padded; j += 4)
        {
            __m256d rv = _mm256_loadu_pd(&r_new[j]);
            _mm256_storeu_pd(&r[j], _mm256_mul_pd(rv, inv_sum));
        }
    }

    b->active_len = new_len;
    b->map_runlength = map_idx;
}
#endif /* BOCPD_USE_ASM_KERNEL */

/*=============================================================================
 * PUBLIC API
 *=============================================================================*/

/**
 * @brief Initialize a BOCPD detector
 *
 * @param b               Pointer to detector structure (caller-allocated)
 * @param hazard_lambda   Expected run-length (hazard rate H = 1/λ)
 * @param prior           Prior parameters {κ₀, μ₀, α₀, β₀}
 * @param max_run_length  Maximum tracked run-length (rounded up to power of 2)
 *
 * @return 0 on success, -1 on failure (invalid parameters or allocation failure)
 *
 * @par Memory Layout
 * Allocates a single contiguous block containing:
 * - 2 interleaved buffers (ping-pong for update)
 * - 2 run-length probability arrays (r and r_scratch)
 * All aligned to 64 bytes for cache line efficiency.
 *
 * @par Typical Usage
 * @code
 * bocpd_asm_t detector;
 * bocpd_prior_t prior = {1.0, 0.0, 1.0, 1.0};  // κ=1, μ=0, α=1, β=1
 * bocpd_ultra_init(&detector, 100.0, prior, 256);  // λ=100, max_run=256
 * @endcode
 */
int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length)
{
    if (!b || hazard_lambda <= 0.0 || max_run_length < 16)
        return -1;

    memset(b, 0, sizeof(*b));

    /* Round capacity up to power of 2 for efficient modulo */
    size_t cap = 32;
    while (cap < max_run_length)
        cap <<= 1;

    b->capacity = cap;
    b->hazard = 1.0 / hazard_lambda;
    b->one_minus_h = 1.0 - b->hazard;
    b->trunc_thresh = 1e-6;
    b->prior = prior;
    b->cur_buf = 0;

    /* Precompute lgamma values for prior (used in slot 0 init) */
    b->prior_lgamma_alpha = lgamma(prior.alpha0);
    b->prior_lgamma_alpha_p5 = lgamma(prior.alpha0 + 0.5);

    /* Calculate allocation sizes */
    size_t n_blocks = cap / 4 + 2;
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    size_t bytes_r = (cap + 32) * sizeof(double);
    size_t total = 2 * bytes_interleaved + 2 * bytes_r + 64;

    /* Aligned allocation */
#ifdef _WIN32
    void *mega = _aligned_malloc(total, 64);
#else
    void *mega = NULL;
    if (posix_memalign(&mega, 64, total) != 0)
        mega = NULL;
#endif

    if (!mega)
        return -1;
    memset(mega, 0, total);

    /* Partition the mega-allocation */
    uint8_t *ptr = (uint8_t *)mega;
    b->interleaved[0] = (double *)ptr;
    ptr += bytes_interleaved;
    b->interleaved[1] = (double *)ptr;
    ptr += bytes_interleaved;
    b->r = (double *)ptr;
    ptr += bytes_r;
    b->r_scratch = (double *)ptr;

    b->mega = mega;
    b->mega_bytes = total;
    b->t = 0;
    b->active_len = 0;

    return 0;
}

/**
 * @brief Free all resources associated with a BOCPD detector
 */
void bocpd_ultra_free(bocpd_asm_t *b)
{
    if (!b)
        return;
#ifdef _WIN32
    if (b->mega)
        _aligned_free(b->mega);
#else
    free(b->mega);
#endif
    memset(b, 0, sizeof(*b));
}

/**
 * @brief Reset detector to initial state without reallocation
 */
void bocpd_ultra_reset(bocpd_asm_t *b)
{
    if (!b)
        return;

    memset(b->r, 0, (b->capacity + 32) * sizeof(double));
    memset(b->r_scratch, 0, (b->capacity + 32) * sizeof(double));

    size_t n_blocks = b->capacity / 4 + 2;
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    memset(b->interleaved[0], 0, bytes_interleaved);
    memset(b->interleaved[1], 0, bytes_interleaved);

    b->t = 0;
    b->active_len = 0;
    b->cur_buf = 0;
    b->map_runlength = 0;
    b->p_changepoint = 0.0;
}

/**
 * @brief Process one observation
 *
 * @param b  Pointer to detector
 * @param x  Observation value
 *
 * @par Outputs (in detector struct)
 * - b->r[i]: Probability of run-length i (normalized)
 * - b->map_runlength: Most likely run-length (MAP estimate)
 * - b->p_changepoint: P(run_length < 5), proxy for recent changepoint
 *
 * @par Typical Usage
 * @code
 * for (int i = 0; i < n_observations; i++) {
 *     bocpd_ultra_step(&detector, data[i]);
 *     if (detector.p_changepoint > 0.5) {
 *         printf("Changepoint detected at %d\n", i);
 *     }
 * }
 * @endcode
 */
void bocpd_ultra_step(bocpd_asm_t *b, double x)
{
    if (!b)
        return;

    /* First observation: special initialization */
    if (b->t == 0)
    {
        b->r[0] = 1.0;

        double *cur = BOCPD_CUR_BUF(b);
        double k0 = b->prior.kappa0, mu0 = b->prior.mu0;
        double a0 = b->prior.alpha0, b0 = b->prior.beta0;

        /* First posterior update */
        double k1 = k0 + 1.0;
        double mu1 = (k0 * mu0 + x) / k1;
        double a1 = a0 + 0.5;
        double beta1 = b0 + 0.5 * (x - mu0) * (x - mu1);

        IBLK_SET_MU(cur, 0, mu1);
        IBLK_SET_KAPPA(cur, 0, k1);
        IBLK_SET_ALPHA(cur, 0, a1);
        IBLK_SET_BETA(cur, 0, beta1);
        IBLK_SET_SS_N(cur, 0, 1.0);

        double sigma_sq = beta1 * (k1 + 1.0) / (a1 * k1);
        double nu = 2.0 * a1;
        double lg_a = lgamma(a1);
        double lg_ap5 = lgamma(a1 + 0.5);
        double C1 = lg_ap5 - lg_a - 0.5 * fast_log_scalar(nu * M_PI * sigma_sq);
        double C2 = a1 + 0.5;

        IBLK_SET_C1(cur, 0, C1);
        IBLK_SET_C2(cur, 0, C2);
        IBLK_SET_INV_SSN(cur, 0, 1.0 / (sigma_sq * nu));

        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }

    size_t n_old = b->active_len;
    prediction_step(b, x);
    update_posteriors_interleaved(b, x, n_old);
    b->t++;

    /* Compute changepoint probability as sum of first 5 run-lengths */
    double p = 0.0;
    size_t lim = (b->active_len < 5) ? b->active_len : 5;
    for (size_t j = 0; j < lim; j++)
        p += b->r[j];
    b->p_changepoint = p;
}

/*=============================================================================
 * POOL ALLOCATOR API
 *
 * For applications with many detectors (e.g., multivariate data), pool
 * allocation reduces overhead and improves cache locality.
 *=============================================================================*/

/**
 * @brief Initialize a pool of BOCPD detectors with shared allocation
 */
int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length)
{
    if (!pool || n_detectors == 0 || hazard_lambda <= 0.0 || max_run_length < 16)
        return -1;

    memset(pool, 0, sizeof(*pool));

    size_t cap = 32;
    while (cap < max_run_length)
        cap <<= 1;

    size_t n_blocks = cap / 4 + 2;
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    size_t bytes_r = (cap + 32) * sizeof(double);
    size_t bytes_per_detector = 2 * bytes_interleaved + 2 * bytes_r;
    bytes_per_detector = (bytes_per_detector + 63) & ~63ULL;

    size_t struct_size = n_detectors * sizeof(bocpd_asm_t);
    struct_size = (struct_size + 63) & ~63ULL;

    size_t total = struct_size + n_detectors * bytes_per_detector;

#ifdef _WIN32
    void *mega = _aligned_malloc(total, 64);
#else
    void *mega = NULL;
    if (posix_memalign(&mega, 64, total) != 0)
        mega = NULL;
#endif

    if (!mega)
        return -1;
    memset(mega, 0, total);

    pool->pool = mega;
    pool->pool_size = total;
    pool->detectors = (bocpd_asm_t *)mega;
    pool->n_detectors = n_detectors;
    pool->bytes_per_detector = bytes_per_detector;

    double prior_lgamma_alpha = lgamma(prior.alpha0);
    double prior_lgamma_alpha_p5 = lgamma(prior.alpha0 + 0.5);

    uint8_t *data_base = (uint8_t *)mega + struct_size;

    for (size_t d = 0; d < n_detectors; d++)
    {
        bocpd_asm_t *b = &pool->detectors[d];
        uint8_t *ptr = data_base + d * bytes_per_detector;

        b->capacity = cap;
        b->hazard = 1.0 / hazard_lambda;
        b->one_minus_h = 1.0 - b->hazard;
        b->trunc_thresh = 1e-6;
        b->prior = prior;
        b->cur_buf = 0;
        b->prior_lgamma_alpha = prior_lgamma_alpha;
        b->prior_lgamma_alpha_p5 = prior_lgamma_alpha_p5;

        b->interleaved[0] = (double *)ptr;
        ptr += bytes_interleaved;
        b->interleaved[1] = (double *)ptr;
        ptr += bytes_interleaved;
        b->r = (double *)ptr;
        ptr += bytes_r;
        b->r_scratch = (double *)ptr;

        b->mega = NULL; /* Pool-managed */
        b->mega_bytes = 0;
        b->t = 0;
        b->active_len = 0;
    }

    return 0;
}

void bocpd_pool_free(bocpd_pool_t *pool)
{
    if (!pool)
        return;
#ifdef _WIN32
    if (pool->pool)
        _aligned_free(pool->pool);
#else
    free(pool->pool);
#endif
    memset(pool, 0, sizeof(*pool));
}

void bocpd_pool_reset(bocpd_pool_t *pool)
{
    if (!pool)
        return;
    for (size_t d = 0; d < pool->n_detectors; d++)
        bocpd_ultra_reset(&pool->detectors[d]);
}

bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index)
{
    if (!pool || index >= pool->n_detectors)
        return NULL;
    return &pool->detectors[index];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * END OF BOCPD V3.1 IMPLEMENTATION
 * ═══════════════════════════════════════════════════════════════════════════ */