/**
 * @file bocpd_ultra_opt_asm.c
 * @brief Ultra-Optimized Bayesian Online Changepoint Detection (BOCPD)
 * @version 3.0 - Native Interleaved Layout
 *
 * @section changes_v3 V3 Changes
 *
 * The major optimization in V3 is eliminating the O(n) build_interleaved()
 * transformation that ran every observation. Instead, posterior parameters
 * are now stored directly in the interleaved SIMD format.
 *
 * @subsection v3_layout New Memory Layout
 *
 * V2 used 13 separate double-buffered arrays (26 total), requiring:
 * 1. Update each array separately
 * 2. Copy 4 arrays to interleaved staging buffer before ASM kernel
 *
 * V3 uses 2 interleaved buffers with 256-byte superblocks containing:
 * - First 128 bytes: prediction params (μ, C1, C2, inv_ssn) - read by ASM
 * - Second 128 bytes: update params (κ, α, β, ss_n) - used by C update
 *
 * @subsection v3_shift Shifted Store with Permute
 *
 * The ping-pong update reads from index i, writes to index i+1. With the
 * interleaved layout, this crosses block boundaries. V3 uses AVX2 vpermpd
 * to rotate values and vblendpd to merge with existing block content.
 *
 * @section perf_v3 Performance Impact
 *
 * | Operation            | V2 Cost      | V3 Cost      |
 * |----------------------|--------------|--------------|
 * | build_interleaved()  | O(n) copy    | Eliminated   |
 * | Posterior update     | 13 stores    | 8 stores     |
 * | ASM kernel read      | Unchanged    | Unchanged    |
 * | Per-step total       | ~1.2 μs      | ~1.0 μs      |
 */

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "bocpd_asm.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#ifndef BOCPD_USE_ASM_KERNEL
#define BOCPD_USE_ASM_KERNEL 1
#endif

/*=============================================================================
 * @defgroup iblk_access Interleaved Block Accessors
 * @brief Scalar and SIMD accessors for the native interleaved layout
 * @{
 *=============================================================================*/

/**
 * @brief Get a scalar value from the interleaved buffer.
 *
 * @param buf          Interleaved buffer base pointer
 * @param idx          Element index (0 to capacity-1)
 * @param field_offset Byte offset of field within superblock (e.g., BOCPD_IBLK_MU)
 * @return The value at buf[block][field][lane]
 *
 * @par Address Calculation
 * @code
 * block = idx / 4
 * lane  = idx % 4
 * address = buf + block * 32 + field_offset/8 + lane
 * @endcode
 */
static inline double iblk_get(const double *buf, size_t idx, size_t field_offset)
{
    size_t block = idx / 4;
    size_t lane = idx & 3;
    return buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane];
}

/**
 * @brief Set a scalar value in the interleaved buffer.
 *
 * @param buf          Interleaved buffer base pointer
 * @param idx          Element index (0 to capacity-1)
 * @param field_offset Byte offset of field within superblock
 * @param val          Value to store
 */
static inline void iblk_set(double *buf, size_t idx, size_t field_offset, double val)
{
    size_t block = idx / 4;
    size_t lane = idx & 3;
    buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane] = val;
}

/* Convenience macros for each field */
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

/** @} */ /* End of iblk_access group */

/*=============================================================================
 * @defgroup simd_math SIMD Mathematical Functions
 * @brief Vectorized mathematical functions optimized for BOCPD workloads
 * @{
 *=============================================================================*/

/**
 * @brief Fast scalar natural logarithm using IEEE-754 bit manipulation.
 *
 * @param x Input value (must be positive)
 * @return Natural logarithm ln(x)
 *
 * Uses the identity: x = 2^e × m where m ∈ [1, 2)
 * Then: ln(x) = e·ln(2) + ln(m)
 * 
 * The mantissa logarithm is computed via arctanh series.
 */
static inline double fast_log_scalar(double x)
{
    union
    {
        double d;
        uint64_t u;
    } u = {.d = x};

    /* Extract exponent (bits 52-62), subtract bias 1023 */
    int64_t e = (int64_t)((u.u >> 52) & 0x7FF) - 1023;

    /* Normalize mantissa to [1, 2) */
    u.u = (u.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = u.d;

    /* Transform for arctanh: t = (m-1)/(m+1) maps [1,2) → [0, 1/3) */
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;

    /* Polynomial: 1 + t²/3 + t⁴/5 + t⁶/7 + t⁸/9 */
    double poly = 1.0 + t2 * (0.3333333333333333 +
                              t2 * (0.2 +
                                    t2 * (0.1428571428571429 +
                                          t2 * 0.1111111111111111)));

    return (double)e * 0.6931471805599453 + 2.0 * t * poly;
}

/**
 * @brief AVX2 SIMD natural logarithm for 4 doubles in parallel.
 */
static inline __m256d fast_log_avx2(__m256d x)
{
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
    const __m256d c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d c5 = _mm256_set1_pd(0.2);
    const __m256d c7 = _mm256_set1_pd(0.1428571428571429);
    const __m256d c9 = _mm256_set1_pd(0.1111111111111111);

    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000ULL);
    const __m256i mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
    const __m256i exp_bias_bits = _mm256_set1_epi64x(0x3FF0000000000000ULL);
    const __m256i magic_i = _mm256_set1_epi64x(0x4330000000000000ULL);
    const __m256d magic_d = _mm256_set1_pd(4503599627370496.0);
    const __m256d bias_1023 = _mm256_set1_pd(1023.0);

    __m256i xi = _mm256_castpd_si256(x);

    /* Extract and convert exponent */
    __m256i exp_bits = _mm256_srli_epi64(_mm256_and_si256(xi, exp_mask), 52);
    __m256i exp_biased = _mm256_or_si256(exp_bits, magic_i);
    __m256d exp_double = _mm256_sub_pd(_mm256_castsi256_pd(exp_biased), magic_d);
    __m256d e = _mm256_sub_pd(exp_double, bias_1023);

    /* Normalize mantissa */
    __m256i mi = _mm256_or_si256(_mm256_and_si256(xi, mantissa_mask), exp_bias_bits);
    __m256d m = _mm256_castsi256_pd(mi);

    /* Arctanh transform */
    __m256d num = _mm256_sub_pd(m, one);
    __m256d den = _mm256_add_pd(m, one);
    __m256d t = _mm256_div_pd(num, den);
    __m256d t2 = _mm256_mul_pd(t, t);

    /* Polynomial evaluation */
    __m256d poly = _mm256_fmadd_pd(t2, c9, c7);
    poly = _mm256_fmadd_pd(t2, poly, c5);
    poly = _mm256_fmadd_pd(t2, poly, c3);
    poly = _mm256_fmadd_pd(t2, poly, one);

    return _mm256_fmadd_pd(e, ln2, _mm256_mul_pd(two, _mm256_mul_pd(t, poly)));
}

/**
 * @brief AVX2 lgamma using Lanczos approximation for x < 40.
 */
static inline __m256d lgamma_lanczos_avx2(__m256d x)
{
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);
    const __m256d g = _mm256_set1_pd(4.7421875);

    /* Lanczos coefficients for g=4.7421875 */
    const __m256d c0 = _mm256_set1_pd(1.000000000190015);
    const __m256d c1 = _mm256_set1_pd(76.18009172947146);
    const __m256d c2 = _mm256_set1_pd(-86.50532032941677);
    const __m256d c3 = _mm256_set1_pd(24.01409824083091);
    const __m256d c4 = _mm256_set1_pd(-1.231739572450155);
    const __m256d c5 = _mm256_set1_pd(0.001208650973866179);

    __m256d xp0 = x;
    __m256d xp1 = _mm256_add_pd(x, one);
    __m256d xp2 = _mm256_add_pd(x, _mm256_set1_pd(2.0));
    __m256d xp3 = _mm256_add_pd(x, _mm256_set1_pd(3.0));
    __m256d xp4 = _mm256_add_pd(x, _mm256_set1_pd(4.0));

    /* Sum rational terms */
    __m256d Ag = c0;
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c1, xp0));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c2, xp1));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c3, xp2));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c4, xp3));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c5, xp4));

    __m256d t = _mm256_add_pd(x, _mm256_sub_pd(g, half));
    __m256d ln_t = fast_log_avx2(t);
    __m256d ln_Ag = fast_log_avx2(Ag);

    __m256d result = half_ln2pi;
    result = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_t, result);
    result = _mm256_sub_pd(result, t);
    result = _mm256_add_pd(result, ln_Ag);

    return result;
}

/**
 * @brief AVX2 lgamma using Stirling's expansion for x > 40.
 */
static inline __m256d lgamma_stirling_avx2(__m256d x)
{
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);

    /* Stirling coefficients */
    const __m256d s1 = _mm256_set1_pd(0.0833333333333333333);
    const __m256d s2 = _mm256_set1_pd(-0.00277777777777777778);
    const __m256d s3 = _mm256_set1_pd(0.000793650793650793651);
    const __m256d s4 = _mm256_set1_pd(-0.000595238095238095238);
    const __m256d s5 = _mm256_set1_pd(0.000841750841750841751);
    const __m256d s6 = _mm256_set1_pd(-0.00191752691752691753);

    __m256d ln_x = fast_log_avx2(x);
    __m256d base = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_x,
                                   _mm256_sub_pd(half_ln2pi, x));

    __m256d inv_x = _mm256_div_pd(one, x);
    __m256d inv_x2 = _mm256_mul_pd(inv_x, inv_x);

    /* Horner evaluation */
    __m256d correction = s6;
    correction = _mm256_fmadd_pd(correction, inv_x2, s5);
    correction = _mm256_fmadd_pd(correction, inv_x2, s4);
    correction = _mm256_fmadd_pd(correction, inv_x2, s3);
    correction = _mm256_fmadd_pd(correction, inv_x2, s2);
    correction = _mm256_fmadd_pd(correction, inv_x2, s1);
    correction = _mm256_mul_pd(correction, inv_x);

    return _mm256_add_pd(base, correction);
}

/**
 * @brief Unified AVX2 lgamma with branchless region selection.
 */
static inline __m256d fast_lgamma_avx2(__m256d x)
{
    const __m256d forty = _mm256_set1_pd(40.0);

    __m256d result_small = lgamma_lanczos_avx2(x);
    __m256d result_large = lgamma_stirling_avx2(x);

    /* Use Lanczos for x <= 40, Stirling for x > 40 */
    __m256d mask_large = _mm256_cmp_pd(x, forty, _CMP_GT_OQ);
    return _mm256_blendv_pd(result_small, result_large, mask_large);
}

/** @} */ /* End of simd_math group */

/*=============================================================================
 * @defgroup shifted_store Shifted Store Operations
 * @brief AVX2 permute-based stores for +1 index offset
 *
 * The ping-pong update reads from CUR[i] and writes to NEXT[i+1]. With the
 * interleaved layout (4 elements per block), this crosses block boundaries.
 *
 * Solution: Use vpermpd to rotate the vector, then blend+store to two blocks.
 * @{
 *=============================================================================*/

/**
 * @brief Store 4 values with +1 index shift using AVX2 permute.
 *
 * @param buf          Destination interleaved buffer
 * @param block_idx    Source block index (i/4 where i is first element)
 * @param field_offset Byte offset of field within superblock
 * @param vals         4 values for indices [i, i+1, i+2, i+3]
 *
 * @par Algorithm
 *
 * Input values are for indices [i, i+1, i+2, i+3] where i is block-aligned.
 * Output should go to indices [i+1, i+2, i+3, i+4], spanning two blocks:
 * - Block k:   lanes 1,2,3 get values 0,1,2
 * - Block k+1: lane 0 gets value 3
 *
 * @par Implementation
 *
 * 1. Rotate right: [v0,v1,v2,v3] → [v3,v0,v1,v2] using vpermpd(0x93)
 * 2. Blend with block k:   keep lane 0, replace lanes 1,2,3
 * 3. Blend with block k+1: replace lane 0, keep lanes 1,2,3
 *
 * @par Performance
 *
 * - 1 vpermpd (3 cycles latency, 1 µop)
 * - 2 vblendpd (1 cycle each, 1 µop each)
 * - 2 loads + 2 stores
 *
 * Total: ~6 cycles vs ~16 cycles for 4 scalar stores with address calculation
 */
static inline void store_shifted_field(double *buf, size_t block_idx,
                                        size_t field_offset, __m256d vals)
{
    /*
     * Rotate right by 1: [v0,v1,v2,v3] → [v3,v0,v1,v2]
     * 
     * vpermpd immediate encoding:
     *   imm8[1:0] selects source for dst[0]
     *   imm8[3:2] selects source for dst[1]
     *   imm8[5:4] selects source for dst[2]
     *   imm8[7:6] selects source for dst[3]
     *
     * To get [src[3], src[0], src[1], src[2]]:
     *   dst[0] = src[3] → bits [1:0] = 3
     *   dst[1] = src[0] → bits [3:2] = 0
     *   dst[2] = src[1] → bits [5:4] = 1
     *   dst[3] = src[2] → bits [7:6] = 2
     *   imm8 = 0b10_01_00_11 = 0x93
     */
    __m256d rotated = _mm256_permute4x64_pd(vals, 0x93);

    /* Calculate block base addresses (in doubles, not bytes) */
    double *block_k = buf + block_idx * BOCPD_IBLK_DOUBLES + field_offset / 8;
    double *block_k1 = buf + (block_idx + 1) * BOCPD_IBLK_DOUBLES + field_offset / 8;

    /* Load existing content of both blocks */
    __m256d existing_k = _mm256_loadu_pd(block_k);
    __m256d existing_k1 = _mm256_loadu_pd(block_k1);

    /*
     * Blend masks for vblendpd:
     *   Bit i = 1 → select from second operand (rotated)
     *   Bit i = 0 → select from first operand (existing)
     *
     * Block k: keep lane 0, replace lanes 1,2,3
     *   mask = 0b1110 = 14
     *
     * Block k+1: replace lane 0, keep lanes 1,2,3
     *   mask = 0b0001 = 1
     */
    __m256d merged_k = _mm256_blend_pd(existing_k, rotated, 0b1110);
    __m256d merged_k1 = _mm256_blend_pd(existing_k1, rotated, 0b0001);

    /* Store merged results */
    _mm256_storeu_pd(block_k, merged_k);
    _mm256_storeu_pd(block_k1, merged_k1);
}

/** @} */ /* End of shifted_store group */

/*=============================================================================
 * @defgroup init_update Initialization and Posterior Update
 * @brief Core BOCPD update operations using native interleaved layout
 * @{
 *=============================================================================*/

/**
 * @brief Initialize slot 0 of NEXT buffer with prior parameters.
 *
 * @param b Pointer to BOCPD detector state
 *
 * Slot 0 represents run length 0 (just had a changepoint). It's initialized
 * with the prior parameters, ready for the next observation.
 */
static inline void init_slot_zero(bocpd_asm_t *b)
{
    double *next = BOCPD_NEXT_BUF(b);

    const double kappa0 = b->prior.kappa0;
    const double mu0 = b->prior.mu0;
    const double alpha0 = b->prior.alpha0;
    const double beta0 = b->prior.beta0;

    /* Write prior parameters to slot 0 */
    IBLK_SET_MU(next, 0, mu0);
    IBLK_SET_KAPPA(next, 0, kappa0);
    IBLK_SET_ALPHA(next, 0, alpha0);
    IBLK_SET_BETA(next, 0, beta0);
    IBLK_SET_SS_N(next, 0, 0.0);

    /* Compute Student-t constants for prior */
    double sigma_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
    double nu = 2.0 * alpha0;
    double ln_nu_pi = fast_log_scalar(nu * M_PI);
    double ln_sigma_sq = fast_log_scalar(sigma_sq);

    double C1 = b->prior_lgamma_alpha_p5 - b->prior_lgamma_alpha -
                0.5 * ln_nu_pi - 0.5 * ln_sigma_sq;
    double C2 = alpha0 + 0.5;

    IBLK_SET_C1(next, 0, C1);
    IBLK_SET_C2(next, 0, C2);
    IBLK_SET_INV_SSN(next, 0, 1.0 / (sigma_sq * nu));
}

/**
 * @brief Fused posterior update directly to interleaved format.
 *
 * @param b     Pointer to BOCPD detector state
 * @param x     New observation value
 * @param n_old Number of active run lengths before this observation
 *
 * @par V3 Optimization
 *
 * This function replaces both update_posteriors_fused() AND build_interleaved()
 * from V2. Everything happens in a single pass with no intermediate buffers.
 *
 * @par Algorithm
 *
 * For each run length i in [0, n_old):
 * 1. Load posterior params from CUR[i]
 * 2. Update using Welford's algorithm
 * 3. Compute Student-t constants
 * 4. Store to NEXT[i+1] using shifted store
 *
 * The +1 shift implements run length growth: after seeing observation x,
 * what was run length i becomes run length i+1.
 */
static void update_posteriors_interleaved(bocpd_asm_t *b, double x, size_t n_old)
{
    /* Initialize slot 0 with prior (changepoint hypothesis) */
    init_slot_zero(b);

    if (n_old == 0)
    {
        b->cur_buf = 1 - b->cur_buf;
        return;
    }

    const double *cur = BOCPD_CUR_BUF(b);
    double *next = BOCPD_NEXT_BUF(b);

    /* SIMD constants */
    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d pi = _mm256_set1_pd(M_PI);

    size_t i = 0;

    /*-------------------------------------------------------------------------
     * SIMD Loop: Process 4 elements per iteration
     *
     * Read from block-aligned positions in CUR, write with +1 shift to NEXT.
     * The shifted store handles the block boundary crossing automatically.
     *-------------------------------------------------------------------------*/
    for (; i + 4 <= n_old; i += 4)
    {
        size_t block = i / 4;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;

        /* Load posterior parameters from current block */
        __m256d mu_old = _mm256_loadu_pd(src + BOCPD_IBLK_MU / 8);
        __m256d kappa_old = _mm256_loadu_pd(src + BOCPD_IBLK_KAPPA / 8);
        __m256d alpha_old = _mm256_loadu_pd(src + BOCPD_IBLK_ALPHA / 8);
        __m256d beta_old = _mm256_loadu_pd(src + BOCPD_IBLK_BETA / 8);
        __m256d ss_n_old = _mm256_loadu_pd(src + BOCPD_IBLK_SS_N / 8);

        /*
         * Welford posterior update:
         *   κₙ = κₙ₋₁ + 1
         *   μₙ = (κₙ₋₁·μₙ₋₁ + x) / κₙ
         *   αₙ = αₙ₋₁ + 0.5
         *   βₙ = βₙ₋₁ + 0.5·(x - μₙ₋₁)·(x - μₙ)
         */
        __m256d ss_n_new = _mm256_add_pd(ss_n_old, one);
        __m256d kappa_new = _mm256_add_pd(kappa_old, one);
        __m256d mu_new = _mm256_div_pd(
            _mm256_fmadd_pd(kappa_old, mu_old, x_vec),
            kappa_new);
        __m256d alpha_new = _mm256_add_pd(alpha_old, half);

        /* Welford β update: avoids catastrophic cancellation */
        __m256d delta1 = _mm256_sub_pd(x_vec, mu_old);
        __m256d delta2 = _mm256_sub_pd(x_vec, mu_new);
        __m256d beta_inc = _mm256_mul_pd(_mm256_mul_pd(delta1, delta2), half);
        __m256d beta_new = _mm256_add_pd(beta_old, beta_inc);

        /*
         * Compute Student-t scale: σ² = β(κ+1)/(ακ)
         * Degrees of freedom: ν = 2α
         */
        __m256d kappa_p1 = _mm256_add_pd(kappa_new, one);
        __m256d sigma_sq = _mm256_div_pd(
            _mm256_mul_pd(beta_new, kappa_p1),
            _mm256_mul_pd(alpha_new, kappa_new));
        __m256d nu = _mm256_mul_pd(two, alpha_new);
        __m256d sigma_sq_nu = _mm256_mul_pd(sigma_sq, nu);
        __m256d inv_ssn = _mm256_div_pd(one, sigma_sq_nu);

        /* lgamma via SIMD approximation */
        __m256d lg_a = fast_lgamma_avx2(alpha_new);
        __m256d alpha_p5 = _mm256_add_pd(alpha_new, half);
        __m256d lg_ap5 = fast_lgamma_avx2(alpha_p5);

        /*
         * Student-t constants:
         *   C1 = lgamma(α+0.5) - lgamma(α) - 0.5·ln(π·ν·σ²)
         *   C2 = α + 0.5
         */
        __m256d nu_pi_s2 = _mm256_mul_pd(_mm256_mul_pd(nu, pi), sigma_sq);
        __m256d ln_term = fast_log_avx2(nu_pi_s2);
        __m256d C1 = _mm256_sub_pd(lg_ap5, lg_a);
        C1 = _mm256_fnmadd_pd(half, ln_term, C1);
        __m256d C2 = alpha_p5;

        /* Store all 8 fields with +1 shift using permute+blend */
        store_shifted_field(next, block, BOCPD_IBLK_MU, mu_new);
        store_shifted_field(next, block, BOCPD_IBLK_KAPPA, kappa_new);
        store_shifted_field(next, block, BOCPD_IBLK_ALPHA, alpha_new);
        store_shifted_field(next, block, BOCPD_IBLK_BETA, beta_new);
        store_shifted_field(next, block, BOCPD_IBLK_SS_N, ss_n_new);
        store_shifted_field(next, block, BOCPD_IBLK_C1, C1);
        store_shifted_field(next, block, BOCPD_IBLK_C2, C2);
        store_shifted_field(next, block, BOCPD_IBLK_INV_SSN, inv_ssn);
    }

    /* Scalar tail for remaining 0-3 elements */
    for (; i < n_old; i++)
    {
        double ss_n_old = IBLK_GET_SS_N(cur, i);
        double kappa_old = IBLK_GET_KAPPA(cur, i);
        double mu_old = IBLK_GET_MU(cur, i);
        double alpha_old = IBLK_GET_ALPHA(cur, i);
        double beta_old = IBLK_GET_BETA(cur, i);

        double ss_n_new = ss_n_old + 1.0;
        double kappa_new = kappa_old + 1.0;
        double mu_new = (kappa_old * mu_old + x) / kappa_new;
        double alpha_new = alpha_old + 0.5;
        double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);

        double sigma_sq = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);
        double nu = 2.0 * alpha_new;
        double inv_ssn = 1.0 / (sigma_sq * nu);

        double lg_a = lgamma(alpha_new);
        double lg_ap5 = lgamma(alpha_new + 0.5);
        double C1 = lg_ap5 - lg_a - 0.5 * fast_log_scalar(nu * M_PI * sigma_sq);
        double C2 = alpha_new + 0.5;

        /* Write to index i+1 (scalar version of shifted store) */
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

    /* Swap buffers: what was NEXT is now CUR */
    b->cur_buf = 1 - b->cur_buf;
}

/** @} */ /* End of init_update group */






/*=============================================================================
 * @defgroup prediction Prediction Step
 * @brief Compute predictive probabilities and update run-length distribution
 * @{
 *=============================================================================*/

/**
 * @brief Prediction step reading directly from native interleaved buffer.
 *
 * @param b Pointer to BOCPD detector state
 * @param x New observation value
 *
 * @par V3 Change
 *
 * The key difference from V2 is that there's no build_interleaved() call.
 * The ASM kernel (or this C fallback) reads directly from BOCPD_CUR_BUF(b).
 *
 * @par Algorithm
 *
 * For each run length i:
 * 1. Load params: μ, C1, C2, inv_ssn from interleaved block
 * 2. Compute Student-t: pp = exp(C1 - C2·log1p((x-μ)²·inv_ssn))
 * 3. Growth: r_new[i+1] = r[i] × pp × (1-h)
 * 4. Changepoint: r0 += r[i] × pp × h
 * 5. Normalize r_new to sum to 1
 */


#if BOCPD_USE_ASM_KERNEL

static void prediction_step(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;
    if (n == 0) return;

    const double thresh = b->trunc_thresh;

    /* V3: No build_interleaved() needed - data already in native format! */
    double *params = BOCPD_CUR_BUF(b);  /* Read directly from interleaved buffer */

    double *r = b->r;
    double *r_new = b->r_scratch;

    const size_t n_padded = (n + 7) & ~7ULL;

    /* Zero-pad input beyond active length */
    for (size_t i = n; i < n_padded + 8; i++)
        r[i] = 0.0;

    /* Zero output buffer */
    memset(r_new, 0, (n_padded + 16) * sizeof(double));

    /* Output variables for kernel */
    double r0_out = 0.0;
    double max_growth_out = 0.0;
    size_t max_idx_out = 0;
    size_t last_valid_out = 0;

    /* Package arguments for assembly kernel */
    bocpd_kernel_args_t args = {
        .lin_interleaved = params,  /* V3: direct pointer, no transformation */
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
        .last_valid_out = &last_valid_out
    };

    /* Call assembly kernel */
    bocpd_fused_loop_avx2(&args);

    /* Assembly writes to *r0_out, not r_new[0] */
    r_new[0] = r0_out;

    if (r0_out > thresh && last_valid_out == 0)
        last_valid_out = 1;

    /* Determine new active length */
    size_t new_len = (last_valid_out > 0) ? last_valid_out + 1 : n + 1;
    if (new_len > b->capacity)
        new_len = b->capacity;

    size_t new_len_padded = (new_len + 7) & ~7ULL;

    /* Normalize distribution */
    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t i = 0; i < new_len_padded; i += 4)
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[i]));

    __m128d lo = _mm256_castpd256_pd128(sum_acc);
    __m128d hi = _mm256_extractf128_pd(sum_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r_sum = _mm_cvtsd_f64(lo);

    if (r_sum > 1e-300) {
        __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
        for (size_t i = 0; i < new_len_padded; i += 4) {
            __m256d rv = _mm256_loadu_pd(&r_new[i]);
            _mm256_storeu_pd(&r[i], _mm256_mul_pd(rv, inv_sum));
        }
    }

    b->active_len = new_len;

    /* Determine MAP run length */
    double r0_normalized = (r_sum > 1e-300) ? r0_out / r_sum : 0.0;
    double max_normalized = (r_sum > 1e-300) ? max_growth_out / r_sum : 0.0;

    if (r0_normalized >= max_normalized)
        b->map_runlength = 0;
    else
        b->map_runlength = max_idx_out;
}

#else /* !BOCPD_USE_ASM_KERNEL - C intrinsics fallback */

static void prediction_step(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;
    if (n == 0)
        return;

    const double h = b->hazard;
    const double omh = b->one_minus_h;
    const double thresh = b->trunc_thresh;

    /* Read directly from current interleaved buffer - NO build_interleaved()! */
    const double *params = BOCPD_CUR_BUF(b);

    double *r = b->r;
    double *r_new = b->r_scratch;

    const size_t n_padded = (n + 7) & ~7ULL;

    /* Zero padding for clean SIMD loads */
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

    /* log1p polynomial coefficients */
    const __m256d log1p_c2 = _mm256_set1_pd(-0.5);
    const __m256d log1p_c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d log1p_c4 = _mm256_set1_pd(-0.25);
    const __m256d log1p_c5 = _mm256_set1_pd(0.2);
    const __m256d log1p_c6 = _mm256_set1_pd(-0.1666666666666667);

    /* exp polynomial coefficients */
    const __m256d exp_inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256d exp_min_x = _mm256_set1_pd(-700.0);
    const __m256d exp_max_x = _mm256_set1_pd(700.0);
    const __m256d exp_c1 = _mm256_set1_pd(0.6931471805599453);
    const __m256d exp_c2 = _mm256_set1_pd(0.24022650695910072);
    const __m256d exp_c3 = _mm256_set1_pd(0.05550410866482158);
    const __m256d exp_c4 = _mm256_set1_pd(0.009618129107628477);
    const __m256d exp_c5 = _mm256_set1_pd(0.0013333558146428443);
    const __m256d exp_c6 = _mm256_set1_pd(0.00015403530393381608);
    const __m256i exp_bias = _mm256_set1_epi64x(1023);

    __m256d r0_acc = _mm256_setzero_pd();
    __m256d max_growth = _mm256_setzero_pd();
    __m256i max_idx_vec = _mm256_setzero_si256();

    __m256i idx_vec = _mm256_set_epi64x(4, 3, 2, 1);
    const __m256i idx_inc = _mm256_set1_epi64x(4);

    size_t last_valid = 0;

    for (size_t i = 0; i < n_padded; i += 4)
    {
        size_t block = i / 4;
        const double *blk = params + block * BOCPD_IBLK_DOUBLES;

        /* Load prediction parameters directly from interleaved block */
        __m256d mu = _mm256_loadu_pd(blk + BOCPD_IBLK_MU / 8);
        __m256d C1 = _mm256_loadu_pd(blk + BOCPD_IBLK_C1 / 8);
        __m256d C2 = _mm256_loadu_pd(blk + BOCPD_IBLK_C2 / 8);
        __m256d inv_ssn = _mm256_loadu_pd(blk + BOCPD_IBLK_INV_SSN / 8);
        __m256d r_old = _mm256_loadu_pd(&r[i]);

        /* Student-t computation: pp = exp(C1 - C2·log1p(t)) where t = (x-μ)²·inv_ssn */
        __m256d z = _mm256_sub_pd(x_vec, mu);
        __m256d z2 = _mm256_mul_pd(z, z);
        __m256d t = _mm256_mul_pd(z2, inv_ssn);

        /* log1p(t) polynomial (Horner's method) */
        __m256d poly = _mm256_fmadd_pd(t, log1p_c6, log1p_c5);
        poly = _mm256_fmadd_pd(t, poly, log1p_c4);
        poly = _mm256_fmadd_pd(t, poly, log1p_c3);
        poly = _mm256_fmadd_pd(t, poly, log1p_c2);
        poly = _mm256_fmadd_pd(t, poly, const_one);
        __m256d log1p_t = _mm256_mul_pd(t, poly);

        /* ln_pp = C1 - C2·log1p(t) */
        __m256d ln_pp = _mm256_fnmadd_pd(C2, log1p_t, C1);

        /* exp(ln_pp) using 2^k × 2^f decomposition */
        __m256d x_clamp = _mm256_max_pd(_mm256_min_pd(ln_pp, exp_max_x), exp_min_x);
        __m256d t_exp = _mm256_mul_pd(x_clamp, exp_inv_ln2);
        __m256d k = _mm256_round_pd(t_exp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d f = _mm256_sub_pd(t_exp, k);

        /* 2^f polynomial (Estrin's scheme) */
        __m256d f2 = _mm256_mul_pd(f, f);
        __m256d p01 = _mm256_fmadd_pd(f, exp_c1, const_one);
        __m256d p23 = _mm256_fmadd_pd(f, exp_c3, exp_c2);
        __m256d p45 = _mm256_fmadd_pd(f, exp_c5, exp_c4);
        __m256d q0123 = _mm256_fmadd_pd(f2, p23, p01);
        __m256d q456 = _mm256_fmadd_pd(f2, exp_c6, p45);
        __m256d f4 = _mm256_mul_pd(f2, f2);
        __m256d exp_p = _mm256_fmadd_pd(f4, q456, q0123);

        /* 2^k via IEEE-754 bit manipulation */
        __m128i k32 = _mm256_cvtpd_epi32(k);
        __m256i k64 = _mm256_cvtepi32_epi64(k32);
        __m256i biased = _mm256_add_epi64(k64, exp_bias);
        __m256i bits = _mm256_slli_epi64(biased, 52);
        __m256d scale = _mm256_castsi256_pd(bits);

        __m256d pp = _mm256_mul_pd(exp_p, scale);
        pp = _mm256_max_pd(pp, min_pp);

        /* BOCPD update */
        __m256d r_pp = _mm256_mul_pd(r_old, pp);
        __m256d growth = _mm256_mul_pd(r_pp, omh_vec);
        __m256d change = _mm256_mul_pd(r_pp, h_vec);

        _mm256_storeu_pd(&r_new[i + 1], growth);
        r0_acc = _mm256_add_pd(r0_acc, change);

        /* Track maximum for MAP */
        __m256d cmp = _mm256_cmp_pd(growth, max_growth, _CMP_GT_OQ);
        max_growth = _mm256_blendv_pd(max_growth, growth, cmp);
        max_idx_vec = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(max_idx_vec),
            _mm256_castsi256_pd(idx_vec), cmp));

        /* Track last index above truncation threshold */
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

    /* Horizontal reduction for r0 */
    __m128d lo = _mm256_castpd256_pd128(r0_acc);
    __m128d hi = _mm256_extractf128_pd(r0_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r0 = _mm_cvtsd_f64(lo);

    r_new[0] = r0;

    if (r0 > thresh && last_valid == 0)
        last_valid = 1;

    /* Find global max for MAP */
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

    /* Normalize distribution */
    size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
    if (new_len > b->capacity)
        new_len = b->capacity;

    size_t new_len_padded = (new_len + 3) & ~3ULL;

    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t j = 0; j < new_len_padded; j += 4)
    {
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[j]));
    }

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
#endif

/** @} */ /* End of prediction group */

/*=============================================================================
 * @defgroup public_api Public API
 * @brief User-facing functions for BOCPD detector lifecycle
 * @{
 *=============================================================================*/

int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length)
{
    if (!b || hazard_lambda <= 0.0 || max_run_length < 16)
        return -1;

    memset(b, 0, sizeof(*b));

    /* Round capacity to power of 2 */
    size_t cap = 32;
    while (cap < max_run_length)
        cap <<= 1;

    b->capacity = cap;
    b->hazard = 1.0 / hazard_lambda;
    b->one_minus_h = 1.0 - b->hazard;
    b->trunc_thresh = 1e-6;
    b->prior = prior;
    b->cur_buf = 0;

    b->prior_lgamma_alpha = lgamma(prior.alpha0);
    b->prior_lgamma_alpha_p5 = lgamma(prior.alpha0 + 0.5);

    /*
     * Memory layout (V3):
     * - 2 interleaved buffers: each (cap/4 + 2) blocks × 256 bytes
     * - r: (cap + 32) doubles
     * - r_scratch: (cap + 32) doubles
     */
    size_t n_blocks = cap / 4 + 2; /* +2 for padding at block boundaries */
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    size_t bytes_r = (cap + 32) * sizeof(double);

    size_t total = 2 * bytes_interleaved + 2 * bytes_r + 64;

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

void bocpd_ultra_reset(bocpd_asm_t *b)
{
    if (!b)
        return;

    memset(b->r, 0, (b->capacity + 32) * sizeof(double));
    b->t = 0;
    b->active_len = 0;
    b->cur_buf = 0;
}

void bocpd_ultra_step(bocpd_asm_t *b, double x)
{
    if (!b)
        return;

    /* First observation: special initialization */
    if (b->t == 0)
    {
        b->r[0] = 1.0;

        double *cur = BOCPD_CUR_BUF(b);

        double k0 = b->prior.kappa0;
        double mu0 = b->prior.mu0;
        double a0 = b->prior.alpha0;
        double b0 = b->prior.beta0;

        /* First posterior after seeing x */
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

    /* Prediction step - reads directly from interleaved buffer */
    prediction_step(b, x);

    /* Posterior update - writes directly to interleaved buffer */
    update_posteriors_interleaved(b, x, n_old);

    b->t++;

    /* Compute changepoint probability (sum of first few run lengths) */
    double p = 0.0;
    size_t lim = (b->active_len < 5) ? b->active_len : 5;
    for (size_t j = 0; j < lim; j++)
        p += b->r[j];
    b->p_changepoint = p;
}

/** @} */ /* End of public_api group */

/*=============================================================================
 * @defgroup pool_api Pool Allocator API
 * @{
 *=============================================================================*/

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
    bytes_per_detector = (bytes_per_detector + 63) & ~63ULL; /* 64-byte align */

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

        b->mega = NULL; /* Pool detectors don't own their memory */
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
    {
        bocpd_ultra_reset(&pool->detectors[d]);
    }
}

bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index)
{
    if (!pool || index >= pool->n_detectors)
        return NULL;
    return &pool->detectors[index];
}

/** @} */ /* End of pool_api group */