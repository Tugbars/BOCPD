/**
 * @file bocpd_asm.h
 * @brief AVX2 Assembly-Optimized BOCPD Implementation
 *
 * Ultra-fast Bayesian Online Changepoint Detection with hand-tuned
 * AVX2 assembly kernels. This is the high-performance path for
 * production trading systems.
 *
 * Features:
 *   - Hand-written AVX2 assembly inner loop
 *   - Two kernel variants: Generic (all CPUs) and Intel-tuned
 *   - ~525K observations/sec throughput
 *   - Sub-2µs latency per observation
 *   - Full Bayesian posterior (no approximations)
 *
 * This header is SEPARATE from bocpd_fast.h (the intrinsics-based version).
 * Choose one implementation for your build, not both.
 */

#ifndef BOCPD_ASM_H
#define BOCPD_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ALIGN64 __attribute__((aligned(64)))

/*=============================================================================
 * Kernel Variant Selection
 *
 * BOCPD_KERNEL_GENERIC (0):
 *   - Conservative scheduling, works well on all x86-64
 *   - Best for: AMD Zen1-4, older Intel, unknown targets
 *   - Throughput: ~510K obs/sec
 *
 * BOCPD_KERNEL_INTEL (1):
 *   - Aggressive ILP with interleaved A/B blocks
 *   - Best for: Intel 12th-14th gen (Alder Lake, Raptor Lake)
 *   - Throughput: ~525K obs/sec (+3%)
 *
 * Set via compile flag: -DBOCPD_KERNEL_VARIANT=1
 *=============================================================================*/

#define BOCPD_KERNEL_GENERIC    0
#define BOCPD_KERNEL_INTEL      1

#ifndef BOCPD_KERNEL_VARIANT
    #define BOCPD_KERNEL_VARIANT BOCPD_KERNEL_GENERIC
#endif

/*=============================================================================
 * Prior Parameters
 *=============================================================================*/

/**
 * @brief Normal-Gamma prior parameters.
 *
 * Conjugate prior for Gaussian data with unknown mean and variance.
 *
 *   μ | τ ~ Normal(μ₀, 1/(κ₀τ))
 *   τ     ~ Gamma(α₀, β₀)
 *
 * where τ = 1/σ² (precision).
 */
typedef struct bocpd_prior {
    double mu0;     /**< Prior mean */
    double kappa0;  /**< Prior mean strength (pseudo-observations) */
    double alpha0;  /**< Precision shape (> 0) */
    double beta0;   /**< Precision rate (> 0), E[σ²] ≈ β₀/(α₀-1) for α₀>1 */
} bocpd_prior_t;

/*=============================================================================
 * Detector State
 *=============================================================================*/

/**
 * @brief BOCPD detector state (ASM-optimized version).
 *
 * All arrays are 64-byte aligned for AVX2/AVX-512.
 * Ring buffer with power-of-2 capacity for fast modular arithmetic.
 */
typedef struct bocpd_asm {
    /*-------------------------------------------------------------------------
     * Configuration
     *-------------------------------------------------------------------------*/
    size_t capacity;            /**< Max run lengths (power of 2) */
    double hazard;              /**< Hazard rate h = 1/λ */
    double one_minus_h;         /**< Precomputed 1-h */
    double trunc_thresh;        /**< Truncation threshold (default 1e-12) */
    bocpd_prior_t prior;        /**< Prior parameters */

    /*-------------------------------------------------------------------------
     * Ring buffer state
     *-------------------------------------------------------------------------*/
    size_t ring_start;          /**< Start index in ring buffer */
    size_t active_len;          /**< Number of active run lengths */

    /*-------------------------------------------------------------------------
     * Sufficient statistics (ring-buffered)
     *-------------------------------------------------------------------------*/
    double *ss_n;               /**< Count per run length */
    double *ss_sum;             /**< Sum of observations per run */
    double *ss_sum2;            /**< Sum of squared observations per run */

    /*-------------------------------------------------------------------------
     * Posterior parameters (ring-buffered)
     *-------------------------------------------------------------------------*/
    double *post_kappa;         /**< Posterior κ */
    double *post_mu;            /**< Posterior μ */
    double *post_alpha;         /**< Posterior α */
    double *post_beta;          /**< Posterior β */

    /*-------------------------------------------------------------------------
     * Precomputed Student-t constants (ring-buffered)
     *-------------------------------------------------------------------------*/
    double *C1;                 /**< Log-pdf constant term */
    double *C2;                 /**< Log-pdf coefficient (α + 0.5) */
    double *sigma_sq;           /**< Scale parameter σ² */
    double *inv_sigma_sq_nu;    /**< Precomputed 1/(σ²ν) */
    double *lgamma_alpha;       /**< Cached lgamma(α) */
    double *lgamma_alpha_p5;    /**< Cached lgamma(α + 0.5) */

    /*-------------------------------------------------------------------------
     * Interleaved scratch buffer (for ASM kernel)
     *
     * Memory layout optimized for cache utilization:
     *   Block k: [μ[4k:4k+3], C1[4k:4k+3], C2[4k:4k+3], inv_σ²ν[4k:4k+3]]
     *
     * Each block = 128 bytes = 2 cache lines.
     * Single memory stream enables efficient prefetching.
     *-------------------------------------------------------------------------*/
    double *lin_interleaved;    /**< Interleaved parameter blocks */

    /*-------------------------------------------------------------------------
     * Legacy pointers (for compatibility, point into lin_interleaved)
     *-------------------------------------------------------------------------*/
    double *lin_mu;             /**< Linearized posterior means (legacy) */
    double *lin_C1;             /**< Linearized C1 constants (legacy) */
    double *lin_C2;             /**< Linearized C2 constants (legacy) */
    double *lin_inv_ssn;        /**< Linearized 1/(σ²ν) values (legacy) */

    /*-------------------------------------------------------------------------
     * Run-length distribution
     *-------------------------------------------------------------------------*/
    double *r;                  /**< Current distribution P(r_t | x_{1:t}) */
    double *r_scratch;          /**< Scratch buffer for updates */

    /*-------------------------------------------------------------------------
     * Output state
     *-------------------------------------------------------------------------*/
    size_t t;                   /**< Current timestep */
    size_t map_runlength;       /**< MAP estimate of run length */
    double p_changepoint;       /**< P(r_t < 5) - quick changepoint indicator */

} bocpd_asm_t;

/*=============================================================================
 * Public API
 *=============================================================================*/

/**
 * @brief Initialize BOCPD detector.
 *
 * @param b              Detector to initialize
 * @param hazard_lambda  Expected run length λ (hazard = 1/λ)
 * @param prior          Prior parameters
 * @param max_run_length Maximum run length to track (rounded up to power of 2)
 *
 * @return 0 on success, -1 on allocation failure
 */
int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);

/**
 * @brief Free detector resources.
 */
void bocpd_ultra_free(bocpd_asm_t *b);

/**
 * @brief Reset detector to initial state (preserves configuration).
 */
void bocpd_ultra_reset(bocpd_asm_t *b);

/**
 * @brief Process one observation.
 *
 * @param b Detector state
 * @param x New observation
 *
 * Complexity: O(active_len), typically 1.5-2.5 µs
 */
void bocpd_ultra_step(bocpd_asm_t *b, double x);

/**
 * @brief Get MAP run length estimate.
 */
static inline size_t bocpd_ultra_get_map(const bocpd_asm_t *b) {
    return b->map_runlength;
}

/**
 * @brief Get current timestep.
 */
static inline size_t bocpd_ultra_get_t(const bocpd_asm_t *b) {
    return b->t;
}

/**
 * @brief Get quick changepoint probability P(run_length < 5).
 */
static inline double bocpd_ultra_get_change_prob(const bocpd_asm_t *b) {
    return b->p_changepoint;
}

/*=============================================================================
 * Assembly Kernel Interface (Internal)
 *
 * These declarations are exposed for advanced users who want to call
 * kernels directly. Normal usage should go through bocpd_ultra_step().
 *=============================================================================*/

/**
 * @brief Arguments passed to assembly kernel.
 *
 * CRITICAL: Field order matches hardcoded offsets in assembly.
 * DO NOT REORDER FIELDS.
 *
 * Offsets (bytes):
 *   +0   lin_interleaved (double*)
 *   +8   r_old (double*)
 *   +16  x (double)
 *   +24  h (double)
 *   +32  one_minus_h (double)
 *   +40  trunc_thresh (double)
 *   +48  n_padded (size_t)
 *   +56  r_new (double*)
 *   +64  r0_out (double*)
 *   +72  max_growth_out (double*)
 *   +80  max_idx_out (size_t*)
 *   +88  last_valid_out (size_t*)
 */
typedef struct bocpd_kernel_args {
    const double *lin_interleaved;  /**< +0   Interleaved parameters */
    const double *r_old;            /**< +8   Input run-length distribution */
    double x;                       /**< +16  Observation */
    double h;                       /**< +24  Hazard rate */
    double one_minus_h;             /**< +32  1 - hazard */
    double trunc_thresh;            /**< +40  Truncation threshold */
    size_t n_padded;                /**< +48  Padded length (multiple of 8) */
    double *r_new;                  /**< +56  Output distribution */
    double *r0_out;                 /**< +64  Output: changepoint probability */
    double *max_growth_out;         /**< +72  Output: max growth value */
    size_t *max_idx_out;            /**< +80  Output: MAP index */
    size_t *last_valid_out;         /**< +88  Output: truncation boundary */
} bocpd_kernel_args_t;

/**
 * @brief Generic AVX2 kernel - works on all x86-64 with AVX2+FMA.
 *
 * Conservative scheduling, aligned loads preferred.
 * Implemented in bocpd_kernel_avx2_generic.asm
 */
extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);

/**
 * @brief Intel-tuned AVX2 kernel - optimized for Alder/Raptor Lake.
 *
 * Aggressive ILP, interleaved A/B scheduling.
 * Implemented in bocpd_kernel_avx2_intel.asm
 */
extern void bocpd_fused_loop_avx2_intel(bocpd_kernel_args_t *args);

/**
 * @brief Kernel dispatch macro.
 *
 * Selects kernel based on BOCPD_KERNEL_VARIANT compile flag.
 */
#if BOCPD_KERNEL_VARIANT == BOCPD_KERNEL_INTEL
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_intel(args)
#else
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
#endif

/*=============================================================================
 * Build Instructions
 *=============================================================================
 *
 * Generic kernel (default):
 *   nasm -f elf64 -o bocpd_kernel_generic.o bocpd_kernel_avx2_generic.asm
 *   gcc -O3 -mavx2 -mfma -c bocpd_ultra_opt_asm.c
 *   ar rcs libbocpd_asm.a bocpd_ultra_opt_asm.o bocpd_kernel_generic.o
 *
 * Intel-tuned kernel:
 *   nasm -f elf64 -o bocpd_kernel_intel.o bocpd_kernel_avx2_intel.asm
 *   gcc -DBOCPD_KERNEL_VARIANT=1 -O3 -mavx2 -mfma -c bocpd_ultra_opt_asm.c
 *   ar rcs libbocpd_asm.a bocpd_ultra_opt_asm.o bocpd_kernel_intel.o
 *
 *=============================================================================*/

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ASM_H */