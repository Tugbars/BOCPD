/**
 * @file bocpd_asm.h
 * @brief Ultra-Optimized Bayesian Online Changepoint Detection (BOCPD)
 * @version 2.0
 *
 * @section overview_sec Overview
 *
 * This header defines the interface for a highly optimized BOCPD implementation
 * suitable for real-time streaming applications. Key features:
 *
 * - **Hand-written AVX2 assembly** inner loop for maximum throughput
 * - **Ping-pong double buffering** eliminates all memmove operations
 * - **Fused shift + update** in single pass reduces memory bandwidth
 * - **Single mega-block allocation** for cache-friendly memory layout
 * - **Pool allocator** for efficient multi-stream monitoring
 * - **SIMD lgamma approximation** removes scalar bottlenecks
 *
 * @section perf_sec Performance
 *
 * Measured on Intel Core i7-10700K @ 3.8GHz:
 *
 * | Metric                  | Value           |
 * |-------------------------|-----------------|
 * | Single-detector         | ~800K obs/sec   |
 * | Pool (100 detectors)    | ~1.2M obs/sec   |
 * | Per-observation latency | ~1.2 μs         |
 * | Memory per detector     | ~52 × capacity  |
 *
 * @section usage_sec Quick Start
 *
 * @subsection single_usage Single Detector
 * @code
 * bocpd_asm_t detector;
 * bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
 *
 * bocpd_ultra_init(&detector, 200.0, prior, 1000);
 *
 * for (int i = 0; i < n_obs; i++) {
 *     bocpd_ultra_step(&detector, data[i]);
 *     if (detector.p_changepoint > 0.5) {
 *         printf("Changepoint at t=%zu\n", detector.t);
 *     }
 * }
 *
 * bocpd_ultra_free(&detector);
 * @endcode
 *
 * @subsection pool_usage Pool of Detectors
 * @code
 * bocpd_pool_t pool;
 * bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
 *
 * bocpd_pool_init(&pool, 100, 200.0, prior, 1000);
 *
 * for (int sensor = 0; sensor < 100; sensor++) {
 *     bocpd_asm_t *det = bocpd_pool_get(&pool, sensor);
 *     bocpd_ultra_step(det, sensor_data[sensor]);
 * }
 *
 * bocpd_pool_free(&pool);
 * @endcode
 *
 * @section math_sec Mathematical Background
 *
 * BOCPD maintains a probability distribution over run lengths (time since last
 * changepoint). For each observation x, it computes:
 *
 * @par Predictive Probability (Student-t)
 * \f[
 *   p(x_t | r_{t-1} = i) = \text{Student-t}_{2\alpha}\left(x; \mu, \frac{\beta(\kappa+1)}{\alpha\kappa}\right)
 * \f]
 *
 * @par Run-Length Update
 * \f[
 *   r_t(i+1) = r_{t-1}(i) \cdot p(x_t | r_{t-1}=i) \cdot (1 - H)
 * \f]
 * \f[
 *   r_t(0) = \sum_i r_{t-1}(i) \cdot p(x_t | r_{t-1}=i) \cdot H
 * \f]
 *
 * Where H = 1/λ is the hazard rate.
 *
 * @section refs_sec References
 *
 * - Adams, R. P., & MacKay, D. J. (2007). "Bayesian Online Changepoint Detection"
 * - Murphy, K. P. (2007). "Conjugate Bayesian analysis of the Gaussian distribution"
 */

#ifndef BOCPD_ASM_H
#define BOCPD_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*=============================================================================
     * @defgroup platform Platform Abstraction
     * @brief Cross-platform memory allocation and compiler compatibility
     * @{
     *=============================================================================*/

#ifdef _WIN32
#include <malloc.h>
/** @brief Aligned allocation wrapper for Windows */
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
/** @brief Aligned free wrapper for Windows */
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
/** @brief Aligned free wrapper for POSIX (uses standard free) */
#define aligned_free(ptr) free(ptr)
#endif

/** @} */ /* End of platform group */

/*=============================================================================
 * @defgroup kernel_config Kernel Configuration
 * @brief Assembly kernel variant selection
 * @{
 *=============================================================================*/

/** @brief Generic AVX2 kernel (Windows x64 ABI) */
#define BOCPD_KERNEL_GENERIC 0
/** @brief Intel-optimized kernel variant */
#define BOCPD_KERNEL_INTEL 1

/**
 * @def BOCPD_KERNEL_VARIANT
 * @brief Select assembly kernel variant.
 *
 * Set to BOCPD_KERNEL_GENERIC or BOCPD_KERNEL_INTEL before including this header.
 * Default is BOCPD_KERNEL_GENERIC.
 */
#ifndef BOCPD_KERNEL_VARIANT
#define BOCPD_KERNEL_VARIANT BOCPD_KERNEL_GENERIC
#endif

    /** @} */ /* End of kernel_config group */

    /*=============================================================================
     * @defgroup types Core Types
     * @brief Data structures for BOCPD state management
     * @{
     *=============================================================================*/

    /**
     * @brief Normal-Inverse-Gamma prior hyperparameters.
     *
     * The conjugate prior for a Gaussian with unknown mean and variance:
     * \f[
     *   \mu | \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2/\kappa_0)
     * \f]
     * \f[
     *   \sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
     * \f]
     *
     * @par Choosing Hyperparameters
     *
     * - **mu0**: Expected mean of the data (0 for centered data)
     * - **kappa0**: Confidence in mu0 (1 = weak, 10+ = strong prior)
     * - **alpha0**: Shape parameter (1 = weak, larger = tighter variance prior)
     * - **beta0**: Rate parameter (controls expected variance scale)
     *
     * @par Default Values
     *
     * For uninformative priors on standardized data: {0.0, 1.0, 1.0, 1.0}
     */
    typedef struct bocpd_prior
    {
        double mu0;    /**< Prior mean */
        double kappa0; /**< Prior mean strength (pseudo-observations, > 0) */
        double alpha0; /**< Precision shape parameter (> 0) */
        double beta0;  /**< Precision rate parameter (> 0) */
    } bocpd_prior_t;

    /**
     * @brief BOCPD detector state with ping-pong double buffering.
     *
     * This structure holds all state for a single BOCPD detector:
     * - Configuration parameters
     * - Double-buffered posterior arrays (for ping-pong optimization)
     * - Run-length distribution
     * - Output statistics
     *
     * @par Ping-Pong Buffering
     *
     * All posterior arrays are doubled: `array[0]` and `array[1]`.
     * The `cur_buf` field indicates which is the current read buffer.
     * Updates read from `cur_buf`, write to `1 - cur_buf`, then swap.
     * This eliminates all memmove operations.
     *
     * @par Memory Layout
     *
     * The `mega` pointer holds a single contiguous allocation containing:
     * - Interleaved SIMD buffer
     * - All double-buffered arrays (26 total)
     * - Run-length distribution and scratch buffer
     *
     * @warning Do not modify fields directly. Use the API functions.
     */
    typedef struct bocpd_asm
    {
        /*--- Configuration (set at initialization, read-only thereafter) ---*/
        size_t capacity;     /**< Maximum run lengths supported */
        double hazard;       /**< Hazard rate H = 1/λ */
        double one_minus_h;  /**< Precomputed 1 - H for efficiency */
        double trunc_thresh; /**< Truncation threshold (default 1e-6) */
        bocpd_prior_t prior; /**< Prior hyperparameters */

        /*--- Precomputed prior lgamma values (avoids lgamma in hot path) ---*/
        double prior_lgamma_alpha;    /**< lgamma(alpha0) */
        double prior_lgamma_alpha_p5; /**< lgamma(alpha0 + 0.5) */

        /*--- Ping-pong buffer state ---*/
        int cur_buf;       /**< Current buffer index: 0 or 1 */
        size_t active_len; /**< Number of active (non-truncated) run lengths */

        /*--- Double-buffered sufficient statistics ---*/
        double *ss_n[2];    /**< Sample count n for each run length */
        double *ss_sum[2];  /**< Sum Σx for each run length */
        double *ss_sum2[2]; /**< Sum of squares Σx² for each run length */

        /*--- Double-buffered posterior parameters ---*/
        double *post_kappa[2]; /**< Posterior κ (pseudo-count) */
        double *post_mu[2];    /**< Posterior mean μ */
        double *post_alpha[2]; /**< Posterior shape α */
        double *post_beta[2];  /**< Posterior rate β */

        /*--- Double-buffered Student-t precomputed constants ---*/
        double *C1[2];              /**< Log normalization constant */
        double *C2[2];              /**< Exponent = α + 0.5 */
        double *sigma_sq[2];        /**< Scale σ² = β(κ+1)/(ακ) */
        double *inv_sigma_sq_nu[2]; /**< Precomputed 1/(σ²ν) */
        double *lgamma_alpha[2];    /**< lgamma(α) cached */
        double *lgamma_alpha_p5[2]; /**< lgamma(α + 0.5) cached */

        /*--- Interleaved buffer for SIMD kernel (rebuilt each step) ---*/
        double *lin_interleaved; /**< [μ×4, C1×4, C2×4, inv×4] blocks */

        /*--- Run-length distribution (single-buffered, kernel uses scratch) ---*/
        double *r;         /**< Run-length distribution P(r_t = i) */
        double *r_scratch; /**< Working buffer for updates */

        /*--- Output state (updated after each observation) ---*/
        size_t t;             /**< Observation count */
        size_t map_runlength; /**< MAP estimate: argmax_i r[i] */
        double p_changepoint; /**< P(changepoint) ≈ r[0] + r[1] + ... */

        /*--- Memory management (for bocpd_ultra_free) ---*/
        void *mega;        /**< Single contiguous allocation */
        size_t mega_bytes; /**< Size of mega allocation */

    } bocpd_asm_t;

/** @} */ /* End of types group */

/*=============================================================================
 * @defgroup macros Buffer Access Macros
 * @brief Convenient access to ping-pong buffers
 * @{
 *=============================================================================*/

/**
 * @def BOCPD_CUR(b, arr)
 * @brief Access the current (read) buffer for array `arr`.
 *
 * @param b   Pointer to bocpd_asm_t
 * @param arr Array name (e.g., post_mu, post_alpha)
 * @return Pointer to current buffer of that array
 *
 * Example: `double *mu = BOCPD_CUR(b, post_mu);`
 */
#define BOCPD_CUR(b, arr) ((b)->arr[(b)->cur_buf])

/**
 * @def BOCPD_NEXT(b, arr)
 * @brief Access the next (write) buffer for array `arr`.
 *
 * @param b   Pointer to bocpd_asm_t
 * @param arr Array name
 * @return Pointer to next buffer (will become current after swap)
 *
 * Example: `BOCPD_NEXT(b, post_mu)[i+1] = new_mu;`
 */
#define BOCPD_NEXT(b, arr) ((b)->arr[1 - (b)->cur_buf])

    /** @} */ /* End of macros group */

    /*=============================================================================
     * @defgroup pool_types Pool Allocator Types
     * @brief Structures for managing multiple detectors efficiently
     * @{
     *=============================================================================*/

    /**
     * @brief Pool of BOCPD detectors with shared memory allocation.
     *
     * For applications monitoring many streams (100+ sensors, instruments, etc.),
     * the pool allocator provides:
     *
     * - **Single allocation**: All detectors share one contiguous memory block
     * - **Fast initialization**: ~100× faster than individual allocations
     * - **Zero fragmentation**: No per-detector malloc overhead
     * - **Cache efficiency**: Sequential memory layout aids prefetching
     *
     * @note Pool-allocated detectors should NOT be freed individually.
     *       Use bocpd_pool_free() to release the entire pool.
     */
    typedef struct bocpd_pool
    {
        void *pool;                /**< Single mega-allocation for all data */
        size_t pool_size;          /**< Total bytes allocated */
        bocpd_asm_t *detectors;    /**< Array of detector structs */
        size_t n_detectors;        /**< Number of detectors in pool */
        size_t bytes_per_detector; /**< Bytes of data per detector */
    } bocpd_pool_t;

    /** @} */ /* End of pool_types group */

    /*=============================================================================
     * @defgroup single_api Single Detector API
     * @brief Functions for managing individual BOCPD detectors
     * @{
     *=============================================================================*/

    /**
     * @brief Initialize a BOCPD detector.
     *
     * @param b               Detector to initialize
     * @param hazard_lambda   Expected run length λ (hazard rate = 1/λ)
     * @param prior           Prior hyperparameters
     * @param max_run_length  Maximum run lengths to track
     *
     * @return 0 on success, -1 on failure (invalid params or allocation)
     *
     * @see bocpd_ultra_free to release resources
     */
    int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                         bocpd_prior_t prior, size_t max_run_length);

    /**
     * @brief Free all resources associated with a detector.
     *
     * @param b Detector to free (may be NULL)
     */
    void bocpd_ultra_free(bocpd_asm_t *b);

    /**
     * @brief Reset detector to initial state without reallocating.
     *
     * @param b Detector to reset
     *
     * Use for processing multiple independent streams with the same detector.
     */
    void bocpd_ultra_reset(bocpd_asm_t *b);

    /**
     * @brief Process a single observation.
     *
     * @param b Detector state
     * @param x New observation value
     *
     * After this call, check:
     * - `b->p_changepoint` for changepoint probability
     * - `b->map_runlength` for MAP run length estimate
     * - `b->r[i]` for full run-length distribution
     */
    void bocpd_ultra_step(bocpd_asm_t *b, double x);

    /**
     * @brief Get MAP (most likely) run length.
     * @param b Detector state
     * @return Run length with highest probability
     */
    static inline size_t bocpd_ultra_get_map(const bocpd_asm_t *b)
    {
        return b->map_runlength;
    }

    /**
     * @brief Get number of observations processed.
     * @param b Detector state
     * @return Observation count
     */
    static inline size_t bocpd_ultra_get_t(const bocpd_asm_t *b)
    {
        return b->t;
    }

    /**
     * @brief Get changepoint probability.
     * @param b Detector state
     * @return Sum of probabilities for small run lengths (≈ P(recent changepoint))
     */
    static inline double bocpd_ultra_get_change_prob(const bocpd_asm_t *b)
    {
        return b->p_changepoint;
    }

    /** @} */ /* End of single_api group */

    /*=============================================================================
     * @defgroup pool_api Pool Allocator API
     * @brief Functions for managing pools of detectors
     * @{
     *=============================================================================*/

    /**
     * @brief Initialize a pool of BOCPD detectors.
     *
     * @param pool           Pool to initialize
     * @param n_detectors    Number of detectors
     * @param hazard_lambda  Expected run length λ (shared by all)
     * @param prior          Prior parameters (shared by all)
     * @param max_run_length Maximum run length per detector
     *
     * @return 0 on success, -1 on failure
     */
    int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                        double hazard_lambda, bocpd_prior_t prior,
                        size_t max_run_length);

    /**
     * @brief Free all pool resources.
     * @param pool Pool to free (may be NULL)
     */
    void bocpd_pool_free(bocpd_pool_t *pool);

    /**
     * @brief Reset all detectors in the pool.
     * @param pool Pool to reset
     */
    void bocpd_pool_reset(bocpd_pool_t *pool);

    /**
     * @brief Get a detector from the pool.
     * @param pool  Pool handle
     * @param index Detector index (0 to n_detectors-1)
     * @return Detector pointer, or NULL if index out of range
     */
    bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index);

    /** @} */ /* End of pool_api group */

    /*=============================================================================
     * @defgroup asm_kernel Assembly Kernel Interface
     * @brief Low-level interface to AVX2 assembly kernel (internal use)
     * @{
     *=============================================================================*/

    /**
     * @brief Arguments structure for assembly kernel.
     *
     * @internal This structure packages all inputs/outputs for the assembly kernel.
     *           Not intended for direct use by application code.
     */
    typedef struct bocpd_kernel_args
    {
        const double *lin_interleaved; /**< Interleaved parameter buffer */
        const double *r_old;           /**< Input run-length distribution */
        double x;                      /**< Current observation */
        double h;                      /**< Hazard rate */
        double one_minus_h;            /**< 1 - hazard rate */
        double trunc_thresh;           /**< Truncation threshold */
        size_t n_padded;               /**< Padded length (multiple of 8) */
        double *r_new;                 /**< Output run-length distribution */
        double *r0_out;                /**< Output: changepoint probability */
        double *max_growth_out;        /**< Output: maximum growth probability */
        size_t *max_idx_out;           /**< Output: index of maximum */
        size_t *last_valid_out;        /**< Output: last index above threshold */
    } bocpd_kernel_args_t;

    /** @brief Generic AVX2 assembly kernel (Windows x64 ABI) */
    extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);

    /** @brief Intel-optimized AVX2 assembly kernel */
    extern void bocpd_fused_loop_avx2_intel(bocpd_kernel_args_t *args);

/** @brief Dispatch macro to selected kernel variant */
#if BOCPD_KERNEL_VARIANT == BOCPD_KERNEL_INTEL
#define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_intel(args)
#else
#define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
#endif

    /** @} */ /* End of asm_kernel group */

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ASM_H */