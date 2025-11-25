/**
 * @file bocpd_asm.h
 * @brief AVX2 Assembly-Optimized BOCPD Implementation
 *
 * Ultra-fast Bayesian Online Changepoint Detection with:
 *   - Hand-written AVX2 assembly inner loop
 *   - Single mega-block allocation per detector
 *   - Pool allocator for multiple detectors
 *   - Precomputed prior lgamma values
 *   - Permanent interleaved layout (no copy overhead)
 */

#ifndef BOCPD_ASM_H
#define BOCPD_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Platform-Specific Aligned Allocation
 *=============================================================================*/

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
#define aligned_free(ptr) free(ptr)
#endif

/*=============================================================================
 * Kernel Variant Selection
 *=============================================================================*/

#define BOCPD_KERNEL_GENERIC    0
#define BOCPD_KERNEL_INTEL      1

#ifndef BOCPD_KERNEL_VARIANT
    #define BOCPD_KERNEL_VARIANT BOCPD_KERNEL_GENERIC
#endif

/*=============================================================================
 * Prior Parameters
 *=============================================================================*/

typedef struct bocpd_prior {
    double mu0;     /**< Prior mean */
    double kappa0;  /**< Prior mean strength (pseudo-observations) */
    double alpha0;  /**< Precision shape (> 0) */
    double beta0;   /**< Precision rate (> 0) */
} bocpd_prior_t;

/*=============================================================================
 * Detector State
 *=============================================================================*/

typedef struct bocpd_asm {
    /*--- Configuration ---*/
    size_t capacity;
    double hazard;
    double one_minus_h;
    double trunc_thresh;
    bocpd_prior_t prior;
    
    /*--- Precomputed prior lgamma values (avoids lgamma in hot path) ---*/
    double prior_lgamma_alpha;
    double prior_lgamma_alpha_p5;
    
    /*--- Ring buffer state ---*/
    size_t ring_start;
    size_t active_len;
    
    /*--- Sufficient statistics ---*/
    double *ss_n;
    double *ss_sum;
    double *ss_sum2;
    
    /*--- Posterior parameters ---*/
    double *post_kappa;
    double *post_mu;
    double *post_alpha;
    double *post_beta;
    
    /*--- Student-t precomputed constants ---*/
    double *C1;
    double *C2;
    double *sigma_sq;
    double *inv_sigma_sq_nu;
    double *lgamma_alpha;
    double *lgamma_alpha_p5;
    
    /*--- Interleaved buffer for SIMD kernel ---*/
    double *lin_interleaved;
    
    /*--- Legacy linear views (unused, kept for compatibility) ---*/
    double *lin_mu;
    double *lin_C1;
    double *lin_C2;
    double *lin_inv_ssn;
    
    /*--- Run-length distribution ---*/
    double *r;
    double *r_scratch;
    
    /*--- Output state ---*/
    size_t t;
    size_t map_runlength;
    double p_changepoint;
    
    /*--- Memory management ---*/
    void *block;
    void *mega;
    size_t mega_bytes;
    
} bocpd_asm_t;

/*=============================================================================
 * Pool Allocator for Multiple Detectors
 *
 * For applications with many detectors (e.g., 100+ instruments),
 * the pool allocator provides:
 *   - Single malloc for all detectors
 *   - ~100x faster initialization
 *   - Zero heap fragmentation
 *   - Single free for cleanup
 *=============================================================================*/

typedef struct bocpd_pool {
    void *pool;                     /**< Single mega-allocation */
    size_t pool_size;               /**< Total bytes allocated */
    bocpd_asm_t *detectors;         /**< Array of detector structs */
    size_t n_detectors;             /**< Number of detectors */
    size_t bytes_per_detector;      /**< Bytes per detector data block */
} bocpd_pool_t;

/*=============================================================================
 * Single Detector API
 *=============================================================================*/

int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);

void bocpd_ultra_free(bocpd_asm_t *b);

void bocpd_ultra_reset(bocpd_asm_t *b);

void bocpd_ultra_step(bocpd_asm_t *b, double x);

static inline size_t bocpd_ultra_get_map(const bocpd_asm_t *b) {
    return b->map_runlength;
}

static inline size_t bocpd_ultra_get_t(const bocpd_asm_t *b) {
    return b->t;
}

static inline double bocpd_ultra_get_change_prob(const bocpd_asm_t *b) {
    return b->p_changepoint;
}

/*=============================================================================
 * Pool Allocator API
 *=============================================================================*/

/**
 * @brief Initialize a pool of BOCPD detectors.
 *
 * @param pool           Pool to initialize
 * @param n_detectors    Number of detectors
 * @param hazard_lambda  Expected run length λ (hazard = 1/λ)
 * @param prior          Prior parameters (shared by all detectors)
 * @param max_run_length Maximum run length to track
 *
 * @return 0 on success, -1 on failure
 *
 * This allocates a single contiguous memory block for all detectors.
 * Initialization time: ~1-2ms for 100 detectors (vs ~300ms with individual allocs)
 */
int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length);

/**
 * @brief Free pool resources.
 *
 * Single free() call for all detectors.
 * Do NOT call bocpd_ultra_free() on pool-managed detectors.
 */
void bocpd_pool_free(bocpd_pool_t *pool);

/**
 * @brief Reset all detectors in pool to initial state.
 */
void bocpd_pool_reset(bocpd_pool_t *pool);

/**
 * @brief Get detector by index.
 *
 * @param pool   Pool
 * @param index  Detector index (0 to n_detectors-1)
 * @return Pointer to detector, or NULL if invalid
 */
bocpd_asm_t* bocpd_pool_get(bocpd_pool_t *pool, size_t index);

/*=============================================================================
 * Assembly Kernel Interface
 *=============================================================================*/

typedef struct bocpd_kernel_args {
    const double *lin_interleaved;
    const double *r_old;
    double x;
    double h;
    double one_minus_h;
    double trunc_thresh;
    size_t n_padded;
    double *r_new;
    double *r0_out;
    double *max_growth_out;
    size_t *max_idx_out;
    size_t *last_valid_out;
} bocpd_kernel_args_t;

extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);
extern void bocpd_fused_loop_avx2_intel(bocpd_kernel_args_t *args);

#if BOCPD_KERNEL_VARIANT == BOCPD_KERNEL_INTEL
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_intel(args)
#else
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
#endif

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ASM_H */