/**
 * @file bocpd_asm.h
 * @brief AVX2 Assembly-Optimized BOCPD Implementation (Ping-Pong Buffering)
 *
 * Ultra-fast Bayesian Online Changepoint Detection with:
 *   - Hand-written AVX2 assembly inner loop
 *   - Ping-pong double buffering (eliminates memmove)
 *   - Fused shift + update in single pass
 *   - Single mega-block allocation per detector
 *   - Pool allocator for multiple detectors
 *   - Precomputed prior lgamma values
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
 * Detector State (Ping-Pong Double Buffering)
 *
 * All posterior arrays are double-buffered [0] and [1].
 * cur_buf indicates which buffer is current (0 or 1).
 * Updates read from cur, write to next (1 - cur_buf), then swap.
 * This eliminates all memmove operations.
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
    
    /*--- Ping-pong buffer state ---*/
    int cur_buf;              /**< Current buffer index: 0 or 1 */
    size_t active_len;        /**< Number of active run lengths */
    
    /*--- Double-buffered sufficient statistics ---*/
    double *ss_n[2];
    double *ss_sum[2];
    double *ss_sum2[2];
    
    /*--- Double-buffered posterior parameters ---*/
    double *post_kappa[2];
    double *post_mu[2];
    double *post_alpha[2];
    double *post_beta[2];
    
    /*--- Double-buffered Student-t precomputed constants ---*/
    double *C1[2];
    double *C2[2];
    double *sigma_sq[2];
    double *inv_sigma_sq_nu[2];
    double *lgamma_alpha[2];
    double *lgamma_alpha_p5[2];
    
    /*--- Interleaved buffer for SIMD kernel (single, rebuilt each step) ---*/
    double *lin_interleaved;
    
    /*--- Run-length distribution (single, kernel uses scratch internally) ---*/
    double *r;
    double *r_scratch;
    
    /*--- Output state ---*/
    size_t t;
    size_t map_runlength;
    double p_changepoint;
    
    /*--- Memory management ---*/
    void *mega;
    size_t mega_bytes;
    
} bocpd_asm_t;

/*=============================================================================
 * Buffer Access Macros
 *
 * CUR(arr)  - Access current buffer for reading
 * NEXT(arr) - Access next buffer for writing
 *=============================================================================*/

#define BOCPD_CUR(b, arr)  ((b)->arr[(b)->cur_buf])
#define BOCPD_NEXT(b, arr) ((b)->arr[1 - (b)->cur_buf])

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
 */
int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length);

void bocpd_pool_free(bocpd_pool_t *pool);

void bocpd_pool_reset(bocpd_pool_t *pool);

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