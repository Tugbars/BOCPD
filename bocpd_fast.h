/**
 * @file bocpd_fast.h
 * @brief Header for ultra-optimized BOCPD implementation
 *
 * Bayesian Online Changepoint Detection with:
 * - Normal-Gamma conjugate prior
 * - Student-t posterior predictive
 * - AVX2 SIMD optimization
 * - Ring buffer for O(1) shifts
 */

#ifndef BOCPD_FAST_H
#define BOCPD_FAST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*==============================================================================
 * Prior Parameters
 *==============================================================================*/

/**
 * @brief Normal-Gamma prior parameters.
 *
 * The conjugate prior for Gaussian data with unknown mean and variance.
 *
 * Prior: μ | σ² ~ N(μ₀, σ²/κ₀)
 *        σ²    ~ InvGamma(α₀, β₀)
 */
typedef struct bocpd_prior {
    double mu0;     /**< Prior mean */
    double kappa0;  /**< Prior precision (pseudocount for mean) */
    double alpha0;  /**< Shape parameter (pseudocount for variance) */
    double beta0;   /**< Rate parameter (sum of squared deviations) */
} bocpd_prior_t;

/*==============================================================================
 * Detector State
 *==============================================================================*/

/**
 * @brief BOCPD detector state.
 *
 * All arrays are ring-buffered with power-of-2 capacity for fast modular
 * arithmetic. Arrays are 64-byte aligned for AVX2 operations.
 */
typedef struct bocpd_ultra {
    /*=========================================================================
     * Configuration
     *=========================================================================*/
    size_t capacity;        /**< Max run lengths (power of 2) */
    double hazard;          /**< Hazard rate h = 1/λ */
    double one_minus_h;     /**< Precomputed 1-h */
    double trunc_thresh;    /**< Truncation threshold (default 1e-6) */
    bocpd_prior_t prior;    /**< Prior parameters */

    /*=========================================================================
     * Ring buffer state
     *=========================================================================*/
    size_t ring_start;      /**< Start index in ring buffer */
    size_t active_len;      /**< Number of active run lengths */

    /*=========================================================================
     * Sufficient statistics (ring-buffered)
     *=========================================================================*/
    double *ss_n;           /**< Count per run */
    double *ss_sum;         /**< Sum of observations per run */
    double *ss_sum2;        /**< Sum of squared observations per run */

    /*=========================================================================
     * Posterior parameters (ring-buffered)
     *=========================================================================*/
    double *post_kappa;     /**< Posterior κ */
    double *post_mu;        /**< Posterior μ */
    double *post_alpha;     /**< Posterior α */
    double *post_beta;      /**< Posterior β */

    /*=========================================================================
     * Precomputed Student-t constants (ring-buffered)
     *=========================================================================*/
    double *C1;             /**< Log-pdf constant term */
    double *C2;             /**< Log-pdf coefficient (α + 0.5) */
    double *sigma_sq;       /**< Scale parameter σ² */
    double *inv_sigma_sq_nu;/**< Precomputed 1/(σ²ν) */
    double *lgamma_alpha;   /**< Cached lgamma(α) */
    double *lgamma_alpha_p5;/**< Cached lgamma(α + 0.5) */

    /*=========================================================================
     * Linearized scratch buffers (for SIMD)
     *
     * OPTIMIZATION: Interleaved block layout for better cache utilization.
     * Instead of 4 separate arrays, we use one interleaved buffer:
     *
     *   Block i: [mu[0:3], C1[0:3], C2[0:3], inv_ssn[0:3]]  (128 bytes)
     *   Block i+1: [mu[4:7], C1[4:7], C2[4:7], inv_ssn[4:7]]  (128 bytes)
     *   ...
     *
     * Benefits:
     * - Single memory stream (better prefetching)
     * - 128 bytes = 2 cache lines (perfect alignment)
     * - Reduced TLB pressure
     *=========================================================================*/
    double *lin_interleaved;    /**< Interleaved [mu,C1,C2,inv_ssn] blocks */
    
    /* Legacy pointers - kept for compatibility, point into lin_interleaved */
    double *lin_mu;         /**< Linearized posterior means (legacy) */
    double *lin_C1;         /**< Linearized C1 constants (legacy) */
    double *lin_C2;         /**< Linearized C2 constants (legacy) */
    double *lin_inv_ssn;    /**< Linearized 1/(σ²ν) values (legacy) */

    /*=========================================================================
     * Run-length distribution
     *=========================================================================*/
    double *r;              /**< Current distribution P(r_t | x_{1:t}) */
    double *r_scratch;      /**< Scratch buffer for updates */

    /*=========================================================================
     * Output state
     *=========================================================================*/
    size_t t;               /**< Current timestep */
    size_t map_runlength;   /**< MAP estimate of run length */
    double p_changepoint;   /**< P(r_t < 5) - quick changepoint indicator */

} bocpd_ultra_t;

/*==============================================================================
 * API Functions
 *==============================================================================*/

/**
 * @brief Initialize BOCPD detector.
 *
 * @param b              Detector state to initialize
 * @param hazard_lambda  Expected run length (1/hazard_rate)
 * @param prior          Prior parameters
 * @param max_run_length Maximum run length to track
 *
 * @return 0 on success, -1 on failure
 *
 * @note max_run_length will be rounded up to next power of 2
 * @note All arrays are allocated with 64-byte alignment
 */
int bocpd_ultra_init(bocpd_ultra_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);

/**
 * @brief Free detector resources.
 *
 * @param b Detector state
 */
void bocpd_ultra_free(bocpd_ultra_t *b);

/**
 * @brief Reset detector to initial state.
 *
 * @param b Detector state
 *
 * @note Preserves configuration, clears all observations
 */
void bocpd_ultra_reset(bocpd_ultra_t *b);

/**
 * @brief Process a new observation.
 *
 * @param b Detector state
 * @param x New observation
 *
 * @post b->t incremented
 * @post b->r contains updated run-length distribution
 * @post b->map_runlength contains MAP estimate
 * @post b->p_changepoint contains P(r < 5)
 */
void bocpd_ultra_step(bocpd_ultra_t *b, double x);

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_FAST_H */
