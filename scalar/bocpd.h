/**
 * @file bocpd_scalar_opt.h
 * @brief Optimized Scalar BOCPD Implementation (No SIMD)
 *
 * Portable fallback for systems without AVX2. Uses the same algorithmic
 * optimizations as the ultra-fast version:
 *
 *   - Ring buffer (O(1) shifts)
 *   - Incremental lgamma via recurrence
 *   - Precomputed Student-t constants
 *   - Fast scalar log/exp approximations
 *
 * Performance: ~50-80K obs/sec
 * Accuracy: Matches SIMD version to ~1e-7 relative error
 */

#ifndef BOCPD_SCALAR_OPT_H
#define BOCPD_SCALAR_OPT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Prior Parameters
 *=============================================================================*/

/**
 * @brief Normal-Gamma prior parameters.
 */
typedef struct bocpd_scalar_prior {
    double mu0;     /**< Prior mean */
    double kappa0;  /**< Prior mean strength (pseudo-observations) */
    double alpha0;  /**< Precision shape (> 0) */
    double beta0;   /**< Precision rate (> 0) */
} bocpd_scalar_prior_t;

/*=============================================================================
 * Detector State
 *=============================================================================*/

/**
 * @brief BOCPD detector state (optimized scalar version).
 */
typedef struct bocpd_scalar {
    /*-------------------------------------------------------------------------
     * Configuration
     *-------------------------------------------------------------------------*/
    size_t capacity;            /**< Max run lengths (power of 2) */
    double hazard;              /**< Hazard rate h = 1/λ */
    double one_minus_h;         /**< Precomputed 1-h */
    double trunc_thresh;        /**< Truncation threshold */
    bocpd_scalar_prior_t prior; /**< Prior parameters */

    /*-------------------------------------------------------------------------
     * Ring buffer state
     *-------------------------------------------------------------------------*/
    size_t ring_start;          /**< Start index in ring buffer */
    size_t active_len;          /**< Number of active run lengths */

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

} bocpd_scalar_t;

/*=============================================================================
 * Public API
 *=============================================================================*/

/**
 * @brief Initialize BOCPD detector.
 *
 * @param b              Detector to initialize
 * @param hazard_lambda  Expected run length λ (hazard = 1/λ)
 * @param prior          Prior parameters
 * @param max_run_length Maximum run length (rounded up to power of 2)
 *
 * @return 0 on success, -1 on failure
 */
int bocpd_scalar_init(bocpd_scalar_t *b, double hazard_lambda,
                      bocpd_scalar_prior_t prior, size_t max_run_length);

/**
 * @brief Free detector resources.
 */
void bocpd_scalar_free(bocpd_scalar_t *b);

/**
 * @brief Reset detector to initial state.
 */
void bocpd_scalar_reset(bocpd_scalar_t *b);

/**
 * @brief Process one observation.
 *
 * @param b Detector state
 * @param x New observation
 */
void bocpd_scalar_step(bocpd_scalar_t *b, double x);

/**
 * @brief Get probability of recent changepoint.
 *
 * @param b      Detector state
 * @param window Run lengths to consider as "recent"
 *
 * @return P(run_length < window)
 */
double bocpd_scalar_change_prob(const bocpd_scalar_t *b, size_t window);

/**
 * @brief Get MAP run length estimate.
 */
static inline size_t bocpd_scalar_get_map(const bocpd_scalar_t *b) {
    return b->map_runlength;
}

/**
 * @brief Get current timestep.
 */
static inline size_t bocpd_scalar_get_t(const bocpd_scalar_t *b) {
    return b->t;
}

/**
 * @brief Get quick changepoint probability P(run_length < 5).
 */
static inline double bocpd_scalar_get_change_prob(const bocpd_scalar_t *b) {
    return b->p_changepoint;
}

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_SCALAR_OPT_H */