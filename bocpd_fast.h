/**
 * @file bocpd_ultra.h
 * @brief Ultra-optimized Bayesian Online Change Point Detection (BOCPD)
 * 
 * @details This implementation is based on the Adams & MacKay (2007) paper
 * "Bayesian Online Changepoint Detection" with extensive optimizations for
 * high-frequency trading applications requiring sub-millisecond latency.
 * 
 * ## Algorithm Overview
 * 
 * BOCPD maintains a probability distribution over "run lengths" - the time
 * since the last change point. At each timestep t, given observation x_t:
 * 
 * 1. **Predictive Step**: For each run length r, compute P(x_t | x_{t-r:t-1})
 *    using the posterior predictive distribution (Student-t for Normal-Gamma)
 * 
 * 2. **Growth Step**: P(r_t = r | x_{1:t}) ∝ P(r_{t-1} = r-1) · P(x_t | run=r-1) · (1-H)
 *    where H is the hazard function (probability of changepoint)
 * 
 * 3. **Changepoint Step**: P(r_t = 0 | x_{1:t}) ∝ Σ_r P(r_{t-1} = r) · P(x_t | run=r) · H
 * 
 * 4. **Normalization**: Ensure Σ_r P(r_t = r) = 1
 * 
 * ## Mathematical Foundation
 * 
 * ### Normal-Gamma Conjugate Prior
 * 
 * For Gaussian data with unknown mean μ and precision τ = 1/σ²:
 * 
 * Prior: p(μ, τ) = Normal(μ | μ₀, (κ₀τ)⁻¹) · Gamma(τ | α₀, β₀)
 * 
 * Posterior after n observations with sufficient stats (n, Σx, Σx²):
 * - κ_n = κ₀ + n
 * - μ_n = (κ₀μ₀ + Σx) / κ_n
 * - α_n = α₀ + n/2
 * - β_n = β₀ + ½(Σx² - n·x̄²) + (κ₀n(x̄ - μ₀)²) / (2κ_n)
 * 
 * ### Posterior Predictive (Student-t)
 * 
 * p(x | data) = Student-t(x | μ_n, σ²_n, ν_n) where:
 * - ν_n = 2α_n (degrees of freedom)
 * - σ²_n = β_n(κ_n + 1) / (α_n · κ_n) (scale parameter)
 * 
 * Log-density: ln p(x) = C₁ - C₂ · ln(1 + z²/ν)
 * where:
 * - z = (x - μ_n) / σ_n
 * - C₁ = lgamma((ν+1)/2) - lgamma(ν/2) - ½ln(νπ) - ½ln(σ²)
 * - C₂ = (ν + 1) / 2
 * 
 * ## Optimizations Applied
 * 
 * | Optimization | Description | Speedup |
 * |--------------|-------------|---------|
 * | Ring buffer + linearization | O(1) shift, contiguous SIMD loads | ~2× |
 * | Precomputed C₁, C₂ | Eliminates lgamma/log from hot loop | ~30% |
 * | Fused SIMD kernel | Single pass, no scalar remainder | ~15% |
 * | Fast log1p polynomial | Direct series for small t | ~20% |
 * | Fast exp polynomial | Fully vectorized 2^k scaling | ~20% |
 * | 2× loop unrolling | Better ILP for polynomial chains | ~20% |
 * | Branchless MAP/truncation | SIMD blend operations | ~5% |
 * | Incremental lgamma | Recurrence: lgamma(a+½) = lgamma(a) + ln(a) | ~10% |
 * 
 * ## Usage Example
 * 
 * @code{.c}
 * bocpd_ultra_t cpd;
 * bocpd_prior_t prior = {0.0, 0.01, 0.5, 0.0001};  // Weakly informative
 * 
 * bocpd_ultra_init(&cpd, 200.0, prior, 512);  // λ=200, max_run=512
 * 
 * for (size_t t = 0; t < n_observations; t++) {
 *     bocpd_ultra_step(&cpd, returns[t]);
 *     
 *     double p_change = bocpd_ultra_change_prob(&cpd, 5);
 *     if (p_change > 0.3) {
 *         // Recent regime change detected
 *         adjust_strategy();
 *     }
 * }
 * 
 * bocpd_ultra_free(&cpd);
 * @endcode
 * 
 * ## Performance Characteristics
 * 
 * - **Time Complexity**: O(active_len) per observation
 * - **Space Complexity**: O(capacity) with ~18 arrays
 * - **Typical Latency**: 2-5 µs for active_len < 200 on modern x86-64
 * - **Throughput**: ~200K observations/second single-threaded
 * 
 * ## References
 * 
 * [1] Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
 *     arXiv:0710.3742
 * 
 * [2] Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution.
 *     Technical report, University of British Columbia.
 * 
 * @author TUGBARS
 * @version 2.1
 * @date 2024
 */

#ifndef BOCPD_ULTRA_H
#define BOCPD_ULTRA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Normal-Gamma prior parameters for the conjugate Bayesian model.
 * 
 * @details The Normal-Gamma distribution is the conjugate prior for a Gaussian
 * likelihood with unknown mean and variance. The joint prior is:
 * 
 *     p(μ, τ) = N(μ | μ₀, (κ₀τ)⁻¹) · Ga(τ | α₀, β₀)
 * 
 * where τ = 1/σ² is the precision.
 * 
 * ### Parameter Interpretation
 * 
 * - **μ₀**: Prior belief about the mean
 * - **κ₀**: "Pseudo-observations" for mean (higher = stronger prior)
 * - **α₀**: Shape for precision prior (α₀ > 0)
 * - **β₀**: Rate for precision prior (β₀ > 0)
 * 
 * ### Prior Variance
 * 
 * The prior expected variance is E[σ²] = β₀ / (α₀ - 1) for α₀ > 1.
 * 
 * ### Recommended Settings for Financial Returns
 * 
 * For daily returns (typically σ ≈ 0.01-0.02):
 * @code
 * bocpd_prior_t prior = {
 *     .mu0 = 0.0,       // Centered at zero
 *     .kappa0 = 0.01,   // Very weak mean prior
 *     .alpha0 = 2.0,    // Allows variance estimation
 *     .beta0 = 0.0002   // E[σ²] ≈ 0.0002, σ ≈ 1.4%
 * };
 * @endcode
 * 
 * For tick-level returns (higher frequency):
 * @code
 * bocpd_prior_t prior = {
 *     .mu0 = 0.0,
 *     .kappa0 = 0.001,
 *     .alpha0 = 1.5,
 *     .beta0 = 1e-6
 * };
 * @endcode
 */
typedef struct {
    double mu0;     /**< Prior mean. Typically 0 for financial returns. */
    double kappa0;  /**< Prior mean strength. Small (0.001-0.1) = weak prior. */
    double alpha0;  /**< Precision shape. Must be > 0. Typically 0.5-5. */
    double beta0;   /**< Precision rate. Scale to expected variance. */
} bocpd_prior_t;

/**
 * @brief Main BOCPD state structure with all precomputed quantities.
 * 
 * @details Memory layout is carefully optimized for AVX2 SIMD operations:
 * 
 * ### Memory Alignment
 * - All arrays are 64-byte aligned (cache line boundary)
 * - Capacity is always power of 2 for fast modular arithmetic: `i & (cap-1)`
 * - Padding ensures SIMD loops have no scalar remainder
 * 
 * ### Ring Buffer Design
 * 
 * Instead of shifting all arrays on each observation (O(n)), we use a ring
 * buffer with a moving `ring_start` pointer. Index mapping:
 * 
 *     physical_index = (ring_start + logical_index) & (capacity - 1)
 * 
 * Before SIMD processing, ring data is linearized into scratch buffers
 * to enable contiguous `_mm256_load_pd` instead of gather operations.
 * 
 * ### Precomputed Constants
 * 
 * The Student-t log-pdf can be written as:
 * 
 *     ln p(x) = C₁ - C₂ · ln(1 + z²/ν)
 * 
 * where C₁ and C₂ depend only on the posterior parameters (not x).
 * By precomputing these during posterior updates, the hot loop reduces to:
 * - 1 subtraction (x - μ)
 * - 1 multiplication (z²)
 * - 1 multiplication (z²/ν via precomputed 1/(σ²ν))
 * - 1 log1p call
 * - 1 FMA (C₁ - C₂ · log1p)
 * - 1 exp call
 * 
 * ### Memory Usage
 * 
 * Total memory ≈ 18 × capacity × 8 bytes + overhead
 * For capacity = 512: ~73 KB
 * For capacity = 2048: ~295 KB
 */
typedef struct {
    /*=========================================================================
     * Configuration
     *=========================================================================*/
    
    /**
     * @brief Hazard rate h = 1/λ.
     * 
     * Probability of a changepoint at any given time step.
     * - λ = 100 → expect changepoints every ~100 observations
     * - λ = 500 → expect changepoints every ~500 observations
     * 
     * For trading: λ = 100-500 is typical for daily data,
     * λ = 1000-5000 for tick data.
     */
    double hazard;
    
    /** @brief Precomputed (1 - h) for the growth probability term. */
    double one_minus_h;
    
    /**
     * @brief Truncation threshold for run-length pruning.
     * 
     * Run lengths with probability < thresh are dropped to bound computation.
     * Typical value: 1e-6 to 1e-8. Lower = more accurate, higher = faster.
     */
    double trunc_thresh;
    
    /** @brief Normal-Gamma prior parameters (immutable after init). */
    bocpd_prior_t prior;

    /*=========================================================================
     * Capacity Management
     *=========================================================================*/
    
    /** @brief Maximum run length capacity (always power of 2). */
    size_t capacity;
    
    /** @brief Current number of active (non-truncated) run lengths. */
    size_t active_len;
    
    /** @brief Ring buffer head index. Advances by 1 each observation. */
    size_t ring_start;

    /*=========================================================================
     * Ring-Buffered Sufficient Statistics
     * 
     * For Gaussian data, the sufficient statistics (n, Σx, Σx²) fully
     * determine the posterior. These are updated in O(active_len) time
     * by adding x to each run's statistics.
     *=========================================================================*/
    
    double *ss_n;       /**< Count of observations in each run: n */
    double *ss_sum;     /**< Sum of observations: Σx */
    double *ss_sum2;    /**< Sum of squares: Σx² (for variance estimation) */

    /*=========================================================================
     * Ring-Buffered Posterior Parameters
     * 
     * Updated via Normal-Gamma conjugate update rules. The key insight
     * is that these can be updated incrementally using Welford's algorithm:
     * 
     *   κ_new = κ_old + 1
     *   μ_new = (κ_old · μ_old + x) / κ_new
     *   α_new = α_old + 0.5
     *   β_new = β_old + 0.5 · (x - μ_old) · (x - μ_new)
     * 
     * The β update is numerically stable (Welford's online variance).
     *=========================================================================*/
    
    double *post_kappa;     /**< Posterior κ = κ₀ + n */
    double *post_mu;        /**< Posterior μ = (κ₀μ₀ + Σx) / κ */
    double *post_alpha;     /**< Posterior α = α₀ + n/2 */
    double *post_beta;      /**< Posterior β (Welford update) */

    /*=========================================================================
     * Precomputed Student-t Constants
     * 
     * The posterior predictive is Student-t with:
     *   ν = 2α (degrees of freedom)
     *   σ² = β(κ+1) / (ακ) (scale)
     * 
     * Log-density: ln p(x) = C₁ - C₂ · log(1 + z²/ν)
     * 
     * By precomputing C₁, C₂, and 1/(σ²ν), the hot loop avoids:
     * - lgamma calls (moved to incremental update)
     * - Multiple log calls (folded into C₁)
     * - Division (precomputed as multiplication by inverse)
     *=========================================================================*/
    
    /**
     * @brief Student-t constant C₁ = lgamma(α+½) - lgamma(α) - ½ln(νπ) - ½ln(σ²)
     * 
     * This absorbs all the "constant" terms in the Student-t log-pdf,
     * leaving only the data-dependent log(1 + z²/ν) term for the hot loop.
     */
    double *C1;
    
    /**
     * @brief Student-t constant C₂ = α + ½ = (ν+1)/2
     * 
     * The exponent in the Student-t density: (1 + z²/ν)^{-(ν+1)/2}
     */
    double *C2;
    
    /**
     * @brief Predictive variance σ² = β(κ+1) / (ακ)
     * 
     * Stored for debugging/inspection. Not used directly in hot loop
     * (we use inv_sigma_sq_nu instead).
     */
    double *sigma_sq;
    
    /**
     * @brief Precomputed 1/(σ²ν) for fast z²/ν calculation.
     * 
     * In the hot loop: t = z² · inv_sigma_sq_nu = z² / (σ²ν)
     * This converts a division to a multiplication.
     */
    double *inv_sigma_sq_nu;

    /*=========================================================================
     * Linearized Scratch Buffers
     * 
     * Ring buffer data is copied here before SIMD processing. This enables:
     * - Contiguous _mm256_load_pd instead of gather instructions
     * - Predictable memory access patterns
     * - Elimination of ring index calculations in hot loop
     * 
     * Cost: O(active_len) memcpy per observation
     * Benefit: ~2× faster SIMD loop (loads are 4× faster than gathers)
     *=========================================================================*/
    
    double *lin_mu;         /**< Linearized posterior means */
    double *lin_C1;         /**< Linearized C₁ constants */
    double *lin_C2;         /**< Linearized C₂ constants */
    double *lin_inv_ssn;    /**< Linearized 1/(σ²ν) values */

    /*=========================================================================
     * Run-Length Distribution
     * 
     * r[i] = P(run_length = i | observations)
     * 
     * This is the core output of BOCPD. After each step:
     * - r[0] = probability of changepoint at current time
     * - r[k] = probability that last changepoint was k steps ago
     * - Σᵢ r[i] = 1 (normalized)
     *=========================================================================*/
    
    double *r;              /**< Current run-length distribution */
    double *r_scratch;      /**< Scratch buffer for update (avoids in-place issues) */

    /*=========================================================================
     * Incremental lgamma Tracking
     * 
     * lgamma is expensive (~50-100 cycles). We use the recurrence:
     *   lgamma(a + 0.5) = lgamma(a) + ln(a)
     * 
     * Since α increases by 0.5 each observation, we can update lgamma
     * incrementally with just one log call instead of one lgamma call.
     *=========================================================================*/
    
    double *lgamma_alpha;       /**< lgamma(α) for each run length */
    double *lgamma_alpha_p5;    /**< lgamma(α + 0.5) for each run length */

    /*=========================================================================
     * Output State
     *=========================================================================*/
    
    size_t t;               /**< Current timestep (observation count) */
    size_t map_runlength;   /**< Most likely run length (MAP estimate) */
    double p_changepoint;   /**< P(run_length < 5), quick change indicator */
    
} bocpd_ultra_t;

/*=============================================================================
 * Public API
 *=============================================================================*/

/**
 * @brief Initialize BOCPD detector.
 * 
 * @param[out] b            Pointer to detector structure (caller-allocated)
 * @param[in]  hazard_lambda Expected run length λ. Hazard h = 1/λ.
 * @param[in]  prior        Normal-Gamma prior parameters
 * @param[in]  max_run_length Maximum run length to track (rounded up to power of 2)
 * 
 * @return 0 on success, -1 on failure (invalid params or allocation failure)
 * 
 * @note Capacity is rounded up to the next power of 2 for efficient ring buffer
 *       indexing. E.g., max_run_length=500 → capacity=512.
 * 
 * @note Memory usage ≈ 18 × capacity × 8 bytes. For max_run_length=512, ~73KB.
 */
int bocpd_ultra_init(bocpd_ultra_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);

/**
 * @brief Free all memory associated with detector.
 * 
 * @param[in,out] b Pointer to detector. Safe to call on NULL or already-freed.
 * 
 * @post All internal arrays freed, structure zeroed.
 */
void bocpd_ultra_free(bocpd_ultra_t *b);

/**
 * @brief Reset detector to initial state (reuse without reallocation).
 * 
 * @param[in,out] b Pointer to initialized detector
 * 
 * @post t=0, active_len=0, ready to process new data stream.
 */
void bocpd_ultra_reset(bocpd_ultra_t *b);

/**
 * @brief Process one observation and update run-length distribution.
 * 
 * @param[in,out] b Pointer to initialized detector
 * @param[in]     x New observation value
 * 
 * @post b->r contains updated run-length distribution
 * @post b->map_runlength contains MAP estimate
 * @post b->p_changepoint contains P(run_length < 5)
 * 
 * @note Typical latency: 2-5 µs for active_len < 200.
 * 
 * ### Algorithm Steps (each observation)
 * 
 * 1. **Linearize**: Copy ring buffer to contiguous scratch (O(active_len))
 * 2. **Fused SIMD**: Compute predictive probs + update r[] in one pass
 * 3. **Normalize**: Scale r[] to sum to 1
 * 4. **Shift**: Advance ring buffer head (O(1))
 * 5. **Observe**: Update all sufficient stats and posteriors (O(active_len))
 */
void bocpd_ultra_step(bocpd_ultra_t *b, double x);

/**
 * @brief Get MAP (most likely) run length.
 * 
 * @param[in] b Pointer to detector
 * @return Run length with highest probability
 * 
 * @note This is a cached value updated during bocpd_ultra_step().
 */
static inline size_t bocpd_ultra_get_map_rl(const bocpd_ultra_t *b) {
    return b->map_runlength;
}

/**
 * @brief Compute probability of recent changepoint.
 * 
 * @param[in] b Pointer to detector
 * @param[in] w Window size (typically 3-10)
 * @return P(run_length < w) = Σᵢ₌₀^{w-1} r[i]
 * 
 * @note This is the recommended signal for changepoint detection.
 *       A spike in this value indicates a likely regime change.
 * 
 * ### Usage for Trading
 * 
 * @code
 * double p_change = bocpd_ultra_change_prob(&cpd, 5);
 * if (p_change > 0.3) {
 *     // Increase Kalman filter process noise
 *     ukf_set_q_scale(&ukf, 1.0 + 5.0 * p_change);
 *     // Reduce position sizes
 *     position_scale = 1.0 - 0.5 * p_change;
 * }
 * @endcode
 */
static inline double bocpd_ultra_change_prob(const bocpd_ultra_t *b, size_t w) {
    double s = 0;
    size_t m = (w < b->active_len) ? w : b->active_len;
    for (size_t i = 0; i < m; i++) s += b->r[i];
    return s;
}

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ULTRA_H */