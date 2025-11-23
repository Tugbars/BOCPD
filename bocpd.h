/**
 * @file bocpd.h
 * @brief Bayesian Online Change Point Detection
 *
 * Based on: "Bayesian Online Changepoint Detection"
 * Ryan Adams, David MacKay; arXiv:0710.3742
 * https://arxiv.org/pdf/0710.3742.pdf
 *
 * This implementation uses NormalGamma conjugate prior for detecting
 * changes in mean and variance of Gaussian data (e.g., returns).
 */

#ifndef BOCPD_H
#define BOCPD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NormalGamma prior parameters
 *
 * Conjugate prior for unknown mean and precision of Gaussian.
 * Posterior predictive is Student-t distribution.
 */
typedef struct {
    double mu;      /* Prior mean location */
    double kappa;   /* Pseudo-observations for mean */
    double alpha;   /* Shape parameter (pseudo-observations / 2) */
    double beta;    /* Rate parameter */
} bocpd_normal_gamma_t;

/**
 * @brief Sufficient statistic for Gaussian data
 */
typedef struct {
    double n;       /* Count */
    double sum_x;   /* Sum of observations */
    double sum_x2;  /* Sum of squared observations */
} bocpd_suffstat_t;

/**
 * @brief BOCPD state container
 */
typedef struct {
    /* Configuration */
    double hazard;                  /* 1/lambda: P(change point) */
    double cdf_threshold;           /* Truncation threshold */
    bocpd_normal_gamma_t prior;     /* Predictive prior */

    /* State */
    size_t t;                       /* Current time step */
    size_t capacity;                /* Allocated capacity */

    double *r;                      /* Run-length probabilities [capacity] */
    bocpd_suffstat_t *suff_stats;   /* Sufficient statistics [capacity] */

    /* Output (updated each step) */
    double p_changepoint;           /* P(change point at current step) = r[0] */
} bocpd_t;

/**
 * @brief Initialize BOCPD detector
 *
 * @param bocpd         Pointer to BOCPD struct (caller allocated)
 * @param hazard_lambda Expected run length. Larger = fewer change points expected.
 *                      P(change) = 1/hazard_lambda. Typical: 100-500
 * @param prior         NormalGamma prior parameters
 * @param max_run_length Maximum run length to track (determines memory usage)
 * @return 0 on success, -1 on failure
 *
 * Example for financial returns:
 *   bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};
 *   bocpd_init(&cpd, 250.0, prior, 1000);
 */
int bocpd_init(bocpd_t *bocpd,
               double hazard_lambda,
               bocpd_normal_gamma_t prior,
               size_t max_run_length);

/**
 * @brief Free BOCPD resources
 */
void bocpd_free(bocpd_t *bocpd);

/**
 * @brief Reset BOCPD to initial state (keeps configuration)
 */
void bocpd_reset(bocpd_t *bocpd);

/**
 * @brief Process a new observation
 *
 * @param bocpd Pointer to BOCPD struct
 * @param x     New observation
 * @return Pointer to run-length distribution (length = t+1 after call)
 *         Also updates bocpd->p_changepoint with P(change at this step)
 */
const double *bocpd_step(bocpd_t *bocpd, double x);

/**
 * @brief Get current change point probability
 *
 * @param bocpd Pointer to BOCPD struct
 * @return P(change point at current step), i.e., r[0]
 */
static inline double bocpd_get_p_changepoint(const bocpd_t *bocpd) {
    return bocpd->p_changepoint;
}

/**
 * @brief Get current time step
 */
static inline size_t bocpd_get_t(const bocpd_t *bocpd) {
    return bocpd->t;
}

/**
 * @brief Get run-length distribution
 *
 * @param bocpd Pointer to BOCPD struct
 * @param len   Output: length of distribution (= t)
 * @return Pointer to run-length distribution
 */
static inline const double *bocpd_get_runlength_dist(const bocpd_t *bocpd,
                                                      size_t *len) {
    if (len) *len = bocpd->t;
    return bocpd->r;
}

/**
 * @brief Find most likely run length (MAP estimate)
 *
 * @param bocpd Pointer to BOCPD struct
 * @return Index of most likely run length
 */
size_t bocpd_map_runlength(const bocpd_t *bocpd);

/**
 * @brief Compute probability mass in short run lengths (change detection signal)
 *
 * This is the probability that a change point occurred recently (within `window` steps).
 * A spike in this value indicates a regime change.
 *
 * @param bocpd  Pointer to BOCPD struct
 * @param window Number of short run lengths to sum (typically 1-5)
 * @return P(run_length < window)
 */
double bocpd_short_run_probability(const bocpd_t *bocpd, size_t window);

/**
 * @brief Compute expected run length
 *
 * @param bocpd Pointer to BOCPD struct
 * @return E[run_length]
 */
double bocpd_expected_runlength(const bocpd_t *bocpd);

/**
 * @brief Check if change point detected (simple threshold)
 *
 * @param bocpd     Pointer to BOCPD struct
 * @param threshold Probability threshold (e.g., 0.5)
 * @return 1 if r[0] > threshold, 0 otherwise
 */
static inline int bocpd_is_changepoint(const bocpd_t *bocpd, double threshold) {
    return bocpd->p_changepoint > threshold ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_H */
