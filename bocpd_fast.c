/**
 * @file bocpd_fast.h
 * @brief Optimized BOCPD with SIMD (AVX2)
 *
 * Optimizations:
 * - Pre-allocated arrays (no malloc in hot path)
 * - Structure-of-Arrays for SIMD-friendly memory layout
 * - AVX2 vectorized loops
 * - Cached lgamma values
 * - Truncation of low-probability run lengths
 */

#ifndef BOCPD_FAST_H
#define BOCPD_FAST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NormalGamma prior parameters
 */
typedef struct {
    double mu;
    double kappa;
    double alpha;
    double beta;
} bocpd_prior_t;

/**
 * @brief Optimized BOCPD state (Structure-of-Arrays layout)
 */
typedef struct {
    /* Configuration */
    double hazard;              /* 1/lambda */
    double log_hazard;          /* log(hazard) for numerical stability */
    double log_one_minus_h;     /* log(1 - hazard) */
    double trunc_threshold;     /* Truncate run lengths below this probability */
    bocpd_prior_t prior;

    /* Capacity */
    size_t capacity;            /* Max run length */
    size_t active_len;          /* Current number of active run lengths */

    /* State: Structure-of-Arrays for SIMD */
    double *r;                  /* Run-length probabilities [capacity] */
    double *ss_n;               /* Sufficient stat: count [capacity] */
    double *ss_sum;             /* Sufficient stat: sum [capacity] */
    double *ss_sum2;            /* Sufficient stat: sum of squares [capacity] */

    /* Pre-allocated work arrays */
    double *pp;                 /* Predictive probabilities [capacity] */
    double *r_new;              /* New run-length dist [capacity] */

    /* Cached values for lgamma (indexed by 2*alpha = n + 2*alpha0) */
    double *lgamma_cache;       /* [capacity + some margin] */
    size_t lgamma_cache_size;

    /* Output */
    size_t t;                   /* Current timestep */
    double p_changepoint;       /* P(recent change) */
    size_t map_runlength;       /* Most likely run length */
} bocpd_fast_t;

/**
 * @brief Initialize optimized BOCPD
 */
int bocpd_fast_init(bocpd_fast_t *b,
                    double hazard_lambda,
                    bocpd_prior_t prior,
                    size_t max_run_length);

/**
 * @brief Free resources
 */
void bocpd_fast_free(bocpd_fast_t *b);

/**
 * @brief Reset to initial state
 */
void bocpd_fast_reset(bocpd_fast_t *b);

/**
 * @brief Process observation (main hot path)
 */
void bocpd_fast_step(bocpd_fast_t *b, double x);

/**
 * @brief Get probability of recent change (within window steps)
 */
double bocpd_fast_change_prob(const bocpd_fast_t *b, size_t window);

/**
 * @brief Get MAP run length
 */
static inline size_t bocpd_fast_get_map_rl(const bocpd_fast_t *b) {
    return b->map_runlength;
}

/**
 * @brief Get current timestep
 */
static inline size_t bocpd_fast_get_t(const bocpd_fast_t *b) {
    return b->t;
}

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_FAST_H */
