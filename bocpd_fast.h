/**
 * @file bocpd_simd.h
 * @brief Fully optimized BOCPD with AVX2 SIMD
 */

#ifndef BOCPD_SIMD_H
#define BOCPD_SIMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double mu0, kappa0, alpha0, beta0;
} bocpd_prior_t;

typedef struct {
    /* Configuration */
    double hazard;
    double one_minus_h;
    double trunc_thresh;
    bocpd_prior_t prior;

    /* Ring buffer state */
    size_t capacity;
    size_t active_len;
    size_t ring_start;      /* Ring buffer head index */

    /* Sufficient stats (ring buffer) */
    double *ss_n;
    double *ss_sum;
    double *ss_sum2;

    /* Incremental posterior params (ring buffer, updated incrementally) */
    double *post_kappa;
    double *post_mu;
    double *post_alpha;
    double *post_beta;

    /* Precomputed per-run-length constants for Student-t */
    double *C1;             /* lgamma terms + constant part */
    double *C2;             /* (nu+1)/2 */
    double *inv_sigma;      /* 1/sigma */
    double *inv_nu;         /* 1/nu */

    /* Work arrays (no ring buffer, linear) */
    double *pp;
    double *r;
    double *r_new;

    /* lgamma cache */
    double *lgamma_cache;
    size_t lgamma_cache_size;

    /* Output */
    size_t t;
    size_t map_runlength;
    double p_changepoint;
} bocpd_simd_t;

int bocpd_simd_init(bocpd_simd_t *b, double hazard_lambda, 
                    bocpd_prior_t prior, size_t max_run_length);
void bocpd_simd_free(bocpd_simd_t *b);
void bocpd_simd_reset(bocpd_simd_t *b);
void bocpd_simd_step(bocpd_simd_t *b, double x);
double bocpd_simd_change_prob(const bocpd_simd_t *b, size_t window);

static inline size_t bocpd_simd_get_map_rl(const bocpd_simd_t *b) {
    return b->map_runlength;
}

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_SIMD_H */
