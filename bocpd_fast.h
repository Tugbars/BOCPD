/**
 * @file bocpd_final.h
 * @brief Production-grade BOCPD with correct AVX2 math
 */

#ifndef BOCPD_FINAL_H
#define BOCPD_FINAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double mu0, kappa0, alpha0, beta0;
} bocpd_prior_t;

typedef struct {
    /* Config */
    double hazard;
    double one_minus_h;
    double trunc_thresh;
    bocpd_prior_t prior;

    /* Capacity */
    size_t capacity;
    size_t active_len;
    size_t ring_start;  /* Ring buffer index */

    /* Ring-buffered arrays (all indexed via ring) */
    double *ss_n;
    double *ss_sum;
    double *ss_sum2;

    /* Incremental posteriors (ring-buffered) */
    double *post_kappa;
    double *post_mu;
    double *post_alpha;
    double *post_beta;

    /* Incremental lgamma values (ring-buffered) */
    double *lgamma_alpha;       /* lgamma(alpha) */
    double *lgamma_alpha_p5;    /* lgamma(alpha + 0.5) */

    /* Incremental log terms (ring-buffered) */
    double *ln_sigma_sq;        /* ln(beta*(kappa+1)/(alpha*kappa)) */
    double *ln_nu_pi;           /* ln(2*alpha*pi) */

    /* Run-length distribution (linear, not ring) */
    double *r;
    double *r_scratch;          /* Scratch buffer instead of alloca */

    /* Output */
    size_t t;
    size_t map_runlength;
    double p_changepoint;
} bocpd_final_t;

int bocpd_final_init(bocpd_final_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);
void bocpd_final_free(bocpd_final_t *b);
void bocpd_final_reset(bocpd_final_t *b);
void bocpd_final_step(bocpd_final_t *b, double x);

static inline size_t bocpd_final_get_map_rl(const bocpd_final_t *b) {
    return b->map_runlength;
}
static inline double bocpd_final_change_prob(const bocpd_final_t *b, size_t w) {
    double s = 0;
    size_t m = (w < b->active_len) ? w : b->active_len;
    for (size_t i = 0; i < m; i++) s += b->r[i];
    return s;
}

#ifdef __cplusplus
}
#endif

#endif
