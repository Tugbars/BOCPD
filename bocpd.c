/**
 * @file bocpd.c
 * @brief Bayesian Online Change Point Detection - Implementation
 */

#include "bocpd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <alloca.h>

/* ============================================================================
 * Math helpers
 * ============================================================================ */

/**
 * @brief Log-gamma function (Stirling approximation for large x, series for small)
 */
static double lgamma_approx(double x) {
    /* Use standard lgamma if available, otherwise approximate */
    return lgamma(x);
}

/**
 * @brief Student-t log-density
 *
 * @param x     Observation
 * @param mu    Location
 * @param sigma Scale (not variance)
 * @param nu    Degrees of freedom
 * @return log p(x | mu, sigma, nu)
 */
static double student_t_ln_pdf(double x, double mu, double sigma, double nu) {
    /* 
     * p(x) = Γ((ν+1)/2) / (Γ(ν/2) * sqrt(νπ) * σ) * (1 + ((x-μ)/σ)²/ν)^(-(ν+1)/2)
     *
     * ln p(x) = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*ln(νπ) - ln(σ)
     *           - ((ν+1)/2) * ln(1 + ((x-μ)/σ)²/ν)
     */
    double z = (x - mu) / sigma;
    double nu_half = nu * 0.5;
    double nu_plus_1_half = (nu + 1.0) * 0.5;

    double ln_pdf = lgamma_approx(nu_plus_1_half)
                  - lgamma_approx(nu_half)
                  - 0.5 * log(nu * M_PI)
                  - log(sigma)
                  - nu_plus_1_half * log(1.0 + (z * z) / nu);

    return ln_pdf;
}

/* ============================================================================
 * NormalGamma conjugate prior operations
 * ============================================================================ */

/**
 * @brief Compute posterior NormalGamma parameters given sufficient statistics
 *
 * @param prior Prior parameters
 * @param ss    Sufficient statistics
 * @param post  Output: Posterior parameters
 */
static void normal_gamma_posterior(const bocpd_normal_gamma_t *prior,
                                   const bocpd_suffstat_t *ss,
                                   bocpd_normal_gamma_t *post) {
    double n = ss->n;

    if (n < 1e-10) {
        /* No data: posterior = prior */
        *post = *prior;
        return;
    }

    double x_bar = ss->sum_x / n;

    /* Posterior parameters (Murphy, "Conjugate Bayesian analysis of the Gaussian") */
    post->kappa = prior->kappa + n;
    post->mu = (prior->kappa * prior->mu + n * x_bar) / post->kappa;
    post->alpha = prior->alpha + n * 0.5;

    /* Sum of squared deviations from sample mean */
    double ss_x = ss->sum_x2 - n * x_bar * x_bar;

    /* Posterior beta */
    double mu_diff = x_bar - prior->mu;
    post->beta = prior->beta
               + 0.5 * ss_x
               + (prior->kappa * n * mu_diff * mu_diff) / (2.0 * post->kappa);
}

/**
 * @brief Compute log posterior predictive probability P(x | data)
 *
 * The posterior predictive of NormalGamma is Student-t:
 *   x | data ~ t_{2α}(μ, β(κ+1)/(ακ))
 *
 * @param prior Prior NormalGamma parameters
 * @param ss    Sufficient statistics (can have n=0)
 * @param x     New observation
 * @return log P(x | data summarized by ss)
 */
static double normal_gamma_ln_pp(const bocpd_normal_gamma_t *prior,
                                 const bocpd_suffstat_t *ss,
                                 double x) {
    bocpd_normal_gamma_t post;
    normal_gamma_posterior(prior, ss, &post);

    /* Student-t parameters from posterior NormalGamma */
    double nu = 2.0 * post.alpha;
    double mu = post.mu;
    double sigma = sqrt(post.beta * (post.kappa + 1.0) / (post.alpha * post.kappa));

    return student_t_ln_pdf(x, mu, sigma, nu);
}

/* ============================================================================
 * Sufficient statistic operations
 * ============================================================================ */

static inline void suffstat_reset(bocpd_suffstat_t *ss) {
    ss->n = 0.0;
    ss->sum_x = 0.0;
    ss->sum_x2 = 0.0;
}

static inline void suffstat_observe(bocpd_suffstat_t *ss, double x) {
    ss->n += 1.0;
    ss->sum_x += x;
    ss->sum_x2 += x * x;
}

/* ============================================================================
 * BOCPD API
 * ============================================================================ */

int bocpd_init(bocpd_t *bocpd,
               double hazard_lambda,
               bocpd_normal_gamma_t prior,
               size_t max_run_length) {
    if (!bocpd || hazard_lambda <= 0.0 || max_run_length == 0) {
        return -1;
    }

    memset(bocpd, 0, sizeof(*bocpd));

    /* Allocate arrays */
    bocpd->r = (double *)calloc(max_run_length, sizeof(double));
    bocpd->suff_stats = (bocpd_suffstat_t *)calloc(max_run_length,
                                                    sizeof(bocpd_suffstat_t));

    if (!bocpd->r || !bocpd->suff_stats) {
        bocpd_free(bocpd);
        return -1;
    }

    /* Configuration */
    bocpd->hazard = 1.0 / hazard_lambda;
    bocpd->cdf_threshold = 1e-3;
    bocpd->prior = prior;
    bocpd->capacity = max_run_length;

    /* Initial state */
    bocpd->t = 0;
    bocpd->p_changepoint = 0.0;

    return 0;
}

void bocpd_free(bocpd_t *bocpd) {
    if (bocpd) {
        free(bocpd->r);
        free(bocpd->suff_stats);
        memset(bocpd, 0, sizeof(*bocpd));
    }
}

void bocpd_reset(bocpd_t *bocpd) {
    if (bocpd) {
        bocpd->t = 0;
        bocpd->p_changepoint = 0.0;
        /* Zero out arrays */
        memset(bocpd->r, 0, bocpd->capacity * sizeof(double));
        for (size_t i = 0; i < bocpd->capacity; i++) {
            suffstat_reset(&bocpd->suff_stats[i]);
        }
    }
}

const double *bocpd_step(bocpd_t *bocpd, double x) {
    if (!bocpd) return NULL;

    const double h = bocpd->hazard;
    const double one_minus_h = 1.0 - h;
    const double cdf_thresh = bocpd->cdf_threshold;

    if (bocpd->t == 0) {
        /*
         * First observation: by definition, this is a change point.
         * r[0] = 1.0 means "run length is 0 with probability 1"
         */
        suffstat_reset(&bocpd->suff_stats[0]);
        bocpd->r[0] = 1.0;
        bocpd->p_changepoint = 1.0;

        /* Observe the first data point */
        suffstat_observe(&bocpd->suff_stats[0], x);
        bocpd->t = 1;

        return bocpd->r;
    }

    /*
     * t >= 1: We have existing run-length distribution
     *
     * Current state:
     *   r[0..t-1] = run-length probabilities
     *   suff_stats[0..t-1] = sufficient statistics for each run length
     *
     * After update:
     *   r[0..t] = new run-length probabilities
     *   suff_stats[0..t] = updated sufficient statistics
     */
    size_t t = bocpd->t;

    /* Bounds check */
    if (t >= bocpd->capacity - 1) {
        t = bocpd->capacity - 2;
    }

    /*
     * Compute predictive probabilities BEFORE updating suff stats.
     * pp[i] = P(x | data accumulated in run of length i)
     */
    double *pp = (double *)alloca((t + 1) * sizeof(double));

    for (size_t i = 0; i < t; i++) {
        double ln_pp = normal_gamma_ln_pp(&bocpd->prior,
                                          &bocpd->suff_stats[i],
                                          x);
        pp[i] = exp(ln_pp);

        /* Clamp for numerical stability */
        if (pp[i] < 1e-300) pp[i] = 1e-300;
        if (pp[i] > 1e300) pp[i] = 1e300;
    }

    /*
     * Core BOCPD update (working backwards to avoid overwriting):
     *
     * r_new[i+1] = r_old[i] * pp[i] * (1-h)   (growth: run continues)
     * r_new[0] = sum_i( r_old[i] * pp[i] * h ) (changepoint: run resets)
     */
    double r0 = 0.0;    /* Accumulator for P(change point) */
    double r_sum = 0.0; /* Normalizer */
    double r_seen = 0.0;

    for (size_t i = t; i > 0; i--) {
        size_t idx = i - 1;  /* Index into old r and suff_stats */
        double r_old = bocpd->r[idx];

        if (r_old < 1e-300) {
            bocpd->r[i] = 0.0;
            continue;
        }

        r_seen += r_old;

        /* Growth: run continues */
        bocpd->r[i] = r_old * pp[idx] * one_minus_h;
        r_sum += bocpd->r[i];

        /* Change: run resets */
        r0 += r_old * pp[idx] * h;

        /* Early termination if we've seen enough probability mass */
        if (1.0 - r_seen < cdf_thresh) {
            break;
        }
    }

    /* r[0] = accumulated change point probability */
    bocpd->r[0] = r0;
    r_sum += r0;

    /* Normalize */
    if (r_sum > 1e-300) {
        double inv_sum = 1.0 / r_sum;
        for (size_t i = 0; i <= t; i++) {
            bocpd->r[i] *= inv_sum;
        }
    }

    bocpd->p_changepoint = bocpd->r[0];

    /*
     * Update sufficient statistics:
     * - Shift existing stats (run length increased by 1)
     * - Add empty stat at position 0 (new potential run starting now)
     * - Observe new data point in all stats
     */
    memmove(&bocpd->suff_stats[1],
            &bocpd->suff_stats[0],
            t * sizeof(bocpd_suffstat_t));
    suffstat_reset(&bocpd->suff_stats[0]);

    /* Observe x in all sufficient statistics */
    for (size_t i = 0; i <= t; i++) {
        suffstat_observe(&bocpd->suff_stats[i], x);
    }

    /* Increment time */
    bocpd->t = t + 1;

    return bocpd->r;
}

size_t bocpd_map_runlength(const bocpd_t *bocpd) {
    if (!bocpd || bocpd->t == 0) return 0;

    size_t best_idx = 0;
    double best_val = bocpd->r[0];

    for (size_t i = 1; i < bocpd->t; i++) {
        if (bocpd->r[i] > best_val) {
            best_val = bocpd->r[i];
            best_idx = i;
        }
    }

    return best_idx;
}

double bocpd_short_run_probability(const bocpd_t *bocpd, size_t window) {
    if (!bocpd || bocpd->t == 0) return 0.0;
    
    double sum = 0.0;
    size_t max_idx = (window < bocpd->t) ? window : bocpd->t;
    
    for (size_t i = 0; i < max_idx; i++) {
        sum += bocpd->r[i];
    }
    
    return sum;
}

double bocpd_expected_runlength(const bocpd_t *bocpd) {
    if (!bocpd || bocpd->t == 0) return 0.0;
    
    double expected = 0.0;
    for (size_t i = 0; i < bocpd->t; i++) {
        expected += i * bocpd->r[i];
    }
    
    return expected;
}

/* ============================================================================
 * Optional: MAP change point extraction (like utils.rs)
 * ============================================================================ */

/**
 * @brief Extract MAP change points from stored run-length history
 *
 * Note: This requires storing the full history of run-length distributions,
 * which the basic bocpd_t doesn't do. This is a utility for offline analysis.
 *
 * @param r_history     Array of run-length distributions [n_steps][max_len]
 * @param n_steps       Number of time steps
 * @param max_len       Maximum run length
 * @param changepoints  Output array (caller allocated, size n_steps)
 * @param n_changepoints Output: number of change points found
 */
void bocpd_map_changepoints(const double *const *r_history,
                            size_t n_steps,
                            size_t max_len,
                            size_t *changepoints,
                            size_t *n_changepoints) {
    if (!r_history || !changepoints || !n_changepoints || n_steps == 0) {
        if (n_changepoints) *n_changepoints = 0;
        return;
    }

    size_t n_cp = 0;
    size_t s = n_steps - 1;

    /* Walk backwards through run-length distributions */
    while (s > 0) {
        /* Find most likely run length at step s */
        const double *r_s = r_history[s];
        size_t len_s = (s + 1 < max_len) ? s + 1 : max_len;

        size_t best_rl = 0;
        double best_val = r_s[0];
        for (size_t i = 1; i < len_s; i++) {
            if (r_s[i] > best_val) {
                best_val = r_s[i];
                best_rl = i;
            }
        }

        if (best_rl == 0) {
            /* Change point at s */
            changepoints[n_cp++] = s;
            s--;
        } else {
            /* Jump back by run length */
            size_t prev_s = (s > best_rl) ? s - best_rl : 0;
            changepoints[n_cp++] = prev_s;
            s = prev_s;
        }
    }

    /* Reverse to get chronological order */
    for (size_t i = 0; i < n_cp / 2; i++) {
        size_t tmp = changepoints[i];
        changepoints[i] = changepoints[n_cp - 1 - i];
        changepoints[n_cp - 1 - i] = tmp;
    }

    *n_changepoints = n_cp;
}
