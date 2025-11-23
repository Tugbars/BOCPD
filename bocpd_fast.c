/**
 * @file bocpd_fast.c
 * @brief Optimized BOCPD with AVX2 SIMD
 */

#include "bocpd_fast.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ============================================================================
 * Constants and fast math
 * ============================================================================ */

#define ALIGN64 __attribute__((aligned(64)))
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double LOG_SQRT_2PI = 0.9189385332046727;

/* ============================================================================
 * Fast approximations
 * ============================================================================ */

/**
 * @brief Fast log approximation (accurate to ~1e-7)
 */
static inline double fast_log(double x) {
    /* Use standard log - compiler will optimize */
    return log(x);
}

/**
 * @brief Fast exp approximation
 */
static inline double fast_exp(double x) {
    /* Clamp to avoid overflow/underflow */
    if (x < -700.0) return 0.0;
    if (x > 700.0) return 1e308;
    return exp(x);
}

/**
 * @brief Precompute lgamma values for indices we'll need
 *
 * For NormalGamma with alpha0, we need lgamma((n + 2*alpha0)/2) and lgamma((n + 2*alpha0 + 1)/2)
 * We cache lgamma(x/2) for x = 1, 2, 3, ... up to some maximum
 */
static void precompute_lgamma_cache(bocpd_fast_t *b) {
    for (size_t i = 0; i < b->lgamma_cache_size; i++) {
        /* Cache lgamma((i + 1) / 2.0) which covers our needs */
        double arg = (double)(i + 1) * 0.5;
        b->lgamma_cache[i] = lgamma(arg);
    }
}

/**
 * @brief Get cached lgamma(x) where x = k/2 for some integer k
 */
static inline double get_lgamma_half(const bocpd_fast_t *b, size_t twice_x) {
    /* twice_x = 2*x, so we want lgamma(twice_x / 2) */
    if (LIKELY(twice_x > 0 && twice_x <= b->lgamma_cache_size)) {
        return b->lgamma_cache[twice_x - 1];
    }
    return lgamma((double)twice_x * 0.5);
}

/* ============================================================================
 * Student-t log density (vectorizable core)
 * ============================================================================ */

/**
 * @brief Compute Student-t log density for a single observation
 *
 * ln p(x | mu, sigma, nu) = lgamma((nu+1)/2) - lgamma(nu/2) 
 *                          - 0.5*ln(nu*pi) - ln(sigma)
 *                          - ((nu+1)/2) * ln(1 + z²/nu)
 * where z = (x - mu) / sigma
 */
static inline double student_t_ln_pdf_fast(double x, double mu, double sigma, 
                                            double nu, double lgamma_nu_half,
                                            double lgamma_nu_plus_1_half) {
    double z = (x - mu) / sigma;
    double z2 = z * z;
    double nu_half = nu * 0.5;
    double nu_plus_1_half = nu_half + 0.5;

    return lgamma_nu_plus_1_half - lgamma_nu_half
           - 0.5 * fast_log(nu * M_PI)
           - fast_log(sigma)
           - nu_plus_1_half * fast_log(1.0 + z2 / nu);
}

/* ============================================================================
 * AVX2 vectorized operations
 * ============================================================================ */

#ifdef __AVX2__

/**
 * @brief Vectorized sum of array
 */
static double avx2_sum(const double *arr, size_t n) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    /* Process 4 doubles at a time */
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&arr[i]);
        sum_vec = _mm256_add_pd(sum_vec, v);
    }

    /* Horizontal sum */
    __m128d lo = _mm256_castpd256_pd128(sum_vec);
    __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd(lo, lo, 1);
    lo = _mm_add_pd(lo, shuf);
    double sum = _mm_cvtsd_f64(lo);

    /* Handle remainder */
    for (; i < n; i++) {
        sum += arr[i];
    }

    return sum;
}

/**
 * @brief Vectorized scale (multiply by scalar)
 */
static void avx2_scale(double *arr, size_t n, double scale) {
    __m256d scale_vec = _mm256_set1_pd(scale);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&arr[i]);
        v = _mm256_mul_pd(v, scale_vec);
        _mm256_storeu_pd(&arr[i], v);
    }

    for (; i < n; i++) {
        arr[i] *= scale;
    }
}

/**
 * @brief Find index of maximum value
 */
static size_t avx2_argmax(const double *arr, size_t n) {
    if (n == 0) return 0;

    size_t best_idx = 0;
    double best_val = arr[0];

    /* Simple loop - argmax doesn't vectorize well */
    for (size_t i = 1; i < n; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best_idx = i;
        }
    }

    return best_idx;
}

/**
 * @brief Vectorized sufficient stat observation
 *
 * For each i: ss_n[i] += 1, ss_sum[i] += x, ss_sum2[i] += x*x
 */
static void avx2_observe_all(double *ss_n, double *ss_sum, double *ss_sum2,
                              size_t n, double x) {
    __m256d one_vec = _mm256_set1_pd(1.0);
    __m256d x_vec = _mm256_set1_pd(x);
    __m256d x2_vec = _mm256_set1_pd(x * x);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        /* ss_n[i] += 1 */
        __m256d n_v = _mm256_loadu_pd(&ss_n[i]);
        n_v = _mm256_add_pd(n_v, one_vec);
        _mm256_storeu_pd(&ss_n[i], n_v);

        /* ss_sum[i] += x */
        __m256d sum_v = _mm256_loadu_pd(&ss_sum[i]);
        sum_v = _mm256_add_pd(sum_v, x_vec);
        _mm256_storeu_pd(&ss_sum[i], sum_v);

        /* ss_sum2[i] += x*x */
        __m256d sum2_v = _mm256_loadu_pd(&ss_sum2[i]);
        sum2_v = _mm256_add_pd(sum2_v, x2_vec);
        _mm256_storeu_pd(&ss_sum2[i], sum2_v);
    }

    for (; i < n; i++) {
        ss_n[i] += 1.0;
        ss_sum[i] += x;
        ss_sum2[i] += x * x;
    }
}

#else
/* Fallback scalar implementations */

static double avx2_sum(const double *arr, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += arr[i];
    return sum;
}

static void avx2_scale(double *arr, size_t n, double scale) {
    for (size_t i = 0; i < n; i++) arr[i] *= scale;
}

static size_t avx2_argmax(const double *arr, size_t n) {
    if (n == 0) return 0;
    size_t best_idx = 0;
    double best_val = arr[0];
    for (size_t i = 1; i < n; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static void avx2_observe_all(double *ss_n, double *ss_sum, double *ss_sum2,
                              size_t n, double x) {
    double x2 = x * x;
    for (size_t i = 0; i < n; i++) {
        ss_n[i] += 1.0;
        ss_sum[i] += x;
        ss_sum2[i] += x2;
    }
}

#endif /* __AVX2__ */

/* ============================================================================
 * Core BOCPD logic
 * ============================================================================ */

/**
 * @brief Compute posterior NormalGamma and predictive probability
 *
 * This is the hot inner loop - compute pp[i] for each run length
 */
static void compute_predictive_probs(bocpd_fast_t *b, double x) {
    const double mu0 = b->prior.mu;
    const double kappa0 = b->prior.kappa;
    const double alpha0 = b->prior.alpha;
    const double beta0 = b->prior.beta;

    const size_t n = b->active_len;
    const double *ss_n = b->ss_n;
    const double *ss_sum = b->ss_sum;
    const double *ss_sum2 = b->ss_sum2;
    double *pp = b->pp;

    /* This loop is the main compute bottleneck */
    for (size_t i = 0; i < n; i++) {
        double ni = ss_n[i];

        double post_kappa, post_mu, post_alpha, post_beta;

        if (ni < 0.5) {
            /* No data: posterior = prior */
            post_kappa = kappa0;
            post_mu = mu0;
            post_alpha = alpha0;
            post_beta = beta0;
        } else {
            double x_bar = ss_sum[i] / ni;

            post_kappa = kappa0 + ni;
            post_mu = (kappa0 * mu0 + ni * x_bar) / post_kappa;
            post_alpha = alpha0 + ni * 0.5;

            double ss_x = ss_sum2[i] - ni * x_bar * x_bar;
            double mu_diff = x_bar - mu0;
            post_beta = beta0 + 0.5 * ss_x
                       + (kappa0 * ni * mu_diff * mu_diff) / (2.0 * post_kappa);
        }

        /* Student-t parameters */
        double nu = 2.0 * post_alpha;
        double sigma = sqrt(post_beta * (post_kappa + 1.0) / (post_alpha * post_kappa));

        /* Get cached lgamma values */
        /* nu = 2 * post_alpha, so we need lgamma(post_alpha) and lgamma(post_alpha + 0.5) */
        size_t nu_int = (size_t)(nu + 0.5);  /* Round to nearest integer */
        double lgamma_nu_half = get_lgamma_half(b, nu_int);
        double lgamma_nu_plus_1_half = get_lgamma_half(b, nu_int + 1);

        /* Compute log predictive probability */
        double ln_pp = student_t_ln_pdf_fast(x, post_mu, sigma, nu,
                                              lgamma_nu_half, lgamma_nu_plus_1_half);

        pp[i] = fast_exp(ln_pp);

        /* Clamp */
        if (pp[i] < 1e-300) pp[i] = 1e-300;
    }
}

/**
 * @brief Update run-length distribution
 */
static void update_runlength_dist(bocpd_fast_t *b) {
    const size_t n = b->active_len;
    const double h = b->hazard;
    const double one_minus_h = 1.0 - h;
    const double *r = b->r;
    const double *pp = b->pp;
    double *r_new = b->r_new;

    double r0 = 0.0;  /* Changepoint accumulator */

    /* Core update: work backwards */
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        double r_old = r[idx];

        if (r_old < 1e-300) {
            r_new[i] = 0.0;
            continue;
        }

        double r_pp = r_old * pp[idx];

        /* Growth: run continues */
        r_new[i] = r_pp * one_minus_h;

        /* Changepoint: run resets */
        r0 += r_pp * h;
    }

    r_new[0] = r0;

    /* Normalize */
    double sum = avx2_sum(r_new, n + 1);
    if (sum > 1e-300) {
        avx2_scale(r_new, n + 1, 1.0 / sum);
    }

    /* Copy back */
    memcpy(b->r, r_new, (n + 1) * sizeof(double));
    b->active_len = n + 1;
}

/**
 * @brief Shift sufficient stats (insert empty at position 0)
 */
static void shift_suffstats(bocpd_fast_t *b) {
    size_t n = b->active_len;

    /* Shift arrays by 1 */
    memmove(&b->ss_n[1], &b->ss_n[0], n * sizeof(double));
    memmove(&b->ss_sum[1], &b->ss_sum[0], n * sizeof(double));
    memmove(&b->ss_sum2[1], &b->ss_sum2[0], n * sizeof(double));

    /* Insert empty at position 0 */
    b->ss_n[0] = 0.0;
    b->ss_sum[0] = 0.0;
    b->ss_sum2[0] = 0.0;
}

/**
 * @brief Truncate low-probability run lengths
 */
static void truncate_runlengths(bocpd_fast_t *b) {
    const double thresh = b->trunc_threshold;
    size_t n = b->active_len;

    /* Find last index with probability above threshold */
    size_t last_valid = 0;
    for (size_t i = 0; i < n; i++) {
        if (b->r[i] > thresh) {
            last_valid = i;
        }
    }

    /* Truncate if beneficial */
    if (last_valid < n - 1 && last_valid > 0) {
        size_t new_len = last_valid + 1;

        /* Renormalize */
        double sum = avx2_sum(b->r, new_len);
        if (sum > 1e-300) {
            avx2_scale(b->r, new_len, 1.0 / sum);
        }

        b->active_len = new_len;
    }

    /* Cap at capacity */
    if (b->active_len >= b->capacity) {
        b->active_len = b->capacity - 1;
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

int bocpd_fast_init(bocpd_fast_t *b,
                    double hazard_lambda,
                    bocpd_prior_t prior,
                    size_t max_run_length) {
    if (!b || hazard_lambda <= 0.0 || max_run_length < 10) {
        return -1;
    }

    memset(b, 0, sizeof(*b));

    b->capacity = max_run_length;
    b->hazard = 1.0 / hazard_lambda;
    b->log_hazard = log(b->hazard);
    b->log_one_minus_h = log(1.0 - b->hazard);
    b->trunc_threshold = 1e-8;
    b->prior = prior;

    /* Allocate aligned arrays */
    size_t alloc_size = max_run_length * sizeof(double);

    b->r = aligned_alloc(64, alloc_size);
    b->ss_n = aligned_alloc(64, alloc_size);
    b->ss_sum = aligned_alloc(64, alloc_size);
    b->ss_sum2 = aligned_alloc(64, alloc_size);
    b->pp = aligned_alloc(64, alloc_size);
    b->r_new = aligned_alloc(64, alloc_size);

    /* lgamma cache: need values up to roughly 2 * (max_run_length + alpha0) */
    b->lgamma_cache_size = 2 * max_run_length + 100;
    b->lgamma_cache = aligned_alloc(64, b->lgamma_cache_size * sizeof(double));

    if (!b->r || !b->ss_n || !b->ss_sum || !b->ss_sum2 ||
        !b->pp || !b->r_new || !b->lgamma_cache) {
        bocpd_fast_free(b);
        return -1;
    }

    /* Zero arrays */
    memset(b->r, 0, alloc_size);
    memset(b->ss_n, 0, alloc_size);
    memset(b->ss_sum, 0, alloc_size);
    memset(b->ss_sum2, 0, alloc_size);
    memset(b->pp, 0, alloc_size);
    memset(b->r_new, 0, alloc_size);

    /* Precompute lgamma cache */
    precompute_lgamma_cache(b);

    /* Initial state */
    b->t = 0;
    b->active_len = 0;
    b->p_changepoint = 0.0;
    b->map_runlength = 0;

    return 0;
}

void bocpd_fast_free(bocpd_fast_t *b) {
    if (b) {
        free(b->r);
        free(b->ss_n);
        free(b->ss_sum);
        free(b->ss_sum2);
        free(b->pp);
        free(b->r_new);
        free(b->lgamma_cache);
        memset(b, 0, sizeof(*b));
    }
}

void bocpd_fast_reset(bocpd_fast_t *b) {
    if (b) {
        size_t alloc_size = b->capacity * sizeof(double);
        memset(b->r, 0, alloc_size);
        memset(b->ss_n, 0, alloc_size);
        memset(b->ss_sum, 0, alloc_size);
        memset(b->ss_sum2, 0, alloc_size);
        b->t = 0;
        b->active_len = 0;
        b->p_changepoint = 0.0;
        b->map_runlength = 0;
    }
}

void bocpd_fast_step(bocpd_fast_t *b, double x) {
    if (!b) return;

    if (b->t == 0) {
        /* First observation */
        b->r[0] = 1.0;
        b->ss_n[0] = 1.0;
        b->ss_sum[0] = x;
        b->ss_sum2[0] = x * x;
        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }

    /* 1. Compute predictive probabilities */
    compute_predictive_probs(b, x);

    /* 2. Update run-length distribution */
    update_runlength_dist(b);

    /* 3. Shift sufficient stats */
    shift_suffstats(b);

    /* 4. Observe x in all sufficient stats */
    avx2_observe_all(b->ss_n, b->ss_sum, b->ss_sum2, b->active_len, x);

    /* 5. Truncate low-probability run lengths */
    truncate_runlengths(b);

    /* 6. Update outputs */
    b->t++;
    b->map_runlength = avx2_argmax(b->r, b->active_len);

    /* Compute P(recent change) = sum of short run length probabilities */
    b->p_changepoint = 0.0;
    size_t window = (b->active_len < 5) ? b->active_len : 5;
    for (size_t i = 0; i < window; i++) {
        b->p_changepoint += b->r[i];
    }
}

double bocpd_fast_change_prob(const bocpd_fast_t *b, size_t window) {
    if (!b || b->active_len == 0) return 0.0;

    double sum = 0.0;
    size_t max_idx = (window < b->active_len) ? window : b->active_len;

    for (size_t i = 0; i < max_idx; i++) {
        sum += b->r[i];
    }

    return sum;
}
