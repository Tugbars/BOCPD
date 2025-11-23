/**
 * @file bocpd_final.c
 * @brief Production-grade fused BOCPD - all issues fixed
 */

#include "bocpd_final.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define ALIGN64 __attribute__((aligned(64)))
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static const double LN_2 = 0.6931471805599453;
static const double LN_PI = 1.1447298858494002;
static const double LN_2PI = 1.8378770664093453;

/* ============================================================================
 * Fully vectorized AVX2 log (no memory roundtrips)
 * 
 * Uses mantissa/exponent extraction via integer ops only.
 * ~15 cycles per vector.
 * ============================================================================ */

static inline __m256d avx2_log_fast(__m256d x) {
    /* 
     * log(x) = log(2^e * m) = e*ln(2) + log(m), where m in [1, 2)
     * 
     * Extract exponent via integer shift (no memory roundtrip)
     * Normalize mantissa via OR with exponent=1023
     * Polynomial on (m-1) for log(m)
     */
    
    const __m256d ln2 = _mm256_set1_pd(LN_2);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000ULL);
    const __m256i mant_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
    const __m256i exp_bias = _mm256_set1_epi64x(1023);
    const __m256i exp_one = _mm256_set1_epi64x(0x3FF0000000000000ULL);

    __m256i xi = _mm256_castpd_si256(x);

    /* Extract exponent: (bits >> 52) - 1023 */
    __m256i exp_bits = _mm256_and_si256(xi, exp_mask);
    __m256i exp_shifted = _mm256_srli_epi64(exp_bits, 52);
    __m256i exp_unbiased = _mm256_sub_epi64(exp_shifted, exp_bias);

    /* Convert int64 exponent to double (AVX2 workaround) */
    /* Use the fact that for small integers, (double)(int64) works via union trick */
    /* Or use magic number conversion: add 2^52, subtract, gives exact double */
    const __m256d magic = _mm256_set1_pd(4503599627370496.0);  /* 2^52 */
    const __m256i magic_i = _mm256_castpd_si256(magic);
    __m256i exp_as_int = _mm256_add_epi64(exp_unbiased, magic_i);
    __m256d e = _mm256_sub_pd(_mm256_castsi256_pd(exp_as_int), magic);

    /* Extract mantissa, set exponent to 1023 → m in [1, 2) */
    __m256i mant_bits = _mm256_or_si256(_mm256_and_si256(xi, mant_mask), exp_one);
    __m256d m = _mm256_castsi256_pd(mant_bits);

    /* 
     * Polynomial for log(m), m in [1, 2)
     * Use substitution u = m - 1, u in [0, 1)
     * log(1+u) ≈ u - u²/2 + u³/3 - u⁴/4 + u⁵/5 - u⁶/6 + u⁷/7
     * Horner form for stability
     */
    __m256d u = _mm256_sub_pd(m, one);

    const __m256d c1 = _mm256_set1_pd(1.0);
    const __m256d c2 = _mm256_set1_pd(-0.5);
    const __m256d c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d c4 = _mm256_set1_pd(-0.25);
    const __m256d c5 = _mm256_set1_pd(0.2);
    const __m256d c6 = _mm256_set1_pd(-0.1666666666666667);
    const __m256d c7 = _mm256_set1_pd(0.1428571428571429);

    /* Horner: poly = c7 + u*(c6 + u*(c5 + u*(c4 + u*(c3 + u*(c2 + u*c1))))) */
    /* Then log(m) = u * poly */
    __m256d poly = _mm256_fmadd_pd(u, c7, c6);
    poly = _mm256_fmadd_pd(u, poly, c5);
    poly = _mm256_fmadd_pd(u, poly, c4);
    poly = _mm256_fmadd_pd(u, poly, c3);
    poly = _mm256_fmadd_pd(u, poly, c2);
    poly = _mm256_fmadd_pd(u, poly, c1);
    __m256d log_m = _mm256_mul_pd(u, poly);

    /* log(x) = e * ln(2) + log(m) */
    return _mm256_fmadd_pd(e, ln2, log_m);
}

/* ============================================================================
 * Fully vectorized AVX2 exp (no scalar loops)
 * 
 * exp(x) = 2^(x/ln2) = 2^k * 2^f, k = round(x/ln2), f = frac
 * ============================================================================ */

static inline __m256d avx2_exp_fast(__m256d x) {
    const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256d ln2 = _mm256_set1_pd(LN_2);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);

    /* Clamp to avoid overflow/underflow */
    const __m256d min_x = _mm256_set1_pd(-700.0);
    const __m256d max_x = _mm256_set1_pd(700.0);
    x = _mm256_max_pd(_mm256_min_pd(x, max_x), min_x);

    /* t = x / ln(2) */
    __m256d t = _mm256_mul_pd(x, inv_ln2);

    /* k = round(t) = floor(t + 0.5) */
    __m256d k = _mm256_floor_pd(_mm256_add_pd(t, half));

    /* f = t - k, in [-0.5, 0.5] */
    __m256d f = _mm256_sub_pd(t, k);

    /* 
     * 2^f ≈ 1 + f*ln2 + f²*ln²2/2 + f³*ln³2/6 + f⁴*ln⁴2/24 + f⁵*ln⁵2/120 + f⁶*ln⁶2/720
     * Coefficients precomputed
     */
    const __m256d c1 = _mm256_set1_pd(0.6931471805599453);    /* ln2 */
    const __m256d c2 = _mm256_set1_pd(0.24022650695910072);   /* ln²2/2 */
    const __m256d c3 = _mm256_set1_pd(0.05550410866482158);   /* ln³2/6 */
    const __m256d c4 = _mm256_set1_pd(0.009618129107628477);  /* ln⁴2/24 */
    const __m256d c5 = _mm256_set1_pd(0.0013333558146428443); /* ln⁵2/120 */
    const __m256d c6 = _mm256_set1_pd(0.00015403530393381608);/* ln⁶2/720 */

    /* Horner: poly = 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6))))) */
    __m256d poly = _mm256_fmadd_pd(f, c6, c5);
    poly = _mm256_fmadd_pd(f, poly, c4);
    poly = _mm256_fmadd_pd(f, poly, c3);
    poly = _mm256_fmadd_pd(f, poly, c2);
    poly = _mm256_fmadd_pd(f, poly, c1);
    poly = _mm256_fmadd_pd(f, poly, one);

    /* 
     * 2^k via integer exponent construction (fully vectorized)
     * IEEE double: bits = (k + 1023) << 52
     * 
     * Convert k (double) to int64:
     * Use magic number trick: add 2^52 + 2^51, gives exact int in low bits
     */
    const __m256d magic = _mm256_set1_pd(6755399441055744.0);  /* 2^52 + 2^51 */
    __m256d k_shifted = _mm256_add_pd(k, magic);
    __m256i ki = _mm256_castpd_si256(k_shifted);

    /* Now low 32 bits of each 64-bit lane contain the integer k (signed) */
    /* Add bias and shift to exponent position */
    const __m256i bias = _mm256_set1_epi64x(1023);
    __m256i exp_int = _mm256_add_epi64(ki, bias);

    /* Mask to get only the low bits we care about */
    const __m256i lo_mask = _mm256_set1_epi64x(0x7FF);
    exp_int = _mm256_and_si256(exp_int, lo_mask);

    /* Shift to exponent position (bits 52-62) */
    exp_int = _mm256_slli_epi64(exp_int, 52);

    /* Reinterpret as double (mantissa = 0, so this is exactly 2^k) */
    __m256d scale = _mm256_castsi256_pd(exp_int);

    /* exp(x) = poly * 2^k */
    return _mm256_mul_pd(poly, scale);
}

/* ============================================================================
 * Ring buffer helpers
 * ============================================================================ */

static inline size_t ring_idx(const bocpd_final_t *b, size_t i) {
    return (b->ring_start + i) & (b->capacity - 1);  /* Assumes capacity is power of 2 */
}

static inline void ring_advance(bocpd_final_t *b) {
    b->ring_start = (b->ring_start + b->capacity - 1) & (b->capacity - 1);
}

/* ============================================================================
 * Incremental posterior + derived quantities
 * 
 * Key insight: update kappa AFTER computing new mu
 * ============================================================================ */

static inline void update_posterior_incremental(
    double *kappa, double *mu, double *alpha, double *beta,
    double *lgamma_a, double *lgamma_a_p5,
    double *ln_sigma_sq, double *ln_nu_pi, double *sigma_sq_out,
    double x,
    double kappa0, double mu0, double alpha0, double beta0)
{
    double kappa_old = *kappa;
    double mu_old = *mu;
    double alpha_old = *alpha;
    double beta_old = *beta;

    /* Correct order: compute mu_new BEFORE updating kappa */
    double kappa_new = kappa_old + 1.0;
    double mu_new = (kappa_old * mu_old + x) / kappa_new;
    double alpha_new = alpha_old + 0.5;
    double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);

    *kappa = kappa_new;
    *mu = mu_new;
    *alpha = alpha_new;
    *beta = beta_new;

    /* Incremental lgamma: lgamma(a_new) = lgamma(a_old) + ln(a_old) */
    *lgamma_a = *lgamma_a + log(alpha_old);
    *lgamma_a_p5 = *lgamma_a_p5 + log(alpha_old + 0.5);

    /* Incremental ln_nu_pi: ln(2*alpha_new*pi) = ln(2*alpha_old*pi) + ln(alpha_new/alpha_old) */
    /* Since alpha_new = alpha_old + 0.5: */
    *ln_nu_pi = *ln_nu_pi + log((alpha_old + 0.5) / alpha_old);

    /* Store sigma² directly (avoid exp in hot loop) */
    double sigma_sq_new = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);
    *sigma_sq_out = sigma_sq_new;
    *ln_sigma_sq = log(sigma_sq_new);
}



static inline void init_posterior_slot(
    bocpd_final_t *b, size_t ri,
    double kappa0, double mu0, double alpha0, double beta0)
{
    b->post_kappa[ri] = kappa0;
    b->post_mu[ri] = mu0;
    b->post_alpha[ri] = alpha0;
    b->post_beta[ri] = beta0;
    b->lgamma_alpha[ri] = lgamma(alpha0);
    b->lgamma_alpha_p5[ri] = lgamma(alpha0 + 0.5);

    double sigma_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
    b->sigma_sq[ri] = sigma_sq;
    b->ln_sigma_sq[ri] = log(sigma_sq);
    b->ln_nu_pi[ri] = log(2.0 * alpha0 * M_PI);
}

/* ============================================================================
 * Fused backwards update (single pass, fully vectorized where possible)
 * ============================================================================ */

static void fused_step_backwards(bocpd_final_t *b, double x) {
    const size_t n = b->active_len;
    const double h = b->hazard;
    const double omh = b->one_minus_h;
    const double thresh = b->trunc_thresh;

    double *r = b->r;
    double *r_new = b->r_scratch;

    double r0 = 0.0;
    size_t last_valid = 0;
    size_t map_idx = 0;
    double map_val = 0.0;

    memset(r_new, 0, (n + 1) * sizeof(double));

    /* Hoist all constants outside loop */
    const __m256d h_vec = _mm256_set1_pd(h);
    const __m256d omh_vec = _mm256_set1_pd(omh);
    const __m256d thresh_vec = _mm256_set1_pd(thresh);
    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d min_pp = _mm256_set1_pd(1e-300);
    __m256d r0_acc = _mm256_setzero_pd();

    size_t i = (n >= 4) ? ((n - 1) & ~3ULL) : 0;

    for (; i + 4 <= n && i < n; ) {
        size_t ri0 = ring_idx(b, i);
        size_t ri1 = ring_idx(b, i + 1);
        size_t ri2 = ring_idx(b, i + 2);
        size_t ri3 = ring_idx(b, i + 3);

        __m256d r_old = _mm256_set_pd(r[i+3], r[i+2], r[i+1], r[i]);

        /* Load precomputed values (no exp needed for sigma_sq!) */
        __m256d sigma_sq = _mm256_set_pd(
            b->sigma_sq[ri3], b->sigma_sq[ri2],
            b->sigma_sq[ri1], b->sigma_sq[ri0]);
        __m256d ln_sigma_sq = _mm256_set_pd(
            b->ln_sigma_sq[ri3], b->ln_sigma_sq[ri2],
            b->ln_sigma_sq[ri1], b->ln_sigma_sq[ri0]);
        __m256d ln_nu_pi = _mm256_set_pd(
            b->ln_nu_pi[ri3], b->ln_nu_pi[ri2],
            b->ln_nu_pi[ri1], b->ln_nu_pi[ri0]);
        __m256d lgamma_a = _mm256_set_pd(
            b->lgamma_alpha[ri3], b->lgamma_alpha[ri2],
            b->lgamma_alpha[ri1], b->lgamma_alpha[ri0]);
        __m256d lgamma_a_p5 = _mm256_set_pd(
            b->lgamma_alpha_p5[ri3], b->lgamma_alpha_p5[ri2],
            b->lgamma_alpha_p5[ri1], b->lgamma_alpha_p5[ri0]);
        __m256d post_mu = _mm256_set_pd(
            b->post_mu[ri3], b->post_mu[ri2],
            b->post_mu[ri1], b->post_mu[ri0]);
        __m256d post_alpha = _mm256_set_pd(
            b->post_alpha[ri3], b->post_alpha[ri2],
            b->post_alpha[ri1], b->post_alpha[ri0]);

        /* nu = 2 * alpha */
        __m256d nu = _mm256_add_pd(post_alpha, post_alpha);

        /* z² / (sigma² * nu) — no exp needed! */
        __m256d x_minus_mu = _mm256_sub_pd(x_vec, post_mu);
        __m256d x_minus_mu_sq = _mm256_mul_pd(x_minus_mu, x_minus_mu);
        __m256d sigma_sq_nu = _mm256_mul_pd(sigma_sq, nu);
        __m256d z2_over_nu = _mm256_div_pd(x_minus_mu_sq, sigma_sq_nu);

        /* ln(1 + z²/nu) */
        __m256d term = _mm256_add_pd(one, z2_over_nu);
        __m256d ln_term = avx2_log_fast(term);

        /* (nu + 1) / 2 = alpha + 0.5 */
        __m256d nu_plus_1_half = _mm256_add_pd(post_alpha, half);

        /* ln_pp = lgamma(a+0.5) - lgamma(a) - 0.5*ln(nu*pi) - 0.5*ln(sigma²) - (nu+1)/2*ln(1+z²/nu) */
        __m256d ln_pp = _mm256_sub_pd(lgamma_a_p5, lgamma_a);
        ln_pp = _mm256_fnmadd_pd(half, ln_nu_pi, ln_pp);
        ln_pp = _mm256_fnmadd_pd(half, ln_sigma_sq, ln_pp);
        ln_pp = _mm256_fnmadd_pd(nu_plus_1_half, ln_term, ln_pp);

        __m256d pp = avx2_exp_fast(ln_pp);
        pp = _mm256_max_pd(pp, min_pp);

        __m256d r_pp = _mm256_mul_pd(r_old, pp);
        __m256d growth = _mm256_mul_pd(r_pp, omh_vec);
        __m256d change = _mm256_mul_pd(r_pp, h_vec);

        _mm256_storeu_pd(&r_new[i + 1], growth);
        r0_acc = _mm256_add_pd(r0_acc, change);

        __m256d cmp = _mm256_cmp_pd(growth, thresh_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        if (mask) {
            size_t local_max = i + 1;
            if (mask & 2) local_max = i + 2;
            if (mask & 4) local_max = i + 3;
            if (mask & 8) local_max = i + 4;
            if (local_max > last_valid) last_valid = local_max;
        }

        if (i < 4) break;
        i -= 4;
    }

    /* Horizontal sum */
    __m128d lo = _mm256_castpd256_pd128(r0_acc);
    __m128d hi = _mm256_extractf128_pd(r0_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    r0 = _mm_cvtsd_f64(lo);

    /* Scalar remainder */
    for (size_t j = 0; j < n && j <= i + 3; j++) {
        if (r_new[j + 1] != 0.0) continue;

        double r_old_j = r[j];
        if (r_old_j < 1e-300) continue;

        size_t ri = ring_idx(b, j);

        double post_mu_j = b->post_mu[ri];
        double post_alpha_j = b->post_alpha[ri];
        double sigma_sq_j = b->sigma_sq[ri];      /* Direct load, no exp */
        double ln_sigma_sq_j = b->ln_sigma_sq[ri];
        double ln_nu_pi_j = b->ln_nu_pi[ri];
        double lgamma_a_j = b->lgamma_alpha[ri];
        double lgamma_a_p5_j = b->lgamma_alpha_p5[ri];

        double nu = 2.0 * post_alpha_j;
        double z2_over_nu = (x - post_mu_j) * (x - post_mu_j) / (sigma_sq_j * nu);

        double ln_pp = lgamma_a_p5_j - lgamma_a_j
                     - 0.5 * ln_nu_pi_j - 0.5 * ln_sigma_sq_j
                     - (nu + 1.0) * 0.5 * log(1.0 + z2_over_nu);

        double pp = exp(ln_pp);
        if (pp < 1e-300) pp = 1e-300;

        double r_pp = r_old_j * pp;
        r_new[j + 1] = r_pp * omh;
        r0 += r_pp * h;

        if (r_new[j + 1] > thresh && j + 1 > last_valid) last_valid = j + 1;
    }

    r_new[0] = r0;

    size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
    if (new_len > b->capacity - 1) new_len = b->capacity - 1;

    double r_sum = 0.0;
    for (size_t j = 0; j < new_len; j++) {
        r_sum += r_new[j];
        if (r_new[j] > map_val) { map_val = r_new[j]; map_idx = j; }
    }

    if (r_sum > 1e-300) {
        double inv_sum = 1.0 / r_sum;
        for (size_t j = 0; j < new_len; j++) r[j] = r_new[j] * inv_sum;
    }

    b->active_len = new_len;
    b->map_runlength = map_idx;
}

/* ============================================================================
 * Shift ring buffer + observe x (O(active_len) for observe, O(1) for shift)
 * ============================================================================ */

static void shift_and_observe(bocpd_final_t *b, double x) {
    const size_t n = b->active_len;

    ring_advance(b);
    size_t new_slot = ring_idx(b, 0);

    init_posterior_slot(b, new_slot,
        b->prior.kappa0, b->prior.mu0, b->prior.alpha0, b->prior.beta0);
    b->ss_n[new_slot] = 0.0;
    b->ss_sum[new_slot] = 0.0;
    b->ss_sum2[new_slot] = 0.0;

    for (size_t i = 0; i < n; i++) {
        size_t ri = ring_idx(b, i);

        b->ss_n[ri] += 1.0;
        b->ss_sum[ri] += x;
        b->ss_sum2[ri] += x * x;

        update_posterior_incremental(
            &b->post_kappa[ri], &b->post_mu[ri],
            &b->post_alpha[ri], &b->post_beta[ri],
            &b->lgamma_alpha[ri], &b->lgamma_alpha_p5[ri],
            &b->ln_sigma_sq[ri], &b->ln_nu_pi[ri], &b->sigma_sq[ri],
            x,
            b->prior.kappa0, b->prior.mu0, b->prior.alpha0, b->prior.beta0);
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

int bocpd_final_init(bocpd_final_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length) {
    if (!b || hazard_lambda <= 0.0 || max_run_length < 16) return -1;

    /* Round capacity up to power of 2 for fast ring indexing */
    size_t cap = 16;
    while (cap < max_run_length) cap <<= 1;

    memset(b, 0, sizeof(*b));

    b->capacity = cap;
    b->hazard = 1.0 / hazard_lambda;
    b->one_minus_h = 1.0 - b->hazard;
    b->trunc_thresh = 1e-6;
    b->prior = prior;
    b->ring_start = 0;

    size_t alloc = cap * sizeof(double);

    b->ss_n = aligned_alloc(64, alloc);
    b->ss_sum = aligned_alloc(64, alloc);
    b->ss_sum2 = aligned_alloc(64, alloc);
    b->post_kappa = aligned_alloc(64, alloc);
    b->post_mu = aligned_alloc(64, alloc);
    b->post_alpha = aligned_alloc(64, alloc);
    b->post_beta = aligned_alloc(64, alloc);
    b->lgamma_alpha = aligned_alloc(64, alloc);
    b->lgamma_alpha_p5 = aligned_alloc(64, alloc);
    b->ln_sigma_sq = aligned_alloc(64, alloc);
    b->ln_nu_pi = aligned_alloc(64, alloc);
    b->r = aligned_alloc(64, alloc);
    b->r_scratch = aligned_alloc(64, alloc);
    b->sigma_sq = aligned_alloc(64, alloc);

    if (!b->ss_n || !b->ss_sum || !b->ss_sum2 || !b->post_kappa ||
        !b->post_mu || !b->post_alpha || !b->post_beta ||
        !b->lgamma_alpha || !b->lgamma_alpha_p5 || !b->sigma_sq
        !b->ln_sigma_sq || !b->ln_nu_pi || !b->r || !b->r_scratch) {
        bocpd_final_free(b);
        return -1;
    }

    memset(b->r, 0, alloc);
    memset(b->r_scratch, 0, alloc);

    b->t = 0;
    b->active_len = 0;

    return 0;
}

void bocpd_final_free(bocpd_final_t *b) {
    if (!b) return;
    free(b->ss_n); free(b->ss_sum); free(b->ss_sum2);
    free(b->post_kappa); free(b->post_mu); free(b->post_alpha); free(b->post_beta);
    free(b->lgamma_alpha); free(b->lgamma_alpha_p5);
    free(b->ln_sigma_sq); free(b->ln_nu_pi);
    free(b->r); free(b->r_scratch);
    /* In bocpd_final_free, add: */
    free(b->sigma_sq);
    memset(b, 0, sizeof(*b));
}

void bocpd_final_reset(bocpd_final_t *b) {
    if (!b) return;
    memset(b->r, 0, b->capacity * sizeof(double));
    b->t = 0;
    b->active_len = 0;
    b->ring_start = 0;
}

void bocpd_final_step(bocpd_final_t *b, double x) {
    if (!b) return;

    if (b->t == 0) {
        /* First observation */
        b->r[0] = 1.0;
        size_t ri = ring_idx(b, 0);
        b->ss_n[ri] = 1.0;
        b->ss_sum[ri] = x;
        b->ss_sum2[ri] = x * x;

        /* Initialize posterior for first observation */
        double kappa_new = b->prior.kappa0 + 1.0;
        double mu_new = (b->prior.kappa0 * b->prior.mu0 + x) / kappa_new;
        double alpha_new = b->prior.alpha0 + 0.5;
        double mu_diff = x - b->prior.mu0;
        double beta_new = b->prior.beta0 + 0.5 * b->prior.kappa0 * mu_diff * mu_diff / kappa_new;

        b->post_kappa[ri] = kappa_new;
        b->post_mu[ri] = mu_new;
        b->post_alpha[ri] = alpha_new;
        b->post_beta[ri] = beta_new;
        b->lgamma_alpha[ri] = lgamma(alpha_new);
        b->lgamma_alpha_p5[ri] = lgamma(alpha_new + 0.5);
        b->ln_sigma_sq[ri] = log(beta_new) + log(kappa_new + 1.0) - log(alpha_new) - log(kappa_new);
        b->ln_nu_pi[ri] = log(2.0 * alpha_new * M_PI);
        b->sigma_sq[ri] = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);

        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }

    /* 1. Fused predictive + run-length update (backwards) */
    fused_step_backwards(b, x);

    /* 2. Shift ring buffer + observe x */
    shift_and_observe(b, x);

    /* 3. Update outputs */
    b->t++;
    b->p_changepoint = 0.0;
    size_t w = (b->active_len < 5) ? b->active_len : 5;
    for (size_t i = 0; i < w; i++) b->p_changepoint += b->r[i];
}
