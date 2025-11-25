/**
 * @file bocpd_ultra_opt_asm.c
 * @brief Ultra-optimized BOCPD implementation with AVX2 SIMD
 *
 * Features:
 *   - Single mega-block allocation (fast init, no fragmentation)
 *   - Interleaved data layout for cache efficiency
 *   - AVX2 SIMD with 2x loop unrolling
 *   - Estrin's scheme for polynomial evaluation
 *   - Fused truncation tracking (no second pass)
 *   - Pool allocator for multi-detector scenarios
 *   - Precomputed prior lgamma values
 */

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "bocpd_asm.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/*=============================================================================
 * Compiler Hints
 *=============================================================================*/

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/*=============================================================================
 * Fast Scalar Log (for posterior updates)
 *=============================================================================*/

static inline double fast_log_scalar(double x)
{
    union
    {
        double d;
        uint64_t u;
    } u = {.d = x};

    int64_t e = (int64_t)((u.u >> 52) & 0x7FF) - 1023;
    u.u = (u.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = u.d;

    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;

    double poly = 1.0 + t2 * (0.3333333333333333 +
                              t2 * (0.2 +
                                    t2 * (0.1428571428571429 +
                                          t2 * 0.1111111111111111)));

    return (double)e * 0.6931471805599453 + 2.0 * t * poly;
}

/*=============================================================================
 * Ring Buffer Helpers
 *=============================================================================*/

static inline size_t ring_idx(const bocpd_asm_t *b, size_t i)
{
    return (b->ring_start + i) & (b->capacity - 1);
}

static inline void ring_advance(bocpd_asm_t *b)
{
    b->ring_start = (b->ring_start + b->capacity - 1) & (b->capacity - 1);
}

/*=============================================================================
 * Linearize Ring Buffer to Interleaved Format
 *=============================================================================*/

static void linearize_ring(bocpd_asm_t *b)
{
    const size_t n = b->active_len;
    const size_t cap = b->capacity;
    const size_t start = b->ring_start;
    const size_t mask = cap - 1;

    double *out = b->lin_interleaved;

    /* Pad to n_padded + 8 for Block B safety */
    size_t padded = ((n + 7) & ~7ULL) + 8;

    for (size_t i = 0; i < padded; i += 4)
    {
        size_t block = i / 4;
        size_t base = block * 16;

        for (size_t j = 0; j < 4; j++)
        {
            size_t idx = i + j;

            if (idx < n)
            {
                size_t ri = (start + idx) & mask;
                out[base + 0 + j] = b->post_mu[ri];
                out[base + 4 + j] = b->C1[ri];
                out[base + 8 + j] = b->C2[ri];
                out[base + 12 + j] = b->inv_sigma_sq_nu[ri];
            }
            else
            {
                /* Safe padding: forces pp → 0 */
                out[base + 0 + j] = 0.0;
                out[base + 4 + j] = -INFINITY;
                out[base + 8 + j] = 1.0;
                out[base + 12 + j] = 1.0;
            }
        }
    }
}

/*=============================================================================
 * Posterior Updates
 *=============================================================================*/

static inline void update_posterior_incremental(bocpd_asm_t *b, size_t ri, double x)
{
    double kappa_old = b->post_kappa[ri];
    double mu_old = b->post_mu[ri];
    double alpha_old = b->post_alpha[ri];
    double beta_old = b->post_beta[ri];

    /* Welford's online update */
    double kappa_new = kappa_old + 1.0;
    double mu_new = (kappa_old * mu_old + x) / kappa_new;
    double alpha_new = alpha_old + 0.5;
    double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);

    b->post_kappa[ri] = kappa_new;
    b->post_mu[ri] = mu_new;
    b->post_alpha[ri] = alpha_new;
    b->post_beta[ri] = beta_new;

    /* Incremental lgamma update */
    b->lgamma_alpha[ri] += fast_log_scalar(alpha_old);
    b->lgamma_alpha_p5[ri] += fast_log_scalar(alpha_old + 0.5);

    /* Precompute Student-t constants */
    double sigma_sq = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);
    double nu = 2.0 * alpha_new;

    b->sigma_sq[ri] = sigma_sq;
    b->inv_sigma_sq_nu[ri] = 1.0 / (sigma_sq * nu);

    double ln_nu_pi = fast_log_scalar(nu * M_PI);
    double ln_sigma_sq = fast_log_scalar(sigma_sq);

    b->C1[ri] = b->lgamma_alpha_p5[ri] - b->lgamma_alpha[ri] - 0.5 * ln_nu_pi - 0.5 * ln_sigma_sq;
    b->C2[ri] = alpha_new + 0.5;
}

static inline void init_posterior_slot(bocpd_asm_t *b, size_t ri)
{
    double kappa0 = b->prior.kappa0;
    double mu0 = b->prior.mu0;
    double alpha0 = b->prior.alpha0;
    double beta0 = b->prior.beta0;

    b->post_kappa[ri] = kappa0;
    b->post_mu[ri] = mu0;
    b->post_alpha[ri] = alpha0;
    b->post_beta[ri] = beta0;

    /* Use precomputed lgamma values from prior */
    b->lgamma_alpha[ri] = b->prior_lgamma_alpha;
    b->lgamma_alpha_p5[ri] = b->prior_lgamma_alpha_p5;

    double sigma_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
    double nu = 2.0 * alpha0;

    b->sigma_sq[ri] = sigma_sq;
    b->inv_sigma_sq_nu[ri] = 1.0 / (sigma_sq * nu);

    double ln_nu_pi = fast_log_scalar(nu * M_PI);
    double ln_sigma_sq = fast_log_scalar(sigma_sq);

    b->C1[ri] = b->prior_lgamma_alpha_p5 - b->prior_lgamma_alpha - 0.5 * ln_nu_pi - 0.5 * ln_sigma_sq;
    b->C2[ri] = alpha0 + 0.5;
}

/*=============================================================================
 * Fused SIMD Kernel
 *=============================================================================*/

static void fused_step_simd(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;
    if (n == 0)
        return;

    const double h = b->hazard;
    const double omh = b->one_minus_h;
    const double thresh = b->trunc_thresh;

    linearize_ring(b);

    double *r = b->r;
    double *r_new = b->r_scratch;

    const size_t n_padded = (n + 7) & ~7ULL;

    /* Zero-pad r beyond active_len */
    for (size_t i = n; i < n_padded + 8; i++)
        r[i] = 0.0;

    /* Zero r_new with margin for Block B writes */
    {
        __m256d zero = _mm256_setzero_pd();
        for (size_t i = 0; i < n_padded + 16; i += 4)
            _mm256_storeu_pd(&r_new[i], zero);
    }

#if BOCPD_USE_ASM
    /*=========================================================================
     * Assembly Kernel Path
     *=========================================================================*/
    {
        double r0_result = 0.0;
        double max_growth_result = 0.0;
        size_t max_idx_result = 0;
        size_t last_valid_result = 0;

        bocpd_kernel_args_t args = {
            .lin_interleaved = b->lin_interleaved,
            .r_old = r,
            .x = x,
            .h = h,
            .one_minus_h = omh,
            .trunc_thresh = thresh,
            .n_padded = n_padded,
            .r_new = r_new,
            .r0_out = &r0_result,
            .max_growth_out = &max_growth_result,
            .max_idx_out = &max_idx_result,
            .last_valid_out = &last_valid_result};

        bocpd_fused_loop_avx2_generic(&args);

        double r0 = r0_result;
        r_new[0] = r0;

        size_t last_valid = last_valid_result;
        if (r0 > thresh && last_valid == 0)
            last_valid = 1;

        size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
        if (new_len > b->capacity)
            new_len = b->capacity;

        size_t new_len_padded = (new_len + 7) & ~7ULL;

        /* Normalize */
        __m256d sum_acc = _mm256_setzero_pd();
        for (size_t i = 0; i < new_len_padded; i += 4)
            sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[i]));

        __m128d lo = _mm256_castpd256_pd128(sum_acc);
        __m128d hi = _mm256_extractf128_pd(sum_acc, 1);
        lo = _mm_add_pd(lo, hi);
        lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
        double r_sum = _mm_cvtsd_f64(lo);

        if (r_sum > 1e-300)
        {
            __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
            for (size_t i = 0; i < new_len_padded; i += 4)
            {
                __m256d rv = _mm256_loadu_pd(&r_new[i]);
                _mm256_storeu_pd(&r[i], _mm256_mul_pd(rv, inv_sum));
            }
        }

        b->active_len = new_len;

        double map_val = r0;
        size_t map_idx = 0;
        if (max_growth_result > map_val)
        {
            map_val = max_growth_result;
            map_idx = max_idx_result;
        }
        b->map_runlength = map_idx;

        return;
    }
#endif /* BOCPD_USE_ASM */

    /*=========================================================================
     * C Implementation Path
     *=========================================================================*/
    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d h_vec = _mm256_set1_pd(h);
    const __m256d omh_vec = _mm256_set1_pd(omh);
    const __m256d thresh_vec = _mm256_set1_pd(thresh);
    const __m256d min_pp = _mm256_set1_pd(1e-300);
    const __m256d const_one = _mm256_set1_pd(1.0);

    __m256d r0_acc_a = _mm256_setzero_pd();
    __m256d r0_acc_b = _mm256_setzero_pd();
    __m256d max_growth_a = _mm256_setzero_pd();
    __m256d max_growth_b = _mm256_setzero_pd();
    __m256i max_idx_a = _mm256_setzero_si256();
    __m256i max_idx_b = _mm256_setzero_si256();

    __m256i idx_vec_a = _mm256_set_epi64x(4, 3, 2, 1);
    __m256i idx_vec_b = _mm256_set_epi64x(8, 7, 6, 5);
    const __m256i idx_increment = _mm256_set1_epi64x(8);

    size_t last_valid = 0;

    /* Polynomial constants */
    const __m256d log1p_c2 = _mm256_set1_pd(-0.5);
    const __m256d log1p_c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d log1p_c4 = _mm256_set1_pd(-0.25);
    const __m256d log1p_c5 = _mm256_set1_pd(0.2);
    const __m256d log1p_c6 = _mm256_set1_pd(-0.1666666666666667);

    const __m256d exp_inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256d exp_min_x = _mm256_set1_pd(-700.0);
    const __m256d exp_max_x = _mm256_set1_pd(700.0);
    const __m256d exp_c1 = _mm256_set1_pd(0.6931471805599453);
    const __m256d exp_c2 = _mm256_set1_pd(0.24022650695910072);
    const __m256d exp_c3 = _mm256_set1_pd(0.05550410866482158);
    const __m256d exp_c4 = _mm256_set1_pd(0.009618129107628477);
    const __m256d exp_c5 = _mm256_set1_pd(0.0013333558146428443);
    const __m256d exp_c6 = _mm256_set1_pd(0.00015403530393381608);
    const __m256i exp_bias_int = _mm256_set1_epi64x(1023);

    for (size_t i = 0; i < n_padded; i += 8)
    {
        /* Block A: indices i..i+3 */
        size_t block_a = i / 4;
        double *base_a = &b->lin_interleaved[block_a * 16];

        __m256d mu_a = _mm256_loadu_pd(base_a + 0);
        __m256d C1_a = _mm256_loadu_pd(base_a + 4);
        __m256d C2_a = _mm256_loadu_pd(base_a + 8);
        __m256d inv_ssn_a = _mm256_loadu_pd(base_a + 12);
        __m256d r_old_a = _mm256_loadu_pd(&r[i]);

        __m256d z_a = _mm256_sub_pd(x_vec, mu_a);
        __m256d z2_a = _mm256_mul_pd(z_a, z_a);
        __m256d t_a = _mm256_mul_pd(z2_a, inv_ssn_a);

        /* log1p polynomial */
        __m256d poly_a = _mm256_fmadd_pd(t_a, log1p_c6, log1p_c5);
        poly_a = _mm256_fmadd_pd(t_a, poly_a, log1p_c4);
        poly_a = _mm256_fmadd_pd(t_a, poly_a, log1p_c3);
        poly_a = _mm256_fmadd_pd(t_a, poly_a, log1p_c2);
        poly_a = _mm256_fmadd_pd(t_a, poly_a, const_one);
        __m256d log1p_t_a = _mm256_mul_pd(t_a, poly_a);

        __m256d ln_pp_a = _mm256_fnmadd_pd(C2_a, log1p_t_a, C1_a);

        /* exp with Estrin's scheme */
        __m256d x_clamped_a = _mm256_max_pd(_mm256_min_pd(ln_pp_a, exp_max_x), exp_min_x);
        __m256d t_exp_a = _mm256_mul_pd(x_clamped_a, exp_inv_ln2);
        __m256d k_a = _mm256_round_pd(t_exp_a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d f_a = _mm256_sub_pd(t_exp_a, k_a);

        __m256d f2_a = _mm256_mul_pd(f_a, f_a);
        __m256d p01_a = _mm256_fmadd_pd(f_a, exp_c1, const_one);
        __m256d p23_a = _mm256_fmadd_pd(f_a, exp_c3, exp_c2);
        __m256d p45_a = _mm256_fmadd_pd(f_a, exp_c5, exp_c4);
        __m256d q0123_a = _mm256_fmadd_pd(f2_a, p23_a, p01_a);
        __m256d q456_a = _mm256_fmadd_pd(f2_a, exp_c6, p45_a);
        __m256d f4_a = _mm256_mul_pd(f2_a, f2_a);
        __m256d exp_poly_a = _mm256_fmadd_pd(f4_a, q456_a, q0123_a);

        __m128i k_int32_a = _mm256_cvtpd_epi32(k_a);
        __m256i k_int64_a = _mm256_cvtepi32_epi64(k_int32_a);
        __m256i exp_biased_a = _mm256_add_epi64(k_int64_a, exp_bias_int);
        __m256i exp_bits_a = _mm256_slli_epi64(exp_biased_a, 52);
        __m256d scale_a = _mm256_castsi256_pd(exp_bits_a);

        __m256d pp_a = _mm256_mul_pd(exp_poly_a, scale_a);
        pp_a = _mm256_max_pd(pp_a, min_pp);

        __m256d r_pp_a = _mm256_mul_pd(r_old_a, pp_a);
        __m256d growth_a = _mm256_mul_pd(r_pp_a, omh_vec);
        __m256d change_a = _mm256_mul_pd(r_pp_a, h_vec);

        _mm256_storeu_pd(&r_new[i + 1], growth_a);
        r0_acc_a = _mm256_add_pd(r0_acc_a, change_a);

        __m256d cmp_a = _mm256_cmp_pd(growth_a, max_growth_a, _CMP_GT_OQ);
        max_growth_a = _mm256_blendv_pd(max_growth_a, growth_a, cmp_a);
        max_idx_a = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(max_idx_a),
            _mm256_castsi256_pd(idx_vec_a), cmp_a));

        __m256d thresh_cmp_a = _mm256_cmp_pd(growth_a, thresh_vec, _CMP_GT_OQ);
        int mask_a = _mm256_movemask_pd(thresh_cmp_a);
        if (mask_a)
        {
            if (mask_a & 8)
                last_valid = i + 4;
            else if (mask_a & 4)
                last_valid = i + 3;
            else if (mask_a & 2)
                last_valid = i + 2;
            else if (mask_a & 1)
                last_valid = i + 1;
        }

        /* Block B: indices i+4..i+7 */
        size_t block_b = (i / 4) + 1;
        double *base_b = &b->lin_interleaved[block_b * 16];

        __m256d mu_b = _mm256_loadu_pd(base_b + 0);
        __m256d C1_b = _mm256_loadu_pd(base_b + 4);
        __m256d C2_b = _mm256_loadu_pd(base_b + 8);
        __m256d inv_ssn_b = _mm256_loadu_pd(base_b + 12);
        __m256d r_old_b = _mm256_loadu_pd(&r[i + 4]);

        __m256d z_b = _mm256_sub_pd(x_vec, mu_b);
        __m256d z2_b = _mm256_mul_pd(z_b, z_b);
        __m256d t_b = _mm256_mul_pd(z2_b, inv_ssn_b);

        __m256d poly_b = _mm256_fmadd_pd(t_b, log1p_c6, log1p_c5);
        poly_b = _mm256_fmadd_pd(t_b, poly_b, log1p_c4);
        poly_b = _mm256_fmadd_pd(t_b, poly_b, log1p_c3);
        poly_b = _mm256_fmadd_pd(t_b, poly_b, log1p_c2);
        poly_b = _mm256_fmadd_pd(t_b, poly_b, const_one);
        __m256d log1p_t_b = _mm256_mul_pd(t_b, poly_b);

        __m256d ln_pp_b = _mm256_fnmadd_pd(C2_b, log1p_t_b, C1_b);

        __m256d x_clamped_b = _mm256_max_pd(_mm256_min_pd(ln_pp_b, exp_max_x), exp_min_x);
        __m256d t_exp_b = _mm256_mul_pd(x_clamped_b, exp_inv_ln2);
        __m256d k_b = _mm256_round_pd(t_exp_b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d f_b = _mm256_sub_pd(t_exp_b, k_b);

        __m256d f2_b = _mm256_mul_pd(f_b, f_b);
        __m256d p01_b = _mm256_fmadd_pd(f_b, exp_c1, const_one);
        __m256d p23_b = _mm256_fmadd_pd(f_b, exp_c3, exp_c2);
        __m256d p45_b = _mm256_fmadd_pd(f_b, exp_c5, exp_c4);
        __m256d q0123_b = _mm256_fmadd_pd(f2_b, p23_b, p01_b);
        __m256d q456_b = _mm256_fmadd_pd(f2_b, exp_c6, p45_b);
        __m256d f4_b = _mm256_mul_pd(f2_b, f2_b);
        __m256d exp_poly_b = _mm256_fmadd_pd(f4_b, q456_b, q0123_b);

        __m128i k_int32_b = _mm256_cvtpd_epi32(k_b);
        __m256i k_int64_b = _mm256_cvtepi32_epi64(k_int32_b);
        __m256i exp_biased_b = _mm256_add_epi64(k_int64_b, exp_bias_int);
        __m256i exp_bits_b = _mm256_slli_epi64(exp_biased_b, 52);
        __m256d scale_b = _mm256_castsi256_pd(exp_bits_b);

        __m256d pp_b = _mm256_mul_pd(exp_poly_b, scale_b);
        pp_b = _mm256_max_pd(pp_b, min_pp);

        __m256d r_pp_b = _mm256_mul_pd(r_old_b, pp_b);
        __m256d growth_b = _mm256_mul_pd(r_pp_b, omh_vec);
        __m256d change_b = _mm256_mul_pd(r_pp_b, h_vec);

        _mm256_storeu_pd(&r_new[i + 5], growth_b);
        r0_acc_b = _mm256_add_pd(r0_acc_b, change_b);

        __m256d cmp_b = _mm256_cmp_pd(growth_b, max_growth_b, _CMP_GT_OQ);
        max_growth_b = _mm256_blendv_pd(max_growth_b, growth_b, cmp_b);
        max_idx_b = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(max_idx_b),
            _mm256_castsi256_pd(idx_vec_b), cmp_b));

        __m256d thresh_cmp_b = _mm256_cmp_pd(growth_b, thresh_vec, _CMP_GT_OQ);
        int mask_b = _mm256_movemask_pd(thresh_cmp_b);
        if (mask_b)
        {
            if (mask_b & 8)
                last_valid = i + 8;
            else if (mask_b & 4)
                last_valid = i + 7;
            else if (mask_b & 2)
                last_valid = i + 6;
            else if (mask_b & 1)
                last_valid = i + 5;
        }

        idx_vec_a = _mm256_add_epi64(idx_vec_a, idx_increment);
        idx_vec_b = _mm256_add_epi64(idx_vec_b, idx_increment);
    }

    /* Horizontal sum for r0 */
    __m256d r0_combined = _mm256_add_pd(r0_acc_a, r0_acc_b);
    __m128d lo = _mm256_castpd256_pd128(r0_combined);
    __m128d hi = _mm256_extractf128_pd(r0_combined, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r0 = _mm_cvtsd_f64(lo);

    r_new[0] = r0;

    if (r0 > thresh && last_valid == 0)
        last_valid = 1;

    /* Extract MAP (unaligned stores - safe) */
    double max_arr_a[4], max_arr_b[4];
    int64_t idx_arr_a[4], idx_arr_b[4];
    _mm256_storeu_pd(max_arr_a, max_growth_a);
    _mm256_storeu_pd(max_arr_b, max_growth_b);
    _mm256_storeu_si256((__m256i *)idx_arr_a, max_idx_a);
    _mm256_storeu_si256((__m256i *)idx_arr_b, max_idx_b);

    double map_val = r0;
    size_t map_idx = 0;

    for (int j = 0; j < 4; j++)
    {
        if (max_arr_a[j] > map_val)
        {
            map_val = max_arr_a[j];
            map_idx = idx_arr_a[j];
        }
        if (max_arr_b[j] > map_val)
        {
            map_val = max_arr_b[j];
            map_idx = idx_arr_b[j];
        }
    }

    size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
    if (new_len > b->capacity)
        new_len = b->capacity;

    size_t new_len_padded = (new_len + 7) & ~7ULL;

    /* Normalize */
    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t i = 0; i < new_len_padded; i += 4)
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[i]));

    lo = _mm256_castpd256_pd128(sum_acc);
    hi = _mm256_extractf128_pd(sum_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r_sum = _mm_cvtsd_f64(lo);

    if (r_sum > 1e-300)
    {
        __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
        for (size_t i = 0; i < new_len_padded; i += 4)
        {
            __m256d rv = _mm256_loadu_pd(&r_new[i]);
            _mm256_storeu_pd(&r[i], _mm256_mul_pd(rv, inv_sum));
        }
    }

    b->active_len = new_len;
    b->map_runlength = map_idx;
}

/*=============================================================================
 * Shift and Observe (CORRECT version - loop 0..n-1)
 *=============================================================================*/

static void shift_and_observe(bocpd_asm_t *b, double x)
{
    const size_t n = b->active_len;

    ring_advance(b);
    size_t new_slot = ring_idx(b, 0);

    init_posterior_slot(b, new_slot);
    b->ss_n[new_slot] = 0.0;
    b->ss_sum[new_slot] = 0.0;
    b->ss_sum2[new_slot] = 0.0;

    /* Update existing slots - these are at logical indices 0..n-1 AFTER advance */
    for (size_t i = 0; i < n; i++)
    {
        size_t ri = ring_idx(b, i);

        b->ss_n[ri] += 1.0;
        b->ss_sum[ri] += x;
        b->ss_sum2[ri] += x * x;

        update_posterior_incremental(b, ri, x);
    }
}

/*=============================================================================
 * Public API
 *=============================================================================*/

int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length)
{
    if (!b || hazard_lambda <= 0.0 || max_run_length < 16)
        return -1;

    memset(b, 0, sizeof(*b));

    /* Round capacity to power of 2 */
    size_t cap = 32;
    while (cap < max_run_length)
        cap <<= 1;

    b->capacity = cap;
    b->hazard = 1.0 / hazard_lambda;
    b->one_minus_h = 1.0 - b->hazard;
    b->trunc_thresh = 1e-6;
    b->prior = prior;
    b->ring_start = 0;

    /* Precompute lgamma values for prior */
    b->prior_lgamma_alpha = lgamma(prior.alpha0);
    b->prior_lgamma_alpha_p5 = lgamma(prior.alpha0 + 0.5);

    /*-------------------------------------------------------------------------
     * Memory Layout (single mega-block)
     *-------------------------------------------------------------------------*/

    size_t bytes_interleaved = (cap + 32) * 4 * sizeof(double);
    size_t bytes_vec = cap * sizeof(double);
    size_t bytes_r = (cap + 32) * sizeof(double);

    size_t total = bytes_interleaved + 13 * bytes_vec + 2 * bytes_r + 64;

#ifdef _WIN32
    void *mega = _aligned_malloc(total, 64);
#else
    void *mega = NULL;
    if (posix_memalign(&mega, 64, total) != 0)
        mega = NULL;
#endif

    if (!mega)
        return -1;

    memset(mega, 0, total);

    uint8_t *ptr = (uint8_t *)mega;

    b->lin_interleaved = (double *)ptr;
    ptr += bytes_interleaved;
    b->ss_n = (double *)ptr;
    ptr += bytes_vec;
    b->ss_sum = (double *)ptr;
    ptr += bytes_vec;
    b->ss_sum2 = (double *)ptr;
    ptr += bytes_vec;
    b->post_kappa = (double *)ptr;
    ptr += bytes_vec;
    b->post_mu = (double *)ptr;
    ptr += bytes_vec;
    b->post_alpha = (double *)ptr;
    ptr += bytes_vec;
    b->post_beta = (double *)ptr;
    ptr += bytes_vec;
    b->C1 = (double *)ptr;
    ptr += bytes_vec;
    b->C2 = (double *)ptr;
    ptr += bytes_vec;
    b->sigma_sq = (double *)ptr;
    ptr += bytes_vec;
    b->inv_sigma_sq_nu = (double *)ptr;
    ptr += bytes_vec;
    b->lgamma_alpha = (double *)ptr;
    ptr += bytes_vec;
    b->lgamma_alpha_p5 = (double *)ptr;
    ptr += bytes_vec;
    b->r = (double *)ptr;
    ptr += bytes_r;
    b->r_scratch = (double *)ptr;
    ptr += bytes_r;

    b->mega = mega;
    b->mega_bytes = total;

    /* Initialize slot 0 with prior */
    init_posterior_slot(b, 0);

    b->t = 0;
    b->active_len = 0;

    return 0;
}

void bocpd_ultra_free(bocpd_asm_t *b)
{
    if (!b)
        return;

#ifdef _WIN32
    if (b->mega)
        _aligned_free(b->mega);
#else
    free(b->mega);
#endif

    memset(b, 0, sizeof(*b));
}

void bocpd_ultra_reset(bocpd_asm_t *b)
{
    if (!b)
        return;

    memset(b->r, 0, (b->capacity + 32) * sizeof(double));
    b->t = 0;
    b->active_len = 0;
    b->ring_start = 0;
}

void bocpd_ultra_step(bocpd_asm_t *b, double x)
{
    if (!b)
        return;

    /* First observation */
    if (b->t == 0)
    {
        b->r[0] = 1.0;
        size_t ri = ring_idx(b, 0);

        b->ss_n[ri] = 1.0;
        b->ss_sum[ri] = x;
        b->ss_sum2[ri] = x * x;

        double k0 = b->prior.kappa0;
        double mu0 = b->prior.mu0;
        double a0 = b->prior.alpha0;
        double b0 = b->prior.beta0;

        double k1 = k0 + 1.0;
        double mu1 = (k0 * mu0 + x) / k1;
        double a1 = a0 + 0.5;
        double beta1 = b0 + 0.5 * (x - mu0) * (x - mu1);

        b->post_kappa[ri] = k1;
        b->post_mu[ri] = mu1;
        b->post_alpha[ri] = a1;
        b->post_beta[ri] = beta1;

        b->lgamma_alpha[ri] = lgamma(a1);
        b->lgamma_alpha_p5[ri] = lgamma(a1 + 0.5);

        double sigma_sq = beta1 * (k1 + 1.0) / (a1 * k1);
        double nu = 2.0 * a1;
        b->sigma_sq[ri] = sigma_sq;
        b->inv_sigma_sq_nu[ri] = 1.0 / (sigma_sq * nu);

        double ln_nupi = fast_log_scalar(nu * M_PI);
        double ln_s2 = fast_log_scalar(sigma_sq);
        b->C1[ri] = b->lgamma_alpha_p5[ri] - b->lgamma_alpha[ri] - 0.5 * ln_nupi - 0.5 * ln_s2;
        b->C2[ri] = a1 + 0.5;

        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }

    /* Normal update */
    fused_step_simd(b, x);
    shift_and_observe(b, x);

    b->t++;

    double p = 0.0;
    size_t lim = (b->active_len < 5) ? b->active_len : 5;
    for (size_t i = 0; i < lim; i++)
        p += b->r[i];
    b->p_changepoint = p;
}

/*=============================================================================
 * Pool Allocator for Multiple Detectors
 *=============================================================================*/

int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length)
{
    if (!pool || n_detectors == 0 || hazard_lambda <= 0.0 || max_run_length < 16)
        return -1;

    memset(pool, 0, sizeof(*pool));

    /* Round capacity to power of 2 */
    size_t cap = 32;
    while (cap < max_run_length)
        cap <<= 1;

    /* Calculate bytes per detector */
    size_t bytes_interleaved = (cap + 32) * 4 * sizeof(double);
    size_t bytes_vec = cap * sizeof(double);
    size_t bytes_r = (cap + 32) * sizeof(double);
    size_t bytes_per_detector = bytes_interleaved + 13 * bytes_vec + 2 * bytes_r;

    /* Align each detector's block to 64 bytes */
    bytes_per_detector = (bytes_per_detector + 63) & ~63ULL;

    /* Total allocation: struct array + all detector data */
    size_t struct_size = n_detectors * sizeof(bocpd_asm_t);
    struct_size = (struct_size + 63) & ~63ULL;

    size_t total = struct_size + n_detectors * bytes_per_detector;

#ifdef _WIN32
    void *mega = _aligned_malloc(total, 64);
#else
    void *mega = NULL;
    if (posix_memalign(&mega, 64, total) != 0)
        mega = NULL;
#endif

    if (!mega)
        return -1;

    memset(mega, 0, total);

    pool->pool = mega;
    pool->pool_size = total;
    pool->detectors = (bocpd_asm_t *)mega;
    pool->n_detectors = n_detectors;
    pool->bytes_per_detector = bytes_per_detector;

    /* Precompute lgamma values once */
    double prior_lgamma_alpha = lgamma(prior.alpha0);
    double prior_lgamma_alpha_p5 = lgamma(prior.alpha0 + 0.5);

    uint8_t *data_base = (uint8_t *)mega + struct_size;

    for (size_t d = 0; d < n_detectors; d++)
    {
        bocpd_asm_t *b = &pool->detectors[d];
        uint8_t *ptr = data_base + d * bytes_per_detector;

        b->capacity = cap;
        b->hazard = 1.0 / hazard_lambda;
        b->one_minus_h = 1.0 - b->hazard;
        b->trunc_thresh = 1e-6;
        b->prior = prior;
        b->ring_start = 0;
        b->prior_lgamma_alpha = prior_lgamma_alpha;
        b->prior_lgamma_alpha_p5 = prior_lgamma_alpha_p5;

        /* Slice the data block */
        b->lin_interleaved = (double *)ptr;
        ptr += bytes_interleaved;
        b->ss_n = (double *)ptr;
        ptr += bytes_vec;
        b->ss_sum = (double *)ptr;
        ptr += bytes_vec;
        b->ss_sum2 = (double *)ptr;
        ptr += bytes_vec;
        b->post_kappa = (double *)ptr;
        ptr += bytes_vec;
        b->post_mu = (double *)ptr;
        ptr += bytes_vec;
        b->post_alpha = (double *)ptr;
        ptr += bytes_vec;
        b->post_beta = (double *)ptr;
        ptr += bytes_vec;
        b->C1 = (double *)ptr;
        ptr += bytes_vec;
        b->C2 = (double *)ptr;
        ptr += bytes_vec;
        b->sigma_sq = (double *)ptr;
        ptr += bytes_vec;
        b->inv_sigma_sq_nu = (double *)ptr;
        ptr += bytes_vec;
        b->lgamma_alpha = (double *)ptr;
        ptr += bytes_vec;
        b->lgamma_alpha_p5 = (double *)ptr;
        ptr += bytes_vec;
        b->r = (double *)ptr;
        ptr += bytes_r;
        b->r_scratch = (double *)ptr;
        ptr += bytes_r;

        /* Mark as pool-managed */
        b->mega = NULL;
        b->mega_bytes = 0;

        /* Initialize slot 0 */
        init_posterior_slot(b, 0);

        b->t = 0;
        b->active_len = 0;
    }

    return 0;
}

void bocpd_pool_free(bocpd_pool_t *pool)
{
    if (!pool)
        return;

#ifdef _WIN32
    if (pool->pool)
        _aligned_free(pool->pool);
#else
    free(pool->pool);
#endif

    memset(pool, 0, sizeof(*pool));
}

void bocpd_pool_reset(bocpd_pool_t *pool)
{
    if (!pool)
        return;

    for (size_t d = 0; d < pool->n_detectors; d++)
    {
        bocpd_ultra_reset(&pool->detectors[d]);
    }
}

bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index)
{
    if (!pool || index >= pool->n_detectors)
        return NULL;
    return &pool->detectors[index];
}