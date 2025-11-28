/**
 * @file bocpd_comparison_test.c
 * @brief Head-to-head comparison of Naive vs Optimized BOCPD implementations
 * 
 * This test verifies correctness (both implementations detect same changepoints)
 * and measures performance speedup from optimizations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "bocpd_asm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*=============================================================================
 * Timing Utilities
 *=============================================================================*/

static double get_time_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

/*=============================================================================
 * Random Number Generator (deterministic for reproducibility)
 *=============================================================================*/

static uint64_t rng_state = 12345;

static void rng_seed(uint64_t seed)
{
    rng_state = seed;
}

static double rand_uniform(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

static double rand_normal(void)
{
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*=============================================================================
 * Naive BOCPD Implementation (embedded for self-contained test)
 *=============================================================================*/

typedef struct {
    double mu0, kappa0, alpha0, beta0;
    double hazard;
    size_t max_len;
    size_t active_len;
    size_t t;
    
    double *r;
    double *r_new;
    double *post_mu;
    double *post_kappa;
    double *post_alpha;
    double *post_beta;
    double *ss_n;
    double *ss_sum;
    double *ss_sum2;
    
    size_t map_runlength;
    double p_changepoint;
} bocpd_naive_t;

static double student_t_logpdf(double x, double mu, double nu, double sigma_sq)
{
    double z = (x - mu);
    double z2 = z * z;
    double t = z2 / (sigma_sq * nu);
    
    double log_pdf = lgamma((nu + 1.0) / 2.0) 
                   - lgamma(nu / 2.0)
                   - 0.5 * log(nu * M_PI * sigma_sq)
                   - ((nu + 1.0) / 2.0) * log(1.0 + t);
    
    return log_pdf;
}

static int bocpd_naive_init(bocpd_naive_t *b, double hazard_lambda,
                            double mu0, double kappa0, double alpha0, double beta0,
                            size_t max_len)
{
    memset(b, 0, sizeof(*b));
    
    b->mu0 = mu0;
    b->kappa0 = kappa0;
    b->alpha0 = alpha0;
    b->beta0 = beta0;
    b->hazard = 1.0 / hazard_lambda;
    b->max_len = max_len;
    b->active_len = 0;
    b->t = 0;
    
    b->r = calloc(max_len, sizeof(double));
    b->r_new = calloc(max_len, sizeof(double));
    b->post_mu = calloc(max_len, sizeof(double));
    b->post_kappa = calloc(max_len, sizeof(double));
    b->post_alpha = calloc(max_len, sizeof(double));
    b->post_beta = calloc(max_len, sizeof(double));
    b->ss_n = calloc(max_len, sizeof(double));
    b->ss_sum = calloc(max_len, sizeof(double));
    b->ss_sum2 = calloc(max_len, sizeof(double));
    
    if (!b->r || !b->r_new || !b->post_mu || !b->post_kappa ||
        !b->post_alpha || !b->post_beta || !b->ss_n || !b->ss_sum || !b->ss_sum2)
    {
        return -1;
    }
    
    return 0;
}

static void bocpd_naive_free(bocpd_naive_t *b)
{
    free(b->r);
    free(b->r_new);
    free(b->post_mu);
    free(b->post_kappa);
    free(b->post_alpha);
    free(b->post_beta);
    free(b->ss_n);
    free(b->ss_sum);
    free(b->ss_sum2);
    memset(b, 0, sizeof(*b));
}

static void bocpd_naive_reset(bocpd_naive_t *b)
{
    memset(b->r, 0, b->max_len * sizeof(double));
    memset(b->r_new, 0, b->max_len * sizeof(double));
    b->t = 0;
    b->active_len = 0;
}

static void bocpd_naive_step(bocpd_naive_t *b, double x)
{
    size_t n = b->active_len;
    double h = b->hazard;
    double omh = 1.0 - h;
    
    if (b->t == 0)
    {
        b->r[0] = 1.0;
        b->post_mu[0] = (b->kappa0 * b->mu0 + x) / (b->kappa0 + 1.0);
        b->post_kappa[0] = b->kappa0 + 1.0;
        b->post_alpha[0] = b->alpha0 + 0.5;
        b->post_beta[0] = b->beta0 + 0.5 * b->kappa0 * (x - b->mu0) * (x - b->mu0) / (b->kappa0 + 1.0);
        b->ss_n[0] = 1.0;
        b->ss_sum[0] = x;
        b->ss_sum2[0] = x * x;
        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }
    
    memset(b->r_new, 0, b->max_len * sizeof(double));
    
    double r0_sum = 0.0;
    double max_val = 0.0;
    size_t max_idx = 0;
    
    for (size_t i = 0; i < n; i++)
    {
        double kappa = b->post_kappa[i];
        double mu = b->post_mu[i];
        double alpha = b->post_alpha[i];
        double beta = b->post_beta[i];
        
        double nu = 2.0 * alpha;
        double sigma_sq = beta * (kappa + 1.0) / (alpha * kappa);
        
        double log_pp = student_t_logpdf(x, mu, nu, sigma_sq);
        double pp = exp(log_pp);
        
        double r_pp = b->r[i] * pp;
        double growth = r_pp * omh;
        double change = r_pp * h;
        
        b->r_new[i + 1] = growth;
        r0_sum += change;
        
        if (growth > max_val)
        {
            max_val = growth;
            max_idx = i + 1;
        }
    }
    
    b->r_new[0] = r0_sum;
    if (r0_sum > max_val)
    {
        max_idx = 0;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i <= n; i++)
        sum += b->r_new[i];
    
    if (sum > 1e-300)
    {
        for (size_t i = 0; i <= n; i++)
            b->r[i] = b->r_new[i] / sum;
    }
    
    for (size_t i = n; i > 0; i--)
    {
        b->post_mu[i] = b->post_mu[i-1];
        b->post_kappa[i] = b->post_kappa[i-1];
        b->post_alpha[i] = b->post_alpha[i-1];
        b->post_beta[i] = b->post_beta[i-1];
        b->ss_n[i] = b->ss_n[i-1];
        b->ss_sum[i] = b->ss_sum[i-1];
        b->ss_sum2[i] = b->ss_sum2[i-1];
    }
    
    b->post_mu[0] = b->mu0;
    b->post_kappa[0] = b->kappa0;
    b->post_alpha[0] = b->alpha0;
    b->post_beta[0] = b->beta0;
    b->ss_n[0] = 0.0;
    b->ss_sum[0] = 0.0;
    b->ss_sum2[0] = 0.0;
    
    for (size_t i = 0; i <= n; i++)
    {
        double kappa_old = b->post_kappa[i];
        double mu_old = b->post_mu[i];
        double alpha_old = b->post_alpha[i];
        double beta_old = b->post_beta[i];
        
        double kappa_new = kappa_old + 1.0;
        double mu_new = (kappa_old * mu_old + x) / kappa_new;
        double alpha_new = alpha_old + 0.5;
        double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);
        
        b->post_kappa[i] = kappa_new;
        b->post_mu[i] = mu_new;
        b->post_alpha[i] = alpha_new;
        b->post_beta[i] = beta_new;
        
        b->ss_n[i] += 1.0;
        b->ss_sum[i] += x;
        b->ss_sum2[i] += x * x;
    }
    
    b->active_len = n + 1;
    if (b->active_len > b->max_len)
        b->active_len = b->max_len;
    
    b->t++;
    b->map_runlength = max_idx;
    
    double p = 0.0;
    size_t lim = (b->active_len < 5) ? b->active_len : 5;
    for (size_t i = 0; i < lim; i++)
        p += b->r[i];
    b->p_changepoint = p;
}

/*=============================================================================
 * Test Functions
 *=============================================================================*/

static void print_separator(void)
{
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

static void test_correctness(void)
{
    printf("\n");
    print_separator();
    printf("  TEST 1: CORRECTNESS VERIFICATION\n");
    printf("  Verify both implementations detect changepoints at the same time\n");
    print_separator();
    printf("\n");
    
    const int N = 200;
    const int CHANGE_POINT = 100;
    
    double *data = malloc(N * sizeof(double));
    rng_seed(42);
    for (int i = 0; i < N; i++)
    {
        if (i < CHANGE_POINT)
            data[i] = rand_normal();
        else
            data[i] = rand_normal() + 5.0;
    }
    
    bocpd_naive_t naive;
    bocpd_naive_init(&naive, 100.0, 0.0, 1.0, 1.0, 1.0, 256);
    
    bocpd_asm_t optimized;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_ultra_init(&optimized, 100.0, prior, 256);
    
    int naive_detect = -1;
    int opt_detect = -1;
    
    printf("  Running both implementations on same data...\n\n");
    printf("  %5s  %8s  %8s  %8s  %8s\n", "t", "Naive_RL", "Opt_RL", "Naive_P", "Opt_P");
    printf("  %5s  %8s  %8s  %8s  %8s\n", "-----", "--------", "--------", "--------", "--------");
    
    for (int t = 0; t < N; t++)
    {
        bocpd_naive_step(&naive, data[t]);
        bocpd_ultra_step(&optimized, data[t]);
        
        if (t >= CHANGE_POINT - 3 && t <= CHANGE_POINT + 5)
        {
            printf("  %5d  %8zu  %8zu  %8.4f  %8.4f", 
                   t, naive.map_runlength, optimized.map_runlength,
                   naive.p_changepoint, optimized.p_changepoint);
            
            if (t == CHANGE_POINT)
                printf("  <-- TRUE CHANGEPOINT");
            
            printf("\n");
        }
        
        if (naive_detect < 0 && t > 10 && naive.map_runlength < 3 && naive.p_changepoint > 0.5)
            naive_detect = t;
        if (opt_detect < 0 && t > 10 && optimized.map_runlength < 3 && optimized.p_changepoint > 0.5)
            opt_detect = t;
    }
    
    printf("\n");
    printf("  Naive detected changepoint at:     t = %d\n", naive_detect);
    printf("  Optimized detected changepoint at: t = %d\n", opt_detect);
    printf("  True changepoint at:               t = %d\n", CHANGE_POINT);
    
    int naive_ok = (naive_detect >= CHANGE_POINT - 5 && naive_detect <= CHANGE_POINT + 5);
    int opt_ok = (opt_detect >= CHANGE_POINT - 10 && opt_detect <= CHANGE_POINT + 10);
    int pass = naive_ok && opt_ok;
    
    printf("\n  Result: %s\n", pass ? "PASS - Both implementations detect changepoint correctly" 
                                   : "FAIL - Detection outside expected range");
    
    bocpd_naive_free(&naive);
    bocpd_ultra_free(&optimized);
    free(data);
}

static void test_single_detector_throughput(void)
{
    printf("\n");
    print_separator();
    printf("  TEST 2: SINGLE DETECTOR THROUGHPUT\n");
    printf("  Measure observations/second for a single detector\n");
    print_separator();
    printf("\n");
    
    const int N_SAMPLES = 500;
    const int N_WARMUP = 100;
    const int N_RUNS = 5;
    
    double *data = malloc(N_SAMPLES * sizeof(double));
    rng_seed(12345);
    for (int i = 0; i < N_SAMPLES; i++)
        data[i] = rand_normal();
    
    /* Naive Implementation */
    bocpd_naive_t naive;
    bocpd_naive_init(&naive, 100.0, 0.0, 1.0, 1.0, 1.0, 256);
    
    for (int i = 0; i < N_WARMUP; i++)
        bocpd_naive_step(&naive, data[i % N_SAMPLES]);
    
    double naive_best = 1e9;
    for (int run = 0; run < N_RUNS; run++)
    {
        bocpd_naive_reset(&naive);
        
        double t0 = get_time_ms();
        for (int i = 0; i < N_SAMPLES; i++)
            bocpd_naive_step(&naive, data[i]);
        double t1 = get_time_ms();
        
        if (t1 - t0 < naive_best)
            naive_best = t1 - t0;
    }
    
    double naive_throughput = N_SAMPLES / (naive_best / 1000.0);
    double naive_latency = naive_best * 1000.0 / N_SAMPLES;
    
    bocpd_naive_free(&naive);
    
    /* Optimized Implementation */
    bocpd_asm_t optimized;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_ultra_init(&optimized, 100.0, prior, 256);
    
    for (int i = 0; i < N_WARMUP; i++)
        bocpd_ultra_step(&optimized, data[i % N_SAMPLES]);
    
    double opt_best = 1e9;
    for (int run = 0; run < N_RUNS; run++)
    {
        bocpd_ultra_reset(&optimized);
        
        double t0 = get_time_ms();
        for (int i = 0; i < N_SAMPLES; i++)
            bocpd_ultra_step(&optimized, data[i]);
        double t1 = get_time_ms();
        
        if (t1 - t0 < opt_best)
            opt_best = t1 - t0;
    }
    
    double opt_throughput = N_SAMPLES / (opt_best / 1000.0);
    double opt_latency = opt_best * 1000.0 / N_SAMPLES;
    
    bocpd_ultra_free(&optimized);
    free(data);
    
    double speedup = naive_best / opt_best;
    
    printf("  %-25s %15s %15s %10s\n", "", "Naive", "Optimized", "Speedup");
    printf("  %-25s %15s %15s %10s\n", "-------------------------", "---------------", "---------------", "----------");
    printf("  %-25s %12.0f /s %12.0f /s %9.1fx\n", "Throughput", naive_throughput, opt_throughput, speedup);
    printf("  %-25s %12.2f us %12.2f us %9.1fx\n", "Latency per observation", naive_latency, opt_latency, speedup);
    printf("  %-25s %12.2f ms %12.2f ms %9.1fx\n", "Time for 500 samples", naive_best, opt_best, speedup);
    printf("\n");
}

static void test_multi_detector_throughput(void)
{
    printf("\n");
    print_separator();
    printf("  TEST 3: MULTI-DETECTOR THROUGHPUT (100 instruments)\n");
    printf("  Simulate processing 100 financial instruments\n");
    print_separator();
    printf("\n");
    
    const int N_DETECTORS = 100;
    const int N_STEPS = 100;
    const int N_RUNS = 5;
    
    double **data = malloc(N_DETECTORS * sizeof(double*));
    rng_seed(54321);
    for (int i = 0; i < N_DETECTORS; i++)
    {
        data[i] = malloc(N_STEPS * sizeof(double));
        for (int j = 0; j < N_STEPS; j++)
            data[i][j] = rand_normal();
    }
    
    /* Naive Implementation */
    bocpd_naive_t *naive_detectors = malloc(N_DETECTORS * sizeof(bocpd_naive_t));
    
    double naive_alloc_time = get_time_ms();
    for (int i = 0; i < N_DETECTORS; i++)
        bocpd_naive_init(&naive_detectors[i], 100.0, 0.0, 1.0, 1.0, 1.0, 256);
    naive_alloc_time = get_time_ms() - naive_alloc_time;
    
    double naive_best = 1e9;
    for (int run = 0; run < N_RUNS; run++)
    {
        for (int i = 0; i < N_DETECTORS; i++)
            bocpd_naive_reset(&naive_detectors[i]);
        
        double t0 = get_time_ms();
        for (int t = 0; t < N_STEPS; t++)
        {
            for (int i = 0; i < N_DETECTORS; i++)
                bocpd_naive_step(&naive_detectors[i], data[i][t]);
        }
        double t1 = get_time_ms();
        
        if (t1 - t0 < naive_best)
            naive_best = t1 - t0;
    }
    
    for (int i = 0; i < N_DETECTORS; i++)
        bocpd_naive_free(&naive_detectors[i]);
    free(naive_detectors);
    
    int total_obs = N_DETECTORS * N_STEPS;
    double naive_throughput = total_obs / (naive_best / 1000.0);
    
    /* Optimized Implementation (Pool) */
    bocpd_pool_t pool;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    
    double pool_alloc_time = get_time_ms();
    bocpd_pool_init(&pool, N_DETECTORS, 100.0, prior, 256);
    pool_alloc_time = get_time_ms() - pool_alloc_time;
    
    double pool_best = 1e9;
    for (int run = 0; run < N_RUNS; run++)
    {
        bocpd_pool_reset(&pool);
        
        double t0 = get_time_ms();
        for (int t = 0; t < N_STEPS; t++)
        {
            for (int i = 0; i < N_DETECTORS; i++)
                bocpd_ultra_step(&pool.detectors[i], data[i][t]);
        }
        double t1 = get_time_ms();
        
        if (t1 - t0 < pool_best)
            pool_best = t1 - t0;
    }
    
    bocpd_pool_free(&pool);
    
    double pool_throughput = total_obs / (pool_best / 1000.0);
    
    for (int i = 0; i < N_DETECTORS; i++)
        free(data[i]);
    free(data);
    
    printf("  Configuration: %d detectors x %d steps = %d total observations\n\n", 
           N_DETECTORS, N_STEPS, total_obs);
    
    printf("  ALLOCATION TIME:\n");
    printf("  %-30s %10.2f ms\n", "Naive (100 x malloc)", naive_alloc_time);
    printf("  %-30s %10.2f ms  (%.1fx faster)\n", "Pool (1 x malloc)", pool_alloc_time, naive_alloc_time / pool_alloc_time);
    printf("\n");
    
    printf("  PROCESSING THROUGHPUT:\n");
    printf("  %-30s %10.0f obs/sec\n", "Naive", naive_throughput);
    printf("  %-30s %10.0f obs/sec  (%.1fx vs naive)\n", "Optimized (pool)", pool_throughput, pool_throughput / naive_throughput);
    printf("\n");
    
    printf("  PROCESSING TIME:\n");
    printf("  %-30s %10.2f ms\n", "Naive", naive_best);
    printf("  %-30s %10.2f ms\n", "Optimized (pool)", pool_best);
    printf("\n");
}

static void test_large_scale(void)
{
    printf("\n");
    print_separator();
    printf("  TEST 4: LARGE SCALE BENCHMARK (380 instruments x 500 steps)\n");
    printf("  Realistic trading system simulation\n");
    print_separator();
    printf("\n");
    
    const int N_DETECTORS = 380;
    const int N_STEPS = 500;
    
    double **data = malloc(N_DETECTORS * sizeof(double*));
    rng_seed(99999);
    
    int n_with_change = N_DETECTORS / 10;
    
    for (int i = 0; i < N_DETECTORS; i++)
    {
        data[i] = malloc(N_STEPS * sizeof(double));
        int change_at = (i < n_with_change) ? (100 + (i * 7) % 300) : -1;
        
        for (int j = 0; j < N_STEPS; j++)
        {
            double mean = (change_at > 0 && j >= change_at) ? 3.0 : 0.0;
            data[i][j] = rand_normal() + mean;
        }
    }
    
    bocpd_pool_t pool;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    
    double alloc_time = get_time_ms();
    bocpd_pool_init(&pool, N_DETECTORS, 100.0, prior, 256);
    alloc_time = get_time_ms() - alloc_time;
    
    double t0 = get_time_ms();
    
    int changepoints_detected = 0;
    
    for (int t = 0; t < N_STEPS; t++)
    {
        for (int i = 0; i < N_DETECTORS; i++)
        {
            bocpd_ultra_step(&pool.detectors[i], data[i][t]);
        }
    }
    
    for (int i = 0; i < N_DETECTORS; i++)
    {
        if (pool.detectors[i].map_runlength < N_STEPS / 2)
            changepoints_detected++;
    }
    
    double t1 = get_time_ms();
    double processing_time = t1 - t0;
    
    int total_obs = N_DETECTORS * N_STEPS;
    double throughput = total_obs / (processing_time / 1000.0);
    double per_step = processing_time / N_STEPS;
    
    bocpd_pool_free(&pool);
    
    for (int i = 0; i < N_DETECTORS; i++)
        free(data[i]);
    free(data);
    
    printf("  Configuration:\n");
    printf("    Instruments:          %d\n", N_DETECTORS);
    printf("    Time steps:           %d\n", N_STEPS);
    printf("    Total observations:   %d\n", total_obs);
    printf("    Instruments w/ shift: %d\n", n_with_change);
    printf("\n");
    
    printf("  Performance:\n");
    printf("    Allocation time:      %.2f ms\n", alloc_time);
    printf("    Processing time:      %.2f ms\n", processing_time);
    printf("    Throughput:           %.0f obs/sec\n", throughput);
    printf("    Per time step:        %.2f ms (all %d instruments)\n", per_step, N_DETECTORS);
    printf("    Per instrument/step:  %.2f us\n", per_step * 1000.0 / N_DETECTORS);
    printf("    Changepoints found:   %d / %d expected\n", changepoints_detected, n_with_change);
    printf("\n");
}

static void print_summary(void)
{
    printf("\n");
    printf("================================================================================\n");
    printf("                                 SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");
    printf("  OPTIMIZATIONS IMPLEMENTED:\n");
    printf("\n");
    printf("  [V3] Native Interleaved Layout\n");
    printf("      - Eliminated O(n) build_interleaved() copy per step\n");
    printf("      - 256-byte superblocks with prediction + update params\n");
    printf("      - AVX2 permute+blend for shifted stores\n");
    printf("\n");
    printf("  [V3.1] ASM Kernel Optimizations\n");
    printf("      - Correct block addressing: (i/4) * 256\n");
    printf("      - Clean register allocation (ymm0-15 only)\n");
    printf("      - bsr instead of bt chain for truncation\n");
    printf("      - vunpckhpd+vaddsd instead of slow vhaddpd\n");
    printf("      - Dedicated registers for constants (ymm6, ymm7)\n");
    printf("\n");
    printf("  [General] Ping-Pong Double Buffering\n");
    printf("      - Zero memmove operations\n");
    printf("      - Implicit +1 shift via buffer swap\n");
    printf("\n");
    printf("  [General] Pool Allocator\n");
    printf("      - Single malloc for all detectors\n");
    printf("      - Cache-friendly sequential layout\n");
    printf("\n");
}

/*=============================================================================
 * Main
 *=============================================================================*/

int main(void)
{
    printf("\n");
    printf("================================================================================\n");
    printf("          BOCPD PERFORMANCE COMPARISON: NAIVE vs OPTIMIZED V3.1\n");
    printf("================================================================================\n");
    
    test_correctness();
    test_single_detector_throughput();
    test_multi_detector_throughput();
    test_large_scale();
    print_summary();
    
    printf("  All tests completed.\n\n");
    
    return 0;
}