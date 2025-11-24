/**
 * @file bocpd_bench.c
 * @brief Performance benchmarks for BOCPD implementation
 *
 * Measures throughput in observations/second for various scenarios.
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "bocpd_fast.h"

/*==============================================================================
 * Timing utilities
 *==============================================================================*/

static inline double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*==============================================================================
 * Random number generation
 *==============================================================================*/

static unsigned int g_seed = 12345;

static inline double fast_rand_uniform(void)
{
    g_seed = g_seed * 1103515245 + 12345;
    return (g_seed >> 16) / 65536.0;
}

static inline double fast_rand_normal(void)
{
    /* Box-Muller */
    double u1 = fast_rand_uniform() + 1e-10;
    double u2 = fast_rand_uniform();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*==============================================================================
 * Benchmark scenarios
 *==============================================================================*/

/**
 * Benchmark: Stationary Gaussian data
 * Tests sustained throughput without changepoints
 */
static double bench_stationary(size_t n_obs, size_t max_run)
{
    bocpd_ultra_t b;
    bocpd_prior_t prior = {
        .mu0 = 0.0,
        .kappa0 = 1.0,
        .alpha0 = 1.0,
        .beta0 = 1.0
    };
    
    bocpd_ultra_init(&b, 100.0, prior, max_run);
    
    /* Warmup */
    for (size_t i = 0; i < 100; i++)
    {
        bocpd_ultra_step(&b, fast_rand_normal());
    }
    bocpd_ultra_reset(&b);
    
    /* Timed run */
    double start = get_time_sec();
    
    for (size_t i = 0; i < n_obs; i++)
    {
        bocpd_ultra_step(&b, fast_rand_normal());
    }
    
    double elapsed = get_time_sec() - start;
    
    bocpd_ultra_free(&b);
    
    return n_obs / elapsed;
}

/**
 * Benchmark: Data with periodic changepoints
 * Tests behavior when truncation is active
 */
static double bench_changepoints(size_t n_obs, size_t max_run, size_t cp_interval)
{
    bocpd_ultra_t b;
    bocpd_prior_t prior = {
        .mu0 = 0.0,
        .kappa0 = 0.1,
        .alpha0 = 1.0,
        .beta0 = 0.1
    };
    
    bocpd_ultra_init(&b, (double)cp_interval, prior, max_run);
    
    /* Warmup */
    for (size_t i = 0; i < 100; i++)
    {
        bocpd_ultra_step(&b, fast_rand_normal());
    }
    bocpd_ultra_reset(&b);
    
    /* Timed run with mean shifts */
    double mean = 0.0;
    double start = get_time_sec();
    
    for (size_t i = 0; i < n_obs; i++)
    {
        if (i > 0 && i % cp_interval == 0)
        {
            mean += 5.0; /* Mean shift */
        }
        bocpd_ultra_step(&b, mean + fast_rand_normal() * 0.1);
    }
    
    double elapsed = get_time_sec() - start;
    
    bocpd_ultra_free(&b);
    
    return n_obs / elapsed;
}

/**
 * Benchmark: Varying max run length
 * Tests how performance scales with active_len
 */
static void bench_scaling(void)
{
    printf("\nScaling benchmark (obs/sec vs max_run_length):\n");
    printf("%-15s %-15s %-15s\n", "max_run", "obs/sec", "avg_active");
    printf("----------------------------------------------\n");
    
    size_t max_runs[] = {64, 128, 256, 512, 1024, 2048, 4096};
    size_t n_sizes = sizeof(max_runs) / sizeof(max_runs[0]);
    
    for (size_t s = 0; s < n_sizes; s++)
    {
        size_t max_run = max_runs[s];
        
        bocpd_ultra_t b;
        bocpd_prior_t prior = {
            .mu0 = 0.0,
            .kappa0 = 1.0,
            .alpha0 = 1.0,
            .beta0 = 1.0
        };
        
        bocpd_ultra_init(&b, 1000.0, prior, max_run); /* Large hazard = long runs */
        
        /* Warmup */
        for (size_t i = 0; i < 100; i++)
        {
            bocpd_ultra_step(&b, fast_rand_normal());
        }
        
        /* Timed run */
        size_t n_obs = 10000;
        size_t active_sum = 0;
        
        double start = get_time_sec();
        
        for (size_t i = 0; i < n_obs; i++)
        {
            bocpd_ultra_step(&b, fast_rand_normal());
            active_sum += b.active_len;
        }
        
        double elapsed = get_time_sec() - start;
        double obs_per_sec = n_obs / elapsed;
        double avg_active = (double)active_sum / n_obs;
        
        printf("%-15zu %-15.0f %-15.1f\n", max_run, obs_per_sec, avg_active);
        
        bocpd_ultra_free(&b);
    }
}

/*==============================================================================
 * Main
 *==============================================================================*/

int main(void)
{
    printf("==============================================\n");
    printf("BOCPD Performance Benchmark\n");
    printf("==============================================\n");
#if BOCPD_USE_ASM
    printf("Implementation: ASM kernel\n");
#else
    printf("Implementation: C (compiler-generated)\n");
#endif
    printf("==============================================\n\n");
    
    /* Primary benchmarks */
    printf("Primary benchmarks (100K observations):\n\n");
    
    double stationary_rate = bench_stationary(100000, 1024);
    printf("  Stationary data:     %10.0f obs/sec\n", stationary_rate);
    
    double cp_rate = bench_changepoints(100000, 1024, 100);
    printf("  With changepoints:   %10.0f obs/sec\n", cp_rate);
    
    /* Scaling analysis */
    bench_scaling();
    
    /* Summary */
    printf("\n==============================================\n");
    printf("Summary\n");
    printf("==============================================\n");
    printf("Peak throughput: %.0f obs/sec\n", stationary_rate > cp_rate ? stationary_rate : cp_rate);
    printf("Latency @ 1024 runs: ~%.2f µs/obs\n", 1e6 / stationary_rate);
    printf("==============================================\n");
    
    return 0;
}
