/**
 * @file test_bocpd_fast.c
 * @brief Test and benchmark optimized BOCPD (ASM version)
 *
 * Build via CMake from root:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build
 *   ./build/test_bocpd_fast
 */

#include "bocpd_asm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* Simple random normal using Box-Muller */
static int randn_have_spare = 0;
static double randn_spare = 0.0;

static void randn_reset(void)
{
    randn_have_spare = 0;
    randn_spare = 0.0;
}

static double randn(double mu, double sigma)
{
    if (randn_have_spare)
    {
        randn_have_spare = 0;
        return randn_spare * sigma + mu;
    }

    double u, v, s;
    do
    {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    randn_spare = v * s;
    randn_have_spare = 1;
    return u * s * sigma + mu;
}

static int test_mean_shift(void)
{
    printf("=== Test: Mean Shift Detection (ASM Kernel) ===\n");
    fflush(stdout);

    bocpd_asm_t cpd;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

    if (bocpd_ultra_init(&cpd, 250.0, prior, 1000) != 0)
    {
        printf("FAIL: Could not initialize BOCPD\n");
        return -1;
    }

    printf("trunc_thresh = %.2e\n", cpd.trunc_thresh);
    printf("hazard = %.6f (1/lambda = 1/250)\n", cpd.hazard);
    printf("capacity = %llu\n", (unsigned long long)cpd.capacity);
    fflush(stdout);

    srand(12345);

    const int n1 = 100;
    const int n2 = 100;
    const int n_total = n1 + n2;

    int change_detected = 0;
    int detection_time = -1;
    size_t prev_map_rl = 0;

    printf("Data: N(0,1) for %d samples, then N(5,1) for %d samples\n", n1, n2);
    printf("True change point at t=%d\n\n", n1);
    fflush(stdout);

    for (int t = 0; t < n_total; t++)
    {
        printf(">>> BEFORE step t=%d\n", t);
        fflush(stdout);

        double x = (t < n1) ? randn(0.0, 1.0) : randn(5.0, 1.0);

        printf(">>> Calling bocpd_ultra_step(x=%.3f), active_len=%llu\n", x, (unsigned long long)cpd.active_len);
        fflush(stdout);

        bocpd_ultra_step(&cpd, x);

        printf(">>> AFTER step t=%d, active_len=%llu\n", t, (unsigned long long)cpd.active_len);
        fflush(stdout);

        size_t map_rl = bocpd_ultra_get_map(&cpd);
        double p_change = bocpd_ultra_get_change_prob(&cpd);

        printf(">>> t=%d: map_rl=%llu, p_change=%.6f, r[0]=%.6e\n", 
               t, (unsigned long long)map_rl, p_change, cpd.r[0]);
        fflush(stdout);

        /* Check for NaN */
        if (isnan(p_change))
        {
            printf("ERROR: p_change is NaN at t=%d!\n", t);
            fflush(stdout);
            break;
        }
        if (isnan(cpd.r[0]))
        {
            printf("ERROR: r[0] is NaN at t=%d!\n", t);
            fflush(stdout);
            break;
        }

        /* Stop detailed output after t=10 to reduce noise */
        if (t >= 10)
        {
            /* Just print summary every 10 steps */
            if (t % 10 == 0)
            {
                printf("t=%d: active_len=%llu, map_rl=%llu, p_change=%.4f\n",
                       t, (unsigned long long)cpd.active_len, 
                       (unsigned long long)map_rl, p_change);
                fflush(stdout);
            }
        }

        if (!change_detected && t >= n1 && map_rl < 5 && prev_map_rl > 10)
        {
            change_detected = 1;
            detection_time = t;
            printf("*** CHANGE DETECTED at t=%d ***\n", t);
            fflush(stdout);
        }

        prev_map_rl = map_rl;
    }

    printf("\nLoop finished.\n");
    fflush(stdout);

    if (change_detected)
    {
        printf("PASS: Change detected at t=%d (true: %d, delay: %d)\n",
               detection_time, n1, detection_time - n1);
    }
    else
    {
        printf("FAIL: Change not detected\n");
    }

    bocpd_ultra_free(&cpd);
    return change_detected ? 0 : -1;
}

static void benchmark_throughput(void)
{
    printf("\n=== Benchmark: ASM Kernel Throughput ===\n");
    fflush(stdout);

    bocpd_asm_t cpd;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    
    printf("Initializing detector...\n");
    fflush(stdout);
    
    bocpd_ultra_init(&cpd, 250.0, prior, 2000);

    printf("Generating data...\n");
    fflush(stdout);

    srand(11111);

    const int n_samples = 500;
    double *data = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++)
    {
        data[i] = randn(0.0, 1.0);
    }

    printf("Warming up...\n");
    fflush(stdout);

    /* Warm up */
    for (int i = 0; i < 100; i++)
    {
        bocpd_ultra_step(&cpd, data[i % n_samples]);
        if (i % 20 == 0) {
            printf("  warmup step %d, active_len=%zu\n", i, cpd.active_len);
            fflush(stdout);
        }
    }
    
    printf("Resetting...\n");
    fflush(stdout);
    
    bocpd_ultra_reset(&cpd);

    /* Timed run */
    double start = get_time_ms();

    for (int i = 0; i < n_samples; i++)
    {
        bocpd_ultra_step(&cpd, data[i]);
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;
    double per_sample_us = elapsed_ms * 1000.0 / n_samples;

    printf("Processed %d samples in %.2f ms\n", n_samples, elapsed_ms);
    printf("Per-sample: %.2f us\n", per_sample_us);
    printf("Throughput: %.0f samples/sec\n", n_samples / (elapsed_ms / 1000.0));
    printf("Active run lengths at end: %zu\n", cpd.active_len);

    free(data);
    bocpd_ultra_free(&cpd);
}

static void benchmark_multiple_instruments_old(void)
{
    printf("\n=== Benchmark: Multi-Instrument (Individual Alloc) ===\n");

    const int n_instruments = 100;
    const int n_steps = 100;

    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

    printf("Allocating %d detectors individually...\n", n_instruments);
    fflush(stdout);

    double alloc_start = get_time_ms();

    bocpd_asm_t *cpds = malloc(n_instruments * sizeof(bocpd_asm_t));
    for (int i = 0; i < n_instruments; i++)
    {
        if (i % 20 == 0) {
            printf("  Alloc detector %d/%d\n", i, n_instruments);
            fflush(stdout);
        }
        bocpd_ultra_init(&cpds[i], 250.0, prior, 500);
    }

    double alloc_end = get_time_ms();
    printf("Individual allocation time: %.2f ms\n", alloc_end - alloc_start);
    fflush(stdout);

    /* Generate test data */
    printf("Generating test data...\n");
    fflush(stdout);
    
    srand(22222);
    randn_reset();
    double **data = malloc(n_instruments * sizeof(double *));
    for (int i = 0; i < n_instruments; i++)
    {
        data[i] = malloc(n_steps * sizeof(double));
        for (int t = 0; t < n_steps; t++)
        {
            data[i][t] = randn(0.0, 1.0);
        }
    }

    printf("Running processing benchmark...\n");
    fflush(stdout);

    /* Timed run */
    double start = get_time_ms();

    for (int t = 0; t < n_steps; t++)
    {
        if (t % 20 == 0) {
            printf("  Step %d/%d\n", t, n_steps);
            fflush(stdout);
        }
        for (int i = 0; i < n_instruments; i++)
        {
            bocpd_ultra_step(&cpds[i], data[i][t]);
        }
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;

    printf("Processing time: %.2f ms\n", elapsed_ms);
    printf("Throughput: %.0f obs/sec\n", 
           (double)(n_instruments * n_steps) / (elapsed_ms / 1000.0));
    fflush(stdout);

    printf("Cleaning up...\n");
    fflush(stdout);

    /* Cleanup */
    for (int i = 0; i < n_instruments; i++)
    {
        free(data[i]);
        bocpd_ultra_free(&cpds[i]);
    }
    free(data);
    free(cpds);
    
    printf("Done with individual alloc benchmark!\n");
    fflush(stdout);
}

static void benchmark_multiple_instruments_pool(void)
{
    printf("\n=== Benchmark: Multi-Instrument (Pool Allocator) ===\n");

    const int n_instruments = 100;
    const int n_steps = 100;

    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_pool_t pool;

    printf("Allocating %d detectors via pool...\n", n_instruments);
    fflush(stdout);

    double alloc_start = get_time_ms();

    if (bocpd_pool_init(&pool, n_instruments, 250.0, prior, 500) != 0)
    {
        printf("FAIL: Could not initialize pool\n");
        return;
    }

    double alloc_end = get_time_ms();
    printf("Pool allocation time: %.2f ms\n", alloc_end - alloc_start);
    printf("Pool size: %.2f KB\n", pool.pool_size / 1024.0);

    /* Generate test data */
    srand(22222);
    randn_reset();
    double **data = malloc(n_instruments * sizeof(double *));
    for (int i = 0; i < n_instruments; i++)
    {
        data[i] = malloc(n_steps * sizeof(double));
        for (int t = 0; t < n_steps; t++)
        {
            data[i][t] = randn(0.0, 1.0);
        }
    }

    /* Timed run */
    double start = get_time_ms();

    for (int t = 0; t < n_steps; t++)
    {
        for (int i = 0; i < n_instruments; i++)
        {
            bocpd_asm_t *det = bocpd_pool_get(&pool, i);
            bocpd_ultra_step(det, data[i][t]);
        }
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;
    double per_step_ms = elapsed_ms / n_steps;
    double per_instrument_us = (elapsed_ms * 1000.0) / (n_steps * n_instruments);

    printf("Processed %d instruments x %d steps = %d total updates\n",
           n_instruments, n_steps, n_instruments * n_steps);
    printf("Processing time: %.2f ms\n", elapsed_ms);
    printf("Per time step (all instruments): %.2f ms\n", per_step_ms);
    printf("Per instrument per step: %.2f us\n", per_instrument_us);
    printf("Throughput: %.0f obs/sec\n", 
           (double)(n_instruments * n_steps) / (elapsed_ms / 1000.0));

    /* Cleanup */
    for (int i = 0; i < n_instruments; i++)
    {
        free(data[i]);
    }
    free(data);
    
    bocpd_pool_free(&pool);
    
    printf("Done!\n");
}

static void benchmark_large_scale(void)
{
    printf("\n=== Benchmark: Large Scale (380 Instruments x 500 Steps) ===\n");

    const int n_instruments = 380;
    const int n_steps = 500;

    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_pool_t pool;

    printf("Allocating %d detectors via pool...\n", n_instruments);
    fflush(stdout);

    double alloc_start = get_time_ms();

    if (bocpd_pool_init(&pool, n_instruments, 250.0, prior, 500) != 0)
    {
        printf("FAIL: Could not initialize pool\n");
        return;
    }

    double alloc_end = get_time_ms();
    printf("Pool allocation time: %.2f ms\n", alloc_end - alloc_start);
    printf("Pool size: %.2f MB\n", pool.pool_size / (1024.0 * 1024.0));

    /* Generate test data */
    printf("Generating test data...\n");
    srand(33333);
    randn_reset();
    double **data = malloc(n_instruments * sizeof(double *));
    for (int i = 0; i < n_instruments; i++)
    {
        data[i] = malloc(n_steps * sizeof(double));
        for (int t = 0; t < n_steps; t++)
        {
            /* Add a changepoint at t=250 for some instruments */
            if (i % 10 == 0 && t >= 250)
                data[i][t] = randn(3.0, 1.0);
            else
                data[i][t] = randn(0.0, 1.0);
        }
    }

    printf("Running benchmark...\n");
    fflush(stdout);

    /* Timed run */
    double start = get_time_ms();

    for (int t = 0; t < n_steps; t++)
    {
        for (int i = 0; i < n_instruments; i++)
        {
            bocpd_asm_t *det = bocpd_pool_get(&pool, i);
            bocpd_ultra_step(det, data[i][t]);
        }
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;
    double per_step_ms = elapsed_ms / n_steps;
    double per_instrument_us = (elapsed_ms * 1000.0) / (n_steps * n_instruments);
    long total_updates = (long)n_instruments * n_steps;

    printf("\nResults:\n");
    printf("  Total updates: %ld\n", total_updates);
    printf("  Total time: %.2f ms\n", elapsed_ms);
    printf("  Per time step (all %d instruments): %.2f ms\n", n_instruments, per_step_ms);
    printf("  Per instrument per step: %.2f us\n", per_instrument_us);
    printf("  Throughput: %.0f obs/sec\n", 
           (double)total_updates / (elapsed_ms / 1000.0));

    /* Check for changepoint detections */
    int changes_detected = 0;
    for (int i = 0; i < n_instruments; i += 10)
    {
        bocpd_asm_t *det = bocpd_pool_get(&pool, i);
        if (det->map_runlength < 200)
            changes_detected++;
    }
    printf("  Changepoints detected: %d / %d instruments with shifts\n", 
           changes_detected, n_instruments / 10);

    /* Cleanup */
    for (int i = 0; i < n_instruments; i++)
    {
        free(data[i]);
    }
    free(data);
    
    bocpd_pool_free(&pool);
    
    printf("Done!\n");
}

int main(void)
{
    printf("BOCPD Ultra - ASM Kernel Test\n");
    printf("=============================\n\n");

#if BOCPD_KERNEL_VARIANT == 1
    printf("Kernel: Intel-tuned (Alder/Raptor Lake)\n\n");
#else
    printf("Kernel: Generic (all AVX2 CPUs)\n\n");
#endif

    int failures = 0;

    /* Correctness test */
    failures += (test_mean_shift() != 0);

    /* Single detector throughput */
    benchmark_throughput();

    /* Compare individual vs pool allocation */
    benchmark_multiple_instruments_old();
    benchmark_multiple_instruments_pool();

    /* Large scale benchmark */
    benchmark_large_scale();

    printf("\n=== Summary ===\n");
    if (failures == 0)
    {
        printf("All tests passed!\n");
    }
    else
    {
        printf("%d test(s) failed\n", failures);
    }

    return failures;
}