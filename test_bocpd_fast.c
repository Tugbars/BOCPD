/**
 * @file test_bocpd_fast.c
 * @brief Test and benchmark optimized BOCPD
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -o test_bocpd_fast test_bocpd_fast.c bocpd_fast.c -lm
 */

#include "bocpd_fast.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Simple random normal using Box-Muller */
static double randn(double mu, double sigma) {
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return spare * sigma + mu;
    }

    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;
    return u * s * sigma + mu;
}

static int test_mean_shift(void) {
    printf("=== Test: Mean Shift Detection (Fast) ===\n");

    bocpd_fast_t cpd;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

    if (bocpd_fast_init(&cpd, 250.0, prior, 1000) != 0) {
        printf("FAIL: Could not initialize BOCPD\n");
        return -1;
    }

    srand(12345);

    const int n1 = 100;
    const int n2 = 100;
    const int n_total = n1 + n2;

    int change_detected = 0;
    int detection_time = -1;
    size_t prev_map_rl = 0;

    printf("Data: N(0,1) for %d samples, then N(5,1) for %d samples\n", n1, n2);
    printf("True change point at t=%d\n\n", n1);

    for (int t = 0; t < n_total; t++) {
        double x = (t < n1) ? randn(0.0, 1.0) : randn(5.0, 1.0);

        bocpd_fast_step(&cpd, x);
        size_t map_rl = bocpd_fast_get_map_rl(&cpd);
        double p_change = bocpd_fast_change_prob(&cpd, 5);

        if (t >= n1 - 5 && t <= n1 + 10) {
            printf("t=%3d: x=%7.3f  MAP_rl=%3zu  P(rl<5)=%.4f %s\n",
                   t, x, map_rl, p_change,
                   (map_rl < 5 && prev_map_rl > 10) ? " <-- CHANGE" : "");
        }

        if (!change_detected && t >= n1 && map_rl < 5 && prev_map_rl > 10) {
            change_detected = 1;
            detection_time = t;
        }

        prev_map_rl = map_rl;
    }

    printf("\n");
    if (change_detected) {
        printf("PASS: Change detected at t=%d (true: %d, delay: %d)\n",
               detection_time, n1, detection_time - n1);
    } else {
        printf("FAIL: Change not detected\n");
    }

    bocpd_fast_free(&cpd);
    return change_detected ? 0 : -1;
}

static void benchmark_comparison(void) {
    printf("\n=== Benchmark: Optimized BOCPD ===\n");

    bocpd_fast_t cpd;
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_fast_init(&cpd, 250.0, prior, 2000);

    srand(11111);

    const int n_samples = 10000;
    double *data = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) {
        data[i] = randn(0.0, 1.0);
    }

    /* Warm up */
    for (int i = 0; i < 100; i++) {
        bocpd_fast_step(&cpd, data[i]);
    }
    bocpd_fast_reset(&cpd);

    /* Timed run */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < n_samples; i++) {
        bocpd_fast_step(&cpd, data[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    double elapsed_ms = elapsed_ns / 1e6;
    double per_sample_us = elapsed_ns / 1000.0 / n_samples;

    printf("Processed %d samples in %.2f ms\n", n_samples, elapsed_ms);
    printf("Per-sample: %.2f µs\n", per_sample_us);
    printf("Throughput: %.0f samples/sec\n", n_samples / (elapsed_ms / 1000.0));
    printf("Active run lengths at end: %zu\n", cpd.active_len);

    free(data);
    bocpd_fast_free(&cpd);
}

static void benchmark_multiple_instruments(void) {
    printf("\n=== Benchmark: 380 Instruments (Simulated) ===\n");

    const int n_instruments = 380;
    const int n_steps = 1000;

    bocpd_fast_t *cpds = malloc(n_instruments * sizeof(bocpd_fast_t));
    bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};

    for (int i = 0; i < n_instruments; i++) {
        bocpd_fast_init(&cpds[i], 250.0, prior, 500);
    }

    srand(22222);

    /* Generate data for all instruments */
    double **data = malloc(n_instruments * sizeof(double *));
    for (int i = 0; i < n_instruments; i++) {
        data[i] = malloc(n_steps * sizeof(double));
        for (int t = 0; t < n_steps; t++) {
            data[i][t] = randn(0.0, 1.0);
        }
    }

    /* Timed run */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int t = 0; t < n_steps; t++) {
        for (int i = 0; i < n_instruments; i++) {
            bocpd_fast_step(&cpds[i], data[i][t]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    double elapsed_ms = elapsed_ns / 1e6;
    double per_step_ms = elapsed_ms / n_steps;
    double per_instrument_us = (elapsed_ns / 1000.0) / (n_steps * n_instruments);

    printf("Processed %d instruments × %d steps = %d total updates\n",
           n_instruments, n_steps, n_instruments * n_steps);
    printf("Total time: %.2f ms\n", elapsed_ms);
    printf("Per time step (all 380 instruments): %.2f ms\n", per_step_ms);
    printf("Per instrument per step: %.2f µs\n", per_instrument_us);

    /* Cleanup */
    for (int i = 0; i < n_instruments; i++) {
        free(data[i]);
        bocpd_fast_free(&cpds[i]);
    }
    free(data);
    free(cpds);
}

int main(void) {
    int failures = 0;

    failures += (test_mean_shift() != 0);

    benchmark_comparison();
    benchmark_multiple_instruments();

    printf("\n=== Summary ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed\n", failures);
    }

    return failures;
}
