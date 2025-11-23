/**
 * @file test_bocpd.c
 * @brief Test and example usage of BOCPD
 *
 * Compile:
 *   gcc -O3 -o test_bocpd test_bocpd.c bocpd.c -lm
 *
 * Run:
 *   ./test_bocpd
 */

#include "bocpd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

/**
 * @brief Test: Detect obvious mean shift
 */
static int test_mean_shift(void) {
    printf("=== Test: Mean Shift Detection ===\n");

    bocpd_t cpd;
    bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};

    if (bocpd_init(&cpd, 250.0, prior, 1000) != 0) {
        printf("FAIL: Could not initialize BOCPD\n");
        return -1;
    }

    srand(12345);

    /* Generate data: N(0,1) for 100 samples, then N(5,1) for 100 samples */
    const int n1 = 100;
    const int n2 = 100;
    const int n_total = n1 + n2;

    int change_detected = 0;
    int detection_time = -1;
    size_t prev_map_rl = 0;

    printf("Data: N(0,1) for %d samples, then N(5,1) for %d samples\n", n1, n2);
    printf("True change point at t=%d\n\n", n1);

    for (int t = 0; t < n_total; t++) {
        double x;
        if (t < n1) {
            x = randn(0.0, 1.0);
        } else {
            x = randn(5.0, 1.0);
        }

        bocpd_step(&cpd, x);
        size_t map_rl = bocpd_map_runlength(&cpd);
        double short_prob = bocpd_short_run_probability(&cpd, 5);

        /* Print around the change point */
        if (t >= n1 - 5 && t <= n1 + 10) {
            printf("t=%3d: x=%7.3f  MAP_rl=%3zu  P(rl<5)=%.4f %s\n",
                   t, x, map_rl, short_prob,
                   (map_rl < 5 && prev_map_rl > 10) ? " <-- CHANGE DETECTED" : "");
        }

        /* Detect change: MAP run length drops significantly */
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

    bocpd_free(&cpd);
    return change_detected ? 0 : -1;
}

/**
 * @brief Test: Detect variance shift
 */
static int test_variance_shift(void) {
    printf("\n=== Test: Variance Shift Detection ===\n");

    bocpd_t cpd;
    bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};

    if (bocpd_init(&cpd, 250.0, prior, 1000) != 0) {
        printf("FAIL: Could not initialize BOCPD\n");
        return -1;
    }

    srand(54321);

    /* Generate data: N(0,1) for 100 samples, then N(0,5) for 100 samples */
    const int n1 = 100;
    const int n2 = 100;
    const int n_total = n1 + n2;

    int change_detected = 0;
    int detection_time = -1;
    size_t prev_map_rl = 0;

    printf("Data: N(0,1) for %d samples, then N(0,5) for %d samples\n", n1, n2);
    printf("True change point at t=%d\n\n", n1);

    for (int t = 0; t < n_total; t++) {
        double x;
        if (t < n1) {
            x = randn(0.0, 1.0);
        } else {
            x = randn(0.0, 5.0);
        }

        bocpd_step(&cpd, x);
        size_t map_rl = bocpd_map_runlength(&cpd);
        double short_prob = bocpd_short_run_probability(&cpd, 5);

        /* Print around the change point */
        if (t >= n1 - 5 && t <= n1 + 10) {
            printf("t=%3d: x=%7.3f  MAP_rl=%3zu  P(rl<5)=%.4f %s\n",
                   t, x, map_rl, short_prob,
                   (map_rl < 5 && prev_map_rl > 10) ? " <-- CHANGE DETECTED" : "");
        }

        /* Detect change: MAP run length drops significantly */
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

    bocpd_free(&cpd);
    return change_detected ? 0 : -1;
}

/**
 * @brief Test: Multiple change points
 */
static int test_multiple_changes(void) {
    printf("\n=== Test: Multiple Change Points ===\n");

    bocpd_t cpd;
    bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};

    if (bocpd_init(&cpd, 100.0, prior, 1000) != 0) {
        printf("FAIL: Could not initialize BOCPD\n");
        return -1;
    }

    srand(98765);

    /* Regime 1: N(0,1), Regime 2: N(3,1), Regime 3: N(-2, 0.5) */
    const int lens[] = {80, 80, 80};
    const double means[] = {0.0, 3.0, -2.0};
    const double stds[] = {1.0, 1.0, 0.5};
    const int n_regimes = 3;

    int t = 0;
    int changes_detected = 0;
    size_t prev_map_rl = 0;

    printf("Regimes:\n");
    for (int r = 0; r < n_regimes; r++) {
        printf("  t=%3d-%3d: N(%.1f, %.1f)\n",
               t, t + lens[r] - 1, means[r], stds[r]);
        t += lens[r];
    }
    printf("\n");

    t = 0;
    for (int r = 0; r < n_regimes; r++) {
        for (int i = 0; i < lens[r]; i++) {
            double x = randn(means[r], stds[r]);
            bocpd_step(&cpd, x);
            size_t map_rl = bocpd_map_runlength(&cpd);

            if (map_rl < 5 && prev_map_rl > 10 && t > 0) {
                printf("Change detected at t=%d (MAP_rl dropped from %zu to %zu)\n", 
                       t, prev_map_rl, map_rl);
                changes_detected++;
            }
            
            prev_map_rl = map_rl;
            t++;
        }
    }

    printf("\nTotal changes detected: %d (expected: %d)\n",
           changes_detected, n_regimes - 1);

    bocpd_free(&cpd);
    return (changes_detected >= 1) ? 0 : -1;
}

/**
 * @brief Benchmark: Throughput test
 */
static void benchmark_throughput(void) {
    printf("\n=== Benchmark: Throughput ===\n");

    bocpd_t cpd;
    bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};
    bocpd_init(&cpd, 250.0, prior, 2000);

    srand(11111);

    const int n_samples = 10000;
    double *data = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) {
        data[i] = randn(0.0, 1.0);
    }

    /* Warm up */
    for (int i = 0; i < 100; i++) {
        bocpd_step(&cpd, data[i]);
    }
    bocpd_reset(&cpd);

    /* Timed run */
    clock_t start = clock();
    for (int i = 0; i < n_samples; i++) {
        bocpd_step(&cpd, data[i]);
    }
    clock_t end = clock();

    double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_sample_us = elapsed_ms * 1000.0 / n_samples;

    printf("Processed %d samples in %.2f ms\n", n_samples, elapsed_ms);
    printf("Per-sample: %.2f µs\n", per_sample_us);
    printf("Throughput: %.0f samples/sec\n", n_samples / (elapsed_ms / 1000.0));

    free(data);
    bocpd_free(&cpd);
}

/**
 * @brief Example: How to use in a trading context
 */
static void example_trading_usage(void) {
    printf("\n=== Example: Trading Usage ===\n\n");

    printf("// Initialize BOCPD for returns data\n");
    printf("bocpd_t cpd;\n");
    printf("bocpd_normal_gamma_t prior = {0.0, 1.0, 1.0, 1.0};\n");
    printf("bocpd_init(&cpd, 250.0, prior, 1000);\n\n");

    printf("// In your main loop:\n");
    printf("double return_t = (price_t - price_t_minus_1) / price_t_minus_1;\n");
    printf("bocpd_step(&cpd, return_t);\n\n");

    printf("double p_change = bocpd_get_p_changepoint(&cpd);\n\n");

    printf("// Use p_change to modulate your SR-UKF:\n");
    printf("if (p_change > 0.3) {\n");
    printf("    // Increase process noise Q\n");
    printf("    srukf_set_q_scale(&ukf, 1.0 + 5.0 * p_change);\n");
    printf("    // Reduce position size\n");
    printf("    position_scale = 1.0 - 0.5 * p_change;\n");
    printf("}\n");
}

int main(void) {
    int failures = 0;

    failures += (test_mean_shift() != 0);
    failures += (test_variance_shift() != 0);
    failures += (test_multiple_changes() != 0);

    benchmark_throughput();
    example_trading_usage();

    printf("\n=== Summary ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed\n", failures);
    }

    return failures;
}
