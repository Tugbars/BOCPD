/**
 * @file bocpd_asm.h
 * @brief Ultra-Optimized Bayesian Online Changepoint Detection - Public API
 * @version 3.2 - Native Interleaved Layout with Platform-Specific ASM Kernels
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * This header provides the public interface for a high-performance BOCPD
 * (Bayesian Online Changepoint Detection) implementation. BOCPD detects
 * abrupt changes in the statistical properties of streaming time series data.
 *
 * KEY FEATURES:
 * - AVX2 SIMD acceleration (processes 4-8 run lengths per cycle)
 * - Hand-tuned assembly kernels for Intel and AMD processors
 * - Native interleaved memory layout (eliminates per-step data transformation)
 * - Pool allocator for multi-detector scenarios (10,000+ detectors)
 * - ~3M observations/second on modern Intel CPUs
 *
 * =============================================================================
 * QUICK START
 * =============================================================================
 *
 * SINGLE DETECTOR:
 * @code
 *     #include "bocpd_asm.h"
 *
 *     // Configure prior (weakly informative)
 *     bocpd_prior_t prior = {
 *         .mu0    = 0.0,    // Prior mean (set to expected data mean)
 *         .kappa0 = 1.0,    // Prior mean confidence (1 = weak)
 *         .alpha0 = 1.0,    // Prior variance shape (1 = weak)
 *         .beta0  = 1.0     // Prior variance rate
 *     };
 *
 *     // Create detector
 *     bocpd_asm_t detector;
 *     int ret = bocpd_ultra_init(&detector,
 *                                100.0,   // Expected run length between changes
 *                                prior,
 *                                1024);   // Max run length to track
 *     if (ret != 0) { handle_error(); }
 *
 *     // Process observations
 *     for (size_t i = 0; i < n_observations; i++) {
 *         bocpd_ultra_step(&detector, data[i]);
 *
 *         // Check for changepoint
 *         if (detector.p_changepoint > 0.5) {
 *             printf("Changepoint detected at t=%zu, confidence=%.2f%%\n",
 *                    i, detector.p_changepoint * 100);
 *         }
 *     }
 *
 *     bocpd_ultra_free(&detector);
 * @endcode
 *
 * MULTIPLE DETECTORS (e.g., monitoring 10,000 sensors):
 * @code
 *     bocpd_pool_t pool;
 *     bocpd_prior_t prior = {0.0, 1.0, 1.0, 1.0};
 *
 *     bocpd_pool_init(&pool, 10000, 100.0, prior, 256);
 *
 *     for (size_t t = 0; t < n_timesteps; t++) {
 *         for (size_t d = 0; d < 10000; d++) {
 *             bocpd_asm_t *det = bocpd_pool_get(&pool, d);
 *             bocpd_ultra_step(det, sensor_data[d][t]);
 *
 *             if (det->p_changepoint > 0.5) {
 *                 printf("Sensor %zu: changepoint at t=%zu\n", d, t);
 *             }
 *         }
 *     }
 *
 *     bocpd_pool_free(&pool);
 * @endcode
 *
 * =============================================================================
 * ALGORITHM BACKGROUND
 * =============================================================================
 *
 * BOCPD (Adams & MacKay, 2007) maintains a probability distribution over
 * "run lengths" — how many observations since the last changepoint.
 *
 * At each timestep, the algorithm:
 *   1. Computes P(observation | run_length) using Student-t predictive
 *   2. Updates the run-length distribution
 *   3. Updates posterior parameters for each run-length hypothesis
 *
 * The output p_changepoint = P(run_length ≤ 4) indicates probability of
 * being within 4 observations of a changepoint.
 *
 * This implementation uses a Normal-Inverse-Gamma conjugate prior, which
 * assumes data between changepoints follows a Normal distribution with
 * unknown mean and variance.
 *
 * =============================================================================
 * CHOOSING PARAMETERS
 * =============================================================================
 *
 * HAZARD LAMBDA (expected run length):
 * - λ = 100  → Expect changepoint every ~100 observations (sensitive)
 * - λ = 1000 → Expect changepoint every ~1000 observations (conservative)
 * - Rule of thumb: Set to minimum expected segment length
 *
 * PRIOR PARAMETERS:
 * - mu0: Set to expected data mean, or 0 if unknown
 * - kappa0: Confidence in mu0 (0.1 = very weak, 10 = strong)
 * - alpha0: Usually 1-2; smaller = heavier tails in predictive
 * - beta0: Related to expected variance; beta0/alpha0 ≈ prior variance
 *
 * WEAK (NON-INFORMATIVE) PRIOR:
 *     bocpd_prior_t weak = {0.0, 0.1, 1.0, 1.0};
 *
 * STRONG PRIOR (if you know the data distribution):
 *     bocpd_prior_t strong = {
 *         .mu0    = known_mean,
 *         .kappa0 = 10.0,                    // Strong confidence in mean
 *         .alpha0 = 5.0,
 *         .beta0  = 5.0 * known_variance    // alpha * expected_variance
 *     };
 *
 * MAX RUN LENGTH:
 * - Set to maximum expected segment length + safety margin
 * - Will be rounded up to next power of 2
 * - Memory usage: ~2KB per detector per 256 max run length
 *
 * =============================================================================
 * INTERPRETING OUTPUT
 * =============================================================================
 *
 * p_changepoint (double):
 *   Probability of being within 4 observations of a changepoint.
 *   - > 0.5: Likely recent changepoint
 *   - > 0.8: Strong evidence of changepoint
 *   - > 0.95: Very confident changepoint
 *
 * map_runlength (size_t):
 *   Most likely current run length (Maximum A Posteriori estimate).
 *   - Sudden drop to 0 indicates detected changepoint
 *   - Gradual increase during stable periods
 *
 * r[] array (double*):
 *   Full probability distribution over run lengths.
 *   - r[0] = probability run length is 0 (just had changepoint)
 *   - r[k] = probability run length is k
 *   - Sum of r[0..active_len-1] ≈ 1.0
 *
 * =============================================================================
 * PERFORMANCE NOTES
 * =============================================================================
 *
 * THROUGHPUT (Intel i9-13900K):
 * - Single detector: ~3M observations/second
 * - Pool of 1000 detectors: ~2.5M total observations/second
 *
 * MEMORY:
 * - Per detector: ~2KB base + ~8 bytes per max_run_length
 * - Pool overhead: Minimal (single allocation, shared metadata)
 *
 * LATENCY:
 * - bocpd_ultra_step: ~300ns typical
 * - Scales with active_len (truncated for negligible probabilities)
 *
 * SIMD REQUIREMENTS:
 * - AVX2 required (Haswell 2013+, Zen 2017+)
 * - FMA3 used for fused multiply-add
 *
 * =============================================================================
 * THREAD SAFETY
 * =============================================================================
 *
 * - Different detectors: Thread-safe (no shared mutable state)
 * - Same detector: NOT thread-safe (use external synchronization)
 * - Pool operations: Thread-safe for bocpd_pool_get() with different indices
 *
 * =============================================================================
 */

#ifndef BOCPD_ASM_H
#define BOCPD_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * PLATFORM-SPECIFIC MEMORY ALLOCATION
 *=============================================================================
 *
 * AVX2 requires 32-byte aligned memory for optimal performance.
 * Windows and POSIX have different aligned allocation APIs.
 *
 * Windows: _aligned_malloc(size, alignment) / _aligned_free(ptr)
 * POSIX:   aligned_alloc(alignment, size) / free(ptr)
 *          (Note: POSIX has alignment first, Windows has size first!)
 *
 * We provide a unified interface via these macros.
 *============================================================================*/

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
#define aligned_free(ptr) free(ptr)
#endif

/*=============================================================================
 * V3 INTERLEAVED BLOCK LAYOUT
 *=============================================================================
 *
 * MOTIVATION:
 * -----------
 * SIMD (AVX2) processes 4 doubles simultaneously. If parameters are stored
 * in separate arrays (μ[], C1[], C2[], ...), loading 4 values of each
 * parameter requires 4 separate gather operations — expensive!
 *
 * SOLUTION: INTERLEAVED "SUPERBLOCKS"
 * -----------------------------------
 * Store 4 consecutive run lengths' parameters together in a 256-byte block.
 * Each aligned load (vmovapd) fetches one parameter for 4 run lengths.
 *
 * MEMORY LAYOUT (256-byte superblock for run lengths [4k, 4k+1, 4k+2, 4k+3]):
 *
 *   Offset   Size   Field     Purpose                        Used By
 *   ─────────────────────────────────────────────────────────────────────
 *   0        32B    μ[0:3]    Posterior means                Prediction
 *   32       32B    C1[0:3]   Student-t normalization        Prediction
 *   64       32B    C2[0:3]   Student-t exponent (α+0.5)     Prediction
 *   96       32B    inv_ssn   Precomputed 1/(σ²ν)            Prediction
 *   ─────────────────────────────────────────────────────────────────────
 *   128      32B    κ[0:3]    Pseudo-count                   Update
 *   160      32B    α[0:3]    Shape parameter                Update
 *   192      32B    β[0:3]    Rate parameter                 Update
 *   224      32B    ss_n      Sample count (ν = 2α)          Update
 *   ─────────────────────────────────────────────────────────────────────
 *   Total:   256B = 4 cache lines = 32 doubles
 *
 * WHY THIS ORDER:
 * ---------------
 * The prediction step (hot loop) only needs μ, C1, C2, inv_ssn — the first
 * 128 bytes (2 cache lines). By placing these first, we maximize cache
 * efficiency and avoid polluting the cache with update parameters.
 *
 * The update step needs all 8 fields but runs less frequently.
 *
 * ADDRESSING:
 * -----------
 * For run length index i:
 *   block_index = i / 4
 *   lane        = i % 4  (which of the 4 elements within the block)
 *   byte_offset = block_index * 256 + field_offset + lane * 8
 *
 * For SIMD (loads 4 consecutive run lengths at once):
 *   byte_offset = block_index * 256 + field_offset
 *   (vmovapd loads all 4 lanes in one instruction)
 *
 *============================================================================*/

/** @name Superblock Size Constants */
/** @{ */
#define BOCPD_IBLK_BYTES     256    /**< Total bytes per superblock */
#define BOCPD_IBLK_DOUBLES   32     /**< Doubles per superblock: 256/8 = 32 */
#define BOCPD_IBLK_ELEMS     4      /**< Run lengths per superblock */
#define BOCPD_IBLK_STRIDE    256    /**< Byte stride between superblocks */
/** @} */

/** @name Field Byte Offsets Within Superblock
 *
 * Use these with IBLK_GET_* and IBLK_SET_* macros for scalar access,
 * or directly in assembly for SIMD loads.
 */
/** @{ */
#define BOCPD_IBLK_MU        0      /**< Posterior mean μ */
#define BOCPD_IBLK_C1        32     /**< Student-t constant C1 */
#define BOCPD_IBLK_C2        64     /**< Student-t exponent C2 = α + 0.5 */
#define BOCPD_IBLK_INV_SSN   96     /**< Inverse scale: 1/(ν×σ²) */
#define BOCPD_IBLK_KAPPA     128    /**< Pseudo-count κ */
#define BOCPD_IBLK_ALPHA     160    /**< Shape parameter α */
#define BOCPD_IBLK_BETA      192    /**< Rate parameter β */
#define BOCPD_IBLK_SS_N      224    /**< Sample count (observation count in run) */
/** @} */

/*=============================================================================
 * PRIOR PARAMETERS
 *=============================================================================
 *
 * The Normal-Inverse-Gamma prior encodes our beliefs about the data
 * distribution BEFORE seeing any observations.
 *
 * MATHEMATICAL INTERPRETATION:
 *
 * We model data as: x ~ Normal(μ, σ²)
 * With prior:       μ | σ² ~ Normal(μ₀, σ²/κ₀)
 *                   σ²     ~ Inverse-Gamma(α₀, β₀)
 *
 * The parameters mean:
 *   μ₀  = Our prior guess for the mean
 *   κ₀  = "Equivalent sample size" for the mean (confidence)
 *   α₀  = Shape of variance prior (controls tail heaviness)
 *   β₀  = Rate of variance prior (β₀/α₀ ≈ prior variance guess)
 *
 * PRACTICAL GUIDANCE:
 *
 * SCENARIO: Monitoring server response times (typically 50-200ms, σ ≈ 30ms)
 *   prior = {
 *       .mu0    = 100.0,   // Expected mean ~100ms
 *       .kappa0 = 2.0,     // Moderate confidence
 *       .alpha0 = 3.0,
 *       .beta0  = 3.0 * 900.0  // 3 * σ² = 3 * 30² = 2700
 *   };
 *
 * SCENARIO: Unknown data distribution (let data speak)
 *   prior = {
 *       .mu0    = 0.0,     // Will adapt quickly
 *       .kappa0 = 0.1,     // Very weak prior on mean
 *       .alpha0 = 1.0,     // Minimal prior on variance
 *       .beta0  = 1.0
 *   };
 *
 *============================================================================*/

/**
 * @brief Normal-Inverse-Gamma prior parameters.
 *
 * These parameters define the prior distribution over the unknown
 * mean and variance of data between changepoints.
 */
typedef struct bocpd_prior {
    double mu0;     /**< Prior mean. Set to expected data mean, or 0 if unknown. */
    double kappa0;  /**< Prior mean confidence. Higher = more confidence in mu0. 
                         Typical range: 0.1 (weak) to 10 (strong). */
    double alpha0;  /**< Variance shape parameter. Usually 1-2. 
                         Smaller values give heavier tails in predictive. */
    double beta0;   /**< Variance rate parameter. 
                         beta0/alpha0 ≈ prior guess for variance. */
} bocpd_prior_t;

/*=============================================================================
 * BOCPD DETECTOR STRUCTURE
 *=============================================================================
 *
 * This structure holds all state for a single BOCPD detector.
 *
 * MEMORY LAYOUT:
 * The detector allocates a single "mega-block" containing:
 *   1. interleaved[0] — Ping buffer for posterior parameters
 *   2. interleaved[1] — Pong buffer for posterior parameters
 *   3. r[]            — Current run-length probability distribution
 *   4. r_scratch[]    — Scratch buffer for probability updates
 *
 * All allocations are 64-byte aligned for AVX2 and cache efficiency.
 *
 * PING-PONG BUFFERING:
 * The algorithm reads from run length r and writes to r+1. Using two
 * buffers prevents read-after-write hazards:
 *   - cur_buf indicates which buffer holds current posteriors
 *   - After each step, cur_buf flips (0→1 or 1→0)
 *
 *============================================================================*/

/**
 * @brief BOCPD detector with native interleaved buffer layout.
 *
 * This structure is typically stack-allocated by the caller, then
 * initialized with bocpd_ultra_init(). The internal buffers are
 * heap-allocated.
 *
 * After use, call bocpd_ultra_free() to release memory.
 *
 * @note For pools of detectors, use bocpd_pool_t instead — it's more
 *       memory-efficient for large numbers of detectors.
 */
typedef struct bocpd_asm {
    /*-------------------------------------------------------------------------
     * Configuration (set once at init, read-only thereafter)
     *------------------------------------------------------------------------*/
    
    size_t capacity;       /**< Maximum run length (rounded to power of 2).
                                Memory allocated for this many run lengths. */
    
    double hazard;         /**< Hazard rate H = 1/λ. Probability of changepoint
                                at each timestep. Typically 0.001 to 0.1. */
    
    double one_minus_h;    /**< Precomputed 1 - H. Used in growth probability:
                                growth = P(continue) = (1-H) × joint_prob */
    
    double trunc_thresh;   /**< Truncation threshold (default 1e-6). Run lengths
                                with probability below this are dropped. */
    
    bocpd_prior_t prior;   /**< Prior parameters. Copied at init time. */
    
    /*-------------------------------------------------------------------------
     * Precomputed Prior Values (avoid redundant lgamma computation)
     *------------------------------------------------------------------------*/
    
    double prior_lgamma_alpha;     /**< ln(Γ(α₀)), precomputed at init */
    double prior_lgamma_alpha_p5;  /**< ln(Γ(α₀ + 0.5)), precomputed at init */
    
    /*-------------------------------------------------------------------------
     * Ping-Pong Buffer State
     *------------------------------------------------------------------------*/
    
    int cur_buf;           /**< Current buffer index (0 or 1). 
                                Use BOCPD_CUR_BUF() macro for access. */
    
    size_t active_len;     /**< Number of active run lengths. Grows with t,
                                but bounded by capacity and truncation. */
    
    double *interleaved[2]; /**< Ping-pong parameter buffers.
                                 Each is an array of 256-byte superblocks.
                                 interleaved[cur_buf] = current posteriors
                                 interleaved[1-cur_buf] = next posteriors */
    
    /*-------------------------------------------------------------------------
     * Run-Length Probability Distribution
     *------------------------------------------------------------------------*/
    
    double *r;             /**< Current probability distribution P(run_length).
                                r[i] = probability that run length equals i.
                                Sum of r[0..active_len-1] ≈ 1.0 */
    
    double *r_scratch;     /**< Scratch buffer for probability updates.
                                Same size as r[], used during step. */
    
    /*-------------------------------------------------------------------------
     * Output State (updated after each bocpd_ultra_step)
     *------------------------------------------------------------------------*/
    
    size_t t;              /**< Current timestep (observation count). */
    
    size_t map_runlength;  /**< MAP (Maximum A Posteriori) run length.
                                The most likely current run length. */
    
    double p_changepoint;  /**< Changepoint probability: sum of r[0..4].
                                High value (>0.5) indicates recent changepoint. */
    
    /*-------------------------------------------------------------------------
     * Memory Management
     *------------------------------------------------------------------------*/
    
    void *mega;            /**< Single allocation holding all buffers.
                                NULL for pool-allocated detectors. */
    
    size_t mega_bytes;     /**< Size of mega allocation in bytes.
                                0 for pool-allocated detectors. */
    
} bocpd_asm_t;

/*=============================================================================
 * BUFFER ACCESS MACROS
 *=============================================================================
 *
 * These macros provide safe access to the ping-pong buffers.
 *
 * USAGE:
 *   double *current = BOCPD_CUR_BUF(detector);   // Read from here
 *   double *next    = BOCPD_NEXT_BUF(detector);  // Write to here
 *
 * After bocpd_ultra_step(), the buffers are swapped internally.
 *
 *============================================================================*/

/** @brief Get pointer to CURRENT posterior buffer (for reading). */
#define BOCPD_CUR_BUF(b)  ((b)->interleaved[(b)->cur_buf])

/** @brief Get pointer to NEXT posterior buffer (for writing). */
#define BOCPD_NEXT_BUF(b) ((b)->interleaved[1 - (b)->cur_buf])

/*=============================================================================
 * POOL ALLOCATOR
 *=============================================================================
 *
 * When monitoring many streams simultaneously (e.g., 10,000 sensors),
 * allocating each detector separately causes:
 *   1. Memory fragmentation (thousands of small allocations)
 *   2. TLB pressure (many different memory pages)
 *   3. Allocation overhead (thousands of malloc calls)
 *
 * The pool allocator solves this by making ONE large allocation and
 * carving it into pieces for each detector.
 *
 * MEMORY LAYOUT:
 *   +------------------------------------------------------------------+
 *   | bocpd_asm_t[0] | bocpd_asm_t[1] | ... | bocpd_asm_t[n-1]         |
 *   +------------------------------------------------------------------+
 *   | detector 0 data buffers | detector 1 data buffers | ...         |
 *   +------------------------------------------------------------------+
 *
 * IMPORTANT: Pool detectors have mega = NULL. Do NOT call bocpd_ultra_free()
 * on them! Use bocpd_pool_free() to free the entire pool.
 *
 *============================================================================*/

/**
 * @brief Pool of BOCPD detectors with shared memory allocation.
 *
 * Use this when you need many detectors with the same configuration.
 * More memory-efficient than individual allocations.
 */
typedef struct bocpd_pool {
    void *pool;              /**< Single allocation for all detector memory. */
    size_t pool_size;        /**< Total bytes allocated. */
    bocpd_asm_t *detectors;  /**< Array of detector structs. */
    size_t n_detectors;      /**< Number of detectors in pool. */
    size_t bytes_per_detector; /**< Bytes of buffer space per detector. */
} bocpd_pool_t;

/*=============================================================================
 * PUBLIC API - SINGLE DETECTOR
 *============================================================================*/

/**
 * @brief Initialize a BOCPD detector.
 *
 * Allocates memory and configures the detector. After this call,
 * the detector is ready to process observations via bocpd_ultra_step().
 *
 * @param b              Pointer to detector struct (caller-allocated)
 * @param hazard_lambda  Expected run length between changepoints (λ > 0).
 *                       Hazard rate H = 1/λ.
 * @param prior          Prior distribution parameters
 * @param max_run_length Maximum run length to track (rounded to power of 2)
 *
 * @return 0 on success, -1 on failure (invalid params or allocation failure)
 *
 * @note max_run_length should be larger than the longest expected segment.
 *       Memory usage scales linearly with max_run_length.
 */
int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);

/**
 * @brief Free all memory associated with a detector.
 *
 * After calling this, the detector is zeroed and must be re-initialized
 * before use.
 *
 * @param b  Pointer to detector (NULL-safe)
 *
 * @warning Do NOT call this on pool-allocated detectors! Use bocpd_pool_free().
 */
void bocpd_ultra_free(bocpd_asm_t *b);

/**
 * @brief Reset detector to initial state without reallocating.
 *
 * Useful for processing multiple independent time series with the
 * same configuration. Faster than free() + init().
 *
 * @param b  Pointer to detector (NULL-safe)
 */
void bocpd_ultra_reset(bocpd_asm_t *b);

/**
 * @brief Process a single observation.
 *
 * This is the main entry point. Each call:
 *   1. Computes predictive probabilities for all active run lengths
 *   2. Updates the run-length probability distribution
 *   3. Updates posterior parameters
 *   4. Updates p_changepoint and map_runlength
 *
 * @param b  Pointer to detector (NULL-safe)
 * @param x  New observation value
 *
 * @note After this call, check b->p_changepoint for changepoint probability.
 */
void bocpd_ultra_step(bocpd_asm_t *b, double x);

/*=============================================================================
 * CONVENIENCE ACCESSORS
 *
 * These inline functions provide read-only access to detector state.
 * They're equivalent to direct field access but may be clearer in code.
 *============================================================================*/

/** @brief Get MAP (most likely) run length. */
static inline size_t bocpd_ultra_get_map(const bocpd_asm_t *b) { 
    return b->map_runlength; 
}

/** @brief Get current timestep (number of observations processed). */
static inline size_t bocpd_ultra_get_t(const bocpd_asm_t *b) { 
    return b->t; 
}

/** @brief Get changepoint probability P(run_length ≤ 4). */
static inline double bocpd_ultra_get_change_prob(const bocpd_asm_t *b) { 
    return b->p_changepoint; 
}

/*=============================================================================
 * PUBLIC API - POOL ALLOCATOR
 *============================================================================*/

/**
 * @brief Initialize a pool of BOCPD detectors.
 *
 * Creates n_detectors, all with the same configuration, in a single
 * contiguous memory allocation.
 *
 * @param pool           Pointer to pool struct (caller-allocated)
 * @param n_detectors    Number of detectors to create
 * @param hazard_lambda  Expected run length (shared by all detectors)
 * @param prior          Prior parameters (shared by all detectors)
 * @param max_run_length Maximum run length (rounded to power of 2)
 *
 * @return 0 on success, -1 on failure
 */
int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length);

/**
 * @brief Free the entire pool.
 *
 * Releases the single allocation holding all detectors.
 * After this call, all detector pointers from bocpd_pool_get() are invalid.
 *
 * @param pool  Pointer to pool (NULL-safe)
 */
void bocpd_pool_free(bocpd_pool_t *pool);

/**
 * @brief Reset all detectors in the pool.
 *
 * Equivalent to calling bocpd_ultra_reset() on each detector.
 *
 * @param pool  Pointer to pool (NULL-safe)
 */
void bocpd_pool_reset(bocpd_pool_t *pool);

/**
 * @brief Get a detector from the pool.
 *
 * The returned pointer is valid until bocpd_pool_free() is called.
 *
 * @param pool   Pointer to pool
 * @param index  Detector index (0 to n_detectors-1)
 *
 * @return Pointer to detector, or NULL if index out of bounds
 *
 * @note Thread-safe: Different indices can be accessed from different threads.
 */
bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index);

/*=============================================================================
 * ASSEMBLY KERNEL INTERFACE
 *=============================================================================
 *
 * This structure passes arguments to the hand-optimized assembly kernel.
 * It's used internally by bocpd_ultra_step() — you shouldn't need to
 * call the kernel directly.
 *
 * The kernel performs the "prediction step": computing Student-t
 * probabilities and updating the run-length distribution.
 *
 *============================================================================*/

/**
 * @brief Arguments for the AVX2 assembly kernel.
 *
 * @internal This is an implementation detail. Users should call
 *           bocpd_ultra_step() instead of invoking the kernel directly.
 */
typedef struct bocpd_kernel_args {
    const double *lin_interleaved;  /**< Interleaved parameters (BOCPD_CUR_BUF) */
    const double *r_old;            /**< Current probability distribution */
    double x;                       /**< New observation */
    double h;                       /**< Hazard rate */
    double one_minus_h;             /**< 1 - hazard rate */
    double trunc_thresh;            /**< Truncation threshold */
    size_t n_padded;                /**< Number of elements (padded to 8) */
    double *r_new;                  /**< Output probability distribution */
    double *r0_out;                 /**< Output: changepoint probability sum */
    double *max_growth_out;         /**< Output: max growth value (for MAP) */
    size_t *max_idx_out;            /**< Output: index of max growth */
    size_t *last_valid_out;         /**< Output: last index above threshold */
} bocpd_kernel_args_t;

/*=============================================================================
 * KERNEL SELECTION
 *=============================================================================
 *
 * This library provides multiple assembly kernels optimized for different
 * CPU architectures:
 *
 *   BOCPD_USE_INTEL_KERNEL=1 (default)
 *     Intel-optimized: Best for i9, Alder Lake, Raptor Lake
 *     Uses aggressive register allocation, Estrin polynomial evaluation
 *
 *   BOCPD_USE_INTEL_KERNEL=0
 *     Generic: Better for AMD Zen, older Intel (pre-Skylake)
 *     More conservative register usage
 *
 * Set via compiler flag:
 *   -DBOCPD_USE_INTEL_KERNEL=0   (for AMD or older Intel)
 *   -DBOCPD_USE_INTEL_KERNEL=1   (for modern Intel, default)
 *
 * PLATFORM-SPECIFIC ENTRY POINTS:
 *   Windows:  bocpd_fused_loop_avx2_win   (RCX = args, saves XMM6-15)
 *   Linux:    bocpd_fused_loop_avx2_sysv  (RDI = args, no XMM saves)
 *
 * The bocpd_fused_loop_avx2() macro automatically selects the right one.
 *
 *============================================================================*/

#ifndef BOCPD_USE_INTEL_KERNEL
#define BOCPD_USE_INTEL_KERNEL 1
#endif

#ifdef _WIN32
    /* Windows x64 ABI: first arg in RCX, XMM6-15 are callee-saved */
    #if BOCPD_USE_INTEL_KERNEL
        extern void bocpd_fused_loop_avx2_win(bocpd_kernel_args_t *args);
        #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_win(args)
    #else
        extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);
        #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
    #endif
#else
    /* System V ABI (Linux/macOS): first arg in RDI, no callee-saved XMM */
    #if BOCPD_USE_INTEL_KERNEL
        extern void bocpd_fused_loop_avx2_sysv(bocpd_kernel_args_t *args);
        #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_sysv(args)
    #else
        extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);
        #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
    #endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ASM_H */