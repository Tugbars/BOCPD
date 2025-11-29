/**
 * @file bocpd_fast_wrapper.h
 * @brief C wrapper for BOCPD V3.3 Fast Direct-Register ABI Kernel
 *
 * This header provides the C interface to the fast-ABI assembly kernel.
 * The kernel receives parameters directly in registers rather than via
 * a struct pointer, saving ~2-3 cycles per call.
 *
 * =============================================================================
 * WHEN TO USE THIS vs STRUCT-POINTER ABI (V3.1/V3.2)
 * =============================================================================
 *
 * USE FAST-ABI (this file) when:
 *   - You need absolute maximum throughput
 *   - Processing millions of observations per second
 *   - Profiling shows kernel call overhead matters
 *
 * USE STRUCT-POINTER ABI (bocpd_asm.h) when:
 *   - Simpler integration is preferred
 *   - Code maintainability is priority
 *   - Performance difference (~1-2%) doesn't matter
 *
 * =============================================================================
 * USAGE
 * =============================================================================
 *
 * @code
 * #include "bocpd_fast_wrapper.h"
 *
 * // Inside your bocpd_ultra_step() or custom loop:
 * double r0, max_growth;
 * size_t max_idx, last_valid;
 *
 * // Using the convenience wrapper (recommended):
 * bocpd_fast_kernel_from_struct(b, x, &r0, &max_growth, &max_idx, &last_valid);
 *
 * // Or calling directly (floats MUST come first for Windows ABI):
 * bocpd_fast_kernel(
 *     x,                      // XMM0
 *     b->hazard,              // XMM1
 *     b->one_minus_h,         // XMM2
 *     b->trunc_thresh,        // XMM3
 *     BOCPD_CUR_BUF(b),       // params (stack on Win)
 *     b->r,                   // r_old
 *     b->r_scratch,           // r_new
 *     n_padded,               // n_padded
 *     &r0,                    // output
 *     &max_growth,            // output
 *     &max_idx,               // output
 *     &last_valid             // output
 * );
 * @endcode
 *
 * =============================================================================
 */

#ifndef BOCPD_FAST_WRAPPER_H
#define BOCPD_FAST_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ASSEMBLY KERNEL DECLARATIONS
 * ============================================================================
 *
 * CRITICAL: Arguments are ordered with floats FIRST to ensure they land
 * in XMM registers on Windows. Windows x64 only puts the first 4 args
 * in registers, regardless of type. By putting floats first, we get:
 *   - x, h, one_minus_h, threshold → XMM0-3 (register)
 *   - All pointers → stack
 *
 * System V (Linux/macOS) counts floats and integers separately, so this
 * ordering works there too.
 *
 * ============================================================================
 */

#ifdef _WIN32

/**
 * @brief Windows x64 fast-ABI kernel entry point.
 *
 * @param x              Current observation (XMM0)
 * @param h              Hazard rate (XMM1)
 * @param one_minus_h    1 - h precomputed (XMM2)
 * @param threshold      Truncation threshold (XMM3)
 * @param params         Interleaved superblock parameters (Stack)
 * @param r_old          Input probability distribution (Stack)
 * @param r_new          Output probability distribution (Stack)
 * @param n_padded       Number of elements, multiple of 8 (Stack)
 * @param r0_out         Output: sum of changepoint contributions (Stack)
 * @param max_growth_out Output: maximum growth probability (Stack)
 * @param max_idx_out    Output: index of maximum growth (Stack)
 * @param last_valid_out Output: last index above threshold (Stack)
 */
extern void bocpd_fast_avx2_win(
    double x,
    double h,
    double one_minus_h,
    double threshold,
    const double *params,
    const double *r_old,
    double *r_new,
    size_t n_padded,
    double *r0_out,
    double *max_growth_out,
    size_t *max_idx_out,
    size_t *last_valid_out
);

#define bocpd_fast_kernel bocpd_fast_avx2_win

#else /* Linux/macOS */

/**
 * @brief Linux/macOS System V fast-ABI kernel entry point.
 *
 * Same parameters as Windows version. System V counts float and integer
 * args separately, so floats go in XMM0-3 and pointers in RDI-R9.
 */
extern void bocpd_fast_avx2_sysv(
    double x,
    double h,
    double one_minus_h,
    double threshold,
    const double *params,
    const double *r_old,
    double *r_new,
    size_t n_padded,
    double *r0_out,
    double *max_growth_out,
    size_t *max_idx_out,
    size_t *last_valid_out
);

#define bocpd_fast_kernel bocpd_fast_avx2_sysv

#endif /* _WIN32 */

/* ============================================================================
 * CONVENIENCE WRAPPER
 * ============================================================================
 *
 * This inline wrapper provides a struct-based interface that internally
 * calls the fast kernel. Use this if you want the fast kernel's performance
 * but prefer the cleaner struct-based API.
 * ============================================================================
 */

#include "bocpd_asm.h"  /* For bocpd_asm_t, BOCPD_CUR_BUF, etc. */

/**
 * @brief Fast kernel call using bocpd_asm_t struct.
 *
 * This is a drop-in replacement for the inner loop of bocpd_ultra_step().
 * It extracts parameters from the detector struct and calls the fast kernel.
 *
 * @param b          Pointer to detector
 * @param x          Current observation
 * @param r0_out     Output: changepoint probability sum
 * @param max_growth Output: maximum growth (for MAP)
 * @param max_idx    Output: index of max growth
 * @param last_valid Output: last valid index (truncation)
 */
static inline void bocpd_fast_kernel_from_struct(
    bocpd_asm_t *b,
    double x,
    double *r0_out,
    double *max_growth,
    size_t *max_idx,
    size_t *last_valid
) {
    /* Round active_len up to multiple of 8 for SIMD */
    size_t n_padded = (b->active_len + 7) & ~7ULL;
    
    /* Note: floats must come first for Windows ABI compatibility */
    bocpd_fast_kernel(
        x,                      /* XMM0 */
        b->hazard,              /* XMM1 */
        b->one_minus_h,         /* XMM2 */
        b->trunc_thresh,        /* XMM3 */
        BOCPD_CUR_BUF(b),       /* params (stack on Win, RDI on SysV) */
        b->r,                   /* r_old */
        b->r_scratch,           /* r_new */
        n_padded,               /* n_padded */
        r0_out,                 /* output */
        max_growth,             /* output */
        max_idx,                /* output */
        last_valid              /* output */
    );
}

/* ============================================================================
 * INTEGRATION EXAMPLE
 * ============================================================================
 *
 * To use the fast kernel in bocpd_ultra_step(), replace the kernel call:
 *
 * BEFORE (struct-pointer ABI):
 *   bocpd_kernel_args_t args = {
 *       .lin_interleaved = BOCPD_CUR_BUF(b),
 *       .r_old = b->r,
 *       // ... etc
 *   };
 *   bocpd_fused_loop_avx2(&args);
 *
 * AFTER (fast direct-register ABI):
 *   double r0, max_growth;
 *   size_t max_idx, last_valid;
 *   bocpd_fast_kernel_from_struct(b, x, &r0, &max_growth, &max_idx, &last_valid);
 *
 * The outputs (r0, max_growth, max_idx, last_valid) are then used for:
 *   - r0 → b->r_scratch[0] (after normalization)
 *   - max_idx → b->map_runlength
 *   - last_valid → b->active_len (truncation)
 *
 * ============================================================================
 */

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_FAST_WRAPPER_H */