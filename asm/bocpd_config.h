/**
 * @file bocpd_config.h
 * @brief Build configuration for BOCPD implementation
 *
 * @details Controls compile-time switches for optimization paths.
 * Use -DBOCPD_USE_ASM=1 to enable hand-optimized assembly kernels.
 */

#ifndef BOCPD_CONFIG_H
#define BOCPD_CONFIG_H

/*=============================================================================
 * Assembly Kernel Configuration
 *
 * When enabled, the hot loop in fused_step_simd() is replaced with
 * hand-optimized AVX2 assembly that provides better instruction scheduling
 * and register allocation than compiler-generated code.
 *
 * Requirements:
 *   - x86-64 architecture
 *   - AVX2 + FMA support
 *   - NASM assembler (for building .asm files)
 *
 * Build:
 *   gcc -DBOCPD_USE_ASM=1 -O3 -mavx2 -mfma bocpd_ultra_opt.c bocpd_kernel_avx2.o
 *   nasm -f elf64 bocpd_kernel_avx2.asm -o bocpd_kernel_avx2.o
 *=============================================================================*/

#ifndef BOCPD_USE_ASM
    #define BOCPD_USE_ASM 0
#endif

/* Architecture gate: ASM only available on x86-64 */
#if BOCPD_USE_ASM
    #if !defined(__x86_64__) && !defined(_M_X64)
        #warning "Assembly kernels only available on x86-64, falling back to C"
        #undef BOCPD_USE_ASM
        #define BOCPD_USE_ASM 0
    #endif
#endif

/* Verify AVX2/FMA support when using ASM path */
#if BOCPD_USE_ASM
    #if !defined(__AVX2__) || !defined(__FMA__)
        #warning "Assembly kernels require AVX2+FMA, falling back to C"
        #undef BOCPD_USE_ASM
        #define BOCPD_USE_ASM 0
    #endif
#endif

/*=============================================================================
 * Kernel Arguments Structure
 *
 * Used to pass parameters to the assembly kernel. Using a struct pointer
 * keeps the interface clean and avoids complex calling convention issues.
 *
 * Memory layout is critical for assembly - do not reorder fields!
 *=============================================================================*/

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Arguments for the fused SIMD kernel.
 *
 * @note Field order matches assembly expectations. Do not reorder!
 * @note All pointers must be 64-byte aligned for optimal performance.
 * @note V2: Uses interleaved layout for better cache utilization.
 */
typedef struct bocpd_kernel_args {
    /*=========================================================================
     * Input arrays (read-only) - INTERLEAVED LAYOUT
     *=========================================================================*/
    const double *lin_interleaved; /**< [0]  Interleaved [mu×4,C1×4,C2×4,inv_ssn×4] blocks */
    const double *r_old;           /**< [8]  Current run-length distribution */

    /*=========================================================================
     * Scalar inputs - offsets 16-55
     *=========================================================================*/
    double x;                   /**< [16] New observation */
    double h;                   /**< [24] Hazard rate */
    double omh;                 /**< [32] 1 - hazard rate */
    double thresh;              /**< [40] Truncation threshold */
    size_t n_padded;            /**< [48] Padded array length (multiple of 8) */

    /*=========================================================================
     * Output array - offset 56
     *=========================================================================*/
    double *r_new;              /**< [56] Output run-length distribution */

    /*=========================================================================
     * Scalar outputs - offsets 64-95 (pointers for assembly to write to)
     *=========================================================================*/
    double *r0_out;             /**< [64]  Pointer to accumulated changepoint probability */
    double *max_growth_out;     /**< [72]  Pointer to maximum growth value (for MAP) */
    size_t *max_idx_out;        /**< [80]  Pointer to index of maximum growth (for MAP) */
    size_t *last_valid_out;     /**< [88]  Pointer to last index above threshold */

} bocpd_kernel_args_t;

/*=============================================================================
 * Assembly Kernel Declaration
 *=============================================================================*/

#if BOCPD_USE_ASM
/**
 * @brief Hand-optimized AVX2 implementation of the fused BOCPD kernel.
 *
 * @param args Pointer to kernel arguments structure
 *
 * @pre All input arrays must be 64-byte aligned
 * @pre args->n_padded must be a multiple of 8
 *
 * @post args->r_new contains unnormalized growth probabilities
 * @post args->r0_out contains accumulated changepoint probability
 * @post args->max_growth_out and args->max_idx_out contain MAP information
 * @post args->last_valid_out contains truncation boundary
 *
 * @note Implemented in bocpd_kernel_avx2.asm
 */
extern void bocpd_fused_loop_avx2(bocpd_kernel_args_t *args);
#endif

#endif /* BOCPD_CONFIG_H */
