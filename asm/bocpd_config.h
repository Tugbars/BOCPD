/**
 * @file bocpd_config.h
 * @brief Build configuration for BOCPD implementation
 *
 * @details Controls compile-time switches for optimization paths.
 *
 * Build options:
 *   -DBOCPD_USE_ASM=1           Enable assembly kernels
 *   -DBOCPD_KERNEL_VARIANT=X    Select kernel variant (see below)
 */

#ifndef BOCPD_CONFIG_H
#define BOCPD_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/*=============================================================================
 * Kernel Variant Selection
 *
 * BOCPD_KERNEL_GENERIC (0):
 *   - Conservative scheduling, works well on all x86-64
 *   - Aligned loads preferred
 *   - Minimal register pressure
 *   - Best for: AMD Zen1-3, older Intel, unknown targets
 *
 * BOCPD_KERNEL_INTEL_PERF (1):
 *   - Aggressive ILP with interleaved A/B blocks
 *   - Unaligned loads (no penalty on Skylake+)
 *   - Tuned for Golden Cove / Raptor Cove microarchitecture
 *   - Best for: Intel 12th-14th gen (Alder Lake, Raptor Lake)
 *
 * BOCPD_KERNEL_ZEN4 (2):  [Future - not yet implemented]
 *   - Tuned for AMD Zen4 (dual 256-bit FMA pipes)
 *   - Best for: AMD Ryzen 7000 series
 *
 *=============================================================================*/

#define BOCPD_KERNEL_GENERIC      0
#define BOCPD_KERNEL_INTEL_PERF   1
#define BOCPD_KERNEL_ZEN4         2  /* Reserved for future */

/* Default to generic if not specified */
#ifndef BOCPD_KERNEL_VARIANT
    #define BOCPD_KERNEL_VARIANT BOCPD_KERNEL_GENERIC
#endif

/*=============================================================================
 * Assembly Kernel Enable/Disable
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

/* Validate kernel variant */
#if BOCPD_USE_ASM
    #if BOCPD_KERNEL_VARIANT == BOCPD_KERNEL_ZEN4
        #warning "Zen4 kernel not yet implemented, falling back to generic"
        #undef BOCPD_KERNEL_VARIANT
        #define BOCPD_KERNEL_VARIANT BOCPD_KERNEL_GENERIC
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

/**
 * @brief Arguments for the fused SIMD kernel.
 *
 * @note Field order matches assembly expectations. Do not reorder!
 * @note All pointers must be 64-byte aligned for optimal performance.
 * @note Uses interleaved layout for better cache utilization.
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
 * Assembly Kernel Declarations
 *=============================================================================*/

#if BOCPD_USE_ASM

/**
 * @brief Generic AVX2 kernel - works well on all x86-64 with AVX2+FMA
 *
 * Conservative scheduling, aligned loads, minimal assumptions.
 * Implemented in bocpd_kernel_avx2_generic.asm
 */
extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);

/**
 * @brief Intel-tuned AVX2 kernel - optimized for Alder Lake / Raptor Lake
 *
 * Aggressive ILP, interleaved A/B scheduling, unaligned loads.
 * Implemented in bocpd_kernel_avx2_intel.asm
 */
extern void bocpd_fused_loop_avx2_intel(bocpd_kernel_args_t *args);

/**
 * @brief Dispatch macro - selects kernel based on BOCPD_KERNEL_VARIANT
 */
#if BOCPD_KERNEL_VARIANT == BOCPD_KERNEL_INTEL_PERF
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_intel(args)
#else
    #define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)
#endif

#endif /* BOCPD_USE_ASM */

/*=============================================================================
 * Build Instructions
 *=============================================================================
 *
 * Generic kernel (default, works everywhere):
 *   nasm -f elf64 -o bocpd_kernel_generic.o bocpd_kernel_avx2_generic.asm
 *   gcc -DBOCPD_USE_ASM=1 -O3 -mavx2 -mfma -c bocpd_ultra_opt.c
 *   ar rcs libbocpd.a bocpd_ultra_opt.o bocpd_kernel_generic.o
 *
 * Intel-optimized kernel:
 *   nasm -f elf64 -o bocpd_kernel_intel.o bocpd_kernel_avx2_intel.asm
 *   gcc -DBOCPD_USE_ASM=1 -DBOCPD_KERNEL_VARIANT=1 -O3 -mavx2 -mfma -c bocpd_ultra_opt.c
 *   ar rcs libbocpd.a bocpd_ultra_opt.o bocpd_kernel_intel.o
 *
 * Both kernels (runtime selection possible with wrapper):
 *   nasm -f elf64 -o bocpd_kernel_generic.o bocpd_kernel_avx2_generic.asm
 *   nasm -f elf64 -o bocpd_kernel_intel.o bocpd_kernel_avx2_intel.asm
 *   gcc -DBOCPD_USE_ASM=1 -O3 -mavx2 -mfma -c bocpd_ultra_opt.c
 *   ar rcs libbocpd.a bocpd_ultra_opt.o bocpd_kernel_generic.o bocpd_kernel_intel.o
 *
 *=============================================================================*/

#endif /* BOCPD_CONFIG_H */
