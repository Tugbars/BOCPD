/**
 * @file bocpd_asm.h (MODIFIED - Native Interleaved Layout)
 */

#ifndef BOCPD_ASM_H
#define BOCPD_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
#define aligned_free(ptr) free(ptr)
#endif

/* Interleaved block layout constants */
#define BOCPD_IBLK_MU       0
#define BOCPD_IBLK_C1       32
#define BOCPD_IBLK_C2       64
#define BOCPD_IBLK_INV_SSN  96
#define BOCPD_IBLK_KAPPA    128
#define BOCPD_IBLK_ALPHA    160
#define BOCPD_IBLK_BETA     192
#define BOCPD_IBLK_SS_N     224
#define BOCPD_IBLK_STRIDE   256

typedef struct bocpd_prior {
    double mu0;
    double kappa0;
    double alpha0;
    double beta0;
} bocpd_prior_t;

/**
 * @brief BOCPD detector with native interleaved buffer layout.
 * 
 * Parameters are stored directly in SIMD-friendly interleaved format,
 * eliminating the need for build_interleaved() transformation each step.
 */
typedef struct bocpd_asm {
    /* Configuration */
    size_t capacity;
    double hazard;
    double one_minus_h;
    double trunc_thresh;
    bocpd_prior_t prior;
    
    double prior_lgamma_alpha;
    double prior_lgamma_alpha_p5;
    
    /* Ping-pong interleaved buffers */
    int cur_buf;
    size_t active_len;
    double *interleaved[2];
    
    /* Run-length distribution */
    double *r;
    double *r_scratch;
    
    /* Output state */
    size_t t;
    size_t map_runlength;
    double p_changepoint;
    
    /* Memory management */
    void *mega;
    size_t mega_bytes;
    
} bocpd_asm_t;

/* Buffer access macros */
#define BOCPD_CUR_BUF(b)  ((b)->interleaved[(b)->cur_buf])
#define BOCPD_NEXT_BUF(b) ((b)->interleaved[1 - (b)->cur_buf])

typedef struct bocpd_pool {
    void *pool;
    size_t pool_size;
    bocpd_asm_t *detectors;
    size_t n_detectors;
    size_t bytes_per_detector;
} bocpd_pool_t;

/* API functions */
int bocpd_ultra_init(bocpd_asm_t *b, double hazard_lambda,
                     bocpd_prior_t prior, size_t max_run_length);
void bocpd_ultra_free(bocpd_asm_t *b);
void bocpd_ultra_reset(bocpd_asm_t *b);
void bocpd_ultra_step(bocpd_asm_t *b, double x);

static inline size_t bocpd_ultra_get_map(const bocpd_asm_t *b) { return b->map_runlength; }
static inline size_t bocpd_ultra_get_t(const bocpd_asm_t *b) { return b->t; }
static inline double bocpd_ultra_get_change_prob(const bocpd_asm_t *b) { return b->p_changepoint; }

int bocpd_pool_init(bocpd_pool_t *pool, size_t n_detectors,
                    double hazard_lambda, bocpd_prior_t prior,
                    size_t max_run_length);
void bocpd_pool_free(bocpd_pool_t *pool);
void bocpd_pool_reset(bocpd_pool_t *pool);
bocpd_asm_t *bocpd_pool_get(bocpd_pool_t *pool, size_t index);

/* Assembly kernel interface - now reads directly from interleaved buffer */
typedef struct bocpd_kernel_args {
    const double *lin_interleaved;  /* Points to BOCPD_CUR_BUF(b) */
    const double *r_old;
    double x;
    double h;
    double one_minus_h;
    double trunc_thresh;
    size_t n_padded;
    double *r_new;
    double *r0_out;
    double *max_growth_out;
    size_t *max_idx_out;
    size_t *last_valid_out;
} bocpd_kernel_args_t;

extern void bocpd_fused_loop_avx2_generic(bocpd_kernel_args_t *args);

#define bocpd_fused_loop_avx2(args) bocpd_fused_loop_avx2_generic(args)

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_ASM_H */