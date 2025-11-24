.intel_syntax noprefix
.text
.global bocpd_kernel_avx512
.type   bocpd_kernel_avx512, @function

# =============================================================================
# void bocpd_kernel_avx512(
#       double *r,             rdi
#       double *r_new,         rsi
#       const double *mu,      rdx
#       const double *C1,      rcx
#       const double *C2,      r8
#       const double *inv_ssn, r9
#       size_t n_padded,       [rbp+16]
#       double x,              [rbp+24]
#       double h,              [rbp+32]
#       double omh,            [rbp+40]
#       double thresh_unused,  [rbp+48]
#       double *change_out     [rbp+56]
# );
#
# Computes for i = 0..n_padded-1:
#   pp[i]     = student_t_pdf(x | mu[i], C1[i], C2[i], inv_ssn[i])
#   growth[i] = r[i] * pp[i] * omh   → stored at r_new[i+1]
#   change   += r[i] * pp[i] * h     → accumulated, stored to *change_out
#
# Register allocation:
#   zmm5        = NEG700 (clamp low)
#   zmm6        = POS700 (clamp high)
#   zmm7-12     = EC1..EC6 (exp poly)
#   zmm13       = MIN_PP
#   zmm15       = change accumulator
#   zmm16       = x (broadcast)
#   zmm17       = h (broadcast)
#   zmm18       = omh (broadcast)
#   zmm19-26    = C1..C8 log1p poly
#   zmm27       = INV_LN2
#   zmm28       = LN2
#   zmm29       = HALF
#   zmm30       = MAGIC
#   zmm31       = ONE
#   zmm0-4,14   = working temporaries
# =============================================================================

bocpd_kernel_avx512:
    push rbp
    mov rbp, rsp
    and rsp, -64                      # 64-byte alignment for ZMM

    # -------------------------------------------------------------------------
    # Load stack arguments
    # -------------------------------------------------------------------------
    mov     r10, [rbp+16]             # n_padded
    mov     r11, [rbp+56]             # change_out ptr

    # -------------------------------------------------------------------------
    # Broadcast scalar arguments
    # -------------------------------------------------------------------------
    vbroadcastsd zmm16, [rbp+24]      # x
    vbroadcastsd zmm17, [rbp+32]      # h
    vbroadcastsd zmm18, [rbp+40]      # omh

    # -------------------------------------------------------------------------
    # log1p polynomial coefficients: log1p(t) ≈ t*(c1 + t*(c2 + ...))
    # -------------------------------------------------------------------------
    vbroadcastsd zmm19, [rip + C1_log1p]   # 1.0
    vbroadcastsd zmm20, [rip + C2_log1p]   # -0.5
    vbroadcastsd zmm21, [rip + C3_log1p]   # 0.333...
    vbroadcastsd zmm22, [rip + C4_log1p]   # -0.25
    vbroadcastsd zmm23, [rip + C5_log1p]   # 0.2
    vbroadcastsd zmm24, [rip + C6_log1p]   # -0.166...
    vbroadcastsd zmm25, [rip + C7_log1p]   # 0.142...
    vbroadcastsd zmm26, [rip + C8_log1p]   # -0.125

    # -------------------------------------------------------------------------
    # exp constants
    # -------------------------------------------------------------------------
    vbroadcastsd zmm27, [rip + INV_LN2]
    vbroadcastsd zmm28, [rip + LN2]
    vbroadcastsd zmm29, [rip + HALF]
    vbroadcastsd zmm30, [rip + MAGIC]
    vbroadcastsd zmm31, [rip + ONE]

    # -------------------------------------------------------------------------
    # Additional constants (clamp bounds, exp poly, min_pp)
    # -------------------------------------------------------------------------
    vbroadcastsd zmm5,  [rip + NEG700]
    vbroadcastsd zmm6,  [rip + POS700]
    vbroadcastsd zmm7,  [rip + EC1]
    vbroadcastsd zmm8,  [rip + EC2]
    vbroadcastsd zmm9,  [rip + EC3]
    vbroadcastsd zmm10, [rip + EC4]
    vbroadcastsd zmm11, [rip + EC5]
    vbroadcastsd zmm12, [rip + EC6]
    vbroadcastsd zmm13, [rip + MIN_PP]

    # -------------------------------------------------------------------------
    # Zero the change accumulator
    # -------------------------------------------------------------------------
    vpxorq zmm15, zmm15, zmm15

    xor rax, rax                      # loop index i = 0

# =============================================================================
# MAIN LOOP — processes 8 run-lengths per iteration
# =============================================================================
.Lloop:
    cmp rax, r10
    jae .Lexit

    # -------------------------------------------------------------------------
    # Load data arrays (8 doubles each)
    # -------------------------------------------------------------------------
    vmovupd zmm0, [rdx + rax*8]       # mu
    vmovupd zmm1, [rcx + rax*8]       # C1 (BOCPD precomputed)
    vmovupd zmm2, [r8  + rax*8]       # C2 (BOCPD precomputed)
    vmovupd zmm3, [r9  + rax*8]       # inv_ssn
    vmovupd zmm4, [rdi + rax*8]       # r_old

    # -------------------------------------------------------------------------
    # Compute t = (x - mu)² * inv_ssn
    # -------------------------------------------------------------------------
    vsubpd zmm14, zmm16, zmm0         # z = x - mu
    vmulpd zmm14, zmm14, zmm14        # z²
    vmulpd zmm14, zmm14, zmm3         # t = z² * inv_ssn

    # -------------------------------------------------------------------------
    # log1p(t) via 8-term Horner polynomial
    # y = c8
    # y = c7 + t*y
    # ...
    # y = c1 + t*y
    # log1p ≈ t * y
    # -------------------------------------------------------------------------
    vmovapd zmm0, zmm26               # y = C8
    vfmadd213pd zmm0, zmm14, zmm25    # y = C7 + t*C8
    vfmadd213pd zmm0, zmm14, zmm24    # y = C6 + t*y
    vfmadd213pd zmm0, zmm14, zmm23    # y = C5 + t*y
    vfmadd213pd zmm0, zmm14, zmm22    # y = C4 + t*y
    vfmadd213pd zmm0, zmm14, zmm21    # y = C3 + t*y
    vfmadd213pd zmm0, zmm14, zmm20    # y = C2 + t*y
    vfmadd213pd zmm0, zmm14, zmm19    # y = C1 + t*y
    vmulpd zmm0, zmm0, zmm14          # log1p(t) = t * y

    # -------------------------------------------------------------------------
    # ln_pp = C1_bocpd - C2_bocpd * log1p
    # Using vfnmadd231pd: zmm1 = zmm1 - zmm2 * zmm0
    # -------------------------------------------------------------------------
    vfnmadd231pd zmm1, zmm2, zmm0     # zmm1 = C1 - C2 * log1p

    # -------------------------------------------------------------------------
    # exp_fast(ln_pp)
    # -------------------------------------------------------------------------

    # Clamp to [-700, 700] to avoid overflow/underflow
    vmaxpd zmm1, zmm1, zmm5           # max with -700
    vminpd zmm1, zmm1, zmm6           # min with +700

    # t = ln_pp * (1/ln2)
    vmulpd zmm3, zmm1, zmm27          # t

    # k = round(t) to nearest integer
    vaddpd zmm0, zmm3, zmm29          # t + 0.5
    vrndscalepd zmm0, zmm0, 0         # round to nearest even

    # f = t - k  (fractional part, in [-0.5, 0.5])
    vsubpd zmm3, zmm3, zmm0           # f

    # z = f * ln2
    vmulpd zmm3, zmm3, zmm28          # z

    # -------------------------------------------------------------------------
    # exp(z) polynomial: 1 + z*(c1 + z*(c2 + z*(c3 + z*(c4 + z*(c5 + z*c6)))))
    # -------------------------------------------------------------------------
    vmovapd zmm14, zmm12              # y = EC6
    vfmadd213pd zmm14, zmm3, zmm11    # y = EC5 + z*y
    vfmadd213pd zmm14, zmm3, zmm10    # y = EC4 + z*y
    vfmadd213pd zmm14, zmm3, zmm9     # y = EC3 + z*y
    vfmadd213pd zmm14, zmm3, zmm8     # y = EC2 + z*y
    vfmadd213pd zmm14, zmm3, zmm7     # y = EC1 + z*y
    vfmadd213pd zmm14, zmm3, zmm31    # y = 1 + z*y

    # -------------------------------------------------------------------------
    # Construct 2^k via bit manipulation
    # After adding MAGIC, the integer k sits in the low bits.
    # We mask, add bias (1023), and shift into the exponent field.
    # -------------------------------------------------------------------------
    vaddpd  zmm0, zmm0, zmm30                  # k + MAGIC
    vpandq  zmm0, zmm0, [rip + LOMASK]         # extract low 11 bits
    vpaddq  zmm0, zmm0, [rip + BIAS_VEC]       # add exponent bias 1023
    vpsllq  zmm0, zmm0, 52                     # shift to exponent position

    # pp = exp(z) * 2^k
    vmulpd zmm14, zmm14, zmm0

    # Clamp pp to minimum (avoid denormals)
    vmaxpd zmm14, zmm14, zmm13

    # -------------------------------------------------------------------------
    # Compute outputs:
    #   r_pp   = r_old * pp
    #   growth = r_pp * omh  → store at r_new[i+1]
    #   change = r_pp * h    → accumulate
    # -------------------------------------------------------------------------
    vmulpd zmm0, zmm4, zmm14          # r_pp = r_old * pp
    vmulpd zmm3, zmm0, zmm18          # growth = r_pp * omh
    vmulpd zmm4, zmm0, zmm17          # change = r_pp * h

    # Store growth at r_new[i+1] (offset by 8 bytes = 1 double)
    vmovupd [rsi + rax*8 + 8], zmm3

    # Accumulate change
    vaddpd zmm15, zmm15, zmm4

    add rax, 8
    jmp .Lloop

# =============================================================================
# EXIT — horizontal reduction of change accumulator
# =============================================================================
.Lexit:
    # -------------------------------------------------------------------------
    # Horizontal sum: zmm15 (8 doubles) → scalar in xmm0
    # -------------------------------------------------------------------------
    vextractf64x4 ymm1, zmm15, 1      # ymm1 = high 256 bits of zmm15
    vaddpd ymm0, ymm15, ymm1          # ymm0 = low256 + high256 (ymm15 aliases low half)

    vextractf128 xmm1, ymm0, 1        # xmm1 = high 128 bits of ymm0
    vaddpd xmm0, xmm0, xmm1           # xmm0 = low128 + high128

    vhaddpd xmm0, xmm0, xmm0          # xmm0[0] = xmm0[0] + xmm0[1]

    # Store result
    movsd [r11], xmm0

    # -------------------------------------------------------------------------
    # Epilogue — clear all vector registers to avoid AVX-SSE penalties
    # -------------------------------------------------------------------------
    vzeroall

    mov rsp, rbp
    pop rbp
    ret


# =============================================================================
# CONSTANTS SECTION (64-byte aligned for ZMM loads)
# =============================================================================
.section .rodata
.align 64

# log1p polynomial coefficients (Taylor series)
C1_log1p: .double 1.0
C2_log1p: .double -0.5
C3_log1p: .double 0.3333333333333333
C4_log1p: .double -0.25
C5_log1p: .double 0.2
C6_log1p: .double -0.1666666666666667
C7_log1p: .double 0.1428571428571429
C8_log1p: .double -0.125

# exp decomposition constants
INV_LN2: .double 1.4426950408889634    # 1/ln(2)
LN2:     .double 0.6931471805599453    # ln(2)
HALF:    .double 0.5
MAGIC:   .double 6755399441055744.0    # 2^52 + 2^51 (for int extraction)
ONE:     .double 1.0

# Clamp bounds
NEG700:  .double -700.0
POS700:  .double 700.0
MIN_PP:  .double 1e-300

# exp polynomial coefficients (minimax on [-ln2/2, ln2/2])
EC1: .double 0.6931471805599453        # ln(2)
EC2: .double 0.24022650695910072       # ln(2)²/2!
EC3: .double 0.05550410866482158       # ln(2)³/3!
EC4: .double 0.009618129107628477      # ln(2)⁴/4!
EC5: .double 0.0013333558146428443     # ln(2)⁵/5!
EC6: .double 0.00015403530393381608    # ln(2)⁶/6!

# 8-element vectors for ZMM integer operations
.align 64
BIAS_VEC:
    .quad 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023

LOMASK:
    .quad 0x7FF, 0x7FF, 0x7FF, 0x7FF, 0x7FF, 0x7FF, 0x7FF, 0x7FF