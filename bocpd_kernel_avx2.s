.intel_syntax noprefix
.text
.global bocpd_kernel_avx2
.type bocpd_kernel_avx2, @function

# =============================================================================
#  bocpd_kernel_avx2 -- AVX2 8-lane unrolled BOCPD compute kernel
#  Computes:
#       growth[i+1] = r[i] * pp[i] * omh
#       change_sum += r[i] * pp[i] * h
#  Inputs:
#       rdi = r_old
#       rsi = r_new
#       rdx = mu
#       rcx = C1
#       r8  = C2
#       r9  = inv_ssn
#       [rbp+16]  size_t n_padded
#       [rbp+24]  double x
#       [rbp+32]  double h
#       [rbp+40]  double omh
#       [rbp+48]  double thresh  (not used here)
#       [rbp+56]  double* change_out
#
#  NOTE: MAP/truncation/normalization left to C layer.
# =============================================================================

bocpd_kernel_avx2:
    push rbp
    mov rbp, rsp
    and rsp, -32            # align stack to 32 bytes for AVX

    # ---------------------------------------------------------------
    # Load scalar arguments from stack
    # ---------------------------------------------------------------
    mov     r10, [rbp+16]       # n_padded
    movsd   xmm0, [rbp+24]      # x
    movsd   xmm1, [rbp+32]      # h
    movsd   xmm2, [rbp+40]      # omh
    movsd   xmm3, [rbp+56]      # *change_out pointer stored in xmm3 temporarily
    mov     r11, [rbp+56]       # r11 = change_out pointer

    # ---------------------------------------------------------------
    # Broadcast scalars to AVX registers
    # ---------------------------------------------------------------
    vbroadcastsd ymm1, xmm0     # ymm1 = x
    vbroadcastsd ymm20, xmm1    # ymm20 = h (reusing xmm1 value)
    vbroadcastsd ymm21, xmm2    # ymm21 = omh

    # ---------------------------------------------------------------
    # Zero change accumulator
    # ---------------------------------------------------------------
    vxorpd ymm30, ymm30, ymm30  # ymm30 = 0 (accumulate r*pp*h)

    # ---------------------------------------------------------------
    # Load log1p polynomial constants into registers (Horner scheme)
    #   c1..c8 = +1, -1/2, +1/3, -1/4, +1/5, -1/6, +1/7, -1/8
    # ---------------------------------------------------------------
    vbroadcastsd ymm8,  [rip + C1_log1p]
    vbroadcastsd ymm9,  [rip + C2_log1p]
    vbroadcastsd ymm10, [rip + C3_log1p]
    vbroadcastsd ymm11, [rip + C4_log1p]
    vbroadcastsd ymm12, [rip + C5_log1p]
    vbroadcastsd ymm13, [rip + C6_log1p]
    vbroadcastsd ymm14, [rip + C7_log1p]
    vbroadcastsd ymm15, [rip + C8_log1p]

    # ---------------------------------------------------------------
    # Load exp polynomial constants
    # ---------------------------------------------------------------
    vbroadcastsd ymm2, [rip + INV_LN2]   # 1/ln2
    vbroadcastsd ymm3, [rip + LN2]       # ln2
    vbroadcastsd ymm4, [rip + EC1]       # exp poly coefficients
    vbroadcastsd ymm5, [rip + EC2]
    vbroadcastsd ymm6, [rip + EC3]
    vbroadcastsd ymm7, [rip + EC4]
    vbroadcastsd ymm22,[rip + EC5]
    vbroadcastsd ymm23,[rip + EC6]

    # ---------------------------------------------------------------
    # Other constants needed in exp(x)
    # ---------------------------------------------------------------
    vbroadcastsd ymm24, [rip + NEG700]    # clamp low
    vbroadcastsd ymm25, [rip + POS700]    # clamp high
    vbroadcastsd ymm26, [rip + HALF]      # 0.5
    vbroadcastsd ymm27, [rip + MAGIC]     # 2^52+2^51 for int extraction
    vbroadcastsd ymm28, [rip + BIAS_VEC]  # +1023 adjustment
    vbroadcastsd ymm29, [rip + LOMASK]    # low 11 bits mask for exponent

    # ---------------------------------------------------------------
    # rax = loop index (0..n_padded)
    # ---------------------------------------------------------------
    xor rax, rax

###############################################################################
# MAIN LOOP — processes 8 run-lengths per iteration (2× unroll)
###############################################################################

.Lloop:

    cmp     rax, r10
    jae     .Lexit

    ###########################################################################
    # --------------------------- BLOCK A  (i .. i+3) -------------------------
    ###########################################################################

    # Load parameters (aligned loads)
    vmovapd ymm0,  [rdx + rax*8]     # mu
    vmovapd ymm10, [rcx + rax*8]     # C1
    vmovapd ymm11, [r8  + rax*8]     # C2
    vmovapd ymm4,  [r9  + rax*8]     # inv_ssn
    vmovapd ymm14, [rdi + rax*8]     # r_old

    # z = x - mu
    vsubpd  ymm2, ymm1, ymm0

    # z²
    vmulpd  ymm3, ymm2, ymm2

    # t = z² * inv_ssn
    vmulpd  ymm5, ymm3, ymm4

    # ----------------------- log1p polynomial ------------------------------
    # Horner: t*(c1 + t*(c2 + t*(c3 + ...)))
    # Start y = c8 + t*c7
    vfmadd213pd ymm15, ymm5, ymm14    # y = c7 + t*c8
    # y = c6 + t*y
    vfmadd213pd ymm13, ymm5, ymm15
    # y = c5 + t*y
    vfmadd213pd ymm12, ymm5, ymm13
    # y = c4 + t*y
    vfmadd213pd ymm11, ymm5, ymm12
    # y = c3 + t*y
    vfmadd213pd ymm10, ymm5, ymm11
    # y = c2 + t*y
    vfmadd213pd ymm9,  ymm5, ymm10
    # y = c1 + t*y
    vfmadd213pd ymm8,  ymm5, ymm9

    # log1p ≈ t * y
    vmulpd  ymm6, ymm8, ymm5

    # ln_pp = C1 - C2 * log1p
    vfnmadd213pd ymm6, ymm11, ymm10   # = -C2*log1p + C1

    # ----------------------- exp_fast approximation ------------------------
    # clamp ln_pp to [-700, 700]
    vmaxpd  ymm6, ymm6, ymm24
    vminpd  ymm6, ymm6, ymm25

    # t = x * 1/ln2
    vmulpd  ymm5, ymm6, ymm2

    # k = round(t)
    vaddpd  ymm9, ymm5, ymm26
    vroundpd ymm9, ymm9, 0      # nearest integer

    # f = t - k
    vsubpd  ymm5, ymm5, ymm9

    # z = f * ln2
    vmulpd  ymm2, ymm5, ymm3

    # Horner: exp(z) poly 6th order
    # y = EC6
    vmovapd ymm12, ymm23
    # y = EC5 + z*y
    vfmadd213pd ymm12, ymm2, ymm22
    # y = EC4 + z*y
    vfmadd213pd ymm12, ymm2, ymm7
    # y = EC3 + z*y
    vfmadd213pd ymm12, ymm2, ymm6
    # y = EC2 + z*y
    vfmadd213pd ymm12, ymm2, ymm5
    # y = EC1 + z*y
    vfmadd213pd ymm12, ymm2, ymm4
    # y = 1 + z*y
    vbroadcastsd ymm15, [rip + ONE]
    vfmadd213pd ymm12, ymm2, ymm15     # EC0 = 1

    # Construct 2^k exponent bits
    vaddpd ymm9, ymm9, ymm27           # k + magic
    vpaddq ymm9, ymm9, ymm28           # add bias 1023
    vpand  ymm9, ymm9, ymm29           # keep exponent bits
    vpsllq ymm9, ymm9, 52              # shift to exponent

    # scale = reinterpret as double
    vmovapd ymm9, ymm9                 # bitwise copy ok

    # pp = y * scale
    vmulpd ymm13, ymm12, ymm9

    # r_pp = r_old * pp
    vmulpd ymm15, ymm14, ymm13

    # growth = r_pp * omh
    vmulpd ymm0, ymm15, ymm21

    # change = r_pp * h
    vmulpd ymm7, ymm15, ymm20

    # store growth at r_new[i+1]
    vmovapd [rsi + (rax+1)*8], ymm0

    # accumulate change
    vaddpd ymm30, ymm30, ymm7


    ###########################################################################
    # --------------------------- BLOCK B  (i+4 .. i+7) -----------------------
    ###########################################################################

    # Load parameters for block B
    vmovapd ymm0,  [rdx + rax*8 + 32]     # mu B
    vmovapd ymm10, [rcx + rax*8 + 32]     # C1 B
    vmovapd ymm11, [r8  + rax*8 + 32]     # C2 B
    vmovapd ymm4,  [r9  + rax*8 + 32]     # inv_ssn B
    vmovapd ymm14, [rdi + rax*8 + 32]     # r_old B

    # z = x - mu
    vsubpd  ymm2, ymm1, ymm0

    # z²
    vmulpd  ymm3, ymm2, ymm2

    # t = z² * inv_ssn
    vmulpd  ymm5, ymm3, ymm4

    # log1p poly
    vfmadd213pd ymm15, ymm5, ymm14
    vfmadd213pd ymm13, ymm5, ymm15
    vfmadd213pd ymm12, ymm5, ymm13
    vfmadd213pd ymm11, ymm5, ymm12
    vfmadd213pd ymm10, ymm5, ymm11
    vfmadd213pd ymm9,  ymm5, ymm10
    vfmadd213pd ymm8,  ymm5, ymm9
    vmulpd      ymm6,  ymm8, ymm5       # log1p

    # ln_pp
    vfnmadd213pd ymm6, ymm11, ymm10     # = C1 - C2*log1p

    # clamp
    vmaxpd  ymm6, ymm6, ymm24
    vminpd  ymm6, ymm6, ymm25

    # t = x * 1/ln2
    vmulpd  ymm5, ymm6, ymm2

    # k = round(t)
    vaddpd  ymm9, ymm5, ymm26
    vroundpd ymm9, ymm9, 0

    # f = t - k
    vsubpd  ymm5, ymm5, ymm9

    # z = f*ln2
    vmulpd  ymm2, ymm5, ymm3

    # exp poly
    vmovapd ymm12, ymm23
    vfmadd213pd ymm12, ymm2, ymm22
    vfmadd213pd ymm12, ymm2, ymm7
    vfmadd213pd ymm12, ymm2, ymm6
    vfmadd213pd ymm12, ymm2, ymm5
    vfmadd213pd ymm12, ymm2, ymm4
    vfmadd213pd ymm12, ymm2, ymm15

    # construct 2^k
    vaddpd ymm9, ymm9, ymm27
    vpaddq ymm9, ymm9, ymm28
    vpand  ymm9, ymm9, ymm29
    vpsllq ymm9, ymm9, 52

    vmovapd ymm9, ymm9      # reinterpret exponent

    # pp = y * scale
    vmulpd ymm13, ymm12, ymm9

    # r_pp
    vmulpd ymm15, ymm14, ymm13

    # growth
    vmulpd ymm0, ymm15, ymm21

    # change
    vmulpd ymm7, ymm15, ymm20

    # store growth at r_new[i+5]
    vmovapd [rsi + (rax+5)*8], ymm0

    # accumulate change
    vaddpd ymm30, ymm30, ymm7

    # increment loop index by 8 elements
    add rax, 8
    jmp .Lloop

###############################################################################
# EXIT + REDUCTION
###############################################################################

.Lexit:

    # ---------------------------------------------------------------
    # Horizontal reduce ymm30 (change accumulator)
    # ---------------------------------------------------------------
    vextractf128 xmm1, ymm30, 1    # hi 128 bits
    vaddpd       xmm0, xmm30, xmm1 # add lo+hi
    vhaddpd      xmm0, xmm0, xmm0  # add lanes

    # xmm0 now holds final change_sum

    # ---------------------------------------------------------------
    # Store the result to change_out pointer (r11)
    # ---------------------------------------------------------------
    movsd [r11], xmm0

    # ---------------------------------------------------------------
    # Epilogue
    # ---------------------------------------------------------------
    vzeroupper       # avoid AVX-SSE transition penalties
    mov rsp, rbp
    pop rbp
    ret


###############################################################################
# CONSTANT TABLE (read-only, aligned)
###############################################################################
.align 32

# log1p polynomial constants
C1_log1p: .double 1.0
C2_log1p: .double -0.5
C3_log1p: .double 0.3333333333333333
C4_log1p: .double -0.25
C5_log1p: .double 0.2
C6_log1p: .double -0.1666666666666667
C7_log1p: .double 0.1428571428571429
C8_log1p: .double -0.125

# exp_fast constants
INV_LN2:  .double 1.4426950408889634
LN2:      .double 0.6931471805599453
HALF:     .double 0.5
NEG700:   .double -700.0
POS700:   .double 700.0
MAGIC:    .double 6755399441055744.0   # 2^52+2^51
ONE:      .double 1.0

# exp polynomial coefficients (6th-order Horner, ln2 powers)
EC1: .double 0.6931471805599453
EC2: .double 0.24022650695910072
EC3: .double 0.05550410866482158
EC4: .double 0.009618129107628477
EC5: .double 0.0013333558146428443
EC6: .double 0.00015403530393381608

# vectors for exponent construction
BIAS_VEC:
    .double 1023.0, 1023.0, 1023.0, 1023.0

LOMASK:
    .quad 0x7FF, 0x7FF, 0x7FF, 0x7FF
