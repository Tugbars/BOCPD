.intel_syntax noprefix
.text
.global bocpd_kernel_avx512
.type bocpd_kernel_avx512, @function

bocpd_kernel_avx512:
    push rbp
    mov rbp, rsp
    and rsp, -64

    # Load stack args
    mov r10, [rbp+16]         # n_padded
    movsd xmm0, [rbp+24]      # x
    movsd xmm1, [rbp+32]      # h
    movsd xmm2, [rbp+40]      # omh
    movsd xmm3, [rbp+48]      # thresh
    mov r11, [rbp+56]         # change_out

    # Broadcast scalars
    vbroadcastsd zmm16, xmm0      # x
    vbroadcastsd zmm17, xmm1      # h
    vbroadcastsd zmm18, xmm2      # omh
    vbroadcastsd zmm14, xmm3      # thresh

    # log1p polynomial constants
    vbroadcastsd zmm19, [rip + C1_log1p]
    vbroadcastsd zmm20, [rip + C2_log1p]
    vbroadcastsd zmm21, [rip + C3_log1p]
    vbroadcastsd zmm22, [rip + C4_log1p]
    vbroadcastsd zmm23, [rip + C5_log1p]
    vbroadcastsd zmm24, [rip + C6_log1p]
    vbroadcastsd zmm25, [rip + C7_log1p]
    vbroadcastsd zmm26, [rip + C8_log1p]

    # exp polynomial constants
    vbroadcastsd zmm27, [rip + INV_LN2]
    vbroadcastsd zmm28, [rip + LN2]
    vbroadcastsd zmm29, [rip + HALF]
    vbroadcastsd zmm30, [rip + MAGIC]
    vbroadcastsd zmm31, [rip + ONE]

    # accumulator
    vpxorq zmm15, zmm15, zmm15

    xor rax, rax               # loop index = 0

# ---------------------------------------------------------------------------
# MAIN LOOP — processes 8 run-length slots per iteration
# ---------------------------------------------------------------------------

.Lloop:
    cmp rax, r10
    jae .Lexit

    # Load data (μ, C1, C2, inv_ssn, r_old)
    vmovapd zmm0,  [rdx + rax*8]      # mu
    vmovapd zmm1,  [rcx + rax*8]      # C1
    vmovapd zmm2,  [r8  + rax*8]      # C2
    vmovapd zmm3,  [r9  + rax*8]      # inv_ssn
    vmovapd zmm4,  [rdi + rax*8]      # r_old

    # -----------------------------------
    # z = x - μ
    # z² = z * z
    # -----------------------------------
    vsubpd zmm5, zmm16, zmm0          # z
    vmulpd zmm6, zmm5, zmm5           # z²

    # t = z² * inv_ssn
    vmulpd zmm7, zmm6, zmm3           # t

    # -----------------------------------
    # log1p(t) via 8-term Horner polynomial
    # poly = C1 + t*(C2 + t*(C3 + ...))
    # -----------------------------------

    # y = C8 + t*C7
    vfmadd213pd zmm8, zmm7, zmm25     # zmm8 = C7 + t*C8

    # y = C6 + t*y
    vfmadd213pd zmm8, zmm7, zmm24

    # y = C5 + t*y
    vfmadd213pd zmm8, zmm7, zmm23

    # y = C4 + t*y
    vfmadd213pd zmm8, zmm7, zmm22

    # y = C3 + t*y
    vfmadd213pd zmm8, zmm7, zmm21

    # y = C2 + t*y
    vfmadd213pd zmm8, zmm7, zmm20

    # y = C1 + t*y
    vfmadd213pd zmm8, zmm7, zmm19

    # log1p ≈ t * y
    vmulpd zmm8, zmm8, zmm7           # log1p(t)

    # -----------------------------------
    # ln_pp = C1_slot − C2_slot * log1p
    # -----------------------------------
    vfnmadd231pd zmm1, zmm2, zmm8     # zmm1 = C1 - C2*log1p

    # -----------------------------------
    # exp_fast(ln_pp)
    # -----------------------------------

    # clamp
    vbroadcastsd zmm9, [rip + NEG700]
    vbroadcastsd zmm10, [rip + POS700]
    vmaxpd zmm1, zmm1, zmm9
    vminpd zmm1, zmm1, zmm10

    # t = x * inv_ln2
    vmulpd zmm11, zmm1, zmm27

    # k = round(t)
    vaddpd zmm12, zmm11, zmm29
    vrndscalepd zmm12, zmm12, 0b0000     # round to nearest integer

    # f = t − k
    vsubpd zmm11, zmm11, zmm12

    # z = f * ln2
    vmulpd zmm13, zmm11, zmm28

    # evaluate exp(z) polynomial
    vbroadcastsd zmm9, [rip + EC6]
    vbroadcastsd zmm10, [rip + EC5]
    vfmadd213pd zmm9, zmm13, zmm10      # EC5 + z*EC6

    vbroadcastsd zmm10, [rip + EC4]
    vfmadd213pd zmm9, zmm13, zmm10

    vbroadcastsd zmm10, [rip + EC3]
    vfmadd213pd zmm9, zmm13, zmm10

    vbroadcastsd zmm10, [rip + EC2]
    vfmadd213pd zmm9, zmm13, zmm10

    vbroadcastsd zmm10, [rip + EC1]
    vfmadd213pd zmm9, zmm13, zmm10

    vfmadd213pd zmm9, zmm13, zmm31      # 1 + z*y

    # 2^k
    vaddpd zmm12, zmm12, zmm30          # add magic

    # reinterpret bits → integer shifts
    vshufi64x2 zmm12, zmm12, zmm12, 0    # unify format
    vpsrlq zmm12, zmm12, 52              # extract exponent
    vpaddq zmm12, zmm12, [rip + BIAS_VEC]
    vpsllq zmm12, zmm12, 52

    # pp = exp(z) * 2^k
    vmulpd zmm1, zmm9, zmm12

    # clamp pp
    vbroadcastsd zmm9, [rip + MIN_PP]
    vmaxpd zmm1, zmm1, zmm9

    # -----------------------------------
    # r_pp = r_old * pp
    # growth = r_pp * omh
    # change = r_pp * h
    # -----------------------------------

    vmulpd zmm9,  zmm4, zmm1          # r_pp
    vmulpd zmm10, zmm9, zmm18         # growth
    vmulpd zmm11, zmm9, zmm17         # change

    # store growth → r_new[i+1]
    vmovapd [rsi + rax*8 + 8], zmm10

    # accumulate change
    vaddpd zmm15, zmm15, zmm11

    add rax, 8
    jmp .Lloop

# ----------------------------
# End of loop
# ----------------------------

.Lexit:
    # horizontal sum of zmm15
    vextractf64x4 ymm0, zmm15, 1
    vaddpd ymm0, ymm0, ymm15

    vhaddpd ymm0, ymm0, ymm0
    vhaddpd ymm0, ymm0, ymm0

    movsd [r11], xmm0

    vzeroall
    mov rsp, rbp
    pop rbp
    ret

.align 64
C1_log1p: .double 1.0
C2_log1p: .double -0.5
C3_log1p: .double 0.3333333333333333
C4_log1p: .double -0.25
C5_log1p: .double 0.2
C6_log1p: .double -0.1666666666666667
C7_log1p: .double 0.1428571428571429
C8_log1p: .double -0.125

INV_LN2:  .double 1.4426950408889634
LN2:      .double 0.6931471805599453
HALF:     .double 0.5
MAGIC:    .double 6755399441055744.0
ONE:      .double 1.0
NEG700:   .double -700.0
POS700:   .double 700.0
MIN_PP:   .double 1e-300

# integer constants
BIAS_VEC:
    .quad 1023, 1023, 1023, 1023
