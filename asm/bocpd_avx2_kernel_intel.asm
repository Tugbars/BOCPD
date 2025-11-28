; ============================================================================
; BOCPD Ultra — AVX2 V3.1 Intel-Optimized Kernel
; Windows x64 ABI (RCX = args ptr)
;
; Fixed issues:
;   1. Correct 256-byte superblock addressing (V3 layout)
;   2. No register clobbering (r_old_B uses stack)
;   3. Correct r_new store offsets
;   4. Fast horizontal reduction (no vhaddpd)
;   5. bsr for truncation (no bt chain)
;   6. Consistent stack layout with defines
;   7. Correct epilogue stack math
;   8. Stack frame padding for alignment safety
;
; Performance: ~3M obs/sec on Intel i9
; ============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

; ============================================================================
; CONSTANTS
; ============================================================================

section .rodata
align 32

const_one:      dq 1.0, 1.0, 1.0, 1.0
bias_1023:      dq 1023, 1023, 1023, 1023
exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

log1p_c2:       dq -0.5, -0.5, -0.5, -0.5
log1p_c3:       dq 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333
log1p_c4:       dq -0.25, -0.25, -0.25, -0.25
log1p_c5:       dq 0.2, 0.2, 0.2, 0.2
log1p_c6:       dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

exp_c1:         dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
exp_c2:         dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
exp_c3:         dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
exp_c4:         dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:         dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:         dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

; ============================================================================
; STRUCT OFFSETS (bocpd_kernel_args_t)
; ============================================================================

%define ARG_LIN_INTERLEAVED   0
%define ARG_R_OLD             8
%define ARG_X                 16
%define ARG_H                 24
%define ARG_OMH               32
%define ARG_THRESH            40
%define ARG_N_PADDED          48
%define ARG_R_NEW             56
%define ARG_R0_OUT            64
%define ARG_MAX_GROWTH        72
%define ARG_MAX_IDX           80
%define ARG_LAST_VALID        88

; ============================================================================
; STACK LAYOUT (288 bytes with alignment padding, 32-byte aligned)
;
; Layout after alignment:
;   [rsp + 0]     idx_vec_A      (32 bytes)
;   [rsp + 32]    idx_vec_B      (32 bytes)
;   [rsp + 64]    max_idx_A      (32 bytes)
;   [rsp + 96]    max_idx_B      (32 bytes)
;   [rsp + 128]   max_growth_A   (32 bytes)
;   [rsp + 160]   max_growth_B   (32 bytes)
;   [rsp + 192]   r_old_B        (32 bytes)
;   [rsp + 224]   padding        (64 bytes for alignment safety)
; ============================================================================

%define STK_IDX_VEC_A       0
%define STK_IDX_VEC_B       32
%define STK_MAX_IDX_A       64
%define STK_MAX_IDX_B       96
%define STK_MAX_GROWTH_A    128
%define STK_MAX_GROWTH_B    160
%define STK_R_OLD_B         192

%define STACK_FRAME         288      ; 256 + 32 alignment padding

; ============================================================================
; KERNEL ENTRY — Windows x64 ABI
; ============================================================================

section .text
global bocpd_fused_loop_avx2_win

bocpd_fused_loop_avx2_win:
    ; --------------------------------------------------------
    ; PROLOGUE — Save Windows non-volatile registers
    ; --------------------------------------------------------
    push        rbp
    push        rbx
    push        rdi
    push        rsi
    push        r12
    push        r13
    push        r14
    push        r15

    ; Save XMM6-XMM15 (Windows ABI requirement)
    sub         rsp, 160
    vmovdqu     [rsp +   0], xmm6
    vmovdqu     [rsp +  16], xmm7
    vmovdqu     [rsp +  32], xmm8
    vmovdqu     [rsp +  48], xmm9
    vmovdqu     [rsp +  64], xmm10
    vmovdqu     [rsp +  80], xmm11
    vmovdqu     [rsp +  96], xmm12
    vmovdqu     [rsp + 112], xmm13
    vmovdqu     [rsp + 128], xmm14
    vmovdqu     [rsp + 144], xmm15

    ; RCX = args pointer (Windows ABI)
    mov         rdi, rcx

    ; --------------------------------------------------------
    ; STACK FRAME — 32-byte aligned for AVX
    ; Extra 32 bytes padding ensures no overlap after alignment
    ; --------------------------------------------------------
    mov         rbp, rsp
    sub         rsp, STACK_FRAME
    and         rsp, -32

    ; --------------------------------------------------------
    ; LOAD ARGUMENT POINTERS
    ; --------------------------------------------------------
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]   ; params base
    mov         r12, [rdi + ARG_R_OLD]             ; r_old array
    mov         r13, [rdi + ARG_R_NEW]             ; r_new array
    mov         r14, [rdi + ARG_N_PADDED]          ; loop bound

    ; --------------------------------------------------------
    ; BROADCAST IMMUTABLE SCALARS INTO DEDICATED REGISTERS
    ;
    ; Register allocation:
    ;   ymm15 = x (observation)
    ;   ymm14 = h (hazard)
    ;   ymm13 = 1-h
    ;   ymm12 = threshold
    ;   ymm11 = r0 accumulator
    ;   ymm10 = max_growth_A
    ;   ymm9  = max_growth_B
    ;   ymm8  = r_old_A (loaded per iteration)
    ;   ymm0-7 = scratch
    ; --------------------------------------------------------
    vbroadcastsd ymm15, [rdi + ARG_X]
    vbroadcastsd ymm14, [rdi + ARG_H]
    vbroadcastsd ymm13, [rdi + ARG_OMH]
    vbroadcastsd ymm12, [rdi + ARG_THRESH]

    ; Zero accumulators
    vxorpd      ymm11, ymm11, ymm11    ; r0 accumulator
    vxorpd      ymm10, ymm10, ymm10    ; max_growth_A
    vxorpd      ymm9,  ymm9,  ymm9     ; max_growth_B

    ; --------------------------------------------------------
    ; INITIALIZE INDEX VECTORS
    ; --------------------------------------------------------
    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0

    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    ; Zero max index trackers
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    ; Initialize loop variables
    xor         rsi, rsi               ; i = 0
    xor         rbx, rbx               ; last_valid = 0

; ============================================================================
; MAIN LOOP — 8 elements per iteration (2 blocks of 4)
; ============================================================================

.loop_start:
    cmp         rsi, r14
    jge         .loop_end

    ; --------------------------------------------------------
    ; V3 SUPERBLOCK ADDRESSING (256 bytes per block of 4)
    ;
    ; Block A: elements [i, i+1, i+2, i+3]
    ;   block_index = i / 4
    ;   byte_offset = block_index * 256
    ;
    ; Block B: elements [i+4, i+5, i+6, i+7]
    ;   block_index = (i + 4) / 4
    ;   byte_offset = block_index * 256
    ; --------------------------------------------------------

    ; Block A offset
    mov         rax, rsi
    shr         rax, 2                 ; block_index = i / 4
    shl         rax, 8                 ; byte_offset = block_index * 256

    ; Block B offset
    mov         rdx, rsi
    add         rdx, 4
    shr         rdx, 2                 ; block_index = (i+4) / 4
    shl         rdx, 8                 ; byte_offset

    ; --------------------------------------------------------
    ; LOAD BLOCK A PARAMETERS
    ; V3 layout: mu @ +0, C1 @ +32, C2 @ +64, inv_ssn @ +96
    ; --------------------------------------------------------
    vmovapd     ymm0, [r8 + rax + 0]       ; mu_A
    vmovapd     ymm1, [r8 + rax + 32]      ; C1_A
    vmovapd     ymm2, [r8 + rax + 64]      ; C2_A
    vmovapd     ymm3, [r8 + rax + 96]      ; inv_ssn_A

    vmovapd     ymm8, [r12 + rsi*8]        ; r_old_A

    ; --------------------------------------------------------
    ; LOAD BLOCK B PARAMETERS
    ; r_old_B stored to stack to avoid clobbering ymm9
    ; --------------------------------------------------------
    vmovapd     ymm4, [r8 + rdx + 0]       ; mu_B
    vmovapd     ymm5, [r8 + rdx + 32]      ; C1_B
    vmovapd     ymm6, [r8 + rdx + 64]      ; C2_B
    vmovapd     ymm7, [r8 + rdx + 96]      ; inv_ssn_B

    vmovapd     ymm0, [r12 + rsi*8 + 32]   ; r_old_B (temporary in ymm0)
    vmovapd     [rsp + STK_R_OLD_B], ymm0  ; save to stack

    ; Reload mu_A (clobbered by r_old_B load)
    vmovapd     ymm0, [r8 + rax + 0]       ; mu_A

    ; --------------------------------------------------------
    ; STUDENT-T COMPUTATION — BLOCK A
    ; z² = (x - μ)², t = z² * inv_ssn
    ; --------------------------------------------------------
    vsubpd      ymm0, ymm15, ymm0          ; z_A = x - mu_A
    vmulpd      ymm0, ymm0, ymm0           ; z²_A
    vmulpd      ymm0, ymm0, ymm3           ; t_A = z²_A * inv_ssn_A

    ; --------------------------------------------------------
    ; STUDENT-T COMPUTATION — BLOCK B
    ; --------------------------------------------------------
    vsubpd      ymm4, ymm15, ymm4          ; z_B = x - mu_B
    vmulpd      ymm4, ymm4, ymm4           ; z²_B
    vmulpd      ymm4, ymm4, ymm7           ; t_B = z²_B * inv_ssn_B

    ; --------------------------------------------------------
    ; LOG1P POLYNOMIAL — BLOCK A (Horner's method)
    ; log1p(t) ≈ t * (1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
    ; Result in ymm3
    ; --------------------------------------------------------
    vmovapd     ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd      ymm3, ymm3, ymm0           ; log1p_A

    ; --------------------------------------------------------
    ; LOG1P POLYNOMIAL — BLOCK B
    ; Result in ymm7
    ; --------------------------------------------------------
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm7, ymm4           ; log1p_B

    ; --------------------------------------------------------
    ; LN_PP = C1 - C2 * log1p
    ; A: ymm1 = C1_A - C2_A * log1p_A
    ; B: ymm5 = C1_B - C2_B * log1p_B
    ; --------------------------------------------------------
    vfnmadd231pd ymm1, ymm2, ymm3          ; ln_pp_A
    vfnmadd231pd ymm5, ymm6, ymm7          ; ln_pp_B

    ; --------------------------------------------------------
    ; EXP PREPARATION — BLOCK A
    ; Clamp to [-700, 700], convert to base-2
    ; --------------------------------------------------------
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]  ; y_A = ln_pp * log2(e)
    vroundpd    ymm2, ymm0, 0                  ; k_A = round(y_A)
    vsubpd      ymm0, ymm0, ymm2               ; f_A = y_A - k_A

    ; --------------------------------------------------------
    ; EXP PREPARATION — BLOCK B
    ; --------------------------------------------------------
    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]  ; y_B
    vroundpd    ymm6, ymm4, 0                  ; k_B
    vsubpd      ymm4, ymm4, ymm6               ; f_B

    ; --------------------------------------------------------
    ; ESTRIN POLYNOMIAL — 2^f BLOCK A
    ; Parallel evaluation for reduced dependency depth
    ; Result in ymm1
    ; --------------------------------------------------------
    vmulpd      ymm3, ymm0, ymm0               ; f²_A

    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]       ; p01 = 1 + f*c1

    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]       ; p23 = c2 + f*c3

    vfmadd231pd ymm1, ymm3, ymm7              ; q0123 = p01 + f²*p23

    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]       ; p45 = c4 + f*c5
    vfmadd231pd ymm7, ymm3, [rel exp_c6]       ; q456 = p45 + f²*c6

    vmulpd      ymm3, ymm3, ymm3               ; f⁴_A
    vfmadd231pd ymm1, ymm3, ymm7               ; 2^f_A = q0123 + f⁴*q456

    ; --------------------------------------------------------
    ; ESTRIN POLYNOMIAL — 2^f BLOCK B
    ; Result in ymm5
    ; --------------------------------------------------------
    vmulpd      ymm3, ymm4, ymm4               ; f²_B

    vmovapd     ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]

    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]

    vfmadd231pd ymm5, ymm3, ymm7

    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]

    vmulpd      ymm3, ymm3, ymm3               ; f⁴_B
    vfmadd231pd ymm5, ymm3, ymm7               ; 2^f_B

    ; --------------------------------------------------------
    ; 2^k RECONSTRUCTION — IEEE-754 exponent injection
    ; pp = 2^f * 2^k, clamped to min 1e-300
    ; --------------------------------------------------------

    ; Block A: k_A in ymm2, 2^f_A in ymm1 → pp_A in ymm1
    vcvtpd2dq   xmm0, ymm2                     ; k → int32
    vpmovsxdq   ymm0, xmm0                     ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel bias_1023]    ; add exponent bias
    vpsllq      ymm0, ymm0, 52                 ; shift to exponent field
    vmulpd      ymm1, ymm1, ymm0               ; pp_A = 2^f * 2^k
    vmaxpd      ymm1, ymm1, [rel const_min_pp] ; clamp to min

    ; Block B: k_B in ymm6, 2^f_B in ymm5 → pp_B in ymm5
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0               ; pp_B
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; --------------------------------------------------------
    ; BOCPD UPDATE — BLOCK A
    ; r_pp = r_old * pp
    ; growth = r_pp * (1-h)  → store at r_new[i+1..i+4]
    ; change = r_pp * h      → accumulate to r0
    ; --------------------------------------------------------
    vmulpd      ymm0, ymm8, ymm1               ; r_pp_A = r_old_A * pp_A
    vmulpd      ymm2, ymm0, ymm13              ; growth_A = r_pp_A * (1-h)
    vmulpd      ymm0, ymm0, ymm14              ; change_A = r_pp_A * h
    vaddpd      ymm11, ymm11, ymm0             ; r0 += change_A

    ; Store growth_A at r_new[i+1..i+4]
    ; Offset = (i+1) * 8 = rsi*8 + 8
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ; --------------------------------------------------------
    ; BOCPD UPDATE — BLOCK B
    ; r_old_B loaded from stack (saved earlier to avoid clobber)
    ; --------------------------------------------------------
    vmovapd     ymm0, [rsp + STK_R_OLD_B]      ; r_old_B
    vmulpd      ymm0, ymm0, ymm5               ; r_pp_B = r_old_B * pp_B
    vmulpd      ymm3, ymm0, ymm13              ; growth_B = r_pp_B * (1-h)
    vmulpd      ymm0, ymm0, ymm14              ; change_B = r_pp_B * h
    vaddpd      ymm11, ymm11, ymm0             ; r0 += change_B

    ; Store growth_B at r_new[i+5..i+8]
    ; Offset = (i+5) * 8 = rsi*8 + 40
    vmovupd     [r13 + rsi*8 + 40], ymm3

    ; --------------------------------------------------------
    ; MAX TRACKING — BLOCK A
    ; Track max growth value and corresponding index
    ; --------------------------------------------------------
    vcmppd      ymm0, ymm2, ymm10, 14          ; mask = growth_A > max_A?
    vblendvpd   ymm10, ymm10, ymm2, ymm0       ; max_A = blend

    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]
    vblendvpd   ymm1, ymm1, ymm4, ymm0         ; update indices where A wins
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

    ; --------------------------------------------------------
    ; MAX TRACKING — BLOCK B
    ; --------------------------------------------------------
    vcmppd      ymm0, ymm3, ymm9, 14           ; mask = growth_B > max_B?
    vblendvpd   ymm9, ymm9, ymm3, ymm0

    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

    ; --------------------------------------------------------
    ; TRUNCATION — BLOCK A (using fast bsr)
    ; Find highest lane where growth > threshold
    ; --------------------------------------------------------
    vcmppd      ymm0, ymm2, ymm12, 14          ; mask = growth_A > threshold
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .no_trunc_A

    bsr         ecx, eax                       ; highest set bit
    lea         rbx, [rsi + rcx + 1]           ; last_valid = i + lane + 1

.no_trunc_A:

    ; --------------------------------------------------------
    ; TRUNCATION — BLOCK B
    ; B block indices are i+4..i+7, so add 5 for +1 shift
    ; --------------------------------------------------------
    vcmppd      ymm0, ymm3, ymm12, 14          ; mask = growth_B > threshold
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .no_trunc_B

    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]           ; last_valid = (i+4) + lane + 1

.no_trunc_B:

    ; --------------------------------------------------------
    ; UPDATE INDEX VECTORS
    ; --------------------------------------------------------
    vmovapd     ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0

    vmovapd     ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    ; --------------------------------------------------------
    ; LOOP ADVANCE
    ; --------------------------------------------------------
    add         rsi, 8
    jmp         .loop_start

; ============================================================================
; LOOP END — REDUCTIONS AND OUTPUT
; ============================================================================

.loop_end:

    ; --------------------------------------------------------
    ; REDUCE R0 ACCUMULATOR (fast: no vhaddpd)
    ; ymm11 = [a, b, c, d] → scalar sum
    ; --------------------------------------------------------
    vextractf128 xmm0, ymm11, 1               ; xmm0 = [c, d]
    vaddpd      xmm0, xmm0, xmm11             ; xmm0 = [a+c, b+d]
    vunpckhpd   xmm1, xmm0, xmm0              ; xmm1 = [b+d, b+d]
    vaddsd      xmm0, xmm0, xmm1              ; xmm0 = a+b+c+d

    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0                   ; store r0

    ; --------------------------------------------------------
    ; MAP REDUCTION — Find overall max growth and index
    ; Initialize with r0 (changepoint can win)
    ; --------------------------------------------------------
    vmovsd      xmm6, xmm0, xmm0              ; best_val = r0
    xor         r15, r15                       ; best_idx = 0

    ; Save max growth vectors to stack for scalar access
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm9

    ; --------------------------------------------------------
    ; SCALAR LOOP — Compare all 8 lanes
    ; --------------------------------------------------------
    xor         rcx, rcx

.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done

    ; Check Block A lane
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .check_B

    vmovsd      xmm6, xmm1, xmm1              ; best_val = this lane
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2                     ; best_idx = index

.check_B:
    ; Check Block B lane
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .next_lane

    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si  r15, xmm2

.next_lane:
    inc         rcx
    jmp         .reduce_loop

.reduce_done:

    ; --------------------------------------------------------
    ; WRITE OUTPUTS
    ; --------------------------------------------------------
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6                   ; max growth value

    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15                    ; MAP run length

    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx                    ; truncation point

; ============================================================================
; EPILOGUE — Restore registers, cleanup
; ============================================================================

    mov         rsp, rbp                      ; restore stack pointer

    ; Restore XMM6-XMM15
    vmovdqu     xmm6,  [rsp +   0]
    vmovdqu     xmm7,  [rsp +  16]
    vmovdqu     xmm8,  [rsp +  32]
    vmovdqu     xmm9,  [rsp +  48]
    vmovdqu     xmm10, [rsp +  64]
    vmovdqu     xmm11, [rsp +  80]
    vmovdqu     xmm12, [rsp +  96]
    vmovdqu     xmm13, [rsp + 112]
    vmovdqu     xmm14, [rsp + 128]
    vmovdqu     xmm15, [rsp + 144]
    add         rsp, 160

    ; Restore GPRs
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rsi
    pop         rdi
    pop         rbx
    pop         rbp

    vzeroupper
    ret

; ============================================================================
; END OF KERNEL
; ============================================================================