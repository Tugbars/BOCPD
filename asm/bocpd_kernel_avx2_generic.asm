;==============================================================================
; BOCPD Fused Prediction Kernel — Generic AVX2 V3.2
; Dual ABI: Windows x64 and Linux System V
;==============================================================================
;
; =============================================================================
; VERSION HISTORY
; =============================================================================
;
; V3.2 (this version):
;   - Eliminated ALL redundant memory loads within the iteration
;   - No reload of C1, C2 after log1p (kept in registers)
;   - No reload of r_old after exp (kept in ymm8/ymm9)
;   - Moved max_growth_B to stack (cold path) to free register for r_old_B
;   - Added Linux System V ABI entry point
;
; V3.1:
;   - Had 6 extra loads per iteration (3 per block)
;   - C1, C2 reloaded after log1p clobbered them
;   - r_old reloaded after Estrin polynomial
;
; =============================================================================
; OPTIMIZATION STRATEGY
; =============================================================================
;
; The key insight is that with careful register allocation, we can keep ALL
; hot values alive across the computation chain:
;
; 1. Load all parameters for BOTH blocks upfront (μ, C1, C2, inv_ssn, r_old)
; 2. Compute Student-t for both blocks (consumes μ, inv_ssn)
; 3. Compute log1p for both blocks (uses scratch registers, preserves C1, C2)
; 4. Compute ln_pp = C1 - C2*log1p (consumes C1, C2)
; 5. Compute exp for both blocks
; 6. BOCPD update (uses r_old from ymm8/ymm9, never reloaded!)
;
; Register allocation is the critical optimization. We dedicate:
;   - ymm8 = r_old_A (persists through entire block computation)
;   - ymm9 = r_old_B (persists through entire block computation)
;   - ymm10 = max_growth_A (register, hot for comparison)
;   - [stack] = max_growth_B (cold, rarely updated)
;
; =============================================================================
; MEMORY LAYOUT: 256-byte Superblocks
; =============================================================================
;
; Each superblock holds parameters for 4 consecutive run lengths:
;
;   Offset   Size   Contents              Used By
;   ──────────────────────────────────────────────────
;   0-31      32    μ[0..3]     (means)   Prediction
;   32-63     32    C1[0..3]    (const)   Prediction
;   64-95     32    C2[0..3]    (exp)     Prediction
;   96-127    32    inv_ssn[0..3]         Prediction
;   128-255  128    κ,α,β,ss_n            Update (not used here)
;
; Block addressing:
;   block_index = i / 4
;   byte_offset = block_index * 256
;
; =============================================================================
; DUAL ABI SUPPORT
; =============================================================================
;
; This file provides TWO entry points:
;
;   bocpd_fused_loop_avx2       — Windows x64 ABI (RCX = args ptr)
;   bocpd_fused_loop_avx2_sysv  — Linux System V ABI (RDI = args ptr)
;
; Windows x64 ABI:
;   - First arg in RCX
;   - Must preserve RBX, RBP, RDI, RSI, R12-R15
;   - Must preserve XMM6-XMM15 (!)
;
; Linux System V ABI:
;   - First arg in RDI
;   - Must preserve RBX, RBP, R12-R15
;   - XMM/YMM registers are ALL caller-saved (no preservation needed)
;
; The Linux version is ~20 cycles faster per call due to simpler prologue/epilogue.
;
; =============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

;==============================================================================
; CONSTANTS (read-only data section)
;==============================================================================

section .rodata
align 32

;------------------------------------------------------------------------------
; General Constants
;------------------------------------------------------------------------------
const_one:      dq 1.0, 1.0, 1.0, 1.0
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

;------------------------------------------------------------------------------
; log1p Polynomial Coefficients
;------------------------------------------------------------------------------
;
; log1p(t) = ln(1+t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
; Taylor series: c2=-1/2, c3=1/3, c4=-1/4, c5=1/5, c6=-1/6
;------------------------------------------------------------------------------
log1p_c2:       dq -0.5, -0.5, -0.5, -0.5
log1p_c3:       dq  0.3333333333333333,  0.3333333333333333,  0.3333333333333333,  0.3333333333333333
log1p_c4:       dq -0.25, -0.25, -0.25, -0.25
log1p_c5:       dq  0.2,  0.2,  0.2,  0.2
log1p_c6:       dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

;------------------------------------------------------------------------------
; exp Polynomial Coefficients (Estrin scheme for 2^f)
;------------------------------------------------------------------------------
;
; exp(x) = 2^(x/ln2) = 2^k × 2^f where k = round(x/ln2), f ∈ [-0.5, 0.5]
; Coefficients from Taylor series of 2^x = e^(x×ln2)
;------------------------------------------------------------------------------
exp_c1:         dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
exp_c2:         dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
exp_c3:         dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
exp_c4:         dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:         dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:         dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

;------------------------------------------------------------------------------
; exp Helper Constants
;------------------------------------------------------------------------------
exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
exp_bias:       dq 1023, 1023, 1023, 1023

;------------------------------------------------------------------------------
; Index Tracking Constants
;------------------------------------------------------------------------------
idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

;==============================================================================
; Structure Offsets (bocpd_kernel_args_t)
;==============================================================================

%define ARG_LIN_INTERLEAVED  0
%define ARG_R_OLD            8
%define ARG_X               16
%define ARG_H               24
%define ARG_OMH             32
%define ARG_THRESH          40
%define ARG_N_PADDED        48
%define ARG_R_NEW           56
%define ARG_R0_OUT          64
%define ARG_MAX_GROWTH      72
%define ARG_MAX_IDX         80
%define ARG_LAST_VALID      88

;==============================================================================
; Stack Frame Layout
;==============================================================================
;
;   Offset      Size    Contents
;   ──────────────────────────────────────────────────────
;   [rsp + 0]     32    idx_vec_A
;   [rsp + 32]    32    idx_vec_B
;   [rsp + 64]    32    max_idx_A
;   [rsp + 96]    32    max_idx_B
;   [rsp + 128]   32    max_growth_A (for final reduction)
;   [rsp + 160]   32    max_growth_B (V3.2: cold path accumulator)
;   [rsp + 192]   64    padding
;   ──────────────────────────────────────────────────────
;==============================================================================

%define STK_IDX_VEC_A       0
%define STK_IDX_VEC_B       32
%define STK_MAX_IDX_A       64
%define STK_MAX_IDX_B       96
%define STK_MAX_GROWTH_A    128
%define STK_MAX_GROWTH_B    160

%define STACK_FRAME         256

;==============================================================================
section .text

global bocpd_fused_loop_avx2
global bocpd_fused_loop_avx2_sysv

;==============================================================================
; WINDOWS x64 ABI ENTRY POINT
;==============================================================================

bocpd_fused_loop_avx2:

    ;==========================================================================
    ; PROLOGUE — Windows x64 ABI
    ;==========================================================================
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

    ; Windows: first arg in RCX, move to RDI
    mov         rdi, rcx

    ;==========================================================================
    ; STACK FRAME SETUP
    ;==========================================================================
    mov         rbp, rsp
    sub         rsp, STACK_FRAME + 32
    and         rsp, -32

    ;==========================================================================
    ; LOAD ARGUMENTS
    ;==========================================================================
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]

    ;==========================================================================
    ; BROADCAST SCALARS
    ;
    ; Register allocation:
    ;   ymm15 = x (observation)
    ;   ymm14 = h (hazard)
    ;   ymm13 = 1-h
    ;   ymm12 = threshold
    ;   ymm11 = r0 accumulator
    ;   ymm10 = max_growth_A
    ;   ymm9  = r_old_B (V3.2: dedicated)
    ;   ymm8  = r_old_A (dedicated)
    ;   ymm0-7 = scratch
    ;==========================================================================
    vbroadcastsd ymm15, qword [rdi + ARG_X]
    vbroadcastsd ymm14, qword [rdi + ARG_H]
    vbroadcastsd ymm13, qword [rdi + ARG_OMH]
    vbroadcastsd ymm12, qword [rdi + ARG_THRESH]

    ;==========================================================================
    ; INITIALIZE ACCUMULATORS
    ;==========================================================================
    vxorpd      ymm11, ymm11, ymm11
    vxorpd      ymm10, ymm10, ymm10
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0

    ;==========================================================================
    ; INITIALIZE INDEX VECTORS
    ;==========================================================================
    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    xor         rsi, rsi
    xor         rbx, rbx

    ;==========================================================================
    ; MAIN LOOP
    ;==========================================================================
.win_loop_start:
    cmp         rsi, r14
    jge         .win_loop_end

    ;--------------------------------------------------------------------------
    ; Calculate superblock offsets
    ;--------------------------------------------------------------------------
    mov         rax, rsi
    shr         rax, 2
    shl         rax, 8

    mov         rdx, rsi
    add         rdx, 4
    shr         rdx, 2
    shl         rdx, 8

    ;--------------------------------------------------------------------------
    ; Load ALL parameters for both blocks (V3.2: no reloads needed later)
    ;
    ; Block A: ymm0=μ, ymm1=C1, ymm2=C2, ymm3=inv_ssn, ymm8=r_old
    ; Block B: ymm4=μ, ymm5=C1, ymm6=C2, ymm7=inv_ssn, ymm9=r_old
    ;--------------------------------------------------------------------------
    vmovapd     ymm0, [r8 + rax + 0]
    vmovapd     ymm1, [r8 + rax + 32]
    vmovapd     ymm2, [r8 + rax + 64]
    vmovapd     ymm3, [r8 + rax + 96]
    vmovapd     ymm8, [r12 + rsi*8]

    vmovapd     ymm4, [r8 + rdx + 0]
    vmovapd     ymm5, [r8 + rdx + 32]
    vmovapd     ymm6, [r8 + rdx + 64]
    vmovapd     ymm7, [r8 + rdx + 96]
    vmovapd     ymm9, [r12 + rsi*8 + 32]

    ;--------------------------------------------------------------------------
    ; Student-t: t = (x - μ)² × inv_ssn
    ; After: ymm0=t_A, ymm4=t_B, C1/C2 preserved in ymm1/2/5/6
    ;--------------------------------------------------------------------------
    vsubpd      ymm0, ymm15, ymm0
    vmulpd      ymm0, ymm0, ymm0
    vmulpd      ymm0, ymm0, ymm3

    vsubpd      ymm4, ymm15, ymm4
    vmulpd      ymm4, ymm4, ymm4
    vmulpd      ymm4, ymm4, ymm7

    ;--------------------------------------------------------------------------
    ; log1p via Horner (uses ymm3, ymm7 as accumulators)
    ; Preserves C1/C2 in ymm1/2/5/6
    ;--------------------------------------------------------------------------
    vmovapd     ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd      ymm3, ymm0, ymm3

    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm4, ymm7

    ;--------------------------------------------------------------------------
    ; ln_pp = C1 - C2 × log1p (consumes C1/C2)
    ;--------------------------------------------------------------------------
    vfnmadd231pd ymm1, ymm2, ymm3
    vfnmadd231pd ymm5, ymm6, ymm7

    ;--------------------------------------------------------------------------
    ; exp preparation
    ;--------------------------------------------------------------------------
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]
    vroundpd    ymm2, ymm0, 0
    vsubpd      ymm0, ymm0, ymm2

    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]
    vroundpd    ymm6, ymm4, 0
    vsubpd      ymm4, ymm4, ymm6

    ;--------------------------------------------------------------------------
    ; Estrin 2^f Block A
    ;--------------------------------------------------------------------------
    vmulpd      ymm3, ymm0, ymm0
    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]
    vfmadd231pd ymm1, ymm3, ymm7
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd      ymm3, ymm3, ymm3
    vfmadd231pd ymm1, ymm3, ymm7

    ;--------------------------------------------------------------------------
    ; Estrin 2^f Block B
    ;--------------------------------------------------------------------------
    vmulpd      ymm3, ymm4, ymm4
    vmovapd     ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]
    vfmadd231pd ymm5, ymm3, ymm7
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd      ymm3, ymm3, ymm3
    vfmadd231pd ymm5, ymm3, ymm7

    ;--------------------------------------------------------------------------
    ; 2^k via IEEE-754 bit manipulation
    ;--------------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ;--------------------------------------------------------------------------
    ; BOCPD update Block A (r_old_A still in ymm8!)
    ;--------------------------------------------------------------------------
    vmulpd      ymm0, ymm8, ymm1
    vmulpd      ymm2, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ;--------------------------------------------------------------------------
    ; BOCPD update Block B (r_old_B still in ymm9!)
    ;--------------------------------------------------------------------------
    vmulpd      ymm0, ymm9, ymm5
    vmulpd      ymm3, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 40], ymm3

    ;--------------------------------------------------------------------------
    ; Max tracking Block A
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm2, ymm10, 14
    vblendvpd   ymm10, ymm10, ymm2, ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

    ;--------------------------------------------------------------------------
    ; Max tracking Block B (V3.2: on stack)
    ;--------------------------------------------------------------------------
    vmovapd     ymm0, [rsp + STK_MAX_GROWTH_B]
    vcmppd      ymm7, ymm3, ymm0, 14
    vblendvpd   ymm0, ymm0, ymm3, ymm7
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm7
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

    ;--------------------------------------------------------------------------
    ; Truncation tracking
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm2, ymm12, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .win_no_trunc_A
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 1]
.win_no_trunc_A:

    vcmppd      ymm0, ymm3, ymm12, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .win_no_trunc_B
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]
.win_no_trunc_B:

    ;--------------------------------------------------------------------------
    ; Update index vectors
    ;--------------------------------------------------------------------------
    vmovapd     ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    add         rsi, 8
    jmp         .win_loop_start

    ;==========================================================================
    ; LOOP END — Reductions
    ;==========================================================================
.win_loop_end:

    ; Reduce r0
    vextractf128 xmm0, ymm11, 1
    vaddpd      xmm0, xmm0, xmm11
    vunpckhpd   xmm1, xmm0, xmm0
    vaddsd      xmm0, xmm0, xmm1

    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0

    ; MAP reduction
    vmovsd      xmm5, xmm0, xmm0
    xor         r15, r15
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10

    xor         rcx, rcx
.win_reduce_loop:
    cmp         rcx, 4
    jge         .win_reduce_done

    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .win_check_B
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2

.win_check_B:
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .win_next_lane
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si  r15, xmm2

.win_next_lane:
    inc         rcx
    jmp         .win_reduce_loop

.win_reduce_done:
    ; Write outputs
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm5
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx

    ;==========================================================================
    ; EPILOGUE — Windows
    ;==========================================================================
    mov         rsp, rbp

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

;==============================================================================
; LINUX SYSTEM V ABI ENTRY POINT
;==============================================================================

bocpd_fused_loop_avx2_sysv:

    ;==========================================================================
    ; PROLOGUE — Linux System V (no XMM saving needed!)
    ;==========================================================================
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    ; RDI already has args pointer

    ;==========================================================================
    ; STACK FRAME SETUP
    ;==========================================================================
    mov         rbp, rsp
    sub         rsp, STACK_FRAME + 32
    and         rsp, -32

    ;==========================================================================
    ; LOAD ARGUMENTS
    ;==========================================================================
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]

    ;==========================================================================
    ; BROADCAST SCALARS
    ;==========================================================================
    vbroadcastsd ymm15, qword [rdi + ARG_X]
    vbroadcastsd ymm14, qword [rdi + ARG_H]
    vbroadcastsd ymm13, qword [rdi + ARG_OMH]
    vbroadcastsd ymm12, qword [rdi + ARG_THRESH]

    ;==========================================================================
    ; INITIALIZE ACCUMULATORS
    ;==========================================================================
    vxorpd      ymm11, ymm11, ymm11
    vxorpd      ymm10, ymm10, ymm10
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0

    ;==========================================================================
    ; INITIALIZE INDEX VECTORS
    ;==========================================================================
    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    xor         rsi, rsi
    xor         rbx, rbx

    ;==========================================================================
    ; MAIN LOOP
    ;==========================================================================
.sysv_loop_start:
    cmp         rsi, r14
    jge         .sysv_loop_end

    ; Calculate superblock offsets
    mov         rax, rsi
    shr         rax, 2
    shl         rax, 8

    mov         rdx, rsi
    add         rdx, 4
    shr         rdx, 2
    shl         rdx, 8

    ; Load ALL parameters (V3.2: no reloads needed)
    vmovapd     ymm0, [r8 + rax + 0]
    vmovapd     ymm1, [r8 + rax + 32]
    vmovapd     ymm2, [r8 + rax + 64]
    vmovapd     ymm3, [r8 + rax + 96]
    vmovapd     ymm8, [r12 + rsi*8]

    vmovapd     ymm4, [r8 + rdx + 0]
    vmovapd     ymm5, [r8 + rdx + 32]
    vmovapd     ymm6, [r8 + rdx + 64]
    vmovapd     ymm7, [r8 + rdx + 96]
    vmovapd     ymm9, [r12 + rsi*8 + 32]

    ; Student-t
    vsubpd      ymm0, ymm15, ymm0
    vmulpd      ymm0, ymm0, ymm0
    vmulpd      ymm0, ymm0, ymm3

    vsubpd      ymm4, ymm15, ymm4
    vmulpd      ymm4, ymm4, ymm4
    vmulpd      ymm4, ymm4, ymm7

    ; log1p A
    vmovapd     ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd      ymm3, ymm0, ymm3

    ; log1p B
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm4, ymm7

    ; ln_pp
    vfnmadd231pd ymm1, ymm2, ymm3
    vfnmadd231pd ymm5, ymm6, ymm7

    ; exp prep A
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]
    vroundpd    ymm2, ymm0, 0
    vsubpd      ymm0, ymm0, ymm2

    ; exp prep B
    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]
    vroundpd    ymm6, ymm4, 0
    vsubpd      ymm4, ymm4, ymm6

    ; Estrin A
    vmulpd      ymm3, ymm0, ymm0
    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]
    vfmadd231pd ymm1, ymm3, ymm7
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd      ymm3, ymm3, ymm3
    vfmadd231pd ymm1, ymm3, ymm7

    ; Estrin B
    vmulpd      ymm3, ymm4, ymm4
    vmovapd     ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]
    vfmadd231pd ymm5, ymm3, ymm7
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd      ymm3, ymm3, ymm3
    vfmadd231pd ymm5, ymm3, ymm7

    ; 2^k A
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    ; 2^k B
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; BOCPD A (r_old_A still in ymm8)
    vmulpd      ymm0, ymm8, ymm1
    vmulpd      ymm2, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ; BOCPD B (r_old_B still in ymm9)
    vmulpd      ymm0, ymm9, ymm5
    vmulpd      ymm3, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 40], ymm3

    ; Max tracking A
    vcmppd      ymm0, ymm2, ymm10, 14
    vblendvpd   ymm10, ymm10, ymm2, ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

    ; Max tracking B (V3.2: on stack)
    vmovapd     ymm0, [rsp + STK_MAX_GROWTH_B]
    vcmppd      ymm7, ymm3, ymm0, 14
    vblendvpd   ymm0, ymm0, ymm3, ymm7
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm7
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

    ; Truncation A
    vcmppd      ymm0, ymm2, ymm12, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .sysv_no_trunc_A
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 1]
.sysv_no_trunc_A:

    ; Truncation B
    vcmppd      ymm0, ymm3, ymm12, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .sysv_no_trunc_B
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]
.sysv_no_trunc_B:

    ; Update index vectors
    vmovapd     ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    add         rsi, 8
    jmp         .sysv_loop_start

    ;==========================================================================
    ; LOOP END — Reductions
    ;==========================================================================
.sysv_loop_end:

    ; Reduce r0
    vextractf128 xmm0, ymm11, 1
    vaddpd      xmm0, xmm0, xmm11
    vunpckhpd   xmm1, xmm0, xmm0
    vaddsd      xmm0, xmm0, xmm1

    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0

    ; MAP reduction
    vmovsd      xmm5, xmm0, xmm0
    xor         r15, r15
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10

    xor         rcx, rcx
.sysv_reduce_loop:
    cmp         rcx, 4
    jge         .sysv_reduce_done

    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .sysv_check_B
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2

.sysv_check_B:
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .sysv_next_lane
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si  r15, xmm2

.sysv_next_lane:
    inc         rcx
    jmp         .sysv_reduce_loop

.sysv_reduce_done:
    ; Write outputs
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm5
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx

    ;==========================================================================
    ; EPILOGUE — Linux (no XMM restore needed!)
    ;==========================================================================
    mov         rsp, rbp

    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp

    vzeroupper
    ret

;==============================================================================
; END OF KERNEL
;==============================================================================