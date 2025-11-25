; =============================================================================
;   BOCPD Ultra — AVX2 Kernel V3 (Corrected, No AVX-512)
; =============================================================================
;
; This version:
;   • AVX2 only (ymm0-15) - works on all modern x86-64
;   • Proper A/B interleaving for ILP
;   • Constants loaded from memory (L1 cache is fast)
;   • Correct register allocation with no clobbering
;   • Stack properly allocated for spills
;
; Register allocation (16 YMM registers):
;   ymm15 = x (broadcast, preserved)
;   ymm14 = h (broadcast, preserved)  
;   ymm13 = 1-h (broadcast, preserved)
;   ymm12 = threshold (broadcast, preserved)
;   ymm11 = r0 accumulator
;   ymm10 = max_growth_A
;   ymm9  = max_growth_B
;   ymm0-8 = scratch (reused each iteration)
;
; Stack layout (after alignment):
;   [rsp + 0]   = idx_vec_A (32 bytes)
;   [rsp + 32]  = idx_vec_B (32 bytes)
;   [rsp + 64]  = max_idx_A (32 bytes)
;   [rsp + 96]  = max_idx_B (32 bytes)
;   [rsp + 128] = scratch (32 bytes)
;
; =============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

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

%define ARG_LIN_INTERLEAVED 0
%define ARG_R_OLD           8
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

%define STACK_SIZE          192

section .text
global bocpd_fused_loop_avx2_intel

bocpd_fused_loop_avx2_intel:
    ; =================================================================
    ; PROLOGUE
    ; =================================================================
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    
    mov         rbp, rsp
    sub         rsp, STACK_SIZE + 32
    and         rsp, -32                    ; 32-byte align
    
    ; =================================================================
    ; Load pointers into callee-saved registers
    ; =================================================================
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]
    
    ; =================================================================
    ; Broadcast scalars - THESE STAY IN REGISTERS FOREVER
    ; =================================================================
    vbroadcastsd ymm15, qword [rdi + ARG_X]
    vbroadcastsd ymm14, qword [rdi + ARG_H]
    vbroadcastsd ymm13, qword [rdi + ARG_OMH]
    vbroadcastsd ymm12, qword [rdi + ARG_THRESH]
    
    ; =================================================================
    ; Initialize accumulators
    ; =================================================================
    vxorpd      ymm11, ymm11, ymm11         ; r0 accumulator
    vxorpd      ymm10, ymm10, ymm10         ; max_growth_A
    vxorpd      ymm9,  ymm9,  ymm9          ; max_growth_B
    
    ; Initialize index vectors on stack
    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp], ymm0                 ; idx_vec_A
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + 32], ymm0            ; idx_vec_B
    
    ; Initialize max_idx accumulators
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + 64], ymm0            ; max_idx_A
    vmovapd     [rsp + 96], ymm0            ; max_idx_B
    
    xor         rbx, rbx                    ; last_valid = 0
    xor         rsi, rsi                    ; i = 0

; =============================================================================
; MAIN LOOP - Process 8 elements per iteration
; =============================================================================
.loop:
    cmp         rsi, r14
    jge         .loop_end
    
    ; Compute block offsets
    mov         rax, rsi
    shl         rax, 5                      ; rax = i * 32 (Block A offset)
    lea         rdx, [rax + 128]            ; rdx = Block B offset
    
    ; =========================================================================
    ; INTERLEAVED LOADS - A and B together for memory parallelism
    ; =========================================================================
    vmovapd     ymm0, [r8 + rax]            ; mu_A
    vmovapd     ymm4, [r8 + rdx]            ; mu_B
    vmovapd     ymm1, [r8 + rax + 32]       ; C1_A
    vmovapd     ymm5, [r8 + rdx + 32]       ; C1_B
    vmovapd     ymm2, [r8 + rax + 64]       ; C2_A
    vmovapd     ymm6, [r8 + rdx + 64]       ; C2_B
    vmovapd     ymm3, [r8 + rax + 96]       ; inv_ssn_A
    vmovapd     ymm7, [r8 + rdx + 96]       ; inv_ssn_B
    
    ; Load r_old
    vmovapd     ymm8, [r12 + rsi*8]         ; r_old_A
    vmovapd     [rsp + 128], ymm8           ; spill r_old_A
    vmovapd     ymm8, [r12 + rsi*8 + 32]    ; r_old_B
    ; r_old_B stays in ymm8
    
    ; =========================================================================
    ; BLOCK A: z² = (x - μ)², t = z² * inv_ssn
    ; =========================================================================
    vsubpd      ymm0, ymm15, ymm0           ; z_A = x - mu_A
    vmulpd      ymm0, ymm0, ymm0            ; z²_A
    vmulpd      ymm0, ymm0, ymm3            ; t_A = z²_A * inv_ssn_A
    
    ; =========================================================================
    ; BLOCK B: z² = (x - μ)², t = z² * inv_ssn  (interleaved)
    ; =========================================================================
    vsubpd      ymm4, ymm15, ymm4           ; z_B = x - mu_B
    vmulpd      ymm4, ymm4, ymm4            ; z²_B
    vmulpd      ymm4, ymm4, ymm7            ; t_B = z²_B * inv_ssn_B
    
    ; =========================================================================
    ; BLOCK A: log1p(t) via Horner - ymm0 = t_A, result in ymm3
    ; =========================================================================
    vmovapd     ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd      ymm3, ymm3, ymm0            ; log1p_A in ymm3
    
    ; =========================================================================
    ; BLOCK B: log1p(t) via Horner - ymm4 = t_B, result in ymm7
    ; =========================================================================
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm7, ymm4            ; log1p_B in ymm7
    
    ; =========================================================================
    ; ln_pp = C1 - C2 * log1p(t)
    ; ymm1 = C1_A, ymm2 = C2_A, ymm3 = log1p_A
    ; ymm5 = C1_B, ymm6 = C2_B, ymm7 = log1p_B
    ; =========================================================================
    vfnmadd231pd ymm1, ymm2, ymm3           ; ln_pp_A = C1_A - C2_A * log1p_A
    vfnmadd231pd ymm5, ymm6, ymm7           ; ln_pp_B = C1_B - C2_B * log1p_B
    
    ; =========================================================================
    ; EXP BLOCK A: clamp, convert to base-2
    ; =========================================================================
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]   ; t = ln_pp * log2(e)
    vroundpd    ymm2, ymm0, 0                   ; k_A = round(t)
    vsubpd      ymm0, ymm0, ymm2                ; f_A = t - k_A
    
    ; =========================================================================
    ; EXP BLOCK B: clamp, convert to base-2
    ; =========================================================================
    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]   ; t = ln_pp * log2(e)
    vroundpd    ymm6, ymm4, 0                   ; k_B = round(t)
    vsubpd      ymm4, ymm4, ymm6                ; f_B = t - k_B
    
    ; =========================================================================
    ; ESTRIN BLOCK A: 2^f_A, f_A in ymm0, result in ymm1
    ; =========================================================================
    vmulpd      ymm3, ymm0, ymm0                ; f²_A
    
    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]        ; p01 = 1 + f*c1
    
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]        ; p23 = c2 + f*c3
    
    vfmadd231pd ymm1, ymm3, ymm7                ; q0123 = p01 + f²*p23
    
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]        ; p45 = c4 + f*c5
    vfmadd231pd ymm7, ymm3, [rel exp_c6]        ; q456 = p45 + f²*c6
    
    vmulpd      ymm3, ymm3, ymm3                ; f⁴_A
    vfmadd231pd ymm1, ymm3, ymm7                ; 2^f_A = q0123 + f⁴*q456
    
    ; =========================================================================
    ; ESTRIN BLOCK B: 2^f_B, f_B in ymm4, result in ymm5
    ; =========================================================================
    vmulpd      ymm3, ymm4, ymm4                ; f²_B
    
    vmovapd     ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]        ; p01
    
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]        ; p23
    
    vfmadd231pd ymm5, ymm3, ymm7                ; q0123
    
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]        ; p45
    vfmadd231pd ymm7, ymm3, [rel exp_c6]        ; q456
    
    vmulpd      ymm3, ymm3, ymm3                ; f⁴_B
    vfmadd231pd ymm5, ymm3, ymm7                ; 2^f_B
    
    ; =========================================================================
    ; 2^k RECONSTRUCTION BLOCK A: k_A in ymm2
    ; =========================================================================
    vcvtpd2dq   xmm0, ymm2                      ; k → int32
    vpmovsxdq   ymm0, xmm0                      ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel bias_1023]     ; add exponent bias
    vpsllq      ymm0, ymm0, 52                  ; shift to exponent field
    vmulpd      ymm1, ymm1, ymm0                ; pp_A = 2^f * 2^k
    vmaxpd      ymm1, ymm1, [rel const_min_pp]  ; clamp
    
    ; =========================================================================
    ; 2^k RECONSTRUCTION BLOCK B: k_B in ymm6
    ; =========================================================================
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0                ; pp_B = 2^f * 2^k
    vmaxpd      ymm5, ymm5, [rel const_min_pp]
    
    ; =========================================================================
    ; BOCPD UPDATE BLOCK A
    ; pp_A in ymm1, r_old_A spilled at [rsp+128]
    ; =========================================================================
    vmovapd     ymm0, [rsp + 128]               ; reload r_old_A
    vmulpd      ymm0, ymm0, ymm1                ; r_pp_A = r_old_A * pp_A
    vmulpd      ymm2, ymm0, ymm13               ; growth_A = r_pp_A * (1-h)
    vmulpd      ymm0, ymm0, ymm14               ; change_A = r_pp_A * h
    
    vmovupd     [r13 + rsi*8 + 8], ymm2         ; store growth_A at r_new[i+1]
    vaddpd      ymm11, ymm11, ymm0              ; r0_accum += change_A
    
    ; =========================================================================
    ; BOCPD UPDATE BLOCK B
    ; pp_B in ymm5, r_old_B in ymm8
    ; =========================================================================
    vmulpd      ymm8, ymm8, ymm5                ; r_pp_B = r_old_B * pp_B
    vmulpd      ymm3, ymm8, ymm13               ; growth_B = r_pp_B * (1-h)
    vmulpd      ymm8, ymm8, ymm14               ; change_B = r_pp_B * h
    
    vmovupd     [r13 + rsi*8 + 40], ymm3        ; store growth_B at r_new[i+5]
    vaddpd      ymm11, ymm11, ymm8              ; r0_accum += change_B
    
    ; =========================================================================
    ; MAX TRACKING BLOCK A: growth_A in ymm2
    ; =========================================================================
    vcmppd      ymm0, ymm2, ymm10, 14           ; mask: growth_A > max_growth_A?
    vblendvpd   ymm10, ymm10, ymm2, ymm0        ; update max_growth_A
    vmovapd     ymm1, [rsp + 64]                ; load max_idx_A
    vmovapd     ymm4, [rsp]                     ; load idx_vec_A
    vblendvpd   ymm1, ymm1, ymm4, ymm0          ; update max_idx_A
    vmovapd     [rsp + 64], ymm1                ; store max_idx_A
    
    ; =========================================================================
    ; MAX TRACKING BLOCK B: growth_B in ymm3
    ; =========================================================================
    vcmppd      ymm0, ymm3, ymm9, 14            ; mask: growth_B > max_growth_B?
    vblendvpd   ymm9, ymm9, ymm3, ymm0          ; update max_growth_B
    vmovapd     ymm1, [rsp + 96]                ; load max_idx_B
    vmovapd     ymm4, [rsp + 32]                ; load idx_vec_B
    vblendvpd   ymm1, ymm1, ymm4, ymm0          ; update max_idx_B
    vmovapd     [rsp + 96], ymm1                ; store max_idx_B
    
    ; =========================================================================
    ; TRUNCATION BLOCK A: growth_A in ymm2, threshold in ymm12
    ; =========================================================================
    vcmppd      ymm0, ymm2, ymm12, 14           ; mask: growth_A > threshold?
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .skip_trunc_a
    
    bt          eax, 3
    jc          .lv_a4
    bt          eax, 2
    jc          .lv_a3
    bt          eax, 1
    jc          .lv_a2
    lea         rbx, [rsi + 1]
    jmp         .skip_trunc_a
.lv_a4:
    lea         rbx, [rsi + 4]
    jmp         .skip_trunc_a
.lv_a3:
    lea         rbx, [rsi + 3]
    jmp         .skip_trunc_a
.lv_a2:
    lea         rbx, [rsi + 2]
.skip_trunc_a:
    
    ; =========================================================================
    ; TRUNCATION BLOCK B: growth_B in ymm3, threshold in ymm12
    ; =========================================================================
    vcmppd      ymm0, ymm3, ymm12, 14           ; mask: growth_B > threshold?
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .skip_trunc_b
    
    bt          eax, 3
    jc          .lv_b8
    bt          eax, 2
    jc          .lv_b7
    bt          eax, 1
    jc          .lv_b6
    lea         rbx, [rsi + 5]
    jmp         .skip_trunc_b
.lv_b8:
    lea         rbx, [rsi + 8]
    jmp         .skip_trunc_b
.lv_b7:
    lea         rbx, [rsi + 7]
    jmp         .skip_trunc_b
.lv_b6:
    lea         rbx, [rsi + 6]
.skip_trunc_b:
    
    ; =========================================================================
    ; UPDATE INDEX VECTORS
    ; =========================================================================
    vmovapd     ymm0, [rsp]                     ; idx_vec_A
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp], ymm0
    
    vmovapd     ymm0, [rsp + 32]                ; idx_vec_B
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + 32], ymm0
    
    ; =========================================================================
    ; LOOP INCREMENT - THIS WAS MISSING IN YOUR VERSION!
    ; =========================================================================
    add         rsi, 8
    jmp         .loop

; =============================================================================
; LOOP END - REDUCTIONS
; =============================================================================
.loop_end:
    
    ; Reduce r0 accumulator
    vextractf128 xmm0, ymm11, 1
    vaddpd      xmm0, xmm0, xmm11
    vhaddpd     xmm0, xmm0, xmm0
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0
    
    ; Use r0 as initial "best" for MAP
    vmovsd      xmm6, xmm0, xmm0
    xor         r15, r15
    
    ; Store max vectors for scalar reduction
    vmovapd     [rsp + 128], ymm10              ; max_growth_A
    vmovapd     [rsp + 160], ymm9               ; max_growth_B (need extra space)
    
    xor         rcx, rcx
.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done
    
    ; Check max_growth_A[j]
    vmovsd      xmm1, [rsp + 128 + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .check_b
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + 64 + rcx*8]        ; max_idx_A[j]
    vcvttsd2si  r15, xmm2
    
.check_b:
    vmovsd      xmm1, [rsp + 160 + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .next_j
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + 96 + rcx*8]        ; max_idx_B[j]
    vcvttsd2si  r15, xmm2
    
.next_j:
    inc         rcx
    jmp         .reduce_loop
    
.reduce_done:
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx
    
    ; =================================================================
    ; EPILOGUE
    ; =================================================================
    mov         rsp, rbp
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    
    vzeroupper
    ret
