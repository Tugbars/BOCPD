;==============================================================================
; BOCPD Fused Prediction Kernel (V3.1, AVX2, Windows x64 ABI)
;
; 256-byte superblock layout:
;   0-31     MU        (prediction)
;   32-63    C1        (prediction)
;   64-95    C2        (prediction)
;   96-127   INV_SSN   (prediction)
;   128-255  unused by kernel (KAPPA, ALPHA, BETA, SS_N)
;
; Block addressing:
;   block_index = i / 4
;   byte_offset = block_index * 256
;
; Processes 8 elements per iteration (Block A + Block B)
;==============================================================================

section .rodata
align 32

; Shared constants
const_one:      dq 1.0, 1.0, 1.0, 1.0

; log1p polynomial: log(1+t) ≈ t*(1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
LOG1P_C2:       dq -0.5, -0.5, -0.5, -0.5
LOG1P_C3:       dq  0.3333333333333333,  0.3333333333333333,  0.3333333333333333,  0.3333333333333333
LOG1P_C4:       dq -0.25, -0.25, -0.25, -0.25
LOG1P_C5:       dq  0.2,  0.2,  0.2,  0.2
LOG1P_C6:       dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

; exp polynomial coefficients
EXP_C1:         dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
EXP_C2:         dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
EXP_C3:         dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
EXP_C4:         dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
EXP_C5:         dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
EXP_C6:         dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

; exp helpers
EXP_INV_LN2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
EXP_MIN_X:      dq -700.0, -700.0, -700.0, -700.0
EXP_MAX_X:      dq 700.0, 700.0, 700.0, 700.0
EXP_BIAS:       dq 1023, 1023, 1023, 1023
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

; Index vectors
idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

; Structure offsets (bocpd_kernel_args_t)
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

; Stack frame layout
%define STK_IDX_VEC_B        0
%define STK_MAX_IDX_A       32
%define STK_MAX_IDX_B       64
%define STK_MAX_GROWTH_A    96
%define STK_MAX_GROWTH_B   128
%define STACK_SIZE         192

;==============================================================================
section .text
global bocpd_fused_loop_avx2_generic

bocpd_fused_loop_avx2_generic:
    ;==========================================================================
    ; PROLOGUE - Windows x64 ABI
    ;==========================================================================
    push        rbp
    push        rbx
    push        rdi
    push        rsi
    push        r12
    push        r13
    push        r14
    push        r15
    
    ; Save XMM6-15 (Windows requires preservation)
    sub         rsp, 168
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
    
    ; Windows x64: first arg in RCX
    mov         rdi, rcx
    
    ; Align stack for AVX
    mov         rbp, rsp
    sub         rsp, STACK_SIZE + 32
    and         rsp, -32
    
    ;==========================================================================
    ; LOAD ARGUMENTS
    ;==========================================================================
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]    ; Interleaved buffer base
    mov         r12, [rdi + ARG_R_OLD]              ; Input distribution
    mov         r13, [rdi + ARG_R_NEW]              ; Output distribution
    mov         r14, [rdi + ARG_N_PADDED]           ; Loop bound
    
    ;==========================================================================
    ; BROADCAST SCALARS TO DEDICATED REGISTERS
    ;
    ; Register allocation:
    ;   ymm6  = const_one (1.0)
    ;   ymm7  = idx_increment (8.0)
    ;   ymm8  = x (observation)
    ;   ymm9  = h (hazard rate)
    ;   ymm10 = 1-h
    ;   ymm11 = threshold
    ;   ymm12 = r0 accumulator
    ;   ymm13 = max_growth_A
    ;   ymm14 = max_growth_B
    ;   ymm15 = idx_vec_A
    ;
    ; Scratch: ymm0-ymm5
    ;==========================================================================
    vmovapd     ymm6, [rel const_one]
    vmovapd     ymm7, [rel idx_increment]
    vbroadcastsd ymm8,  qword [rdi + ARG_X]
    vbroadcastsd ymm9,  qword [rdi + ARG_H]
    vbroadcastsd ymm10, qword [rdi + ARG_OMH]
    vbroadcastsd ymm11, qword [rdi + ARG_THRESH]
    
    ;==========================================================================
    ; INITIALIZE ACCUMULATORS
    ;==========================================================================
    vxorpd      ymm12, ymm12, ymm12                 ; r0_acc = 0
    vxorpd      ymm13, ymm13, ymm13                 ; max_growth_A = 0
    vxorpd      ymm14, ymm14, ymm14                 ; max_growth_B = 0
    
    vmovapd     ymm15, [rel idx_init_a]             ; idx_vec_A = [1,2,3,4]
    
    ; idx_vec_B on stack
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0         ; idx_vec_B = [5,6,7,8]
    
    ; max_idx accumulators on stack (initialized to 0)
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0
    
    ; Scalar tracking
    xor         rbx, rbx                            ; last_valid = 0
    xor         rsi, rsi                            ; i = 0
    
    ;==========================================================================
    ; MAIN LOOP
    ;==========================================================================
.loop_start:
    cmp         rsi, r14
    jge         .loop_end
    
    ;==========================================================================
    ; BLOCK A: Elements [i, i+1, i+2, i+3]
    ;
    ; Address calculation for V3 layout:
    ;   block_index = i / 4
    ;   byte_offset = block_index * 256
    ;==========================================================================
    mov         rax, rsi
    shr         rax, 2                              ; block_index = i / 4
    shl         rax, 8                              ; byte_offset = block_index * 256
    
    ; Load prediction parameters
    vmovapd     ymm0, [r8 + rax]                    ; mu
    vmovapd     ymm1, [r8 + rax + 32]               ; C1
    vmovapd     ymm2, [r8 + rax + 64]               ; C2
    vmovapd     ymm3, [r8 + rax + 96]               ; inv_ssn
    vmovapd     ymm4, [r12 + rsi*8]                 ; r_old[i:i+3]
    
    ;--------------------------------------------------------------------------
    ; Student-t: z² = (x - μ)², t = z² * inv_ssn
    ;--------------------------------------------------------------------------
    vsubpd      ymm5, ymm8, ymm0                    ; z = x - mu
    vmulpd      ymm5, ymm5, ymm5                    ; z²
    vmulpd      ymm5, ymm5, ymm3                    ; t = z² * inv_ssn
    
    ;--------------------------------------------------------------------------
    ; log1p(t) via Horner (using ymm0 as accumulator)
    ; log1p(t) ≈ t * (1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
    ;--------------------------------------------------------------------------
    vmovapd     ymm0, [rel LOG1P_C6]
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C5]          ; c5 + t*c6
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C4]          ; c4 + t*(...)
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C3]          ; c3 + t*(...)
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C2]          ; c2 + t*(...)
    vfmadd213pd ymm0, ymm5, ymm6                    ; 1 + t*(...) [ymm6 = const_one]
    vmulpd      ymm0, ymm5, ymm0                    ; log1p(t) = t * poly
    
    ;--------------------------------------------------------------------------
    ; ln_pp = C1 - C2 * log1p(t)
    ; Reload C1, C2 since they may have been clobbered
    ;--------------------------------------------------------------------------
    vmovapd     ymm1, [r8 + rax + 32]               ; C1
    vmovapd     ymm2, [r8 + rax + 64]               ; C2
    vfnmadd231pd ymm1, ymm2, ymm0                   ; C1 = C1 - C2*log1p(t)
    
    ;--------------------------------------------------------------------------
    ; exp(ln_pp): clamp, split into k and f
    ;--------------------------------------------------------------------------
    vmaxpd      ymm1, ymm1, [rel EXP_MIN_X]
    vminpd      ymm1, ymm1, [rel EXP_MAX_X]
    vmulpd      ymm0, ymm1, [rel EXP_INV_LN2]       ; y = ln_pp * log2(e)
    vroundpd    ymm2, ymm0, 0                       ; k = round(y)
    vsubpd      ymm0, ymm0, ymm2                    ; f = y - k
    
    ;--------------------------------------------------------------------------
    ; 2^f via Estrin polynomial
    ;--------------------------------------------------------------------------
    vmulpd      ymm3, ymm0, ymm0                    ; f²
    
    ; p01 = 1 + f*c1
    vmovapd     ymm1, ymm6                          ; 1.0
    vfmadd231pd ymm1, ymm0, [rel EXP_C1]            ; p01 = 1 + f*c1
    
    ; p23 = c2 + f*c3
    vmovapd     ymm4, [rel EXP_C2]
    vfmadd231pd ymm4, ymm0, [rel EXP_C3]            ; p23 = c2 + f*c3
    
    ; p45 = c4 + f*c5
    vmovapd     ymm5, [rel EXP_C4]
    vfmadd231pd ymm5, ymm0, [rel EXP_C5]            ; p45 = c4 + f*c5
    
    ; q0123 = p01 + f²*p23
    vfmadd231pd ymm1, ymm3, ymm4                    ; q0123
    
    ; q456 = p45 + f²*c6
    vfmadd231pd ymm5, ymm3, [rel EXP_C6]            ; q456
    
    ; f⁴
    vmulpd      ymm3, ymm3, ymm3                    ; f⁴
    
    ; exp_p = q0123 + f⁴*q456
    vfmadd231pd ymm1, ymm3, ymm5                    ; 2^f
    
    ;--------------------------------------------------------------------------
    ; 2^k via IEEE-754 bit manipulation
    ;--------------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm2                          ; k → int32
    vpmovsxdq   ymm0, xmm0                          ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel EXP_BIAS]          ; + 1023
    vpsllq      ymm0, ymm0, 52                      ; << 52
    
    ; pp = 2^f * 2^k
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]      ; clamp to min
    
    ;--------------------------------------------------------------------------
    ; BOCPD update
    ;--------------------------------------------------------------------------
    ; Reload r_old (was clobbered)
    vmovapd     ymm4, [r12 + rsi*8]
    
    vmulpd      ymm0, ymm4, ymm1                    ; r_pp = r_old * pp
    vmulpd      ymm1, ymm0, ymm10                   ; growth = r_pp * (1-h)
    vmulpd      ymm0, ymm0, ymm9                    ; change = r_pp * h
    
    ; Store growth at r_new[i+1:i+4]
    vmovupd     [r13 + rsi*8 + 8], ymm1
    
    ; Accumulate change into r0
    vaddpd      ymm12, ymm12, ymm0
    
    ;--------------------------------------------------------------------------
    ; MAX tracking A
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm13, 14               ; growth > max_growth_A?
    vblendvpd   ymm13, ymm13, ymm1, ymm0            ; update max
    vmovapd     ymm2, [rsp + STK_MAX_IDX_A]
    vblendvpd   ymm2, ymm2, ymm15, ymm0             ; update indices
    vmovapd     [rsp + STK_MAX_IDX_A], ymm2
    
    ;--------------------------------------------------------------------------
    ; Truncation tracking A (using bsr)
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm11, 14               ; growth > threshold?
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .no_trunc_A
    bsr         ecx, eax                            ; highest set bit
    lea         rbx, [rsi + rcx + 1]                ; last_valid = i + bit + 1
.no_trunc_A:
    
    ;==========================================================================
    ; BLOCK B: Elements [i+4, i+5, i+6, i+7]
    ;==========================================================================
    mov         rax, rsi
    add         rax, 4                              ; i + 4
    shr         rax, 2                              ; block_index = (i+4) / 4
    shl         rax, 8                              ; byte_offset = block_index * 256
    
    ; Load prediction parameters
    vmovapd     ymm0, [r8 + rax]                    ; mu
    vmovapd     ymm1, [r8 + rax + 32]               ; C1
    vmovapd     ymm2, [r8 + rax + 64]               ; C2
    vmovapd     ymm3, [r8 + rax + 96]               ; inv_ssn
    vmovapd     ymm4, [r12 + rsi*8 + 32]            ; r_old[i+4:i+7]
    
    ;--------------------------------------------------------------------------
    ; Student-t: z², t
    ;--------------------------------------------------------------------------
    vsubpd      ymm5, ymm8, ymm0
    vmulpd      ymm5, ymm5, ymm5
    vmulpd      ymm5, ymm5, ymm3
    
    ;--------------------------------------------------------------------------
    ; log1p(t) via Horner
    ;--------------------------------------------------------------------------
    vmovapd     ymm0, [rel LOG1P_C6]
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C5]
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C4]
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C3]
    vfmadd213pd ymm0, ymm5, [rel LOG1P_C2]
    vfmadd213pd ymm0, ymm5, ymm6
    vmulpd      ymm0, ymm5, ymm0
    
    ;--------------------------------------------------------------------------
    ; ln_pp = C1 - C2 * log1p(t)
    ;--------------------------------------------------------------------------
    vmovapd     ymm1, [r8 + rax + 32]
    vmovapd     ymm2, [r8 + rax + 64]
    vfnmadd231pd ymm1, ymm2, ymm0
    
    ;--------------------------------------------------------------------------
    ; exp(ln_pp)
    ;--------------------------------------------------------------------------
    vmaxpd      ymm1, ymm1, [rel EXP_MIN_X]
    vminpd      ymm1, ymm1, [rel EXP_MAX_X]
    vmulpd      ymm0, ymm1, [rel EXP_INV_LN2]
    vroundpd    ymm2, ymm0, 0
    vsubpd      ymm0, ymm0, ymm2
    
    ; 2^f via Estrin
    vmulpd      ymm3, ymm0, ymm0                    ; f²
    
    vmovapd     ymm1, ymm6
    vfmadd231pd ymm1, ymm0, [rel EXP_C1]
    
    vmovapd     ymm4, [rel EXP_C2]
    vfmadd231pd ymm4, ymm0, [rel EXP_C3]
    
    vmovapd     ymm5, [rel EXP_C4]
    vfmadd231pd ymm5, ymm0, [rel EXP_C5]
    
    vfmadd231pd ymm1, ymm3, ymm4
    vfmadd231pd ymm5, ymm3, [rel EXP_C6]
    
    vmulpd      ymm3, ymm3, ymm3                    ; f⁴
    vfmadd231pd ymm1, ymm3, ymm5
    
    ; 2^k
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel EXP_BIAS]
    vpsllq      ymm0, ymm0, 52
    
    ; pp = 2^f * 2^k
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]
    
    ;--------------------------------------------------------------------------
    ; BOCPD update B
    ;--------------------------------------------------------------------------
    vmovapd     ymm4, [r12 + rsi*8 + 32]            ; reload r_old
    
    vmulpd      ymm0, ymm4, ymm1
    vmulpd      ymm1, ymm0, ymm10                   ; growth
    vmulpd      ymm0, ymm0, ymm9                    ; change
    
    ; Store growth at r_new[i+5:i+8]
    vmovupd     [r13 + rsi*8 + 40], ymm1
    
    ; Accumulate change
    vaddpd      ymm12, ymm12, ymm0
    
    ;--------------------------------------------------------------------------
    ; MAX tracking B
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm14, 14
    vblendvpd   ymm14, ymm14, ymm1, ymm0
    vmovapd     ymm2, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm3, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm2, ymm2, ymm3, ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm2
    
    ;--------------------------------------------------------------------------
    ; Truncation tracking B
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm11, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .no_trunc_B
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]                ; last_valid = i + 4 + bit + 1
.no_trunc_B:
    
    ;--------------------------------------------------------------------------
    ; Update index vectors (ymm3 to avoid false deps on ymm0)
    ;--------------------------------------------------------------------------
    vaddpd      ymm15, ymm15, ymm7                  ; idx_vec_A += 8
    vmovapd     ymm3, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm3, ymm3, ymm7
    vmovapd     [rsp + STK_IDX_VEC_B], ymm3
    
    ;--------------------------------------------------------------------------
    ; Loop increment
    ;--------------------------------------------------------------------------
    add         rsi, 8
    jmp         .loop_start
    
    ;==========================================================================
    ; LOOP END - HORIZONTAL REDUCTIONS
    ;==========================================================================
.loop_end:
    
    ;--------------------------------------------------------------------------
    ; Reduce r0: [a,b,c,d] → a+b+c+d
    ;--------------------------------------------------------------------------
    vextractf128 xmm0, ymm12, 1                     ; [c, d]
    vaddpd      xmm0, xmm0, xmm12                   ; [a+c, b+d]
    vunpckhpd   xmm1, xmm0, xmm0                    ; [b+d, b+d]
    vaddsd      xmm0, xmm0, xmm1                    ; a+b+c+d
    
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0
    
    ;--------------------------------------------------------------------------
    ; Find global max across A and B
    ;--------------------------------------------------------------------------
    vmovsd      xmm5, xmm0, xmm0                    ; best_val = r0
    xor         r15, r15                            ; best_idx = 0
    
    ; Store max vectors for scalar reduction
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm13
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm14
    
    xor         rcx, rcx
.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done
    
    ; Check max_growth_A[j]
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .check_b_reduce
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2
    
.check_b_reduce:
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd    xmm1, xmm5
    jbe         .next_reduce
    vmovsd      xmm5, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si  r15, xmm2
    
.next_reduce:
    inc         rcx
    jmp         .reduce_loop
    
.reduce_done:
    ;--------------------------------------------------------------------------
    ; Store outputs
    ;--------------------------------------------------------------------------
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm5
    
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15
    
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx
    
    ;==========================================================================
    ; EPILOGUE
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
    add         rsp, 168
    
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