
;==============================================================================
; BOCPD Ultra-Optimized AVX2 Kernel - CORRECTED VERSION
;
; This kernel implements the inner loop of the BOCPD algorithm with:
; - Correct IEEE-754 2^k reconstruction (no magic number hack)
; - Correct truncation logic (matching C version exactly)
; - Correct stack alignment (32-byte for AVX2)
; - Integer constants stored and used correctly
;
; Calling convention: System V AMD64 ABI
; void bocpd_kernel_avx2(const bocpd_kernel_args_t *args);
;
; Input structure (bocpd_kernel_args_t):
;   +0   double* lin_mu
;   +8   double* lin_C1
;   +16  double* lin_C2
;   +24  double* lin_inv_ssn
;   +32  double* r_old
;   +40  double  x
;   +48  double  h
;   +56  double  one_minus_h
;   +64  double  trunc_thresh
;   +72  size_t  n_padded
;   +80  double* r_new
;   +88  double* r0_out
;   +96  double* max_growth_out
;   +104 size_t* max_idx_out
;   +112 size_t* last_valid_out
;==============================================================================

section .rodata
align 32

;------------------------------------------------------------------------------
; log1p polynomial coefficients (for t in [0, ~2])
; log(1+t) ≈ t * (c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
;------------------------------------------------------------------------------
log1p_c1:   dq 1.0, 1.0, 1.0, 1.0
log1p_c2:   dq -0.5, -0.5, -0.5, -0.5
log1p_c3:   dq 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333
log1p_c4:   dq -0.25, -0.25, -0.25, -0.25
log1p_c5:   dq 0.2, 0.2, 0.2, 0.2
log1p_c6:   dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

;------------------------------------------------------------------------------
; exp polynomial coefficients (for 2^f where f in [-0.5, 0.5])
; 2^f ≈ 1 + f*c1 + f²*c2 + f³*c3 + f⁴*c4 + f⁵*c5 + f⁶*c6
; where c_i = ln(2)^i / i!
;------------------------------------------------------------------------------
exp_c1:     dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
exp_c2:     dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
exp_c3:     dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
exp_c4:     dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:     dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:     dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

;------------------------------------------------------------------------------
; exp() range constants
;------------------------------------------------------------------------------
exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
const_one:      dq 1.0, 1.0, 1.0, 1.0

;------------------------------------------------------------------------------
; INTEGER constants (stored as int64, NOT doubles!)
; These are used with vpaddq, NOT vaddpd
;------------------------------------------------------------------------------
bias_1023:      dq 1023, 1023, 1023, 1023      ; Exponent bias for IEEE-754

;------------------------------------------------------------------------------
; Other constants
;------------------------------------------------------------------------------
const_zero:     dq 0.0, 0.0, 0.0, 0.0
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

;------------------------------------------------------------------------------
; Index offset vectors (as doubles for blendv compatibility)
; Block A: i+1, i+2, i+3, i+4
; Block B: i+5, i+6, i+7, i+8
;------------------------------------------------------------------------------
idx_offset_a:   dq 1.0, 2.0, 3.0, 4.0
idx_offset_b:   dq 5.0, 6.0, 7.0, 8.0

;------------------------------------------------------------------------------
; Structure offsets
;------------------------------------------------------------------------------
%define ARG_LIN_MU          0
%define ARG_LIN_C1          8
%define ARG_LIN_C2          16
%define ARG_LIN_INV_SSN     24
%define ARG_R_OLD           32
%define ARG_X               40
%define ARG_H               48
%define ARG_OMH             56
%define ARG_THRESH          64
%define ARG_N_PADDED        72
%define ARG_R_NEW           80
%define ARG_R0_OUT          88
%define ARG_MAX_GROWTH      96
%define ARG_MAX_IDX         104
%define ARG_LAST_VALID      112

;------------------------------------------------------------------------------
; Stack layout (must be multiple of 32 for alignment)
;
; [rsp + 0]   = max_idx_B (32 bytes, 4 doubles)
; [rsp + 32]  = max_growth_A (32 bytes, 4 doubles)  
; [rsp + 64]  = max_growth_B (32 bytes, 4 doubles)
; [rsp + 96]  = max_idx_A (32 bytes, 4 doubles)
; [rsp + 128] = scratch space
;------------------------------------------------------------------------------
%define STACK_SIZE          192     ; Space for YMM spills and scratch

section .text
global bocpd_kernel_avx2

;==============================================================================
; Main kernel entry point
;==============================================================================
bocpd_kernel_avx2:
    ;--------------------------------------------------------------------------
    ; Prologue with correct 32-byte stack alignment
    ;
    ; Order of operations:
    ; 1. Push all callee-saved registers
    ; 2. Set up frame pointer
    ; 3. Align stack to 32 bytes
    ; 4. Allocate local space
    ;--------------------------------------------------------------------------
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    
    ; Set up frame pointer AFTER all pushes
    mov         rbp, rsp
    
    ; Align stack to 32 bytes and allocate local space
    sub         rsp, STACK_SIZE + 32
    and         rsp, -32
    
    ;--------------------------------------------------------------------------
    ; Load scalar arguments into preserved registers
    ;--------------------------------------------------------------------------
    mov         r8,  [rdi + ARG_LIN_MU]
    mov         r9,  [rdi + ARG_LIN_C1]
    mov         r10, [rdi + ARG_LIN_C2]
    mov         r11, [rdi + ARG_LIN_INV_SSN]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]
    
    ;--------------------------------------------------------------------------
    ; Load broadcast scalars into YMM registers
    ;
    ; Register allocation:
    ;   ymm8  = x (observation value, broadcast)
    ;   ymm9  = h (hazard rate, broadcast)
    ;   ymm10 = 1-h (one minus hazard, broadcast)
    ;   ymm11 = threshold (truncation threshold, broadcast)
    ;   ymm12 = r0 accumulator (changepoint probability sum)
    ;   ymm13 = max_growth_A accumulator (block A running max)
    ;   ymm14 = max_growth_B accumulator (block B running max)
    ;   ymm15 = max_idx_A accumulator (block A index of max, as doubles for blend)
    ;   
    ; For max_idx_B, we'll use stack storage since we're out of preserved YMM regs
    ;--------------------------------------------------------------------------
    vbroadcastsd ymm8,  qword [rdi + ARG_X]         ; x (observation)
    vbroadcastsd ymm9,  qword [rdi + ARG_H]         ; h (hazard)
    vbroadcastsd ymm10, qword [rdi + ARG_OMH]       ; 1-h
    vbroadcastsd ymm11, qword [rdi + ARG_THRESH]    ; truncation threshold
    
    ;--------------------------------------------------------------------------
    ; Initialize accumulators
    ;--------------------------------------------------------------------------
    vxorpd      ymm12, ymm12, ymm12                 ; r0 accumulator = 0
    vxorpd      ymm13, ymm13, ymm13                 ; max_growth_A = 0  
    vxorpd      ymm14, ymm14, ymm14                 ; max_growth_B = 0
    vxorpd      ymm15, ymm15, ymm15                 ; max_idx_A = 0 (as doubles for blendv)
    
    ; Initialize max_idx_B on stack
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp], ymm0                         ; max_idx_B = 0
    
    xor         rbx, rbx                             ; last_valid = 0
    
    ;--------------------------------------------------------------------------
    ; Main loop: process 8 elements per iteration (2 blocks of 4)
    ;--------------------------------------------------------------------------
    xor         rsi, rsi                             ; i = 0
    
.loop:
    cmp         rsi, r14
    jge         .loop_end
    
    ;==========================================================================
    ; Block A: indices i+0 to i+3
    ;==========================================================================
    
    ; Load data for block A
    vmovapd     ymm0, [r8  + rsi*8]                 ; lin_mu[i..i+3]
    vmovapd     ymm1, [r9  + rsi*8]                 ; lin_C1[i..i+3]
    vmovapd     ymm2, [r10 + rsi*8]                 ; lin_C2[i..i+3]
    vmovapd     ymm3, [r11 + rsi*8]                 ; lin_inv_ssn[i..i+3]
    vmovapd     ymm4, [r12 + rsi*8]                 ; r_old[i..i+3]
    
    ; Compute z² = (x - μ)²
    vsubpd      ymm5, ymm8, ymm0                    ; z = x - mu
    vmulpd      ymm5, ymm5, ymm5                    ; z² = z * z
    
    ; t = z² * inv_sigma_sq_nu
    vmulpd      ymm5, ymm5, ymm3                    ; t = z² * inv_ssn
    
    ; log1p(t) via Horner's method
    ; poly = c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))
    vmovapd     ymm6, [rel log1p_c6]
    vfmadd213pd ymm6, ymm5, [rel log1p_c5]          ; c5 + t*c6
    vfmadd213pd ymm6, ymm5, [rel log1p_c4]          ; c4 + t*(...)
    vfmadd213pd ymm6, ymm5, [rel log1p_c3]          ; c3 + t*(...)
    vfmadd213pd ymm6, ymm5, [rel log1p_c2]          ; c2 + t*(...)
    vfmadd213pd ymm6, ymm5, [rel log1p_c1]          ; c1 + t*(...)
    vmulpd      ymm6, ymm6, ymm5                    ; log1p(t) = t * poly
    
    ; ln_pp = C1 + C2 * log1p(t)
    vfmadd231pd ymm1, ymm2, ymm6                    ; ln_pp = C1 + C2*log1p
    
    ;--------------------------------------------------------------------------
    ; exp(ln_pp) with CORRECT 2^k reconstruction
    ;--------------------------------------------------------------------------
    
    ; Clamp input to [-700, 700]
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    
    ; t = ln_pp * log2(e)
    vmulpd      ymm5, ymm1, [rel exp_inv_ln2]
    
    ; k = round(t), f = t - k
    vroundpd    ymm6, ymm5, 0                       ; k = round(t)
    vsubpd      ymm5, ymm5, ymm6                    ; f = t - k
    
    ; 2^f via polynomial (Horner)
    vmovapd     ymm7, [rel exp_c6]
    vfmadd213pd ymm7, ymm5, [rel exp_c5]
    vfmadd213pd ymm7, ymm5, [rel exp_c4]
    vfmadd213pd ymm7, ymm5, [rel exp_c3]
    vfmadd213pd ymm7, ymm5, [rel exp_c2]
    vfmadd213pd ymm7, ymm5, [rel exp_c1]
    vfmadd213pd ymm7, ymm5, [rel const_one]         ; 2^f ≈ poly result
    
    ;--------------------------------------------------------------------------
    ; CORRECT 2^k reconstruction (IEEE-754 compliant)
    ;
    ; A double 2^k has: sign=0, mantissa=0, exponent=(k+1023)
    ; So the bit pattern is simply: (k + 1023) << 52
    ;
    ; Steps:
    ; 1. Convert double k to int32 (k is in valid range)
    ; 2. Sign-extend int32 to int64
    ; 3. Add bias 1023
    ; 4. Shift to exponent position
    ;--------------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm6                          ; 4 doubles -> 4 int32 in xmm0
    vpmovsxdq   ymm0, xmm0                          ; sign-extend to 4 int64
    vpaddq      ymm0, ymm0, [rel bias_1023]         ; add exponent bias
    vpsllq      ymm0, ymm0, 52                      ; shift to exponent field
    ; ymm0 now contains exact 2^k bit patterns
    
    ; pp = 2^f * 2^k
    vmulpd      ymm7, ymm7, ymm0                    ; pp = exp(ln_pp)
    
    ; Clamp pp to minimum
    vmaxpd      ymm7, ymm7, [rel const_min_pp]
    
    ; growth = pp * r_old * (1-h)
    vmulpd      ymm0, ymm7, ymm4                    ; pp * r_old
    vmulpd      ymm0, ymm0, ymm10                   ; * (1-h)
    
    ; r0 += pp * r_old * h
    vmulpd      ymm1, ymm7, ymm4
    vfmadd231pd ymm12, ymm1, ymm9                   ; r0 += pp * r_old * h
    
    ; Store growth to r_new[i+1..i+4]
    vmovupd     [r13 + rsi*8 + 8], ymm0
    
    ;--------------------------------------------------------------------------
    ; Block A: Truncation tracking (find highest index above threshold)
    ;--------------------------------------------------------------------------
    vcmppd      ymm1, ymm0, ymm11, 14               ; growth > thresh? (14 = GT)
    vmovmskpd   eax, ymm1
    
    test        eax, eax
    jz          .skip_trunc_a
    
    ; Find highest set bit -> highest valid index
    ; Bit 3 = lane 3 = index i+4
    ; Bit 0 = lane 0 = index i+1
    bt          eax, 3
    jc          .last_valid_a4
    bt          eax, 2
    jc          .last_valid_a3
    bt          eax, 1
    jc          .last_valid_a2
    ; bit 0 must be set
    lea         rbx, [rsi + 1]
    jmp         .skip_trunc_a
.last_valid_a4:
    lea         rbx, [rsi + 4]
    jmp         .skip_trunc_a
.last_valid_a3:
    lea         rbx, [rsi + 3]
    jmp         .skip_trunc_a
.last_valid_a2:
    lea         rbx, [rsi + 2]
.skip_trunc_a:
    
    ;--------------------------------------------------------------------------
    ; Block A: MAX tracking (correct per-block accumulator with index)
    ;
    ; We maintain max_growth_A (ymm13) and max_idx_A (ymm15).
    ; For each lane where growth > current max, update both value and index.
    ; This matches the C version exactly.
    ;--------------------------------------------------------------------------
    
    ; Build index vector for block A: [i+1, i+2, i+3, i+4]
    ; Convert loop counter to double, then add offsets
    vcvtsi2sd   xmm1, xmm1, rsi                     ; convert i to double
    vbroadcastsd ymm1, xmm1                         ; broadcast i
    vaddpd      ymm2, ymm1, [rel idx_offset_a]      ; ymm2 = [i+1, i+2, i+3, i+4]
    
    ; Compare growth_a > max_growth_A
    vcmppd      ymm1, ymm0, ymm13, 14               ; ymm1 = mask where growth > max
    
    ; Blend in new maxima where mask is set
    vblendvpd   ymm13, ymm13, ymm0, ymm1            ; max_growth_A = blend(old, growth, mask)
    vblendvpd   ymm15, ymm15, ymm2, ymm1            ; max_idx_A = blend(old, indices, mask)
    
    ;==========================================================================
    ; Block B: indices i+4 to i+7 (writes to r_new[i+5..i+8])
    ;==========================================================================
    
    ; Load data for block B
    vmovapd     ymm0, [r8  + rsi*8 + 32]            ; lin_mu[i+4..i+7]
    vmovapd     ymm1, [r9  + rsi*8 + 32]            ; lin_C1[i+4..i+7]
    vmovapd     ymm2, [r10 + rsi*8 + 32]            ; lin_C2[i+4..i+7]
    vmovapd     ymm3, [r11 + rsi*8 + 32]            ; lin_inv_ssn[i+4..i+7]
    vmovapd     ymm4, [r12 + rsi*8 + 32]            ; r_old[i+4..i+7]
    
    ; Compute z² = (x - μ)²
    vsubpd      ymm5, ymm8, ymm0
    vmulpd      ymm5, ymm5, ymm5
    
    ; t = z² * inv_sigma_sq_nu
    vmulpd      ymm5, ymm5, ymm3
    
    ; log1p(t)
    vmovapd     ymm6, [rel log1p_c6]
    vfmadd213pd ymm6, ymm5, [rel log1p_c5]
    vfmadd213pd ymm6, ymm5, [rel log1p_c4]
    vfmadd213pd ymm6, ymm5, [rel log1p_c3]
    vfmadd213pd ymm6, ymm5, [rel log1p_c2]
    vfmadd213pd ymm6, ymm5, [rel log1p_c1]
    vmulpd      ymm6, ymm6, ymm5
    
    ; ln_pp = C1 + C2 * log1p(t)
    vfmadd231pd ymm1, ymm2, ymm6
    
    ; exp(ln_pp) - clamp
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    
    ; t = ln_pp * log2(e)
    vmulpd      ymm5, ymm1, [rel exp_inv_ln2]
    
    ; k = round(t), f = t - k
    vroundpd    ymm6, ymm5, 0
    vsubpd      ymm5, ymm5, ymm6
    
    ; 2^f polynomial
    vmovapd     ymm7, [rel exp_c6]
    vfmadd213pd ymm7, ymm5, [rel exp_c5]
    vfmadd213pd ymm7, ymm5, [rel exp_c4]
    vfmadd213pd ymm7, ymm5, [rel exp_c3]
    vfmadd213pd ymm7, ymm5, [rel exp_c2]
    vfmadd213pd ymm7, ymm5, [rel exp_c1]
    vfmadd213pd ymm7, ymm5, [rel const_one]
    
    ; CORRECT 2^k reconstruction for block B
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    
    ; pp = 2^f * 2^k
    vmulpd      ymm7, ymm7, ymm0
    vmaxpd      ymm7, ymm7, [rel const_min_pp]
    
    ; growth = pp * r_old * (1-h)
    vmulpd      ymm0, ymm7, ymm4
    vmulpd      ymm0, ymm0, ymm10
    
    ; r0 += pp * r_old * h
    vmulpd      ymm1, ymm7, ymm4
    vfmadd231pd ymm12, ymm1, ymm9
    
    ; Store growth to r_new[i+5..i+8]
    vmovupd     [r13 + rsi*8 + 40], ymm0
    
    ;--------------------------------------------------------------------------
    ; Block B: Truncation tracking
    ;--------------------------------------------------------------------------
    vcmppd      ymm1, ymm0, ymm11, 14
    vmovmskpd   eax, ymm1
    
    test        eax, eax
    jz          .skip_trunc_b
    
    bt          eax, 3
    jc          .last_valid_b8
    bt          eax, 2
    jc          .last_valid_b7
    bt          eax, 1
    jc          .last_valid_b6
    lea         rbx, [rsi + 5]
    jmp         .skip_trunc_b
.last_valid_b8:
    lea         rbx, [rsi + 8]
    jmp         .skip_trunc_b
.last_valid_b7:
    lea         rbx, [rsi + 7]
    jmp         .skip_trunc_b
.last_valid_b6:
    lea         rbx, [rsi + 6]
.skip_trunc_b:
    
    ;--------------------------------------------------------------------------
    ; Block B: MAX tracking (correct per-block accumulator with index)
    ;
    ; We maintain max_growth_B (ymm14) and max_idx_B (on stack).
    ; For each lane where growth > current max, update both value and index.
    ;--------------------------------------------------------------------------
    
    ; Build index vector for block B: [i+5, i+6, i+7, i+8]
    vcvtsi2sd   xmm1, xmm1, rsi
    vbroadcastsd ymm1, xmm1
    vaddpd      ymm2, ymm1, [rel idx_offset_b]      ; ymm2 = [i+5, i+6, i+7, i+8]
    
    ; Load current max_idx_B from stack
    vmovapd     ymm3, [rsp]                         ; ymm3 = max_idx_B
    
    ; Compare growth_b > max_growth_B
    vcmppd      ymm1, ymm0, ymm14, 14               ; ymm1 = mask where growth > max
    
    ; Blend in new maxima where mask is set
    vblendvpd   ymm14, ymm14, ymm0, ymm1            ; max_growth_B = blend(old, growth, mask)
    vblendvpd   ymm3, ymm3, ymm2, ymm1              ; max_idx_B = blend(old, indices, mask)
    
    ; Store updated max_idx_B back to stack
    vmovapd     [rsp], ymm3
    
    ;--------------------------------------------------------------------------
    ; Loop increment
    ;--------------------------------------------------------------------------
    add         rsi, 8
    jmp         .loop
    
.loop_end:
    
    ;==========================================================================
    ; Horizontal reductions
    ;==========================================================================
    
    ;--------------------------------------------------------------------------
    ; Horizontal sum of r0 accumulator (ymm12)
    ;--------------------------------------------------------------------------
    vextractf128 xmm0, ymm12, 1
    vaddpd      xmm0, xmm0, xmm12
    vhaddpd     xmm0, xmm0, xmm0
    
    ; Store r0
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0
    
    ; Save r0 value for MAP comparison
    vmovsd      xmm6, xmm0, xmm0                    ; xmm6 = r0 (best value so far)
    xor         r15, r15                             ; r15 = 0 (best index so far)
    
    ;--------------------------------------------------------------------------
    ; Horizontal reduction for MAP: find global max across both blocks
    ;
    ; We have:
    ;   ymm13 = max_growth_A (4 lanes)
    ;   ymm14 = max_growth_B (4 lanes)
    ;   ymm15 = max_idx_A (4 lanes, as doubles)
    ;   [rsp] = max_idx_B (4 lanes, as doubles)
    ;
    ; Must compare all 8 lanes to find global maximum, matching C version.
    ;--------------------------------------------------------------------------
    
    ; Store all to stack for scalar reduction
    vmovapd     [rsp + 32], ymm13                   ; max_growth_A
    vmovapd     [rsp + 64], ymm14                   ; max_growth_B
    vmovapd     [rsp + 96], ymm15                   ; max_idx_A
    ; max_idx_B already at [rsp]
    
    ; Scalar loop: compare all 8 lanes to find global max
    ; This matches the C version exactly
    xor         rcx, rcx                             ; j = 0
    
.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done
    
    ; Check max_growth_A[j]
    vmovsd      xmm1, [rsp + 32 + rcx*8]            ; max_growth_A[j]
    vucomisd    xmm1, xmm6                          ; compare to best
    jbe         .check_b
    
    vmovsd      xmm6, xmm1, xmm1                    ; update best value
    vmovsd      xmm2, [rsp + 96 + rcx*8]            ; max_idx_A[j]
    vcvttsd2si  r15, xmm2                           ; update best index
    
.check_b:
    ; Check max_growth_B[j]
    vmovsd      xmm1, [rsp + 64 + rcx*8]            ; max_growth_B[j]
    vucomisd    xmm1, xmm6                          ; compare to best
    jbe         .next_j
    
    vmovsd      xmm6, xmm1, xmm1                    ; update best value
    vmovsd      xmm2, [rsp + rcx*8]                 ; max_idx_B[j]
    vcvttsd2si  r15, xmm2                           ; update best index
    
.next_j:
    inc         rcx
    jmp         .reduce_loop
    
.reduce_done:
    ; Store max_growth
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6
    
    ; Store max_idx
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15
    
    ; Store last_valid
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx
    
    ;--------------------------------------------------------------------------
    ; Epilogue with correct stack restoration
    ;
    ; Restore RSP from RBP (undoes alignment and local allocation),
    ; then pop callee-saved registers in reverse order.
    ;--------------------------------------------------------------------------
    mov         rsp, rbp
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    
    vzeroupper
    ret
