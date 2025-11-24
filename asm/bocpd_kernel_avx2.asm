;==============================================================================
; BOCPD Ultra-Optimized AVX2 Kernel - V2 (Interleaved Layout)
;
; ALGORITHM OVERVIEW
; ==================
; Bayesian Online Changepoint Detection (BOCPD) maintains a probability
; distribution r[i] over "run lengths" - how long since the last changepoint.
;
; For each new observation x, we update:
;   1. Compute predictive probability: pp[i] = P(x | run_length=i)
;      Using Student-t: pp = exp(C1 - C2 * log1p((x-μ)² / (σ²ν)))
;   2. Growth probability:   growth[i] = r[i] * pp[i] * (1-h)
;   3. Changepoint prob:     r_new[0] += r[i] * pp[i] * h
;   4. Shift and store:      r_new[i+1] = growth[i]
;
; This kernel processes 8 elements per iteration (2 SIMD blocks of 4).
;
; KEY OPTIMIZATIONS
; =================
; 1. INTERLEAVED MEMORY LAYOUT
;    Instead of 4 separate arrays, parameters are interleaved:
;      Block k: [mu[4k:4k+3], C1[4k:4k+3], C2[4k:4k+3], inv_ssn[4k:4k+3]]
;    Each block = 128 bytes = 2 cache lines (perfect spatial locality)
;
; 2. RUNNING INDEX VECTORS
;    Instead of: idx = broadcast(i) + offset  (2 µops, 3-cycle latency)
;    We use:     idx_vec += 8 each iteration  (1 µop, 1-cycle latency)
;
; 3. ESTRIN'S POLYNOMIAL SCHEME
;    For exp(), groups terms to reduce dependency depth from 6 to 4 FMAs
;
; 4. IEEE-754 DIRECT EXPONENT MANIPULATION
;    Computes 2^k by directly constructing the bit pattern, avoiding libm
;
; 5. SHARED CONSTANTS
;    const_one used by both log1p (c1=1) and exp (base=1)
;
; REGISTER ALLOCATION
; ===================
; Callee-saved (preserved):
;   r8=lin_interleaved, r12=r_old, r13=r_new, r14=n_padded, rdi=args
;   rsi=loop_counter, rbx=last_valid
;
; Dedicated YMM (never spilled):
;   ymm8=x, ymm9=h, ymm10=1-h, ymm11=threshold
;   ymm12=r0_accumulator, ymm13=max_growth_A, ymm14=max_growth_B
;   ymm15=idx_vec_A
;
; Scratch: ymm0-ymm7 (reused freely each iteration)
;
; Calling convention: System V AMD64 ABI
; void bocpd_fused_loop_avx2(const bocpd_kernel_args_t *args);
;
; Input structure (bocpd_kernel_args_t):
;   +0   double* lin_interleaved  ; Interleaved [mu×4, C1×4, C2×4, inv_ssn×4]
;   +8   double* r_old            ; Current run-length distribution
;   +16  double  x                ; New observation
;   +24  double  h                ; Hazard rate P(changepoint)
;   +32  double  one_minus_h      ; 1 - h (precomputed)
;   +40  double  trunc_thresh     ; Truncation threshold
;   +48  size_t  n_padded         ; Array length (multiple of 8)
;   +56  double* r_new            ; Output distribution
;   +64  double* r0_out           ; Output: sum of changepoint probs
;   +72  double* max_growth_out   ; Output: max growth value (for MAP)
;   +80  size_t* max_idx_out      ; Output: index of max growth
;   +88  size_t* last_valid_out   ; Output: last index above threshold
;==============================================================================

; Mark stack as non-executable (ELF security best practice)
section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
align 32                            ; AVX2 requires 32-byte alignment

;------------------------------------------------------------------------------
; SHARED CONSTANT: 1.0
; Used as c1 in log1p polynomial AND as base value in exp polynomial
; Saves one constant slot (register pressure reduction)
;------------------------------------------------------------------------------
const_one:      dq 1.0, 1.0, 1.0, 1.0

;------------------------------------------------------------------------------
; LOG1P POLYNOMIAL: log(1+t) ≈ t * (c1 + t*(c2 + t*(c3 + ...)))
; Taylor series: log(1+t) = t - t²/2 + t³/3 - t⁴/4 + ...
; Coefficients: c1=1, c2=-1/2, c3=1/3, c4=-1/4, c5=1/5, c6=-1/6
; (c1 uses const_one above)
;------------------------------------------------------------------------------
log1p_c2:   dq -0.5, -0.5, -0.5, -0.5
log1p_c3:   dq 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333
log1p_c4:   dq -0.25, -0.25, -0.25, -0.25
log1p_c5:   dq 0.2, 0.2, 0.2, 0.2
log1p_c6:   dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

;------------------------------------------------------------------------------
; EXP POLYNOMIAL: 2^f ≈ 1 + f*c1 + f²*c2 + f³*c3 + f⁴*c4 + f⁵*c5 + f⁶*c6
;
; Strategy: exp(x) = 2^(x * log2(e)) = 2^t where t = x / ln(2)
; Split: t = k + f where k = round(t), f = t - k, |f| ≤ 0.5
; Then: exp(x) = 2^k * 2^f
;   - 2^k computed via IEEE-754 exponent bit manipulation
;   - 2^f approximated by this polynomial (accurate for |f| ≤ 0.5)
;
; Coefficients: c_i = ln(2)^i / i!
;------------------------------------------------------------------------------
exp_c1:     dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453      ; ln(2)
exp_c2:     dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072  ; ln(2)²/2
exp_c3:     dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158  ; ln(2)³/6
exp_c4:     dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:     dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:     dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

;------------------------------------------------------------------------------
; EXP HELPER CONSTANTS
;------------------------------------------------------------------------------
exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634  ; log2(e) = 1/ln(2)
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0      ; Clamp floor: exp(-700) ≈ 1e-304
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0          ; Clamp ceil: exp(700) ≈ 1e304

;------------------------------------------------------------------------------
; IEEE-754 EXPONENT BIAS (as int64, NOT float!)
; Double format: [sign:1][exponent:11][mantissa:52]
; Exponent uses bias-1023: stored_exp = actual_exp + 1023
; To create 2^k: set exponent = k+1023, shift left 52 bits
;------------------------------------------------------------------------------
bias_1023:      dq 1023, 1023, 1023, 1023

;------------------------------------------------------------------------------
; NUMERICAL FLOOR: minimum predictive probability
; Prevents log(0) or divide-by-zero in downstream calculations
;------------------------------------------------------------------------------
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

;------------------------------------------------------------------------------
; Running index vectors (as doubles for blendv)
; Initial values: Block A = [1,2,3,4], Block B = [5,6,7,8]
; Increment = 8 each iteration
;------------------------------------------------------------------------------
idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

;------------------------------------------------------------------------------
; Structure offsets - UPDATED FOR INTERLEAVED LAYOUT
;------------------------------------------------------------------------------
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

;------------------------------------------------------------------------------
; Stack layout (must be multiple of 32 for alignment)
;
; [rsp + 0]   = max_idx_B (32 bytes, 4 doubles)
; [rsp + 32]  = max_growth_A (32 bytes, 4 doubles)  
; [rsp + 64]  = max_growth_B (32 bytes, 4 doubles)
; [rsp + 96]  = max_idx_A (32 bytes, 4 doubles)
; [rsp + 128] = scratch space
; [rsp + 160] = idx_vec_B (32 bytes)
;------------------------------------------------------------------------------
%define STACK_SIZE          224

section .text
global bocpd_fused_loop_avx2

bocpd_fused_loop_avx2:
    ;--------------------------------------------------------------------------
    ; Prologue with correct stack alignment
    ;--------------------------------------------------------------------------
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    
    mov         rbp, rsp
    sub         rsp, STACK_SIZE + 32
    and         rsp, -32
    
    ;--------------------------------------------------------------------------
    ; Load pointers into preserved registers
    ;--------------------------------------------------------------------------
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]
    
    ;--------------------------------------------------------------------------
    ; Load broadcast scalars into YMM registers
    ;--------------------------------------------------------------------------
    vbroadcastsd ymm8,  qword [rdi + ARG_X]
    vbroadcastsd ymm9,  qword [rdi + ARG_H]
    vbroadcastsd ymm10, qword [rdi + ARG_OMH]
    vbroadcastsd ymm11, qword [rdi + ARG_THRESH]
    
    ;--------------------------------------------------------------------------
    ; Initialize accumulators
    ;--------------------------------------------------------------------------
    vxorpd      ymm12, ymm12, ymm12                 ; r0 accumulator = 0
    vxorpd      ymm13, ymm13, ymm13                 ; max_growth_A = 0  
    vxorpd      ymm14, ymm14, ymm14                 ; max_growth_B = 0
    
    ; Initialize running index vectors
    vmovapd     ymm15, [rel idx_init_a]             ; idx_vec_A = [1,2,3,4]
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + 160], ymm0                   ; idx_vec_B on stack
    
    ; Initialize max_idx accumulators
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp], ymm0                         ; max_idx_B = 0
    vmovapd     [rsp + 96], ymm0                    ; max_idx_A = 0
    
    xor         rbx, rbx                             ; last_valid = 0
    xor         rsi, rsi                             ; i = 0
    
.loop:
    cmp         rsi, r14
    jge         .loop_end
    
    ;==========================================================================
    ; Block A: indices i+0 to i+3
    ; Base offset = (i/4) * 128 = i * 32
    ;==========================================================================
    
    mov         rax, rsi
    shl         rax, 5                              ; rax = i * 32
    
    ; Load from interleaved buffer
    vmovapd     ymm0, [r8 + rax]                    ; mu_a
    vmovapd     ymm1, [r8 + rax + 32]               ; C1_a
    vmovapd     ymm2, [r8 + rax + 64]               ; C2_a
    vmovapd     ymm3, [r8 + rax + 96]               ; inv_ssn_a
    vmovapd     ymm4, [r12 + rsi*8]                 ; r_old_a
    
    ; z² = (x - μ)²
    vsubpd      ymm5, ymm8, ymm0
    vmulpd      ymm5, ymm5, ymm5
    
    ; t = z² * inv_ssn
    vmulpd      ymm5, ymm5, ymm3
    
    ; log1p(t) via Horner
    vmovapd     ymm6, [rel log1p_c6]
    vfmadd213pd ymm6, ymm5, [rel log1p_c5]
    vfmadd213pd ymm6, ymm5, [rel log1p_c4]
    vfmadd213pd ymm6, ymm5, [rel log1p_c3]
    vfmadd213pd ymm6, ymm5, [rel log1p_c2]
    vfmadd213pd ymm6, ymm5, [rel const_one]
    vmulpd      ymm6, ymm6, ymm5
    
    ; ln_pp = C1 - C2 * log1p(t)
    vfnmadd231pd ymm1, ymm2, ymm6
    
    ; exp(ln_pp) - clamp
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm5, ymm1, [rel exp_inv_ln2]
    vroundpd    ymm6, ymm5, 0
    vsubpd      ymm5, ymm5, ymm6
    
    ; Estrin's scheme for 2^f
    vmovapd     ymm7, [rel const_one]
    vfmadd231pd ymm7, ymm5, [rel exp_c1]
    vmovapd     ymm0, [rel exp_c2]
    vfmadd231pd ymm0, ymm5, [rel exp_c3]
    vmovapd     ymm1, [rel exp_c4]
    vfmadd231pd ymm1, ymm5, [rel exp_c5]
    vmulpd      ymm2, ymm5, ymm5
    vfmadd231pd ymm7, ymm2, ymm0
    vfmadd231pd ymm1, ymm2, [rel exp_c6]
    vmulpd      ymm2, ymm2, ymm2
    vfmadd231pd ymm7, ymm2, ymm1
    
    ; 2^k reconstruction
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    
    ; pp = 2^f * 2^k
    vmulpd      ymm7, ymm7, ymm0
    vmaxpd      ymm7, ymm7, [rel const_min_pp]
    
    ; BOCPD update
    vmulpd      ymm0, ymm7, ymm4
    vmulpd      ymm1, ymm0, ymm10                   ; growth_a
    vmulpd      ymm0, ymm0, ymm9                    ; change_a
    
    vmovupd     [r13 + rsi*8 + 8], ymm1
    vaddpd      ymm12, ymm12, ymm0
    
    ; MAX tracking A
    vcmppd      ymm0, ymm1, ymm13, 14
    vblendvpd   ymm13, ymm13, ymm1, ymm0
    vmovapd     ymm2, [rsp + 96]
    vblendvpd   ymm2, ymm2, ymm15, ymm0
    vmovapd     [rsp + 96], ymm2
    
    ; Truncation A
    vcmppd      ymm0, ymm1, ymm11, 14
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
    
    ;==========================================================================
    ; Block B: indices i+4 to i+7
    ; Base offset = (i/4 + 1) * 128 = i*32 + 128
    ;==========================================================================
    
    mov         rax, rsi
    shl         rax, 5
    add         rax, 128
    
    vmovapd     ymm0, [r8 + rax]
    vmovapd     ymm1, [r8 + rax + 32]
    vmovapd     ymm2, [r8 + rax + 64]
    vmovapd     ymm3, [r8 + rax + 96]
    vmovapd     ymm4, [r12 + rsi*8 + 32]
    
    vsubpd      ymm5, ymm8, ymm0
    vmulpd      ymm5, ymm5, ymm5
    vmulpd      ymm5, ymm5, ymm3
    
    vmovapd     ymm6, [rel log1p_c6]
    vfmadd213pd ymm6, ymm5, [rel log1p_c5]
    vfmadd213pd ymm6, ymm5, [rel log1p_c4]
    vfmadd213pd ymm6, ymm5, [rel log1p_c3]
    vfmadd213pd ymm6, ymm5, [rel log1p_c2]
    vfmadd213pd ymm6, ymm5, [rel const_one]
    vmulpd      ymm6, ymm6, ymm5
    
    vfnmadd231pd ymm1, ymm2, ymm6
    
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm5, ymm1, [rel exp_inv_ln2]
    vroundpd    ymm6, ymm5, 0
    vsubpd      ymm5, ymm5, ymm6
    
    vmovapd     ymm7, [rel const_one]
    vfmadd231pd ymm7, ymm5, [rel exp_c1]
    vmovapd     ymm0, [rel exp_c2]
    vfmadd231pd ymm0, ymm5, [rel exp_c3]
    vmovapd     ymm1, [rel exp_c4]
    vfmadd231pd ymm1, ymm5, [rel exp_c5]
    vmulpd      ymm2, ymm5, ymm5
    vfmadd231pd ymm7, ymm2, ymm0
    vfmadd231pd ymm1, ymm2, [rel exp_c6]
    vmulpd      ymm2, ymm2, ymm2
    vfmadd231pd ymm7, ymm2, ymm1
    
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    
    vmulpd      ymm7, ymm7, ymm0
    vmaxpd      ymm7, ymm7, [rel const_min_pp]
    
    vmulpd      ymm0, ymm7, ymm4
    vmulpd      ymm1, ymm0, ymm10
    vmulpd      ymm0, ymm0, ymm9
    
    vmovupd     [r13 + rsi*8 + 40], ymm1
    vaddpd      ymm12, ymm12, ymm0
    
    ; MAX tracking B
    vcmppd      ymm0, ymm1, ymm14, 14
    vblendvpd   ymm14, ymm14, ymm1, ymm0
    vmovapd     ymm2, [rsp]
    vmovapd     ymm3, [rsp + 160]
    vblendvpd   ymm2, ymm2, ymm3, ymm0
    vmovapd     [rsp], ymm2
    
    ; Truncation B
    vcmppd      ymm0, ymm1, ymm11, 14
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
    
    ; Update running index vectors
    vaddpd      ymm15, ymm15, [rel idx_increment]
    vmovapd     ymm0, [rsp + 160]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + 160], ymm0
    
    add         rsi, 8
    jmp         .loop
    
.loop_end:
    
    ;==========================================================================
    ; Horizontal reductions
    ;==========================================================================
    
    ; r0 sum
    vextractf128 xmm0, ymm12, 1
    vaddpd      xmm0, xmm0, xmm12
    vhaddpd     xmm0, xmm0, xmm0
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0
    
    vmovsd      xmm6, xmm0, xmm0
    xor         r15, r15
    
    ; Store max vectors for reduction
    vmovapd     [rsp + 32], ymm13
    vmovapd     [rsp + 64], ymm14
    
    xor         rcx, rcx
.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done
    
    vmovsd      xmm1, [rsp + 32 + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .check_b
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + 96 + rcx*8]
    vcvttsd2si  r15, xmm2
    
.check_b:
    vmovsd      xmm1, [rsp + 64 + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .next_j
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + rcx*8]
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
    
    ;--------------------------------------------------------------------------
    ; Epilogue
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
