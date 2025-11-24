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
; void bocpd_fused_loop_avx2_generic(const bocpd_kernel_args_t *args);
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
; RUNNING INDEX VECTORS (stored as doubles for vblendvpd compatibility)
;
; OPTIMIZATION: Instead of expensive per-iteration broadcast:
;   Old: idx = _mm256_set1_epi64x(i) + offset  ; 2 µops, 3 cycles
;   New: idx_vec += increment each iteration   ; 1 µop, 1 cycle
;
; Block A processes indices [1,2,3,4] + 8*iteration
; Block B processes indices [5,6,7,8] + 8*iteration
; (1-based because run length 0 is handled separately via r0 accumulator)
;------------------------------------------------------------------------------
idx_init_a:     dq 1.0, 2.0, 3.0, 4.0      ; Initial indices for Block A
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0      ; Initial indices for Block B
idx_increment:  dq 8.0, 8.0, 8.0, 8.0      ; Add 8 each iteration (2 blocks × 4)

;------------------------------------------------------------------------------
; STRUCTURE FIELD OFFSETS (must match bocpd_kernel_args_t in C exactly!)
;------------------------------------------------------------------------------
%define ARG_LIN_INTERLEAVED 0              ; double* - interleaved params
%define ARG_R_OLD           8              ; double* - input distribution
%define ARG_X               16             ; double  - observation value
%define ARG_H               24             ; double  - hazard rate
%define ARG_OMH             32             ; double  - 1 - hazard (precomputed)
%define ARG_THRESH          40             ; double  - truncation threshold
%define ARG_N_PADDED        48             ; size_t  - loop bound (multiple of 8)
%define ARG_R_NEW           56             ; double* - output distribution
%define ARG_R0_OUT          64             ; double* - output: changepoint prob
%define ARG_MAX_GROWTH      72             ; double* - output: max growth value
%define ARG_MAX_IDX         80             ; size_t* - output: index of max
%define ARG_LAST_VALID      88             ; size_t* - output: truncation boundary

;------------------------------------------------------------------------------
; STACK FRAME LAYOUT
;
; We need stack storage for vectors that don't fit in registers:
; [rsp + 0]   = max_idx_B    (32 bytes) - indices where Block B max occurred
; [rsp + 32]  = max_growth_A (32 bytes) - max values from Block A (for reduction)
; [rsp + 64]  = max_growth_B (32 bytes) - max values from Block B (for reduction)
; [rsp + 96]  = max_idx_A    (32 bytes) - indices where Block A max occurred
; [rsp + 160] = idx_vec_B    (32 bytes) - running index vector for Block B
;
; Total: 192 bytes, rounded to 224 for 32-byte alignment
;------------------------------------------------------------------------------
%define STACK_SIZE          224

;==============================================================================
; CODE SECTION
;==============================================================================
section .text
global bocpd_fused_loop_avx2_generic

bocpd_fused_loop_avx2_generic:
    ;==========================================================================
    ; PROLOGUE: Save callee-saved registers, set up aligned stack frame
    ;==========================================================================
    
    ; System V AMD64: rbx, rbp, r12-r15 are callee-saved
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    
    ; Set up aligned stack frame (AVX2 requires 32-byte alignment)
    mov         rbp, rsp                            ; Save original RSP for epilogue
    sub         rsp, STACK_SIZE + 32                ; Allocate local storage
    and         rsp, -32                            ; Align to 32 bytes
    
    ;==========================================================================
    ; LOAD ARGUMENTS: Move frequently-accessed values into registers
    ;==========================================================================
    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]    ; r8  = interleaved params base
    mov         r12, [rdi + ARG_R_OLD]              ; r12 = input distribution
    mov         r13, [rdi + ARG_R_NEW]              ; r13 = output distribution
    mov         r14, [rdi + ARG_N_PADDED]           ; r14 = loop bound
    
    ;==========================================================================
    ; BROADCAST SCALARS: Replicate scalar parameters to all SIMD lanes
    ; vbroadcastsd: load scalar double, replicate to all 4 lanes of YMM
    ;==========================================================================
    vbroadcastsd ymm8,  qword [rdi + ARG_X]         ; ymm8  = [x, x, x, x]
    vbroadcastsd ymm9,  qword [rdi + ARG_H]         ; ymm9  = [h, h, h, h]
    vbroadcastsd ymm10, qword [rdi + ARG_OMH]       ; ymm10 = [1-h, 1-h, 1-h, 1-h]
    vbroadcastsd ymm11, qword [rdi + ARG_THRESH]    ; ymm11 = [thresh, ...]
    
    ;==========================================================================
    ; INITIALIZE ACCUMULATORS
    ;==========================================================================
    vxorpd      ymm12, ymm12, ymm12                 ; r0 accumulator = 0 (changepoint prob sum)
    vxorpd      ymm13, ymm13, ymm13                 ; max_growth_A = 0 (for MAP tracking)
    vxorpd      ymm14, ymm14, ymm14                 ; max_growth_B = 0 (separate to avoid deps)
    
    ; Running index vectors (stored as doubles for vblendvpd)
    vmovapd     ymm15, [rel idx_init_a]             ; idx_vec_A = [1,2,3,4] (in register)
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + 160], ymm0                   ; idx_vec_B = [5,6,7,8] (on stack)
    
    ; Max index accumulators (track WHERE maximum occurred)
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp], ymm0                         ; max_idx_B = [0,0,0,0]
    vmovapd     [rsp + 96], ymm0                    ; max_idx_A = [0,0,0,0]
    
    ; Scalar tracking
    xor         rbx, rbx                            ; last_valid = 0 (truncation boundary)
    xor         rsi, rsi                            ; i = 0 (loop counter)
    
    ;==========================================================================
    ; MAIN LOOP: Process 8 elements per iteration (Block A + Block B)
    ; for (i = 0; i < n_padded; i += 8)
    ;==========================================================================
.loop:
    cmp         rsi, r14                            ; if i >= n_padded
    jge         .loop_end                           ;   exit loop
    
    ;==========================================================================
    ; BLOCK A: Process elements [i, i+1, i+2, i+3]
    ;
    ; INTERLEAVED LAYOUT ADDRESS CALCULATION:
    ;   Block number = i / 4 (since i is multiple of 8, i/4 is even)
    ;   Byte offset = block_number * 128 = (i/4) * 128 = i * 32
    ;
    ; Within each 128-byte block:
    ;   mu[0:3]      at offset 0   (32 bytes)
    ;   C1[0:3]      at offset 32  (32 bytes)
    ;   C2[0:3]      at offset 64  (32 bytes)
    ;   inv_ssn[0:3] at offset 96  (32 bytes)
    ;==========================================================================
    
    mov         rax, rsi
    shl         rax, 5                              ; rax = i * 32 = byte offset
    
    ; Load all 4 parameter vectors from interleaved block (2 cache lines)
    vmovapd     ymm0, [r8 + rax]                    ; ymm0 = mu[i:i+3]
    vmovapd     ymm1, [r8 + rax + 32]               ; ymm1 = C1[i:i+3]
    vmovapd     ymm2, [r8 + rax + 64]               ; C2_a
    vmovapd     ymm3, [r8 + rax + 96]               ; inv_ssn_a
    vmovapd     ymm4, [r12 + rsi*8]                 ; ymm4 = r_old[i:i+3] (input distribution)
    
    ;--------------------------------------------------------------------------
    ; STEP 1: Compute z² = (x - μ)²
    ; Squared deviation of observation from posterior mean
    ;--------------------------------------------------------------------------
    vsubpd      ymm5, ymm8, ymm0                    ; ymm5 = x - mu (deviation)
    vmulpd      ymm5, ymm5, ymm5                    ; ymm5 = z² = (x - mu)²
    
    ;--------------------------------------------------------------------------
    ; STEP 2: Compute t = z² / (σ²ν) = z² * inv_ssn
    ; This is the argument to log1p in Student-t formula
    ;--------------------------------------------------------------------------
    vmulpd      ymm5, ymm5, ymm3                    ; ymm5 = t = z² * inv_ssn
    
    ;--------------------------------------------------------------------------
    ; STEP 3: Compute log1p(t) via Horner's method
    ; log(1+t) ≈ t * (c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
    ; vfmadd213pd ymm6, ymm5, [mem]: ymm6 = ymm5*ymm6 + [mem]
    ;--------------------------------------------------------------------------
    vmovapd     ymm6, [rel log1p_c6]                ; Start with c6 (innermost)
    vfmadd213pd ymm6, ymm5, [rel log1p_c5]          ; ymm6 = t*c6 + c5
    vfmadd213pd ymm6, ymm5, [rel log1p_c4]          ; ymm6 = t*(t*c6+c5) + c4
    vfmadd213pd ymm6, ymm5, [rel log1p_c3]          ; ...continue unwinding
    vfmadd213pd ymm6, ymm5, [rel log1p_c2]
    vfmadd213pd ymm6, ymm5, [rel const_one]         ; + c1 (c1=1.0)
    vmulpd      ymm6, ymm6, ymm5                    ; ymm6 = t * poly = log1p(t)
    
    ;--------------------------------------------------------------------------
    ; STEP 4: Compute ln_pp = C1 - C2 * log1p(t)
    ; This is log of predictive probability from Student-t distribution
    ; vfnmadd231pd: ymm1 = ymm1 - ymm2*ymm6 (fused negative multiply-add)
    ;--------------------------------------------------------------------------
    vfnmadd231pd ymm1, ymm2, ymm6                   ; ymm1 = C1 - C2*log1p(t) = ln_pp
    
    ;--------------------------------------------------------------------------
    ; STEP 5: Compute exp(ln_pp) = pp (predictive probability)
    ;
    ; Strategy: exp(x) = 2^k * 2^f where k=round(x/ln2), f=x/ln2-k
    ; Clamp input to [-700, 700] to avoid overflow/underflow
    ;--------------------------------------------------------------------------
    vmaxpd      ymm1, ymm1, [rel exp_min_x]         ; Clamp: max(ln_pp, -700)
    vminpd      ymm1, ymm1, [rel exp_max_x]         ; Clamp: min(ln_pp, 700)
    vmulpd      ymm5, ymm1, [rel exp_inv_ln2]       ; ymm5 = ln_pp * log2(e)
    vroundpd    ymm6, ymm5, 0                       ; ymm6 = k = round(ymm5)
    vsubpd      ymm5, ymm5, ymm6                    ; ymm5 = f = fractional part, |f|≤0.5
    
    ;--------------------------------------------------------------------------
    ; STEP 6: Compute 2^f using Estrin's scheme (reduced dependency depth)
    ;
    ; Polynomial: 2^f ≈ 1 + f*c1 + f²*c2 + f³*c3 + f⁴*c4 + f⁵*c5 + f⁶*c6
    ; Estrin groups: p01=1+f*c1, p23=c2+f*c3, p45=c4+f*c5
    ; Then combines: q0123 = p01 + f²*p23, q456 = p45 + f²*c6
    ; Final: 2^f = q0123 + f⁴*q456
    ;--------------------------------------------------------------------------
    vmovapd     ymm7, [rel const_one]
    vfmadd231pd ymm7, ymm5, [rel exp_c1]            ; p01 = 1 + f*c1
    vmovapd     ymm0, [rel exp_c2]
    vfmadd231pd ymm0, ymm5, [rel exp_c3]            ; p23 = c2 + f*c3
    vmovapd     ymm1, [rel exp_c4]
    vfmadd231pd ymm1, ymm5, [rel exp_c5]            ; p45 = c4 + f*c5
    vmulpd      ymm2, ymm5, ymm5                    ; f² = f * f
    vfmadd231pd ymm7, ymm2, ymm0                    ; q0123 = p01 + f²*p23
    vfmadd231pd ymm1, ymm2, [rel exp_c6]            ; q456 = p45 + f²*c6
    vmulpd      ymm2, ymm2, ymm2                    ; f⁴ = f² * f²
    vfmadd231pd ymm7, ymm2, ymm1                    ; 2^f = q0123 + f⁴*q456
    
    ;--------------------------------------------------------------------------
    ; STEP 7: Compute 2^k via IEEE-754 bit manipulation
    ;
    ; IEEE-754 double: [sign:1][exponent:11][mantissa:52]
    ; For 2^k: exponent_field = k + 1023, shift left 52 bits
    ; vcvtpd2dq: convert double→int32, vpmovsxdq: sign-extend to int64
    ;--------------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm6                          ; Convert k to int32
    vpmovsxdq   ymm0, xmm0                          ; Sign-extend to int64
    vpaddq      ymm0, ymm0, [rel bias_1023]         ; Add exponent bias
    vpsllq      ymm0, ymm0, 52                      ; Shift to exponent field
    
    ;--------------------------------------------------------------------------
    ; STEP 8: Combine and clamp: pp = 2^f * 2^k, pp = max(pp, 1e-300)
    ;--------------------------------------------------------------------------
    vmulpd      ymm7, ymm7, ymm0                    ; pp = 2^f * 2^k = exp(ln_pp)
    vmaxpd      ymm7, ymm7, [rel const_min_pp]      ; Floor at 1e-300
    
    ;--------------------------------------------------------------------------
    ; STEP 9: BOCPD update equations
    ;   r_pp = r_old * pp
    ;   growth = r_pp * (1-h)    → store at r_new[i+1] (run length increases)
    ;   change = r_pp * h        → accumulate to r0 (changepoint contribution)
    ;--------------------------------------------------------------------------
    vmulpd      ymm0, ymm7, ymm4                    ; ymm0 = r_old * pp = r_pp
    vmulpd      ymm1, ymm0, ymm10                   ; ymm1 = growth = r_pp * (1-h)
    vmulpd      ymm0, ymm0, ymm9                    ; ymm0 = change = r_pp * h
    
    vmovupd     [r13 + rsi*8 + 8], ymm1             ; Store growth at r_new[i+1:i+4]
    vaddpd      ymm12, ymm12, ymm0                  ; r0_accumulator += change
    
    ;--------------------------------------------------------------------------
    ; STEP 10: Branchless MAP tracking (find run length with max probability)
    ;
    ; vcmppd with predicate 14 = "greater than, ordered, quiet"
    ; vblendvpd: for each lane, select new value where mask bit is set
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm13, 14               ; mask where growth > max_growth_A
    vblendvpd   ymm13, ymm13, ymm1, ymm0            ; Update max_growth_A where larger
    vmovapd     ymm2, [rsp + 96]                    ; Load max_idx_A
    vblendvpd   ymm2, ymm2, ymm15, ymm0             ; Update indices where new max found
    vmovapd     [rsp + 96], ymm2                    ; Store updated max_idx_A
    
    ;--------------------------------------------------------------------------
    ; STEP 11: Truncation tracking (find last significant probability)
    ; Track highest index where growth > threshold
    ;--------------------------------------------------------------------------
    vcmppd      ymm0, ymm1, ymm11, 14               ; mask where growth > threshold
    vmovmskpd   eax, ymm0                           ; Extract 4-bit mask to scalar
    test        eax, eax                            ; Any lane above threshold?
    jz          .skip_trunc_a                       ; Skip if none
    
    ; Find highest set bit (highest valid index in this block)
    ; Mask bits: [bit3=i+3, bit2=i+2, bit1=i+1, bit0=i+0]
    bt          eax, 3                              ; Test bit 3?
    jc          .lv_a4                              ;   → last_valid = i+4
    bt          eax, 2                              ; Test bit 2?
    jc          .lv_a3                              ;   → last_valid = i+3
    bt          eax, 1                              ; Test bit 1?
    jc          .lv_a2                              ;   → last_valid = i+2
    lea         rbx, [rsi + 1]                      ; Only bit 0 → last_valid = i+1
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
    ; BLOCK B: Process elements [i+4, i+5, i+6, i+7]
    ;
    ; Same algorithm as Block A, but for the next interleaved block.
    ; Block B uses separate max_growth_B accumulator (ymm14) to avoid
    ; read-after-write dependencies with Block A's ymm13.
    ;
    ; Byte offset = (i/4 + 1) * 128 = i*32 + 128
    ;==========================================================================
    
    mov         rax, rsi
    shl         rax, 5
    add         rax, 128                            ; Next interleaved block
    
    ; Load Block B parameters
    vmovapd     ymm0, [r8 + rax]                    ; mu[i+4:i+7]
    vmovapd     ymm1, [r8 + rax + 32]               ; C1[i+4:i+7]
    vmovapd     ymm2, [r8 + rax + 64]               ; C2[i+4:i+7]
    vmovapd     ymm3, [r8 + rax + 96]               ; inv_ssn[i+4:i+7]
    vmovapd     ymm4, [r12 + rsi*8 + 32]            ; r_old[i+4:i+7]
    
    ; z² = (x - μ)², t = z² * inv_ssn
    vsubpd      ymm5, ymm8, ymm0
    vmulpd      ymm5, ymm5, ymm5
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
    
    ; exp(ln_pp): clamp, convert to base 2, split k and f
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
    vmulpd      ymm2, ymm5, ymm5                    ; f²
    vfmadd231pd ymm7, ymm2, ymm0
    vfmadd231pd ymm1, ymm2, [rel exp_c6]
    vmulpd      ymm2, ymm2, ymm2                    ; f⁴
    vfmadd231pd ymm7, ymm2, ymm1                    ; 2^f
    
    ; 2^k via IEEE-754 bit manipulation
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    
    ; pp = 2^f * 2^k, clamped
    vmulpd      ymm7, ymm7, ymm0
    vmaxpd      ymm7, ymm7, [rel const_min_pp]
    
    ; BOCPD update: growth and change
    vmulpd      ymm0, ymm7, ymm4                    ; r_pp = r_old * pp
    vmulpd      ymm1, ymm0, ymm10                   ; growth = r_pp * (1-h)
    vmulpd      ymm0, ymm0, ymm9                    ; change = r_pp * h
    
    vmovupd     [r13 + rsi*8 + 40], ymm1            ; Store at r_new[i+5:i+8] (offset: (i+4)*8+8)
    vaddpd      ymm12, ymm12, ymm0                  ; Accumulate to r0
    
    ; MAX tracking B (uses ymm14 to avoid dependency with Block A's ymm13)
    vcmppd      ymm0, ymm1, ymm14, 14               ; mask where growth > max_growth_B
    vblendvpd   ymm14, ymm14, ymm1, ymm0            ; Update max_growth_B
    vmovapd     ymm2, [rsp]                         ; Load max_idx_B
    vmovapd     ymm3, [rsp + 160]                   ; Load idx_vec_B
    vblendvpd   ymm2, ymm2, ymm3, ymm0              ; Update indices where new max
    vmovapd     [rsp], ymm2                         ; Store max_idx_B
    
    ; Truncation B: find highest valid index
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
    lea         rbx, [rsi + 5]                      ; Bit 0 only → i+5
    jmp         .skip_trunc_b
.lv_b8:
    lea         rbx, [rsi + 8]                      ; Bit 3 → i+8
    jmp         .skip_trunc_b
.lv_b7:
    lea         rbx, [rsi + 7]                      ; Bit 2 → i+7
    jmp         .skip_trunc_b
.lv_b6:
    lea         rbx, [rsi + 6]                      ; Bit 1 → i+6
.skip_trunc_b:
    
    ;--------------------------------------------------------------------------
    ; Update running index vectors: add 8 for next iteration
    ; This replaces expensive broadcast+add with simple vector add
    ;--------------------------------------------------------------------------
    vaddpd      ymm15, ymm15, [rel idx_increment]   ; idx_vec_A += 8
    vmovapd     ymm0, [rsp + 160]
    vaddpd      ymm0, ymm0, [rel idx_increment]     ; idx_vec_B += 8
    vmovapd     [rsp + 160], ymm0
    
    add         rsi, 8                              ; i += 8 (processed 8 elements)
    jmp         .loop
    
.loop_end:
    
    ;==========================================================================
    ; HORIZONTAL REDUCTIONS
    ;
    ; After the loop, we have:
    ;   ymm12 = 4 partial sums of changepoint probability
    ;   ymm13 = 4 max growth values from Block A
    ;   ymm14 = 4 max growth values from Block B
    ;   [rsp+96] = 4 indices for max in Block A
    ;   [rsp] = 4 indices for max in Block B
    ;   rbx = last_valid (highest index above threshold)
    ;
    ; Need to reduce these to scalar outputs.
    ;==========================================================================
    
    ;--------------------------------------------------------------------------
    ; Reduce r0 accumulator: sum 4 doubles → 1 double
    ; [a, b, c, d] → a+b+c+d
    ;--------------------------------------------------------------------------
    vextractf128 xmm0, ymm12, 1                     ; xmm0 = [c, d]
    vaddpd      xmm0, xmm0, xmm12                   ; xmm0 = [a+c, b+d]
    vhaddpd     xmm0, xmm0, xmm0                    ; xmm0 = [a+b+c+d, a+b+c+d]
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0                         ; *r0_out = sum
    
    ; Keep r0 as initial "best" for MAP comparison
    ; r0 represents run_length=0 (changepoint just occurred)
    vmovsd      xmm6, xmm0, xmm0                    ; xmm6 = current best value
    xor         r15, r15                            ; r15 = 0 (best index = 0 initially)
    
    ;--------------------------------------------------------------------------
    ; Reduce MAX across all 8 lanes to find global maximum
    ; Compare each lane against r0 to find overall MAP run length
    ;
    ; We use a scalar loop since SIMD horizontal max is complex
    ; and we only have 8 elements to check.
    ;--------------------------------------------------------------------------
    vmovapd     [rsp + 32], ymm13                   ; Save max_growth_A for scalar access
    vmovapd     [rsp + 64], ymm14                   ; Save max_growth_B for scalar access
    
    xor         rcx, rcx                            ; j = 0
.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done
    
    ; Check max_growth_A[j]
    vmovsd      xmm1, [rsp + 32 + rcx*8]            ; Load max_growth_A[j]
    vucomisd    xmm1, xmm6                          ; Compare with current best
    jbe         .check_b                            ; Skip if not greater
    vmovsd      xmm6, xmm1, xmm1                    ; New best value
    vmovsd      xmm2, [rsp + 96 + rcx*8]            ; Load max_idx_A[j]
    vcvttsd2si  r15, xmm2                           ; Convert to integer
    
.check_b:
    ; Check max_growth_B[j]
    vmovsd      xmm1, [rsp + 64 + rcx*8]            ; Load max_growth_B[j]
    vucomisd    xmm1, xmm6                          ; Compare with current best
    jbe         .next_j                             ; Skip if not greater
    vmovsd      xmm6, xmm1, xmm1                    ; New best value
    vmovsd      xmm2, [rsp + rcx*8]                 ; Load max_idx_B[j]
    vcvttsd2si  r15, xmm2                           ; Convert to integer
    
.next_j:
    inc         rcx                                 ; j++
    jmp         .reduce_loop
    
.reduce_done:
    ;--------------------------------------------------------------------------
    ; Store final outputs
    ;--------------------------------------------------------------------------
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6                         ; *max_growth_out = best value
    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15                          ; *max_idx_out = best index (MAP)
    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx                          ; *last_valid_out = truncation boundary
    
    ;==========================================================================
    ; EPILOGUE: Restore callee-saved registers and return
    ;==========================================================================
    mov         rsp, rbp                            ; Restore original stack pointer
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    
    ; vzeroupper: Clear upper 128 bits of all YMM registers
    ; Required when transitioning from AVX to SSE code to avoid
    ; performance penalty from "AVX-SSE transition stall"
    vzeroupper
    
    ret
