; ============================================================================
; BOCPD ULTRA — AVX2 MEGA-KERNEL V3.3 (Fast Direct-Register ABI)
; ============================================================================
;
; This kernel receives parameters directly in registers rather than via a
; struct pointer. This eliminates one level of indirection but requires a
; C wrapper to set up the registers correctly.
;
; =============================================================================
; PERFORMANCE COMPARISON
; =============================================================================
;
; Struct-pointer ABI (V3.1/V3.2):
;   - One pointer load to get args struct
;   - Individual loads from struct fields
;   - Simpler C calling convention
;   - ~3M obs/sec
;
; Direct-register ABI (this version):
;   - No struct indirection
;   - Saves ~2-3 cycles per call
;   - Requires C wrapper for register setup
;   - ~3.05M obs/sec (marginal improvement)
;
; VERDICT: The performance gain is minimal (~1-2%). Use this only if you
; need every last cycle. The struct-pointer version is easier to maintain.
;
; =============================================================================
; CALLING CONVENTION
; =============================================================================
;
; This file provides TWO entry points:
;
; WINDOWS x64 (bocpd_fast_avx2_win):
;   RCX = params (interleaved superblocks pointer)
;   RDX = r_old (input probability array)
;   R8  = r_new (output probability array)  
;   R9  = n_padded (number of elements, padded to 8)
;   XMM0 = x (observation)
;   XMM1 = h (hazard rate)
;   XMM2 = one_minus_h (1 - h)
;   XMM3 = threshold (truncation threshold)
;   Stack [RSP+40] = r0_out (pointer to output r0)
;   Stack [RSP+48] = max_growth_out (pointer to output max growth)
;   Stack [RSP+56] = max_idx_out (pointer to output max index)
;   Stack [RSP+64] = last_valid_out (pointer to output last valid)
;
; LINUX/MACOS System V (bocpd_fast_avx2_sysv):
;   RDI = params
;   RSI = r_old
;   RDX = r_new
;   RCX = n_padded
;   XMM0 = x
;   XMM1 = h
;   XMM2 = one_minus_h
;   XMM3 = threshold
;   R8  = r0_out
;   R9  = max_growth_out
;   Stack [RSP+8]  = max_idx_out
;   Stack [RSP+16] = last_valid_out
;
; =============================================================================
; C WRAPPER EXAMPLE (Windows)
; =============================================================================
;
; extern void bocpd_fast_avx2_win(
;     const double *params,      // RCX
;     const double *r_old,       // RDX  
;     double *r_new,             // R8
;     size_t n_padded,           // R9
;     double x,                  // XMM0
;     double h,                  // XMM1
;     double one_minus_h,        // XMM2
;     double threshold,          // XMM3
;     double *r0_out,            // Stack
;     double *max_growth_out,    // Stack
;     size_t *max_idx_out,       // Stack
;     size_t *last_valid_out     // Stack
; );
;
; =============================================================================
; REGISTER ALLOCATION (during main loop)
; =============================================================================
;
; PRESERVED (callee-saved, hold values across entire function):
;   RBP = frame pointer (for stack restoration)
;   RBX = last_valid (truncation tracking)
;   R12 = n_padded (loop bound) 
;   R13 = params pointer
;   R14 = loop counter (i)
;   R15 = best_idx (MAP tracking)
;
; PARAMETERS (loaded once, used throughout):
;   YMM15 = x (broadcast observation)
;   YMM14 = h (broadcast hazard)
;   YMM13 = 1-h (broadcast continuation probability)
;   YMM12 = threshold (broadcast truncation threshold)
;
; ACCUMULATORS (persist across loop iterations):
;   YMM11 = r0 accumulator (sum of changepoint contributions)
;   YMM10 = max_growth_A (running max for Block A)
;   YMM9  = max_growth_B (running max for Block B)
;
; SCRATCH (reused freely within each iteration):
;   RAX, RCX, RDX, R8, R9, R10, R11
;   YMM0-YMM8
;
; STACK SLOTS:
;   [RSP + STK_IDX_VEC_A]   = current index vector for Block A
;   [RSP + STK_IDX_VEC_B]   = current index vector for Block B
;   [RSP + STK_MAX_IDX_A]   = indices where Block A achieved max
;   [RSP + STK_MAX_IDX_B]   = indices where Block B achieved max
;   [RSP + STK_R_OLD_B]     = spilled r_old_B (ymm9 conflict)
;   [RSP + STK_R_OLD_PTR]   = saved r_old pointer
;   [RSP + STK_R_NEW_PTR]   = saved r_new pointer
;   [RSP + STK_OUTPUTS]     = saved output pointers
;
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

; log1p(t) Taylor coefficients: t - t²/2 + t³/3 - t⁴/4 + t⁵/5 - t⁶/6
log1p_c2:       dq -0.5, -0.5, -0.5, -0.5
log1p_c3:       dq 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333
log1p_c4:       dq -0.25, -0.25, -0.25, -0.25
log1p_c5:       dq 0.2, 0.2, 0.2, 0.2
log1p_c6:       dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

; 2^f polynomial coefficients (Estrin evaluation)
exp_c1:         dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
exp_c2:         dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
exp_c3:         dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
exp_c4:         dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:         dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:         dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

; Index tracking vectors
idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

; ============================================================================
; STACK FRAME LAYOUT
; ============================================================================
;
; After alignment, RSP points to:
;   [RSP + 0]     idx_vec_A      (32 bytes) - current indices for Block A
;   [RSP + 32]    idx_vec_B      (32 bytes) - current indices for Block B  
;   [RSP + 64]    max_idx_A      (32 bytes) - best indices for Block A
;   [RSP + 96]    max_idx_B      (32 bytes) - best indices for Block B
;   [RSP + 128]   max_growth_A   (32 bytes) - for final reduction
;   [RSP + 160]   max_growth_B   (32 bytes) - for final reduction
;   [RSP + 192]   r_old_B        (32 bytes) - spilled because ymm9 conflict
;   [RSP + 224]   r_old_ptr      (8 bytes)  - saved pointer
;   [RSP + 232]   r_new_ptr      (8 bytes)  - saved pointer
;   [RSP + 240]   r0_out         (8 bytes)  - output pointer
;   [RSP + 248]   max_growth_out (8 bytes)  - output pointer
;   [RSP + 256]   max_idx_out    (8 bytes)  - output pointer
;   [RSP + 264]   last_valid_out (8 bytes)  - output pointer
;   [RSP + 272]   padding        (48 bytes) - alignment safety
;
; Total: 320 bytes
; ============================================================================

%define STK_IDX_VEC_A       0
%define STK_IDX_VEC_B       32
%define STK_MAX_IDX_A       64
%define STK_MAX_IDX_B       96
%define STK_MAX_GROWTH_A    128
%define STK_MAX_GROWTH_B    160
%define STK_R_OLD_B         192
%define STK_R_OLD_PTR       224
%define STK_R_NEW_PTR       232
%define STK_R0_OUT          240
%define STK_MAX_GROWTH_OUT  248
%define STK_MAX_IDX_OUT     256
%define STK_LAST_VALID_OUT  264
%define STACK_FRAME         320

; ============================================================================
; WINDOWS x64 ENTRY POINT
; ============================================================================

section .text
global bocpd_fast_avx2_win
global bocpd_fast_avx2_sysv

bocpd_fast_avx2_win:
    ; -----------------------------------------------------------------------
    ; PROLOGUE - Save Windows non-volatile registers
    ; -----------------------------------------------------------------------
    push    rbp
    push    rbx
    push    rdi
    push    rsi
    push    r12
    push    r13
    push    r14
    push    r15

    ; Save XMM6-XMM15 (Windows ABI requirement)
    sub     rsp, 160
    vmovdqu [rsp +   0], xmm6
    vmovdqu [rsp +  16], xmm7
    vmovdqu [rsp +  32], xmm8
    vmovdqu [rsp +  48], xmm9
    vmovdqu [rsp +  64], xmm10
    vmovdqu [rsp +  80], xmm11
    vmovdqu [rsp +  96], xmm12
    vmovdqu [rsp + 112], xmm13
    vmovdqu [rsp + 128], xmm14
    vmovdqu [rsp + 144], xmm15

    ; Save frame pointer for epilogue
    mov     rbp, rsp

    ; Allocate and align stack frame
    sub     rsp, STACK_FRAME
    and     rsp, -32

    ; -----------------------------------------------------------------------
    ; UNPACK WINDOWS ARGUMENTS
    ; -----------------------------------------------------------------------
    ; Register args: RCX=params, RDX=r_old, R8=r_new, R9=n_padded
    ; XMM args: XMM0=x, XMM1=h, XMM2=one_minus_h, XMM3=threshold
    ; Stack args (relative to RBP before our allocations):
    ;   [RBP + 160 + 64 + 40] = r0_out
    ;   [RBP + 160 + 64 + 48] = max_growth_out  
    ;   [RBP + 160 + 64 + 56] = max_idx_out
    ;   [RBP + 160 + 64 + 64] = last_valid_out
    ; (160 = XMM saves, 64 = GPR pushes)

    mov     r13, rcx                    ; r13 = params (preserved)
    mov     [rsp + STK_R_OLD_PTR], rdx  ; save r_old pointer
    mov     [rsp + STK_R_NEW_PTR], r8   ; save r_new pointer
    mov     r12, r9                     ; r12 = n_padded (preserved)

    ; Load stack arguments (account for pushes + XMM saves)
    %define WIN_STACK_OFFSET (160 + 64 + 32)  ; XMM + GPR + shadow space
    mov     rax, [rbp + WIN_STACK_OFFSET + 8]
    mov     [rsp + STK_R0_OUT], rax
    mov     rax, [rbp + WIN_STACK_OFFSET + 16]
    mov     [rsp + STK_MAX_GROWTH_OUT], rax
    mov     rax, [rbp + WIN_STACK_OFFSET + 24]
    mov     [rsp + STK_MAX_IDX_OUT], rax
    mov     rax, [rbp + WIN_STACK_OFFSET + 32]
    mov     [rsp + STK_LAST_VALID_OUT], rax

    ; Broadcast scalar parameters to YMM registers
    vbroadcastsd ymm15, xmm0            ; x
    vbroadcastsd ymm14, xmm1            ; h
    vbroadcastsd ymm13, xmm2            ; 1-h
    vbroadcastsd ymm12, xmm3            ; threshold

    jmp     .common_init

; ============================================================================
; COMMON INITIALIZATION (Windows only - SysV has its own complete kernel below)
; ============================================================================

.common_init:
    ; Zero accumulators
    vxorpd  ymm11, ymm11, ymm11         ; r0 accumulator
    vxorpd  ymm10, ymm10, ymm10         ; max_growth_A
    vxorpd  ymm9,  ymm9,  ymm9          ; max_growth_B

    ; Initialize index vectors
    vmovapd ymm0, [rel idx_init_a]
    vmovapd [rsp + STK_IDX_VEC_A], ymm0

    vmovapd ymm0, [rel idx_init_b]
    vmovapd [rsp + STK_IDX_VEC_B], ymm0

    ; Zero max index trackers
    vxorpd  ymm0, ymm0, ymm0
    vmovapd [rsp + STK_MAX_IDX_A], ymm0
    vmovapd [rsp + STK_MAX_IDX_B], ymm0

    ; Initialize loop variables (in callee-saved registers!)
    xor     r14, r14                    ; r14 = i = 0 (loop counter)
    xor     rbx, rbx                    ; rbx = last_valid = 0
    xor     r15, r15                    ; r15 = best_idx = 0

; ============================================================================
; MAIN LOOP - 8 elements per iteration (2 blocks of 4)
; ============================================================================

.loop_start:
    cmp     r14, r12                    ; if i >= n_padded, exit
    jge     .loop_end

    ; Load pointers (they're in stack slots now)
    mov     rsi, [rsp + STK_R_OLD_PTR]  ; rsi = r_old
    mov     rdi, [rsp + STK_R_NEW_PTR]  ; rdi = r_new

    ; -----------------------------------------------------------------------
    ; COMPUTE SUPERBLOCK OFFSETS
    ; Block A: elements [i, i+1, i+2, i+3]
    ; Block B: elements [i+4, i+5, i+6, i+7]
    ; -----------------------------------------------------------------------
    mov     rax, r14
    shr     rax, 2
    shl     rax, 8                      ; blockA_offset = (i/4) * 256

    mov     rdx, r14
    add     rdx, 4
    shr     rdx, 2
    shl     rdx, 8                      ; blockB_offset = ((i+4)/4) * 256

    ; -----------------------------------------------------------------------
    ; LOAD BLOCK A PARAMETERS
    ; -----------------------------------------------------------------------
    vmovapd ymm0, [r13 + rax + 0]       ; mu_A
    vmovapd ymm1, [r13 + rax + 32]      ; C1_A
    vmovapd ymm2, [r13 + rax + 64]      ; C2_A
    vmovapd ymm3, [r13 + rax + 96]      ; inv_ssn_A
    vmovapd ymm8, [rsi + r14*8]         ; r_old_A

    ; -----------------------------------------------------------------------
    ; LOAD BLOCK B PARAMETERS
    ; -----------------------------------------------------------------------
    vmovapd ymm4, [r13 + rdx + 0]       ; mu_B
    vmovapd ymm5, [r13 + rdx + 32]      ; C1_B
    vmovapd ymm6, [r13 + rdx + 64]      ; C2_B
    vmovapd ymm7, [r13 + rdx + 96]      ; inv_ssn_B

    ; Spill r_old_B to stack (can't keep in register, need ymm0-7 for computation)
    vmovapd ymm0, [rsi + r14*8 + 32]    ; r_old_B
    vmovapd [rsp + STK_R_OLD_B], ymm0

    ; Reload mu_A (was clobbered)
    vmovapd ymm0, [r13 + rax + 0]

    ; -----------------------------------------------------------------------
    ; STUDENT-T COMPUTATION - BLOCK A
    ; t = (x - μ)² × inv_ssn
    ; -----------------------------------------------------------------------
    vsubpd  ymm0, ymm15, ymm0           ; z_A = x - mu_A
    vmulpd  ymm0, ymm0, ymm0            ; z²_A
    vmulpd  ymm0, ymm0, ymm3            ; t_A = z²_A × inv_ssn_A

    ; -----------------------------------------------------------------------
    ; STUDENT-T COMPUTATION - BLOCK B
    ; -----------------------------------------------------------------------
    vsubpd  ymm4, ymm15, ymm4           ; z_B = x - mu_B
    vmulpd  ymm4, ymm4, ymm4            ; z²_B
    vmulpd  ymm4, ymm4, ymm7            ; t_B = z²_B × inv_ssn_B

    ; -----------------------------------------------------------------------
    ; LOG1P POLYNOMIAL - BLOCK A (Horner's method)
    ; log1p(t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
    ; -----------------------------------------------------------------------
    vmovapd ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd  ymm3, ymm3, ymm0            ; log1p_A

    ; -----------------------------------------------------------------------
    ; LOG1P POLYNOMIAL - BLOCK B
    ; -----------------------------------------------------------------------
    vmovapd ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd  ymm7, ymm7, ymm4            ; log1p_B

    ; -----------------------------------------------------------------------
    ; LN_PP = C1 - C2 × log1p
    ; -----------------------------------------------------------------------
    vfnmadd231pd ymm1, ymm2, ymm3       ; ln_pp_A = C1_A - C2_A × log1p_A
    vfnmadd231pd ymm5, ymm6, ymm7       ; ln_pp_B = C1_B - C2_B × log1p_B

    ; -----------------------------------------------------------------------
    ; EXP PREPARATION - BLOCK A
    ; Clamp, convert to base-2
    ; -----------------------------------------------------------------------
    vmaxpd  ymm1, ymm1, [rel exp_min_x]
    vminpd  ymm1, ymm1, [rel exp_max_x]
    vmulpd  ymm0, ymm1, [rel exp_inv_ln2]   ; y_A = ln_pp × log₂(e)
    vroundpd ymm2, ymm0, 0                  ; k_A = round(y_A)
    vsubpd  ymm0, ymm0, ymm2                ; f_A = y_A - k_A

    ; -----------------------------------------------------------------------
    ; EXP PREPARATION - BLOCK B
    ; -----------------------------------------------------------------------
    vmaxpd  ymm5, ymm5, [rel exp_min_x]
    vminpd  ymm5, ymm5, [rel exp_max_x]
    vmulpd  ymm4, ymm5, [rel exp_inv_ln2]   ; y_B
    vroundpd ymm6, ymm4, 0                  ; k_B
    vsubpd  ymm4, ymm4, ymm6                ; f_B

    ; -----------------------------------------------------------------------
    ; ESTRIN POLYNOMIAL - 2^f BLOCK A
    ; -----------------------------------------------------------------------
    vmulpd  ymm3, ymm0, ymm0                ; f²_A

    vmovapd ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]    ; p01 = 1 + f×c1

    vmovapd ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]    ; p23 = c2 + f×c3
    vfmadd231pd ymm1, ymm3, ymm7            ; q0123 = p01 + f²×p23

    vmovapd ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]    ; p45 = c4 + f×c5
    vfmadd231pd ymm7, ymm3, [rel exp_c6]    ; q456 = p45 + f²×c6

    vmulpd  ymm3, ymm3, ymm3                ; f⁴_A
    vfmadd231pd ymm1, ymm3, ymm7            ; 2^f_A

    ; -----------------------------------------------------------------------
    ; ESTRIN POLYNOMIAL - 2^f BLOCK B
    ; -----------------------------------------------------------------------
    vmulpd  ymm3, ymm4, ymm4                ; f²_B

    vmovapd ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]

    vmovapd ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]
    vfmadd231pd ymm5, ymm3, ymm7

    vmovapd ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]

    vmulpd  ymm3, ymm3, ymm3                ; f⁴_B
    vfmadd231pd ymm5, ymm3, ymm7            ; 2^f_B

    ; -----------------------------------------------------------------------
    ; 2^k RECONSTRUCTION - BLOCK A
    ; -----------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm2                  ; k → int32
    vpmovsxdq   ymm0, xmm0                  ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel bias_1023] ; add exponent bias
    vpsllq      ymm0, ymm0, 52              ; shift to exponent field
    vmulpd      ymm1, ymm1, ymm0            ; pp_A = 2^f × 2^k
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    ; -----------------------------------------------------------------------
    ; 2^k RECONSTRUCTION - BLOCK B
    ; -----------------------------------------------------------------------
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0            ; pp_B
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; -----------------------------------------------------------------------
    ; BOCPD UPDATE - BLOCK A
    ; growth = r_old × pp × (1-h)
    ; change = r_old × pp × h  → accumulate to r0
    ; -----------------------------------------------------------------------
    vmulpd  ymm0, ymm8, ymm1                ; r_pp_A = r_old_A × pp_A
    vmulpd  ymm2, ymm0, ymm13               ; growth_A = r_pp_A × (1-h)
    vmulpd  ymm0, ymm0, ymm14               ; change_A = r_pp_A × h
    vaddpd  ymm11, ymm11, ymm0              ; r0_acc += change_A

    ; Store growth_A at r_new[i+1..i+4]
    vmovupd [rdi + r14*8 + 8], ymm2

    ; -----------------------------------------------------------------------
    ; BOCPD UPDATE - BLOCK B
    ; -----------------------------------------------------------------------
    vmovapd ymm0, [rsp + STK_R_OLD_B]       ; reload r_old_B
    vmulpd  ymm0, ymm0, ymm5                ; r_pp_B
    vmulpd  ymm3, ymm0, ymm13               ; growth_B
    vmulpd  ymm0, ymm0, ymm14               ; change_B
    vaddpd  ymm11, ymm11, ymm0              ; r0_acc += change_B

    ; Store growth_B at r_new[i+5..i+8]
    vmovupd [rdi + r14*8 + 40], ymm3

    ; -----------------------------------------------------------------------
    ; MAX TRACKING - BLOCK A
    ; -----------------------------------------------------------------------
    vcmppd  ymm0, ymm2, ymm10, 14           ; mask = growth_A > max_A?
    vblendvpd ymm10, ymm10, ymm2, ymm0      ; update max where mask set

    vmovapd ymm1, [rsp + STK_MAX_IDX_A]     ; current best indices
    vmovapd ymm4, [rsp + STK_IDX_VEC_A]     ; this iteration's indices
    vblendvpd ymm1, ymm1, ymm4, ymm0        ; update indices where mask set
    vmovapd [rsp + STK_MAX_IDX_A], ymm1

    ; -----------------------------------------------------------------------
    ; MAX TRACKING - BLOCK B
    ; -----------------------------------------------------------------------
    vcmppd  ymm0, ymm3, ymm9, 14            ; mask = growth_B > max_B?
    vblendvpd ymm9, ymm9, ymm3, ymm0

    vmovapd ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd ymm1, ymm1, ymm4, ymm0
    vmovapd [rsp + STK_MAX_IDX_B], ymm1

    ; -----------------------------------------------------------------------
    ; DYNAMIC TRUNCATION - BLOCK A
    ; Find highest lane where growth > threshold
    ; -----------------------------------------------------------------------
    vcmppd  ymm0, ymm2, ymm12, 14           ; mask = growth_A > threshold?
    vmovmskpd eax, ymm0                     ; extract mask to 4-bit int
    test    eax, eax
    jz      .skip_trunc_A

    bsr     ecx, eax                        ; ecx = highest set bit (0-3)
    lea     rbx, [r14 + rcx + 1]            ; last_valid = i + lane + 1

.skip_trunc_A:

    ; -----------------------------------------------------------------------
    ; DYNAMIC TRUNCATION - BLOCK B
    ; -----------------------------------------------------------------------
    vcmppd  ymm0, ymm3, ymm12, 14
    vmovmskpd eax, ymm0
    test    eax, eax
    jz      .skip_trunc_B

    bsr     ecx, eax
    lea     rbx, [r14 + rcx + 5]            ; last_valid = (i+4) + lane + 1

.skip_trunc_B:

    ; -----------------------------------------------------------------------
    ; UPDATE INDEX VECTORS
    ; -----------------------------------------------------------------------
    vmovapd ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd  ymm0, ymm0, [rel idx_increment]
    vmovapd [rsp + STK_IDX_VEC_A], ymm0

    vmovapd ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd  ymm0, ymm0, [rel idx_increment]
    vmovapd [rsp + STK_IDX_VEC_B], ymm0

    ; -----------------------------------------------------------------------
    ; LOOP ADVANCE
    ; -----------------------------------------------------------------------
    add     r14, 8
    jmp     .loop_start

; ============================================================================
; LOOP END - REDUCTIONS AND OUTPUT
; ============================================================================

.loop_end:

    ; -----------------------------------------------------------------------
    ; REDUCE R0 ACCUMULATOR
    ; ymm11 = [a, b, c, d] → scalar sum
    ; -----------------------------------------------------------------------
    vextractf128 xmm0, ymm11, 1             ; xmm0 = [c, d]
    vaddpd  xmm0, xmm0, xmm11               ; xmm0 = [a+c, b+d]
    vunpckhpd xmm1, xmm0, xmm0              ; xmm1 = [b+d, b+d]
    vaddsd  xmm0, xmm0, xmm1                ; xmm0 = a+b+c+d = r0

    ; Save r0 for later (we need it for MAP comparison)
    vmovsd  xmm8, xmm0, xmm0                ; xmm8 = r0 (preserved)

    ; -----------------------------------------------------------------------
    ; MAP REDUCTION - Find overall max
    ; Compare r0 against all 8 lanes of max_growth
    ; -----------------------------------------------------------------------
    vmovsd  xmm6, xmm0, xmm0                ; best_val = r0
    xor     r15, r15                        ; best_idx = 0

    ; Spill max vectors for scalar access
    vmovapd [rsp + STK_MAX_GROWTH_A], ymm10
    vmovapd [rsp + STK_MAX_GROWTH_B], ymm9

    xor     ecx, ecx                        ; lane counter

.reduce_loop:
    cmp     ecx, 4
    jge     .reduce_done

    ; Check Block A lane
    vmovsd  xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd xmm1, xmm6
    jbe     .check_B

    vmovsd  xmm6, xmm1, xmm1                ; update best_val
    vmovsd  xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si r15, xmm2                    ; update best_idx

.check_B:
    ; Check Block B lane
    vmovsd  xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd xmm1, xmm6
    jbe     .next_lane

    vmovsd  xmm6, xmm1, xmm1
    vmovsd  xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si r15, xmm2

.next_lane:
    inc     ecx
    jmp     .reduce_loop

.reduce_done:

    ; -----------------------------------------------------------------------
    ; WRITE OUTPUTS
    ; -----------------------------------------------------------------------
    mov     rax, [rsp + STK_R0_OUT]
    vmovsd  [rax], xmm8                     ; *r0_out = r0

    mov     rax, [rsp + STK_MAX_GROWTH_OUT]
    vmovsd  [rax], xmm6                     ; *max_growth_out = best_val

    mov     rax, [rsp + STK_MAX_IDX_OUT]
    mov     [rax], r15                      ; *max_idx_out = best_idx

    mov     rax, [rsp + STK_LAST_VALID_OUT]
    mov     [rax], rbx                      ; *last_valid_out = last_valid

; ============================================================================
; EPILOGUE - Windows x64
; ============================================================================

.epilogue:
    mov     rsp, rbp                        ; restore stack pointer

    ; Restore XMM6-XMM15 (Windows ABI requirement)
    vmovdqu xmm6,  [rsp +   0]
    vmovdqu xmm7,  [rsp +  16]
    vmovdqu xmm8,  [rsp +  32]
    vmovdqu xmm9,  [rsp +  48]
    vmovdqu xmm10, [rsp +  64]
    vmovdqu xmm11, [rsp +  80]
    vmovdqu xmm12, [rsp +  96]
    vmovdqu xmm13, [rsp + 112]
    vmovdqu xmm14, [rsp + 128]
    vmovdqu xmm15, [rsp + 144]
    add     rsp, 160

    ; Restore GPRs (reverse order of push)
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rdi
    pop     rbx
    pop     rbp

    vzeroupper
    ret

; ============================================================================
; LINUX/MACOS System V - COMPLETE SEPARATE IMPLEMENTATION
; ============================================================================
;
; This is a complete duplicate of the kernel with System V ABI.
; Duplicating ~400 lines of loop code is unfortunate but necessary for
; correct prologue/epilogue pairing without complex macro gymnastics.
;
; The alternative would be storing an ABI flag, but that adds runtime
; overhead to what should be a tight loop.
; ============================================================================

bocpd_fast_avx2_sysv:
    ; -----------------------------------------------------------------------
    ; PROLOGUE - System V (NO XMM saves required!)
    ; -----------------------------------------------------------------------
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbp, rsp
    sub     rsp, STACK_FRAME
    and     rsp, -32

    ; -----------------------------------------------------------------------
    ; UNPACK SYSTEM V ARGUMENTS
    ; RDI=params, RSI=r_old, RDX=r_new, RCX=n_padded
    ; R8=r0_out, R9=max_growth_out
    ; XMM0=x, XMM1=h, XMM2=one_minus_h, XMM3=threshold
    ; Stack: [RBP+56]=max_idx_out, [RBP+64]=last_valid_out
    ; -----------------------------------------------------------------------
    mov     r13, rdi
    mov     [rsp + STK_R_OLD_PTR], rsi
    mov     [rsp + STK_R_NEW_PTR], rdx
    mov     r12, rcx

    mov     [rsp + STK_R0_OUT], r8
    mov     [rsp + STK_MAX_GROWTH_OUT], r9
    mov     rax, [rbp + 56]
    mov     [rsp + STK_MAX_IDX_OUT], rax
    mov     rax, [rbp + 64]
    mov     [rsp + STK_LAST_VALID_OUT], rax

    vbroadcastsd ymm15, xmm0
    vbroadcastsd ymm14, xmm1
    vbroadcastsd ymm13, xmm2
    vbroadcastsd ymm12, xmm3

    ; Initialize accumulators
    vxorpd  ymm11, ymm11, ymm11
    vxorpd  ymm10, ymm10, ymm10
    vxorpd  ymm9,  ymm9,  ymm9

    vmovapd ymm0, [rel idx_init_a]
    vmovapd [rsp + STK_IDX_VEC_A], ymm0
    vmovapd ymm0, [rel idx_init_b]
    vmovapd [rsp + STK_IDX_VEC_B], ymm0

    vxorpd  ymm0, ymm0, ymm0
    vmovapd [rsp + STK_MAX_IDX_A], ymm0
    vmovapd [rsp + STK_MAX_IDX_B], ymm0

    xor     r14, r14
    xor     rbx, rbx
    xor     r15, r15

; -----------------------------------------------------------------------
; MAIN LOOP - System V version (identical math, different registers)
; -----------------------------------------------------------------------

.sysv_loop_start:
    cmp     r14, r12
    jge     .sysv_loop_end

    mov     rsi, [rsp + STK_R_OLD_PTR]
    mov     rdi, [rsp + STK_R_NEW_PTR]

    mov     rax, r14
    shr     rax, 2
    shl     rax, 8

    mov     rdx, r14
    add     rdx, 4
    shr     rdx, 2
    shl     rdx, 8

    ; Load Block A
    vmovapd ymm0, [r13 + rax + 0]
    vmovapd ymm1, [r13 + rax + 32]
    vmovapd ymm2, [r13 + rax + 64]
    vmovapd ymm3, [r13 + rax + 96]
    vmovapd ymm8, [rsi + r14*8]

    ; Load Block B
    vmovapd ymm4, [r13 + rdx + 0]
    vmovapd ymm5, [r13 + rdx + 32]
    vmovapd ymm6, [r13 + rdx + 64]
    vmovapd ymm7, [r13 + rdx + 96]
    vmovapd ymm0, [rsi + r14*8 + 32]
    vmovapd [rsp + STK_R_OLD_B], ymm0
    vmovapd ymm0, [r13 + rax + 0]

    ; Student-t A
    vsubpd  ymm0, ymm15, ymm0
    vmulpd  ymm0, ymm0, ymm0
    vmulpd  ymm0, ymm0, ymm3

    ; Student-t B
    vsubpd  ymm4, ymm15, ymm4
    vmulpd  ymm4, ymm4, ymm4
    vmulpd  ymm4, ymm4, ymm7

    ; log1p A
    vmovapd ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd  ymm3, ymm3, ymm0

    ; log1p B
    vmovapd ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd  ymm7, ymm7, ymm4

    ; ln_pp
    vfnmadd231pd ymm1, ymm2, ymm3
    vfnmadd231pd ymm5, ymm6, ymm7

    ; exp prep A
    vmaxpd  ymm1, ymm1, [rel exp_min_x]
    vminpd  ymm1, ymm1, [rel exp_max_x]
    vmulpd  ymm0, ymm1, [rel exp_inv_ln2]
    vroundpd ymm2, ymm0, 0
    vsubpd  ymm0, ymm0, ymm2

    ; exp prep B
    vmaxpd  ymm5, ymm5, [rel exp_min_x]
    vminpd  ymm5, ymm5, [rel exp_max_x]
    vmulpd  ymm4, ymm5, [rel exp_inv_ln2]
    vroundpd ymm6, ymm4, 0
    vsubpd  ymm4, ymm4, ymm6

    ; Estrin 2^f A
    vmulpd  ymm3, ymm0, ymm0
    vmovapd ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]
    vmovapd ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]
    vfmadd231pd ymm1, ymm3, ymm7
    vmovapd ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd  ymm3, ymm3, ymm3
    vfmadd231pd ymm1, ymm3, ymm7

    ; Estrin 2^f B
    vmulpd  ymm3, ymm4, ymm4
    vmovapd ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]
    vmovapd ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]
    vfmadd231pd ymm5, ymm3, ymm7
    vmovapd ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]
    vmulpd  ymm3, ymm3, ymm3
    vfmadd231pd ymm5, ymm3, ymm7

    ; 2^k A
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    ; 2^k B
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; BOCPD update A
    vmulpd  ymm0, ymm8, ymm1
    vmulpd  ymm2, ymm0, ymm13
    vmulpd  ymm0, ymm0, ymm14
    vaddpd  ymm11, ymm11, ymm0
    vmovupd [rdi + r14*8 + 8], ymm2

    ; BOCPD update B
    vmovapd ymm0, [rsp + STK_R_OLD_B]
    vmulpd  ymm0, ymm0, ymm5
    vmulpd  ymm3, ymm0, ymm13
    vmulpd  ymm0, ymm0, ymm14
    vaddpd  ymm11, ymm11, ymm0
    vmovupd [rdi + r14*8 + 40], ymm3

    ; Max tracking A
    vcmppd  ymm0, ymm2, ymm10, 14
    vblendvpd ymm10, ymm10, ymm2, ymm0
    vmovapd ymm1, [rsp + STK_MAX_IDX_A]
    vmovapd ymm4, [rsp + STK_IDX_VEC_A]
    vblendvpd ymm1, ymm1, ymm4, ymm0
    vmovapd [rsp + STK_MAX_IDX_A], ymm1

    ; Max tracking B
    vcmppd  ymm0, ymm3, ymm9, 14
    vblendvpd ymm9, ymm9, ymm3, ymm0
    vmovapd ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd ymm1, ymm1, ymm4, ymm0
    vmovapd [rsp + STK_MAX_IDX_B], ymm1

    ; Truncation A
    vcmppd  ymm0, ymm2, ymm12, 14
    vmovmskpd eax, ymm0
    test    eax, eax
    jz      .sysv_skip_trunc_A
    bsr     ecx, eax
    lea     rbx, [r14 + rcx + 1]
.sysv_skip_trunc_A:

    ; Truncation B
    vcmppd  ymm0, ymm3, ymm12, 14
    vmovmskpd eax, ymm0
    test    eax, eax
    jz      .sysv_skip_trunc_B
    bsr     ecx, eax
    lea     rbx, [r14 + rcx + 5]
.sysv_skip_trunc_B:

    ; Update index vectors
    vmovapd ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd  ymm0, ymm0, [rel idx_increment]
    vmovapd [rsp + STK_IDX_VEC_A], ymm0
    vmovapd ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd  ymm0, ymm0, [rel idx_increment]
    vmovapd [rsp + STK_IDX_VEC_B], ymm0

    add     r14, 8
    jmp     .sysv_loop_start

; -----------------------------------------------------------------------
; LOOP END - System V version
; -----------------------------------------------------------------------

.sysv_loop_end:
    ; Reduce r0
    vextractf128 xmm0, ymm11, 1
    vaddpd  xmm0, xmm0, xmm11
    vunpckhpd xmm1, xmm0, xmm0
    vaddsd  xmm0, xmm0, xmm1
    vmovsd  xmm8, xmm0, xmm0

    ; MAP reduction
    vmovsd  xmm6, xmm0, xmm0
    xor     r15, r15

    vmovapd [rsp + STK_MAX_GROWTH_A], ymm10
    vmovapd [rsp + STK_MAX_GROWTH_B], ymm9

    xor     ecx, ecx

.sysv_reduce_loop:
    cmp     ecx, 4
    jge     .sysv_reduce_done

    vmovsd  xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd xmm1, xmm6
    jbe     .sysv_check_B
    vmovsd  xmm6, xmm1, xmm1
    vmovsd  xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si r15, xmm2

.sysv_check_B:
    vmovsd  xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd xmm1, xmm6
    jbe     .sysv_next_lane
    vmovsd  xmm6, xmm1, xmm1
    vmovsd  xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si r15, xmm2

.sysv_next_lane:
    inc     ecx
    jmp     .sysv_reduce_loop

.sysv_reduce_done:
    ; Write outputs
    mov     rax, [rsp + STK_R0_OUT]
    vmovsd  [rax], xmm8
    mov     rax, [rsp + STK_MAX_GROWTH_OUT]
    vmovsd  [rax], xmm6
    mov     rax, [rsp + STK_MAX_IDX_OUT]
    mov     [rax], r15
    mov     rax, [rsp + STK_LAST_VALID_OUT]
    mov     [rax], rbx

; -----------------------------------------------------------------------
; EPILOGUE - System V (much simpler - no XMM restore needed!)
; -----------------------------------------------------------------------

    mov     rsp, rbp

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp

    vzeroupper
    ret

; ============================================================================
; END OF FILE
; ============================================================================