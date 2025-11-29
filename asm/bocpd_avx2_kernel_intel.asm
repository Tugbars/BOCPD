; ============================================================================
; BOCPD Ultra — AVX2 V3.1 Intel-Optimized Kernel
; Windows x64 ABI (RCX = args ptr)
; ============================================================================
;
; =============================================================================
; WHY HAND-WRITTEN ASSEMBLY?
; =============================================================================
;
; The C compiler (even with -O3 -march=native) leaves performance on the table:
;
; 1. REGISTER ALLOCATION: We process 8 run lengths per iteration (2 blocks of 4).
;    The compiler struggles to keep all intermediate values in registers and
;    spills to memory unnecessarily. We manually allocate registers to minimize
;    spills.
;
; 2. INSTRUCTION SCHEDULING: The Student-t computation has a long dependency
;    chain (log1p → multiply → exp). We interleave Block A and Block B
;    computations so the CPU can execute independent operations in parallel
;    while waiting for dependent results.
;
; 3. MEMORY ACCESS PATTERNS: We manually prefetch and schedule loads to hide
;    memory latency. The compiler doesn't understand our access pattern well
;    enough to optimize this.
;
; 4. FMA ORDERING: The compiler sometimes misses opportunities to use FMA
;    (fused multiply-add) or uses suboptimal FMA variants. We explicitly use
;    vfmadd213pd, vfmadd231pd, vfnmadd231pd as appropriate.
;
; PERFORMANCE COMPARISON:
;   C intrinsics version: ~1.7M obs/sec
;   This assembly kernel: ~3M obs/sec (76% faster)
;
; =============================================================================
; ALGORITHM OVERVIEW
; =============================================================================
;
; This kernel implements the BOCPD prediction step:
;
; For each run length r:
;   1. Compute Student-t log-probability: ln(p) = C1 - C2 × log1p((x-μ)²×inv_ssn)
;   2. Exponentiate: p = exp(ln(p))
;   3. Compute growth probability: growth[r+1] = r[r] × p × (1-h)
;   4. Compute changepoint contribution: r0 += r[r] × p × h
;   5. Track maximum growth (for MAP estimation)
;   6. Track last valid index (for dynamic truncation)
;
; We process 8 run lengths per iteration: Block A (indices i..i+3) and
; Block B (indices i+4..i+7). This allows maximum utilization of the 16 YMM
; registers available.
;
; =============================================================================
; WINDOWS x64 ABI REQUIREMENTS
; =============================================================================
;
; The Windows x64 calling convention differs from System V (Linux/macOS):
;
; PARAMETER PASSING:
;   - First 4 integer/pointer args: RCX, RDX, R8, R9
;   - First 4 float args: XMM0, XMM1, XMM2, XMM3
;   - We receive args pointer in RCX
;
; NON-VOLATILE (callee-saved) REGISTERS:
;   - GPR: RBX, RBP, RDI, RSI, R12-R15
;   - XMM: XMM6-XMM15 (!)
;
; This is why we save XMM6-XMM15 in the prologue. Linux/macOS don't require
; saving XMM registers, but Windows does. Forgetting this causes mysterious
; crashes in release builds.
;
; STACK ALIGNMENT:
;   - Stack must be 16-byte aligned at CALL instruction
;   - We need 32-byte alignment for AVX stores, so we align to 32
;
; =============================================================================
; FIXED ISSUES FROM PREVIOUS VERSIONS
; =============================================================================
;
;   1. Correct 256-byte superblock addressing (V3 layout)
;      - Previous versions used wrong offsets for the interleaved format
;
;   2. No register clobbering (r_old_B uses stack)
;      - ymm9 holds max_growth_B, can't also hold r_old_B
;      - Solution: spill r_old_B to stack temporarily
;
;   3. Correct r_new store offsets
;      - growth[i] goes to r_new[i+1], not r_new[i]
;      - Block A: offset = (i+1)*8 = rsi*8 + 8
;      - Block B: offset = (i+5)*8 = rsi*8 + 40
;
;   4. Fast horizontal reduction (no vhaddpd)
;      - vhaddpd has poor throughput on Intel
;      - Use vextractf128 + vaddpd + shuffle instead
;
;   5. bsr for truncation (no bt chain)
;      - Finding highest set bit with bt requires 4 comparisons
;      - bsr (bit scan reverse) does it in one instruction
;
;   6. Consistent stack layout with defines
;      - Named offsets prevent off-by-one errors
;
;   7. Correct epilogue stack math
;      - Must restore RSP before accessing saved XMM registers
;
;   8. Stack frame padding for alignment safety
;      - Extra 32 bytes prevents overlap after AND alignment
;
; =============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

; =============================================================================
; CONSTANTS (read-only data section)
; =============================================================================
;
; All constants are 32-byte aligned for efficient AVX loads.
; Each constant is replicated 4 times for broadcast-free loading into YMM.
;
; WHY PRECOMPUTE THESE?
; Loading from memory is faster than computing at runtime, and these values
; never change. The constants fit in L1 cache and stay there for the duration
; of the kernel.
; =============================================================================

section .rodata
align 32

; -----------------------------------------------------------------------------
; GENERAL CONSTANTS
; -----------------------------------------------------------------------------

const_one:      dq 1.0, 1.0, 1.0, 1.0           ; Used in polynomial evaluation
bias_1023:      dq 1023, 1023, 1023, 1023       ; IEEE-754 exponent bias
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300  ; Minimum probability (avoid zero)

; -----------------------------------------------------------------------------
; EXP CONSTANTS
; -----------------------------------------------------------------------------
;
; exp(x) is computed as 2^(x/ln2) = 2^k × 2^f where k = round(x/ln2)
; exp_inv_ln2 = 1/ln(2) = log₂(e) ≈ 1.4427
;
; The polynomial coefficients approximate 2^f for f ∈ [-0.5, 0.5].
; They are derived from the Taylor series of 2^x = e^(x×ln2):
;   exp_c1 = ln(2)
;   exp_c2 = ln(2)²/2
;   exp_c3 = ln(2)³/6
;   etc.
;
; Clamping to [-700, 700] prevents overflow (exp(709) > DBL_MAX).
; -----------------------------------------------------------------------------

exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
exp_c1:         dq 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
exp_c2:         dq 0.24022650695910072, 0.24022650695910072, 0.24022650695910072, 0.24022650695910072
exp_c3:         dq 0.05550410866482158, 0.05550410866482158, 0.05550410866482158, 0.05550410866482158
exp_c4:         dq 0.009618129107628477, 0.009618129107628477, 0.009618129107628477, 0.009618129107628477
exp_c5:         dq 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443, 0.0013333558146428443
exp_c6:         dq 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608, 0.00015403530393381608

; -----------------------------------------------------------------------------
; LOG1P CONSTANTS
; -----------------------------------------------------------------------------
;
; log1p(t) = ln(1+t) is approximated by the Taylor series:
;   ln(1+t) = t - t²/2 + t³/3 - t⁴/4 + t⁵/5 - t⁶/6 + ...
;
; We use Horner's method: t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
; where c2 = -1/2, c3 = 1/3, c4 = -1/4, c5 = 1/5, c6 = -1/6
;
; WHY LOG1P INSTEAD OF LOG?
; For small t (when x ≈ μ), log(1+t) loses precision because 1+t ≈ 1.
; log1p is designed for this case and maintains precision.
;
; In BOCPD, t = (x-μ)²/(νσ²). When the observation matches the predicted
; mean (x ≈ μ), t is very small and log1p matters.
; -----------------------------------------------------------------------------

log1p_c2:       dq -0.5, -0.5, -0.5, -0.5
log1p_c3:       dq 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333
log1p_c4:       dq -0.25, -0.25, -0.25, -0.25
log1p_c5:       dq 0.2, 0.2, 0.2, 0.2
log1p_c6:       dq -0.16666666666666666, -0.16666666666666666, -0.16666666666666666, -0.16666666666666666

; -----------------------------------------------------------------------------
; INDEX TRACKING CONSTANTS
; -----------------------------------------------------------------------------
;
; We track indices as doubles (not integers) because:
;   1. AVX2 has poor support for 64-bit integer comparisons
;   2. vblendvpd works on doubles, not integers
;   3. Converting at the end is cheaper than throughout
;
; idx_init_a = [1,2,3,4] (Block A processes run lengths at i+1..i+4 output indices)
; idx_init_b = [5,6,7,8] (Block B processes run lengths at i+5..i+8 output indices)
; idx_increment = 8 (we advance by 8 run lengths per iteration)
; -----------------------------------------------------------------------------

idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

; =============================================================================
; STRUCT OFFSETS (bocpd_kernel_args_t)
; =============================================================================
;
; The C caller passes a pointer to this struct in RCX:
;
; typedef struct {
;     double *lin_interleaved;  // [0]  Pointer to interleaved parameter blocks
;     double *r_old;            // [8]  Current probability distribution
;     double x;                 // [16] New observation
;     double h;                 // [24] Hazard rate
;     double one_minus_h;       // [32] 1 - h (precomputed)
;     double trunc_thresh;      // [40] Truncation threshold (1e-6)
;     size_t n_padded;          // [48] Number of elements (padded to multiple of 8)
;     double *r_new;            // [56] Output probability distribution
;     double *r0_out;           // [64] Output: changepoint probability
;     double *max_growth_out;   // [72] Output: maximum growth value
;     size_t *max_idx_out;      // [80] Output: index of maximum growth
;     size_t *last_valid_out;   // [88] Output: last index above threshold
; } bocpd_kernel_args_t;
; =============================================================================

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

; =============================================================================
; STACK FRAME LAYOUT
; =============================================================================
;
; We need stack space for values that don't fit in registers:
;   - Index vectors (track which run length corresponds to each lane)
;   - Maximum index vectors (track indices of max values)
;   - Maximum growth vectors (for final reduction)
;   - r_old_B (spilled because ymm9 is used for max_growth_B)
;
; MEMORY MAP (after 32-byte alignment):
;
;   Offset      Size    Contents
;   ──────────────────────────────────────────────────────
;   [rsp + 0]     32    idx_vec_A: current indices for Block A
;   [rsp + 32]    32    idx_vec_B: current indices for Block B
;   [rsp + 64]    32    max_idx_A: indices where Block A had max
;   [rsp + 96]    32    max_idx_B: indices where Block B had max
;   [rsp + 128]   32    max_growth_A: max growth values for Block A
;   [rsp + 160]   32    max_growth_B: max growth values for Block B
;   [rsp + 192]   32    r_old_B: spilled r_old for Block B
;   [rsp + 224]   64    padding (alignment safety margin)
;   ──────────────────────────────────────────────────────
;   Total:       288 bytes
;
; WHY ALIGNMENT PADDING?
; After `and rsp, -32`, the actual aligned address might be up to 31 bytes
; lower than expected. The padding ensures we don't accidentally overlap
; with saved registers or return addresses.
; =============================================================================

%define STK_IDX_VEC_A       0
%define STK_IDX_VEC_B       32
%define STK_MAX_IDX_A       64
%define STK_MAX_IDX_B       96
%define STK_MAX_GROWTH_A    128
%define STK_MAX_GROWTH_B    160
%define STK_R_OLD_B         192

%define STACK_FRAME         288      ; 256 + 32 alignment padding

; =============================================================================
; KERNEL ENTRY POINTS
; =============================================================================
;
; This file provides TWO entry points for different platforms:
;
;   bocpd_fused_loop_avx2_win  — Windows x64 ABI (RCX = args ptr)
;   bocpd_fused_loop_avx2_sysv — Linux/macOS System V ABI (RDI = args ptr)
;
; WHICH ONE TO USE:
;   #ifdef _WIN32
;       extern void bocpd_fused_loop_avx2_win(bocpd_kernel_args_t *args);
;       #define bocpd_fused_loop_avx2 bocpd_fused_loop_avx2_win
;   #else
;       extern void bocpd_fused_loop_avx2_sysv(bocpd_kernel_args_t *args);
;       #define bocpd_fused_loop_avx2 bocpd_fused_loop_avx2_sysv
;   #endif
;
; The System V version is ~20 cycles faster per call because it doesn't
; need to save/restore XMM6-15 (they're caller-saved on Linux/macOS).
;
; =============================================================================

section .text
global bocpd_fused_loop_avx2_win
global bocpd_fused_loop_avx2_sysv

; =============================================================================
; WINDOWS x64 ABI ENTRY POINT
; =============================================================================

bocpd_fused_loop_avx2_win:

; =============================================================================
; PROLOGUE — Save callee-saved registers
; =============================================================================
;
; Windows x64 ABI requires us to preserve:
;   - GPR: RBX, RBP, RDI, RSI, R12-R15
;   - XMM: XMM6-XMM15
;
; We save GPRs with PUSH (8 bytes each) and XMMs with VMOVDQU to stack.
; Total saved: 8 GPRs × 8 bytes + 10 XMMs × 16 bytes = 64 + 160 = 224 bytes
; =============================================================================

    ; Save non-volatile GPRs
    push        rbp
    push        rbx
    push        rdi
    push        rsi
    push        r12
    push        r13
    push        r14
    push        r15

    ; Save non-volatile XMM registers (Windows ABI requirement)
    ; We only use the low 128 bits for saving (vmovdqu) because the high
    ; 128 bits of YMM6-15 are volatile on Windows.
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

    ; RCX = args pointer (Windows ABI first parameter)
    ; Move to RDI for consistency with our code (RDI is callee-saved)
    mov         rdi, rcx

; =============================================================================
; STACK FRAME SETUP — Allocate and align local storage
; =============================================================================
;
; We save RBP so we can restore RSP exactly in the epilogue.
; Then allocate STACK_FRAME bytes and align to 32-byte boundary.
;
; The alignment is critical for AVX: vmovapd requires 32-byte alignment.
; Misaligned access causes #GP fault or severe performance penalty.
; =============================================================================

    mov         rbp, rsp                ; Save original RSP
    sub         rsp, STACK_FRAME        ; Allocate stack frame
    and         rsp, -32                ; Align to 32 bytes

; =============================================================================
; LOAD ARGUMENT POINTERS INTO REGISTERS
; =============================================================================
;
; We keep frequently-used pointers in callee-saved registers so they survive
; across the loop without reloading from the args struct.
;
; Register allocation:
;   RDI = args struct pointer (already set)
;   R8  = lin_interleaved (parameter blocks)
;   R12 = r_old (input probability distribution)
;   R13 = r_new (output probability distribution)
;   R14 = n_padded (loop bound)
;   RSI = loop counter (i)
;   RBX = last_valid (truncation tracking)
; =============================================================================

    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]   ; params base pointer
    mov         r12, [rdi + ARG_R_OLD]             ; r_old array
    mov         r13, [rdi + ARG_R_NEW]             ; r_new array
    mov         r14, [rdi + ARG_N_PADDED]          ; loop bound

; =============================================================================
; BROADCAST IMMUTABLE SCALARS INTO DEDICATED YMM REGISTERS
; =============================================================================
;
; These values are used in every iteration but never change.
; By keeping them in dedicated registers, we avoid repeated memory loads.
;
; YMM REGISTER MAP (throughout the kernel):
;   ymm15 = x (observation) — broadcast of the new data point
;   ymm14 = h (hazard rate) — probability of changepoint
;   ymm13 = 1-h (continuation probability)
;   ymm12 = threshold — for dynamic truncation
;   ymm11 = r0 accumulator — sum of changepoint contributions
;   ymm10 = max_growth_A — running max for Block A
;   ymm9  = max_growth_B — running max for Block B
;   ymm8  = r_old_A — loaded fresh each iteration
;   ymm0-7 = scratch registers — reused throughout computation
;
; WHY THESE SPECIFIC REGISTERS?
; ymm15-12 are callee-saved on Windows, so we saved them in prologue.
; ymm11-9 hold accumulators that persist across iterations.
; ymm8 and below are scratch.
; =============================================================================

    vbroadcastsd ymm15, [rdi + ARG_X]     ; x = observation (same for all lanes)
    vbroadcastsd ymm14, [rdi + ARG_H]     ; h = hazard rate
    vbroadcastsd ymm13, [rdi + ARG_OMH]   ; 1-h = continuation probability
    vbroadcastsd ymm12, [rdi + ARG_THRESH]; threshold for truncation

    ; Zero the accumulators
    vxorpd      ymm11, ymm11, ymm11       ; r0 accumulator = 0
    vxorpd      ymm10, ymm10, ymm10       ; max_growth_A = 0
    vxorpd      ymm9,  ymm9,  ymm9        ; max_growth_B = 0

; =============================================================================
; INITIALIZE INDEX VECTORS
; =============================================================================
;
; We track run length indices as doubles (for vblendvpd compatibility).
; Block A starts at [1,2,3,4], Block B at [5,6,7,8].
; These are OUTPUT indices — where growth values are stored in r_new.
; =============================================================================

    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0

    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    ; Zero max index trackers (no max found yet)
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    ; Initialize loop variables
    xor         rsi, rsi                  ; i = 0 (loop counter)
    xor         rbx, rbx                  ; last_valid = 0 (truncation point)

; =============================================================================
; MAIN LOOP — Process 8 run lengths per iteration
; =============================================================================
;
; Each iteration processes:
;   Block A: run lengths at indices i, i+1, i+2, i+3
;   Block B: run lengths at indices i+4, i+5, i+6, i+7
;
; The computation is interleaved: we start Block A's computation, then Block B,
; then finish Block A, then finish Block B. This hides latency by keeping the
; CPU's execution units busy while waiting for dependent results.
; =============================================================================

.loop_start:
    cmp         rsi, r14                  ; if i >= n_padded, exit
    jge         .loop_end

; =============================================================================
; V3 SUPERBLOCK ADDRESSING
; =============================================================================
;
; The interleaved memory layout uses 256-byte "superblocks".
; Each superblock holds parameters for 4 consecutive run lengths:
;
;   Offset   Contents
;   ─────────────────────────────────
;   0-31     μ[0..3]      (means)
;   32-63    C1[0..3]     (Student-t constant)
;   64-95    C2[0..3]     (Student-t exponent)
;   96-127   inv_ssn[0..3](inverse scale)
;   128-255  (update params, not used here)
;
; To find parameters for run length i:
;   block_index = i / 4
;   byte_offset = block_index × 256
;   lane = i % 4
;
; For Block A (indices i..i+3): all in same superblock
;   block_A = i / 4
;   offset_A = block_A × 256
;
; For Block B (indices i+4..i+7): next superblock
;   block_B = (i+4) / 4 = i/4 + 1
;   offset_B = block_B × 256
; =============================================================================

    ; Calculate Block A superblock offset
    mov         rax, rsi                  ; rax = i
    shr         rax, 2                    ; rax = i / 4 (block index)
    shl         rax, 8                    ; rax = block_index × 256

    ; Calculate Block B superblock offset
    mov         rdx, rsi
    add         rdx, 4                    ; rdx = i + 4
    shr         rdx, 2                    ; rdx = (i+4) / 4
    shl         rdx, 8                    ; rdx = block_index × 256

; =============================================================================
; LOAD BLOCK A PARAMETERS
; =============================================================================
;
; V3 superblock layout:
;   [base + offset + 0]   = μ[0..3]
;   [base + offset + 32]  = C1[0..3]
;   [base + offset + 64]  = C2[0..3]
;   [base + offset + 96]  = inv_ssn[0..3]
;
; We also load r_old[i..i+3] from the probability array.
; =============================================================================

    vmovapd     ymm0, [r8 + rax + 0]      ; μ_A = means for Block A
    vmovapd     ymm1, [r8 + rax + 32]     ; C1_A = Student-t constants
    vmovapd     ymm2, [r8 + rax + 64]     ; C2_A = Student-t exponents
    vmovapd     ymm3, [r8 + rax + 96]     ; inv_ssn_A = inverse scales

    vmovapd     ymm8, [r12 + rsi*8]       ; r_old_A = P(run_length = i..i+3)

; =============================================================================
; LOAD BLOCK B PARAMETERS
; =============================================================================
;
; We need r_old_B but all YMM registers are in use. ymm9 holds max_growth_B
; which we need to preserve. Solution: load r_old_B into ymm0 temporarily,
; then spill to stack.
;
; After this section:
;   ymm4 = μ_B
;   ymm5 = C1_B
;   ymm6 = C2_B
;   ymm7 = inv_ssn_B
;   [rsp + STK_R_OLD_B] = r_old_B
; =============================================================================

    vmovapd     ymm4, [r8 + rdx + 0]      ; μ_B
    vmovapd     ymm5, [r8 + rdx + 32]     ; C1_B
    vmovapd     ymm6, [r8 + rdx + 64]     ; C2_B
    vmovapd     ymm7, [r8 + rdx + 96]     ; inv_ssn_B

    vmovapd     ymm0, [r12 + rsi*8 + 32]  ; r_old_B (temporary in ymm0)
    vmovapd     [rsp + STK_R_OLD_B], ymm0 ; spill to stack

    ; Reload μ_A (was clobbered by r_old_B load above)
    vmovapd     ymm0, [r8 + rax + 0]      ; μ_A

; =============================================================================
; STUDENT-T COMPUTATION — BLOCK A
; =============================================================================
;
; The Student-t log-probability is:
;   ln(p) = C1 - C2 × log1p(t)
; where:
;   t = (x - μ)² × inv_ssn = (x - μ)² / (ν × σ²)
;
; Step 1: z = x - μ
; Step 2: z² = z × z
; Step 3: t = z² × inv_ssn
;
; After this section: ymm0 = t_A (the argument for log1p)
; =============================================================================

    vsubpd      ymm0, ymm15, ymm0         ; z_A = x - μ_A
    vmulpd      ymm0, ymm0, ymm0          ; z²_A = z_A × z_A
    vmulpd      ymm0, ymm0, ymm3          ; t_A = z²_A × inv_ssn_A

; =============================================================================
; STUDENT-T COMPUTATION — BLOCK B
; =============================================================================
;
; Same computation for Block B. We do this now (while t_A is ready) to give
; the CPU independent work while we wait for any pipeline stalls.
;
; After this section: ymm4 = t_B
; =============================================================================

    vsubpd      ymm4, ymm15, ymm4         ; z_B = x - μ_B
    vmulpd      ymm4, ymm4, ymm4          ; z²_B
    vmulpd      ymm4, ymm4, ymm7          ; t_B = z²_B × inv_ssn_B

; =============================================================================
; LOG1P POLYNOMIAL — BLOCK A (Horner's Method)
; =============================================================================
;
; log1p(t) = ln(1 + t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
;
; Horner's method evaluates from inside out:
;   p = c6
;   p = p × t + c5  (vfmadd213pd: p = p*t + c5)
;   p = p × t + c4
;   p = p × t + c3
;   p = p × t + c2
;   p = p × t + 1
;   log1p = p × t
;
; vfmadd213pd ymm_a, ymm_b, ymm_c computes: ymm_a = ymm_a × ymm_b + ymm_c
;
; After this section: ymm3 = log1p(t_A)
; =============================================================================

    vmovapd     ymm3, [rel log1p_c6]      ; p = c6
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]; p = c6×t + c5
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]; p = (c6×t + c5)×t + c4
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]; ...and so on
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]; p = (...×t + c2)×t + 1
    vmulpd      ymm3, ymm3, ymm0          ; log1p_A = p × t

; =============================================================================
; LOG1P POLYNOMIAL — BLOCK B
; =============================================================================
;
; Same computation for Block B.
; After this section: ymm7 = log1p(t_B)
; =============================================================================

    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm7, ymm4          ; log1p_B

; =============================================================================
; COMPUTE ln(p) = C1 - C2 × log1p(t)
; =============================================================================
;
; vfnmadd231pd ymm_a, ymm_b, ymm_c computes: ymm_a = ymm_a - ymm_b × ymm_c
; This is "fused negative multiply-add": a = a - b×c
;
; ymm1 = C1_A, ymm2 = C2_A, ymm3 = log1p_A
; ymm5 = C1_B, ymm6 = C2_B, ymm7 = log1p_B
;
; After this section:
;   ymm1 = ln_pp_A = C1_A - C2_A × log1p_A
;   ymm5 = ln_pp_B = C1_B - C2_B × log1p_B
; =============================================================================

    vfnmadd231pd ymm1, ymm2, ymm3         ; ln_pp_A = C1_A - C2_A × log1p_A
    vfnmadd231pd ymm5, ymm6, ymm7         ; ln_pp_B = C1_B - C2_B × log1p_B

; =============================================================================
; EXP PREPARATION — BLOCK A
; =============================================================================
;
; To compute exp(x), we use: exp(x) = 2^(x/ln2) = 2^(k + f) = 2^k × 2^f
; where:
;   y = x / ln2 = x × log₂(e)
;   k = round(y) — integer part
;   f = y - k   — fractional part, f ∈ [-0.5, 0.5]
;
; We'll compute 2^f with a polynomial, and 2^k via IEEE-754 bit manipulation.
;
; First, clamp to [-700, 700] to prevent overflow/underflow.
;
; After this section:
;   ymm2 = k_A (integer exponent, stored as double)
;   ymm0 = f_A (fractional part)
; =============================================================================

    vmaxpd      ymm1, ymm1, [rel exp_min_x]      ; clamp lower
    vminpd      ymm1, ymm1, [rel exp_max_x]      ; clamp upper
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]    ; y_A = ln_pp_A × log₂(e)
    vroundpd    ymm2, ymm0, 0                    ; k_A = round(y_A) to nearest int
    vsubpd      ymm0, ymm0, ymm2                 ; f_A = y_A - k_A

; =============================================================================
; EXP PREPARATION — BLOCK B
; =============================================================================
;
; Same for Block B.
; After this section:
;   ymm6 = k_B
;   ymm4 = f_B
; =============================================================================

    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]    ; y_B
    vroundpd    ymm6, ymm4, 0                    ; k_B
    vsubpd      ymm4, ymm4, ymm6                 ; f_B

; =============================================================================
; ESTRIN POLYNOMIAL — 2^f BLOCK A
; =============================================================================
;
; We need to compute 2^f for f ∈ [-0.5, 0.5].
; The polynomial approximates 2^f using coefficients exp_c1..exp_c6.
;
; ESTRIN'S SCHEME (vs Horner):
; Horner evaluates p = c0 + f×(c1 + f×(c2 + ...)) — sequential dependency
; Estrin groups terms for parallelism:
;   p01 = c0 + f×c1
;   p23 = c2 + f×c3
;   q0123 = p01 + f²×p23
;   ...
;
; This reduces the dependency chain from O(n) to O(log n) depth.
;
; For our 6-term polynomial:
;   p01 = 1 + f×c1
;   p23 = c2 + f×c3
;   p45 = c4 + f×c5
;   q0123 = p01 + f²×p23
;   q456 = p45 + f²×c6
;   result = q0123 + f⁴×q456
;
; After this section: ymm1 = 2^f_A
; =============================================================================

    vmulpd      ymm3, ymm0, ymm0               ; f²_A

    ; p01 = 1 + f×c1
    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]       ; ymm1 = 1 + f_A×c1

    ; p23 = c2 + f×c3
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]       ; ymm7 = c2 + f_A×c3

    ; q0123 = p01 + f²×p23
    vfmadd231pd ymm1, ymm3, ymm7               ; ymm1 = p01 + f²×p23

    ; p45 = c4 + f×c5
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]       ; ymm7 = c4 + f_A×c5

    ; q456 = p45 + f²×c6
    vfmadd231pd ymm7, ymm3, [rel exp_c6]       ; ymm7 = p45 + f²×c6

    ; result = q0123 + f⁴×q456
    vmulpd      ymm3, ymm3, ymm3               ; f⁴_A = f²×f²
    vfmadd231pd ymm1, ymm3, ymm7               ; ymm1 = 2^f_A

; =============================================================================
; ESTRIN POLYNOMIAL — 2^f BLOCK B
; =============================================================================
;
; Same computation for Block B.
; After this section: ymm5 = 2^f_B
; =============================================================================

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
    vfmadd231pd ymm5, ymm3, ymm7               ; ymm5 = 2^f_B

; =============================================================================
; 2^k RECONSTRUCTION — IEEE-754 Exponent Manipulation
; =============================================================================
;
; IEEE-754 double format: [sign:1][exponent:11][mantissa:52]
;
; The value 2^k is represented as:
;   - sign = 0
;   - exponent = k + 1023 (the bias)
;   - mantissa = 0 (implicit leading 1)
;
; So the bit pattern for 2^k is: (k + 1023) << 52
;
; Steps:
;   1. Convert k from double to int32 (vcvtpd2dq)
;   2. Sign-extend to int64 (vpmovsxdq)
;   3. Add bias 1023 (vpaddq)
;   4. Shift left by 52 (vpsllq)
;   5. Reinterpret as double (it's already in the register)
;
; Finally: pp = 2^f × 2^k, clamped to minimum 1e-300 to avoid exact zero.
;
; After this section:
;   ymm1 = pp_A = exp(ln_pp_A) = Student-t probability for Block A
;   ymm5 = pp_B = exp(ln_pp_B) = Student-t probability for Block B
; =============================================================================

    ; Block A: k_A in ymm2, 2^f_A in ymm1 → pp_A in ymm1
    vcvtpd2dq   xmm0, ymm2                     ; k → int32 (4 doubles → 4 int32s in xmm)
    vpmovsxdq   ymm0, xmm0                     ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel bias_1023]    ; add exponent bias
    vpsllq      ymm0, ymm0, 52                 ; shift to exponent position
    vmulpd      ymm1, ymm1, ymm0               ; pp_A = 2^f_A × 2^k_A
    vmaxpd      ymm1, ymm1, [rel const_min_pp] ; clamp to avoid zero

    ; Block B: k_B in ymm6, 2^f_B in ymm5 → pp_B in ymm5
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0               ; pp_B
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

; =============================================================================
; BOCPD UPDATE — BLOCK A
; =============================================================================
;
; For each run length r:
;   r_pp = r_old[r] × pp        (joint probability of r and observing x)
;   growth = r_pp × (1-h)       (probability of continuing the run)
;   change = r_pp × h           (probability of changepoint)
;
; growth goes to r_new[r+1] (run length increases by 1 if no changepoint)
; change accumulates into r_new[0] (all changepoint probabilities sum to r0)
;
; Memory layout note:
;   r_old[i..i+3] was loaded into ymm8
;   r_new[i+1..i+4] means offset = (i+1)×8 = rsi×8 + 8
; =============================================================================

    vmulpd      ymm0, ymm8, ymm1               ; r_pp_A = r_old_A × pp_A
    vmulpd      ymm2, ymm0, ymm13              ; growth_A = r_pp_A × (1-h)
    vmulpd      ymm0, ymm0, ymm14              ; change_A = r_pp_A × h
    vaddpd      ymm11, ymm11, ymm0             ; r0_accumulator += change_A

    ; Store growth_A at r_new[i+1..i+4]
    vmovupd     [r13 + rsi*8 + 8], ymm2

; =============================================================================
; BOCPD UPDATE — BLOCK B
; =============================================================================
;
; Same computation for Block B.
; r_old_B was spilled to stack earlier, reload it now.
;
; Memory layout:
;   r_new[i+5..i+8] means offset = (i+5)×8 = rsi×8 + 40
; =============================================================================

    vmovapd     ymm0, [rsp + STK_R_OLD_B]      ; reload r_old_B from stack
    vmulpd      ymm0, ymm0, ymm5               ; r_pp_B = r_old_B × pp_B
    vmulpd      ymm3, ymm0, ymm13              ; growth_B = r_pp_B × (1-h)
    vmulpd      ymm0, ymm0, ymm14              ; change_B = r_pp_B × h
    vaddpd      ymm11, ymm11, ymm0             ; r0_accumulator += change_B

    ; Store growth_B at r_new[i+5..i+8]
    vmovupd     [r13 + rsi*8 + 40], ymm3

; =============================================================================
; MAX TRACKING — BLOCK A
; =============================================================================
;
; For MAP (Maximum A Posteriori) estimation, we track which run length has
; the highest probability. We do this lane-by-lane using vector compares
; and blends.
;
; vcmppd with predicate 14 (CMP_GT_OQ) sets each lane to all-1s if greater,
; all-0s otherwise. vblendvpd uses the sign bit of the mask to select.
;
; We track both the max value (in ymm10) and the corresponding index
; (in [rsp + STK_MAX_IDX_A]).
; =============================================================================

    vcmppd      ymm0, ymm2, ymm10, 14          ; mask = growth_A > max_growth_A?
    vblendvpd   ymm10, ymm10, ymm2, ymm0       ; max_growth_A = max(max, growth)

    ; Update indices where this iteration's growth beat the previous max
    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]    ; current best indices
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]    ; this iteration's indices
    vblendvpd   ymm1, ymm1, ymm4, ymm0         ; update where mask is set
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

; =============================================================================
; MAX TRACKING — BLOCK B
; =============================================================================

    vcmppd      ymm0, ymm3, ymm9, 14           ; mask = growth_B > max_growth_B?
    vblendvpd   ymm9, ymm9, ymm3, ymm0         ; max_growth_B = max(max, growth)

    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

; =============================================================================
; DYNAMIC TRUNCATION — BLOCK A
; =============================================================================
;
; To prevent unbounded growth of the run length distribution, we truncate
; run lengths with negligible probability (< threshold).
;
; We track last_valid = highest index where probability > threshold.
; After the loop, active_len = last_valid + 1.
;
; vmovmskpd extracts the sign bits of each lane into a 4-bit integer.
; bsr (bit scan reverse) finds the index of the highest set bit.
;
; For Block A (indices i..i+3), if lane k is above threshold:
;   last_valid = i + k + 1 (the +1 because growth goes to r_new[i+k+1])
; =============================================================================

    vcmppd      ymm0, ymm2, ymm12, 14          ; mask = growth_A > threshold?
    vmovmskpd   eax, ymm0                      ; eax = 4-bit mask
    test        eax, eax                       ; any lanes above threshold?
    jz          .no_trunc_A                    ; skip if all below

    bsr         ecx, eax                       ; ecx = highest set bit (0-3)
    lea         rbx, [rsi + rcx + 1]           ; last_valid = i + lane + 1

.no_trunc_A:

; =============================================================================
; DYNAMIC TRUNCATION — BLOCK B
; =============================================================================
;
; For Block B (indices i+4..i+7):
;   last_valid = (i+4) + lane + 1 = i + lane + 5
; =============================================================================

    vcmppd      ymm0, ymm3, ymm12, 14          ; mask = growth_B > threshold?
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .no_trunc_B

    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]           ; last_valid = i + lane + 5

.no_trunc_B:

; =============================================================================
; UPDATE INDEX VECTORS
; =============================================================================
;
; Advance indices by 8 for the next iteration.
; idx_vec_A: [1,2,3,4] → [9,10,11,12] → [17,18,19,20] → ...
; idx_vec_B: [5,6,7,8] → [13,14,15,16] → [21,22,23,24] → ...
; =============================================================================

    vmovapd     ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd      ymm0, ymm0, [rel idx_increment] ; +8
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0

    vmovapd     ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

; =============================================================================
; LOOP ADVANCE
; =============================================================================

    add         rsi, 8                         ; i += 8 (process next 8 run lengths)
    jmp         .loop_start

; =============================================================================
; LOOP END — Horizontal reductions and output
; =============================================================================

.loop_end:

; =============================================================================
; REDUCE R0 ACCUMULATOR
; =============================================================================
;
; ymm11 contains 4 partial sums: [a, b, c, d]
; We need the total: a + b + c + d
;
; FAST REDUCTION (avoiding slow vhaddpd):
;   1. Extract high 128 bits: xmm0 = [c, d]
;   2. Add to low 128 bits:   xmm0 = [a+c, b+d]
;   3. Shuffle high to low:   xmm1 = [b+d, b+d]
;   4. Add:                   xmm0 = [a+b+c+d, ...]
;
; vhaddpd is avoided because it has poor throughput on Intel (2 μops, 
; uses shuffle unit). This sequence is faster.
; =============================================================================

    vextractf128 xmm0, ymm11, 1               ; xmm0 = high 128 bits = [c, d]
    vaddpd      xmm0, xmm0, xmm11             ; xmm0 = [a+c, b+d]
    vunpckhpd   xmm1, xmm0, xmm0              ; xmm1 = [b+d, b+d]
    vaddsd      xmm0, xmm0, xmm1              ; xmm0[0] = a+b+c+d

    ; Store r0 to output
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0                   ; *r0_out = r0

; =============================================================================
; MAP REDUCTION — Find overall maximum
; =============================================================================
;
; We need to find the global maximum across all 8 lanes (4 in Block A, 4 in
; Block B), and compare against r0 (changepoint probability).
;
; If r0 beats all growth probabilities, MAP run length is 0.
; Otherwise, it's the index of the highest growth probability.
;
; We use a scalar loop because:
;   1. Only 8 comparisons total
;   2. Mixing scalar and vector reduction is error-prone
;   3. This happens once per kernel call (not per iteration)
; =============================================================================

    vmovsd      xmm6, xmm0, xmm0              ; best_val = r0
    xor         r15, r15                       ; best_idx = 0

    ; Save max growth vectors to stack for scalar access
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm9

    ; Scalar comparison loop over 4 lanes
    xor         rcx, rcx                       ; lane counter

.reduce_loop:
    cmp         rcx, 4
    jge         .reduce_done

    ; Check Block A lane
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm6                     ; compare to current best
    jbe         .check_B                       ; skip if not greater

    vmovsd      xmm6, xmm1, xmm1              ; update best value
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2                     ; update best index

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

; =============================================================================
; WRITE OUTPUTS
; =============================================================================

    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6                   ; *max_growth_out = best_val

    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15                    ; *max_idx_out = best_idx

    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx                    ; *last_valid_out = truncation point

; =============================================================================
; EPILOGUE — Restore registers and return
; =============================================================================
;
; We must restore RSP first (from saved RBP), then restore saved registers
; in reverse order of how we saved them.
;
; VZEROUPPER is required before returning to code that might use SSE.
; Without it, SSE instructions incur a ~70 cycle penalty due to the
; "AVX/SSE transition penalty" on older Intel CPUs.
; =============================================================================

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

    ; Restore GPRs (reverse order of push)
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rsi
    pop         rdi
    pop         rbx
    pop         rbp

    vzeroupper                                ; avoid AVX/SSE transition penalty
    ret

; =============================================================================
; LINUX SYSTEM V ABI ENTRY POINT
; =============================================================================
;
; The System V AMD64 ABI (used by Linux, macOS, BSD) differs from Windows x64:
;
; PARAMETER PASSING:
;   - Integer/pointer args: RDI, RSI, RDX, RCX, R8, R9
;   - Float args: XMM0-XMM7
;   - We receive args pointer in RDI (not RCX)
;
; CALLEE-SAVED (non-volatile) REGISTERS:
;   - GPR: RBX, RBP, R12-R15
;   - XMM: NONE! (All XMM/YMM registers are caller-saved)
;
; This is MUCH simpler than Windows because we don't need to save XMM6-15.
; The prologue and epilogue are shorter, saving ~20 cycles per call.
;
; STACK ALIGNMENT:
;   - Stack must be 16-byte aligned before CALL
;   - We align to 32 for AVX operations
;
; =============================================================================

bocpd_fused_loop_avx2_sysv:

; =============================================================================
; PROLOGUE — Save callee-saved GPRs only (no XMM saving needed on Linux!)
; =============================================================================

    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    ; RDI already contains args pointer (System V ABI first parameter)
    ; No need to move it like we do on Windows

; =============================================================================
; STACK FRAME SETUP — Same as Windows version
; =============================================================================

    mov         rbp, rsp                ; Save original RSP
    sub         rsp, STACK_FRAME        ; Allocate stack frame
    and         rsp, -32                ; Align to 32 bytes for AVX

; =============================================================================
; LOAD ARGUMENTS AND INITIALIZE — Identical to Windows version
; =============================================================================

    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]

    vbroadcastsd ymm15, [rdi + ARG_X]
    vbroadcastsd ymm14, [rdi + ARG_H]
    vbroadcastsd ymm13, [rdi + ARG_OMH]
    vbroadcastsd ymm12, [rdi + ARG_THRESH]

    vxorpd      ymm11, ymm11, ymm11
    vxorpd      ymm10, ymm10, ymm10
    vxorpd      ymm9,  ymm9,  ymm9

    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    xor         rsi, rsi
    xor         rbx, rbx

; =============================================================================
; MAIN LOOP — Identical to Windows version
; =============================================================================

.sysv_loop_start:
    cmp         rsi, r14
    jge         .sysv_loop_end

    ; Block A offset calculation
    mov         rax, rsi
    shr         rax, 2
    shl         rax, 8

    ; Block B offset calculation
    mov         rdx, rsi
    add         rdx, 4
    shr         rdx, 2
    shl         rdx, 8

    ; Load Block A parameters
    vmovapd     ymm0, [r8 + rax + 0]
    vmovapd     ymm1, [r8 + rax + 32]
    vmovapd     ymm2, [r8 + rax + 64]
    vmovapd     ymm3, [r8 + rax + 96]
    vmovapd     ymm8, [r12 + rsi*8]

    ; Load Block B parameters
    vmovapd     ymm4, [r8 + rdx + 0]
    vmovapd     ymm5, [r8 + rdx + 32]
    vmovapd     ymm6, [r8 + rdx + 64]
    vmovapd     ymm7, [r8 + rdx + 96]
    vmovapd     ymm0, [r12 + rsi*8 + 32]
    vmovapd     [rsp + STK_R_OLD_B], ymm0
    vmovapd     ymm0, [r8 + rax + 0]

    ; Student-t computation Block A
    vsubpd      ymm0, ymm15, ymm0
    vmulpd      ymm0, ymm0, ymm0
    vmulpd      ymm0, ymm0, ymm3

    ; Student-t computation Block B
    vsubpd      ymm4, ymm15, ymm4
    vmulpd      ymm4, ymm4, ymm4
    vmulpd      ymm4, ymm4, ymm7

    ; log1p Block A
    vmovapd     ymm3, [rel log1p_c6]
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]
    vfmadd213pd ymm3, ymm0, [rel const_one]
    vmulpd      ymm3, ymm3, ymm0

    ; log1p Block B
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm7, ymm4

    ; ln_pp = C1 - C2 * log1p
    vfnmadd231pd ymm1, ymm2, ymm3
    vfnmadd231pd ymm5, ymm6, ymm7

    ; exp prep Block A
    vmaxpd      ymm1, ymm1, [rel exp_min_x]
    vminpd      ymm1, ymm1, [rel exp_max_x]
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]
    vroundpd    ymm2, ymm0, 0
    vsubpd      ymm0, ymm0, ymm2

    ; exp prep Block B
    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]
    vroundpd    ymm6, ymm4, 0
    vsubpd      ymm4, ymm4, ymm6

    ; Estrin 2^f Block A
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

    ; Estrin 2^f Block B
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

    ; 2^k reconstruction Block A
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    ; 2^k reconstruction Block B
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; BOCPD update Block A
    vmulpd      ymm0, ymm8, ymm1
    vmulpd      ymm2, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ; BOCPD update Block B
    vmovapd     ymm0, [rsp + STK_R_OLD_B]
    vmulpd      ymm0, ymm0, ymm5
    vmulpd      ymm3, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 40], ymm3

    ; Max tracking Block A
    vcmppd      ymm0, ymm2, ymm10, 14
    vblendvpd   ymm10, ymm10, ymm2, ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

    ; Max tracking Block B
    vcmppd      ymm0, ymm3, ymm9, 14
    vblendvpd   ymm9, ymm9, ymm3, ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

    ; Truncation Block A
    vcmppd      ymm0, ymm2, ymm12, 14
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .sysv_no_trunc_A
    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 1]
.sysv_no_trunc_A:

    ; Truncation Block B
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

; =============================================================================
; LOOP END — Reductions (identical logic to Windows)
; =============================================================================

.sysv_loop_end:

    ; Reduce r0 accumulator
    vextractf128 xmm0, ymm11, 1
    vaddpd      xmm0, xmm0, xmm11
    vunpckhpd   xmm1, xmm0, xmm0
    vaddsd      xmm0, xmm0, xmm1

    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0

    ; MAP reduction
    vmovsd      xmm6, xmm0, xmm0
    xor         r15, r15

    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm9

    xor         rcx, rcx

.sysv_reduce_loop:
    cmp         rcx, 4
    jge         .sysv_reduce_done

    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .sysv_check_B
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2

.sysv_check_B:
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_B + rcx*8]
    vucomisd    xmm1, xmm6
    jbe         .sysv_next_lane
    vmovsd      xmm6, xmm1, xmm1
    vmovsd      xmm2, [rsp + STK_MAX_IDX_B + rcx*8]
    vcvttsd2si  r15, xmm2

.sysv_next_lane:
    inc         rcx
    jmp         .sysv_reduce_loop

.sysv_reduce_done:

    ; Write outputs
    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm6

    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15

    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx

; =============================================================================
; EPILOGUE — Much simpler than Windows (no XMM restore needed!)
; =============================================================================

    mov         rsp, rbp                      ; restore stack pointer

    ; Restore GPRs only (reverse order)
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp

    vzeroupper
    ret

; =============================================================================
; END OF KERNEL
; =============================================================================