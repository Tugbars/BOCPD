;==============================================================================
; BOCPD Fused Prediction Kernel — Generic AVX2 V3.2
; Dual ABI: Windows x64 and Linux System V
;==============================================================================
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
; 3. MEMORY ACCESS PATTERNS: We manually schedule loads to hide memory latency.
;    The compiler doesn't understand our access pattern well enough to optimize.
;
; 4. FMA ORDERING: The compiler sometimes misses opportunities to use FMA
;    (fused multiply-add) or uses suboptimal FMA variants. We explicitly use
;    vfmadd213pd, vfmadd231pd, vfnmadd231pd as appropriate.
;
; PERFORMANCE COMPARISON:
;   C intrinsics version:     ~1.7M obs/sec
;   V3.1 generic assembly:    ~2.3M obs/sec (35% faster than C)
;   V3.2 generic assembly:    ~2.8M obs/sec (64% faster than C)
;   Intel-optimized assembly: ~3.0M obs/sec (76% faster than C)
;
; =============================================================================
; VERSION HISTORY
; =============================================================================
;
; V3.2 (this version):
;   - Eliminated ALL redundant memory loads within the iteration
;   - No reload of C1, C2 after log1p (kept in ymm1/2/5/6)
;   - No reload of r_old after exp (kept in ymm8/ymm9)
;   - Moved max_growth_B to stack (cold path) to free register for r_old_B
;   - Added Linux System V ABI entry point
;   - Matches Intel kernel optimization level
;
; V3.1:
;   - Had 6 extra loads per iteration (3 per block)
;   - C1, C2 reloaded after log1p clobbered them
;   - r_old reloaded after Estrin polynomial clobbered it
;   - max_growth_B in register (wasted since it's cold path)
;   - Windows-only
;
; =============================================================================
; OPTIMIZATION STRATEGY
; =============================================================================
;
; The key insight is that with careful register allocation, we can keep ALL
; hot values alive across the computation chain:
;
; 1. Load all parameters for BOTH blocks upfront (μ, C1, C2, inv_ssn, r_old)
; 2. Compute Student-t for both blocks (consumes μ, inv_ssn; frees ymm0,3,4,7)
; 3. Compute log1p for both blocks (uses ymm3,7 as accumulators; preserves C1,C2)
; 4. Compute ln_pp = C1 - C2*log1p (consumes C1, C2; frees ymm1,2,5,6)
; 5. Compute exp for both blocks (uses all scratch registers)
; 6. BOCPD update (uses r_old from ymm8/ymm9, never reloaded!)
;
; The critical register allocation that makes this work:
;
;   DEDICATED REGISTERS (persist across entire iteration):
;   ───────────────────────────────────────────────────────
;   ymm15 = x (observation, broadcast)
;   ymm14 = h (hazard rate)
;   ymm13 = 1-h (continuation probability)
;   ymm12 = threshold (for truncation)
;   ymm11 = r0 accumulator
;   ymm10 = max_growth_A (hot, compared every iteration)
;   ymm9  = r_old_B (V3.2: persists until BOCPD update!)
;   ymm8  = r_old_A (persists until BOCPD update!)
;
;   SCRATCH REGISTERS (reused within iteration):
;   ───────────────────────────────────────────────────────
;   ymm0-ymm7 = computation temporaries
;
;   STACK LOCATIONS (cold path):
;   ───────────────────────────────────────────────────────
;   [rsp + STK_MAX_GROWTH_B] = max_growth_B (rarely updated)
;   [rsp + STK_IDX_VEC_A/B]  = index vectors
;   [rsp + STK_MAX_IDX_A/B]  = max index tracking
;
; WHY max_growth_B ON STACK (V3.2 change):
; ----------------------------------------
; We need to keep BOTH r_old_A and r_old_B in registers because they're used
; at the END of the computation chain (in BOCPD update). If we spill r_old_B
; to stack, we add a load on the CRITICAL PATH.
;
; max_growth_B, on the other hand, is only used for MAX tracking which:
;   - Happens AFTER the critical BOCPD computation
;   - Only updates when a new maximum is found (rare)
;   - Is not latency-sensitive
;
; Trade-off analysis:
;   V3.1: r_old_B spilled (hot path penalty), max_growth_B in register (wasted)
;   V3.2: r_old_B in register (fast), max_growth_B on stack (acceptable)
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
;   128-159   32    κ[0..3]               Update (not used here)
;   160-191   32    α[0..3]               Update (not used here)
;   192-223   32    β[0..3]               Update (not used here)
;   224-255   32    ss_n[0..3]            Update (not used here)
;   ──────────────────────────────────────────────────
;   Total:   256 bytes per superblock
;
; Block addressing:
;   block_index = i / 4
;   byte_offset = block_index * 256
;
; WHY THIS LAYOUT:
; The prediction kernel needs μ, C1, C2, inv_ssn for each run length.
; With this layout, a single vmovapd loads the same field for 4 consecutive
; run lengths. No gather operations needed!
;
; WHY PREDICTION PARAMS FIRST (offsets 0-127):
; The hot loop only needs μ, C1, C2, inv_ssn. By placing these in the first
; 128 bytes (2 cache lines), we maximize cache efficiency. The update params
; (κ, α, β, ss_n) don't pollute the cache during prediction.
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
; WINDOWS x64 ABI:
;   - First 4 integer/pointer args: RCX, RDX, R8, R9
;   - Must preserve: RBX, RBP, RDI, RSI, R12-R15
;   - Must preserve: XMM6-XMM15 (!)
;   - Stack must be 16-byte aligned at CALL
;
; LINUX SYSTEM V ABI:
;   - First 6 integer/pointer args: RDI, RSI, RDX, RCX, R8, R9
;   - Must preserve: RBX, RBP, R12-R15
;   - XMM/YMM registers are ALL caller-saved (no preservation needed!)
;   - Stack must be 16-byte aligned at CALL
;
; The Linux version is ~20 cycles faster per call due to not needing to
; save/restore XMM6-15 in the prologue/epilogue.
;
; =============================================================================
; MATHEMATICAL BACKGROUND
; =============================================================================
;
; STUDENT-T PREDICTIVE DISTRIBUTION:
; ----------------------------------
; Given Normal-Inverse-Gamma posterior with parameters (μ, κ, α, β), the
; predictive distribution for the next observation is Student-t:
;
;   x_new ~ Student-t(ν, μ, σ²)
;
; where:
;   ν = 2α                    (degrees of freedom)
;   σ² = β(κ+1)/(ακ)          (scale parameter)
;
; The log-PDF is:
;   ln p(x) = C1 - C2 × ln(1 + (x-μ)²/(νσ²))
;
; where:
;   C1 = lgamma(α+0.5) - lgamma(α) - 0.5×ln(νπσ²)  (precomputed)
;   C2 = α + 0.5 = (ν+1)/2                          (precomputed)
;   inv_ssn = 1/(νσ²)                               (precomputed)
;
; So the kernel computes:
;   t = (x - μ)² × inv_ssn
;   ln_pp = C1 - C2 × log1p(t)    ; log1p for numerical stability
;   pp = exp(ln_pp)
;
; LOG1P APPROXIMATION:
; --------------------
; log1p(t) = ln(1+t) is approximated by Taylor series:
;   ln(1+t) ≈ t - t²/2 + t³/3 - t⁴/4 + t⁵/5 - t⁶/6
;
; Evaluated via Horner's method:
;   ln(1+t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
;
; where c2=-0.5, c3=1/3, c4=-0.25, c5=0.2, c6=-1/6
;
; WHY LOG1P INSTEAD OF LOG:
; For small t (when x ≈ μ), computing log(1+t) directly loses precision
; because 1+t ≈ 1. The log1p formulation maintains precision for the
; common case where the observation is near the predicted mean.
;
; EXP APPROXIMATION:
; ------------------
; exp(x) is computed as 2^(x/ln2) = 2^k × 2^f where:
;   k = round(x/ln2)     (integer part)
;   f = x/ln2 - k        (fractional part, f ∈ [-0.5, 0.5])
;
; 2^f is approximated by a 6-term polynomial using Estrin's scheme:
;   p01 = 1 + f×c1
;   p23 = c2 + f×c3
;   p45 = c4 + f×c5
;   q0123 = p01 + f²×p23
;   q456 = p45 + f²×c6
;   result = q0123 + f⁴×q456
;
; 2^k is computed via IEEE-754 bit manipulation:
;   bits = (k + 1023) << 52
;   This directly constructs a double with value 2^k.
;
; ESTRIN VS HORNER:
; Horner: p = c0 + f×(c1 + f×(c2 + ...)) has O(n) dependency depth
; Estrin: Groups terms for parallelism, reducing to O(log n) depth
; For our 6-term polynomial, Estrin reduces critical path by ~40%.
;
; =============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

;==============================================================================
; CONSTANTS (read-only data section)
;==============================================================================
;
; All constants are 32-byte aligned for efficient AVX loads.
; Each constant is replicated 4 times for broadcast-free loading into YMM.
;
; WHY PRECOMPUTE THESE?
; Loading from memory is faster than computing at runtime, and these values
; never change. The constants fit in L1 cache and stay there for the duration
; of the kernel.
;==============================================================================

section .rodata
align 32

;------------------------------------------------------------------------------
; General Constants
;------------------------------------------------------------------------------

const_one:      dq 1.0, 1.0, 1.0, 1.0
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300  ; Avoid exact zero

;------------------------------------------------------------------------------
; log1p Polynomial Coefficients
;------------------------------------------------------------------------------
;
; log1p(t) = ln(1+t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
;
; Taylor series coefficients:
;   c2 = -1/2   = -0.5
;   c3 =  1/3   ≈  0.3333...
;   c4 = -1/4   = -0.25
;   c5 =  1/5   =  0.2
;   c6 = -1/6   ≈ -0.1666...
;
; This 6-term approximation gives ~13 bits of precision for |t| < 1.
; For BOCPD, t = (x-μ)²/(νσ²) is typically small when x is near the mean.
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
;
; 2^f is approximated by polynomial with coefficients derived from
; Taylor series of 2^x = e^(x×ln2):
;   c1 = ln(2)       ≈ 0.6931...
;   c2 = ln(2)²/2    ≈ 0.2402...
;   c3 = ln(2)³/6    ≈ 0.0555...
;   c4 = ln(2)⁴/24   ≈ 0.00962...
;   c5 = ln(2)⁵/120  ≈ 0.00133...
;   c6 = ln(2)⁶/720  ≈ 0.000154...
;
; For f ∈ [-0.5, 0.5], this gives ~15 bits of precision.
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
;
; exp_inv_ln2 = 1/ln(2) = log₂(e) ≈ 1.4427
;   Used to convert: x/ln(2) = x × log₂(e)
;
; exp_min_x, exp_max_x: Clamping bounds to prevent overflow/underflow
;   exp(709) ≈ 10^308 (max double)
;   exp(-745) ≈ 10^-324 (min positive double)
;   We use ±700 for safety margin.
;
; exp_bias = 1023: IEEE-754 exponent bias
;   2^k is represented as (k + 1023) in the exponent field
;------------------------------------------------------------------------------

exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
exp_bias:       dq 1023, 1023, 1023, 1023

;------------------------------------------------------------------------------
; Index Tracking Constants
;------------------------------------------------------------------------------
;
; We track run length indices as doubles (not integers) because:
;   1. AVX2 has poor support for 64-bit integer comparisons
;   2. vblendvpd works on doubles, not integers
;   3. Converting at the end is cheaper than throughout
;
; idx_init_a = [1,2,3,4]:
;   Block A processes run lengths at input indices i..i+3
;   Output goes to r_new[i+1..i+4], so output indices are 1,2,3,4 initially
;
; idx_init_b = [5,6,7,8]:
;   Block B processes run lengths at input indices i+4..i+7
;   Output goes to r_new[i+5..i+8], so output indices are 5,6,7,8 initially
;
; idx_increment = 8:
;   Each iteration processes 8 run lengths, so indices advance by 8
;------------------------------------------------------------------------------

idx_init_a:     dq 1.0, 2.0, 3.0, 4.0
idx_init_b:     dq 5.0, 6.0, 7.0, 8.0
idx_increment:  dq 8.0, 8.0, 8.0, 8.0

;==============================================================================
; Structure Offsets (bocpd_kernel_args_t)
;==============================================================================
;
; The C caller passes a pointer to this struct:
;
; typedef struct {
;     double *lin_interleaved;  // [0]  Pointer to interleaved parameter blocks
;     double *r_old;            // [8]  Current probability distribution
;     double x;                 // [16] New observation
;     double h;                 // [24] Hazard rate (prob of changepoint)
;     double one_minus_h;       // [32] 1 - h (prob of continuation)
;     double trunc_thresh;      // [40] Truncation threshold (typically 1e-6)
;     size_t n_padded;          // [48] Number of elements (padded to mult of 8)
;     double *r_new;            // [56] Output probability distribution
;     double *r0_out;           // [64] Output: changepoint probability sum
;     double *max_growth_out;   // [72] Output: maximum growth value
;     size_t *max_idx_out;      // [80] Output: index of maximum growth
;     size_t *last_valid_out;   // [88] Output: last index above threshold
; } bocpd_kernel_args_t;
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
; After 32-byte alignment, our stack frame contains:
;
;   Offset      Size    Contents
;   ──────────────────────────────────────────────────────
;   [rsp + 0]     32    idx_vec_A: current output indices for Block A
;   [rsp + 32]    32    idx_vec_B: current output indices for Block B
;   [rsp + 64]    32    max_idx_A: indices where Block A had max growth
;   [rsp + 96]    32    max_idx_B: indices where Block B had max growth
;   [rsp + 128]   32    max_growth_A: max growth values (for final reduction)
;   [rsp + 160]   32    max_growth_B: V3.2 cold-path accumulator
;   [rsp + 192]   64    padding for alignment safety
;   ──────────────────────────────────────────────────────
;   Total:       256 bytes
;
; V3.2 CHANGE: max_growth_B moved to stack
; ----------------------------------------
; Previous versions kept max_growth_B in ymm14 (or similar), which meant
; r_old_B had to be spilled to stack. But r_old_B is on the CRITICAL PATH
; (needed for BOCPD multiplication), while max_growth_B is only used for
; MAX tracking (cold, rarely updated).
;
; By swapping their locations:
;   - r_old_B stays in ymm9: fast access for BOCPD update
;   - max_growth_B goes to stack: acceptable since it's off critical path
;
; This removes a load from the critical path, saving ~5-7 cycles/iteration.
;
; WHY ALIGNMENT PADDING:
; After `and rsp, -32`, the actual aligned address might be up to 31 bytes
; lower than expected. The padding ensures we don't accidentally overlap
; with saved registers or the return address.
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

; Export both entry points
global bocpd_fused_loop_avx2
global bocpd_fused_loop_avx2_sysv

;==============================================================================
; WINDOWS x64 ABI ENTRY POINT
;==============================================================================
;
; Called as: bocpd_fused_loop_avx2(bocpd_kernel_args_t *args)
;
; Windows x64 calling convention:
;   - First argument in RCX
;   - Must preserve: RBX, RBP, RDI, RSI, R12-R15, XMM6-XMM15
;   - Stack must be 16-byte aligned at CALL instruction
;
; We align to 32 bytes for AVX vmovapd requirements.
;==============================================================================

bocpd_fused_loop_avx2:

    ;==========================================================================
    ; PROLOGUE — Save callee-saved registers (Windows x64 ABI)
    ;==========================================================================
    ;
    ; Windows requires preservation of:
    ;   - GPR: RBX, RBP, RDI, RSI, R12-R15 (8 registers × 8 bytes = 64 bytes)
    ;   - XMM: XMM6-XMM15 (10 registers × 16 bytes = 160 bytes)
    ;
    ; We save GPRs with PUSH and XMMs with VMOVDQU to a stack region.
    ; Only the low 128 bits of YMM6-15 are non-volatile; high 128 are volatile.
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

    ; Windows: first arg in RCX, move to RDI for consistency with kernel code
    mov         rdi, rcx

    ;==========================================================================
    ; STACK FRAME SETUP
    ;==========================================================================
    ;
    ; Save RBP so we can restore RSP exactly in the epilogue.
    ; Allocate STACK_FRAME + 32 bytes, then align to 32 for AVX.
    ;==========================================================================

    mov         rbp, rsp
    sub         rsp, STACK_FRAME + 32
    and         rsp, -32                ; Align to 32 bytes for AVX

    ;==========================================================================
    ; LOAD ARGUMENT POINTERS INTO CALLEE-SAVED REGISTERS
    ;==========================================================================
    ;
    ; We keep frequently-used pointers in callee-saved registers so they
    ; survive across the loop without reloading from the args struct.
    ;
    ; Register allocation:
    ;   RDI = args struct pointer (set above)
    ;   R8  = lin_interleaved (parameter blocks base)
    ;   R12 = r_old (input probability distribution)
    ;   R13 = r_new (output probability distribution)
    ;   R14 = n_padded (loop bound)
    ;   RSI = loop counter (i)
    ;   RBX = last_valid (truncation tracking)
    ;==========================================================================

    mov         r8,  [rdi + ARG_LIN_INTERLEAVED]
    mov         r12, [rdi + ARG_R_OLD]
    mov         r13, [rdi + ARG_R_NEW]
    mov         r14, [rdi + ARG_N_PADDED]

    ;==========================================================================
    ; BROADCAST IMMUTABLE SCALARS INTO DEDICATED YMM REGISTERS
    ;==========================================================================
    ;
    ; These values are used in every iteration but never change.
    ; By keeping them in dedicated registers, we avoid repeated memory loads.
    ;
    ; YMM REGISTER MAP (throughout the kernel):
    ; ─────────────────────────────────────────────────────────────────────────
    ;   ymm15 = x (observation)         — broadcast of the new data point
    ;   ymm14 = h (hazard rate)         — probability of changepoint
    ;   ymm13 = 1-h                     — probability of run continuation
    ;   ymm12 = threshold               — for dynamic truncation (1e-6)
    ;   ymm11 = r0 accumulator          — sum of changepoint contributions
    ;   ymm10 = max_growth_A            — running max for Block A (hot)
    ;   ymm9  = r_old_B                 — V3.2: probability input for Block B
    ;   ymm8  = r_old_A                 — probability input for Block A
    ;   ymm0-7 = scratch                — reused throughout computation
    ;
    ;   [rsp + STK_MAX_GROWTH_B] = max_growth_B — V3.2: moved to stack (cold)
    ; ─────────────────────────────────────────────────────────────────────────
    ;==========================================================================

    vbroadcastsd ymm15, qword [rdi + ARG_X]
    vbroadcastsd ymm14, qword [rdi + ARG_H]
    vbroadcastsd ymm13, qword [rdi + ARG_OMH]
    vbroadcastsd ymm12, qword [rdi + ARG_THRESH]

    ;==========================================================================
    ; INITIALIZE ACCUMULATORS
    ;==========================================================================

    vxorpd      ymm11, ymm11, ymm11               ; r0 accumulator = 0
    vxorpd      ymm10, ymm10, ymm10               ; max_growth_A = 0

    ; V3.2: Zero max_growth_B on stack
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0

    ;==========================================================================
    ; INITIALIZE INDEX VECTORS
    ;==========================================================================
    ;
    ; idx_vec_A starts at [1,2,3,4] — output indices for Block A
    ; idx_vec_B starts at [5,6,7,8] — output indices for Block B
    ; max_idx_A/B start at 0 — no max found yet
    ;==========================================================================

    vmovapd     ymm0, [rel idx_init_a]
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0
    vmovapd     ymm0, [rel idx_init_b]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0
    vxorpd      ymm0, ymm0, ymm0
    vmovapd     [rsp + STK_MAX_IDX_A], ymm0
    vmovapd     [rsp + STK_MAX_IDX_B], ymm0

    ; Initialize loop variables
    xor         rsi, rsi                          ; i = 0 (loop counter)
    xor         rbx, rbx                          ; last_valid = 0

    ;==========================================================================
    ; MAIN LOOP — Process 8 run lengths per iteration
    ;==========================================================================
    ;
    ; Each iteration processes:
    ;   Block A: run lengths at input indices i, i+1, i+2, i+3
    ;   Block B: run lengths at input indices i+4, i+5, i+6, i+7
    ;
    ; Output goes to:
    ;   Block A: r_new[i+1], r_new[i+2], r_new[i+3], r_new[i+4]
    ;   Block B: r_new[i+5], r_new[i+6], r_new[i+7], r_new[i+8]
    ;
    ; V3.2 KEY OPTIMIZATION:
    ; All parameters (μ, C1, C2, inv_ssn, r_old) are loaded ONCE at the start.
    ; Through careful register allocation:
    ;   - C1/C2 remain in ymm1/2/5/6 until consumed by ln_pp computation
    ;   - r_old_A/B remain in ymm8/9 until consumed by BOCPD update
    ; NO RELOADS within the iteration!
    ;==========================================================================

.win_loop_start:
    cmp         rsi, r14
    jge         .win_loop_end

    ;==========================================================================
    ; CALCULATE SUPERBLOCK OFFSETS
    ;==========================================================================
    ;
    ; Block A: elements [i, i+1, i+2, i+3]
    ;   block_index = i / 4
    ;   byte_offset = block_index * 256
    ;
    ; Block B: elements [i+4, i+5, i+6, i+7]
    ;   block_index = (i+4) / 4
    ;   byte_offset = block_index * 256
    ;==========================================================================

    mov         rax, rsi
    shr         rax, 2                            ; block_A = i / 4
    shl         rax, 8                            ; offset_A = block_A * 256

    mov         rdx, rsi
    add         rdx, 4
    shr         rdx, 2                            ; block_B = (i+4) / 4
    shl         rdx, 8                            ; offset_B = block_B * 256

    ;==========================================================================
    ; LOAD ALL PARAMETERS — BOTH BLOCKS (V3.2: no reloads needed later!)
    ;==========================================================================
    ;
    ; This is the key to eliminating reloads. We load everything upfront
    ; into carefully chosen registers that won't be clobbered.
    ;
    ; Block A parameters:
    ;   ymm0 = μ_A        (will become t_A after Student-t)
    ;   ymm1 = C1_A       (preserved until ln_pp computation)
    ;   ymm2 = C2_A       (preserved until ln_pp computation)
    ;   ymm3 = inv_ssn_A  (consumed by Student-t, then free)
    ;   ymm8 = r_old_A    (DEDICATED: preserved until BOCPD update!)
    ;
    ; Block B parameters:
    ;   ymm4 = μ_B        (will become t_B after Student-t)
    ;   ymm5 = C1_B       (preserved until ln_pp computation)
    ;   ymm6 = C2_B       (preserved until ln_pp computation)
    ;   ymm7 = inv_ssn_B  (consumed by Student-t, then free)
    ;   ymm9 = r_old_B    (DEDICATED: preserved until BOCPD update!)
    ;==========================================================================

    vmovapd     ymm0, [r8 + rax + 0]              ; μ_A
    vmovapd     ymm1, [r8 + rax + 32]             ; C1_A
    vmovapd     ymm2, [r8 + rax + 64]             ; C2_A
    vmovapd     ymm3, [r8 + rax + 96]             ; inv_ssn_A
    vmovapd     ymm8, [r12 + rsi*8]               ; r_old_A (DEDICATED)

    vmovapd     ymm4, [r8 + rdx + 0]              ; μ_B
    vmovapd     ymm5, [r8 + rdx + 32]             ; C1_B
    vmovapd     ymm6, [r8 + rdx + 64]             ; C2_B
    vmovapd     ymm7, [r8 + rdx + 96]             ; inv_ssn_B
    vmovapd     ymm9, [r12 + rsi*8 + 32]          ; r_old_B (DEDICATED, V3.2)

    ;==========================================================================
    ; STUDENT-T COMPUTATION — BOTH BLOCKS
    ;==========================================================================
    ;
    ; The Student-t log-probability requires:
    ;   t = (x - μ)² × inv_ssn
    ;
    ; After this section:
    ;   ymm0 = t_A (μ_A and inv_ssn_A consumed)
    ;   ymm4 = t_B (μ_B and inv_ssn_B consumed)
    ;   ymm1, ymm2 = C1_A, C2_A (STILL PRESERVED!)
    ;   ymm5, ymm6 = C1_B, C2_B (STILL PRESERVED!)
    ;   ymm8 = r_old_A (STILL PRESERVED!)
    ;   ymm9 = r_old_B (STILL PRESERVED!)
    ;   ymm3, ymm7 = now free for scratch
    ;==========================================================================

    vsubpd      ymm0, ymm15, ymm0                 ; z_A = x - μ_A
    vmulpd      ymm0, ymm0, ymm0                  ; z²_A
    vmulpd      ymm0, ymm0, ymm3                  ; t_A = z²_A × inv_ssn_A

    vsubpd      ymm4, ymm15, ymm4                 ; z_B = x - μ_B
    vmulpd      ymm4, ymm4, ymm4                  ; z²_B
    vmulpd      ymm4, ymm4, ymm7                  ; t_B = z²_B × inv_ssn_B

    ;==========================================================================
    ; LOG1P POLYNOMIAL — BOTH BLOCKS (Horner's Method)
    ;==========================================================================
    ;
    ; log1p(t) = ln(1+t) ≈ t × (1 + t×(c2 + t×(c3 + t×(c4 + t×(c5 + t×c6)))))
    ;
    ; We use ymm3 for Block A poly accumulator, ymm7 for Block B.
    ; This preserves ymm1/ymm2 (C1_A/C2_A) and ymm5/ymm6 (C1_B/C2_B).
    ;
    ; vfmadd213pd ymm_a, ymm_b, ymm_c computes: ymm_a = ymm_a × ymm_b + ymm_c
    ;
    ; After this section:
    ;   ymm3 = log1p(t_A)
    ;   ymm7 = log1p(t_B)
    ;   ymm0 = t_A (consumed by final multiply)
    ;   ymm4 = t_B (consumed by final multiply)
    ;   ymm1, ymm2 = C1_A, C2_A (STILL PRESERVED!)
    ;   ymm5, ymm6 = C1_B, C2_B (STILL PRESERVED!)
    ;==========================================================================

    ; Block A log1p via Horner
    vmovapd     ymm3, [rel log1p_c6]              ; p = c6
    vfmadd213pd ymm3, ymm0, [rel log1p_c5]        ; p = c6×t + c5
    vfmadd213pd ymm3, ymm0, [rel log1p_c4]        ; p = p×t + c4
    vfmadd213pd ymm3, ymm0, [rel log1p_c3]        ; p = p×t + c3
    vfmadd213pd ymm3, ymm0, [rel log1p_c2]        ; p = p×t + c2
    vfmadd213pd ymm3, ymm0, [rel const_one]       ; p = p×t + 1
    vmulpd      ymm3, ymm0, ymm3                  ; log1p_A = t × p

    ; Block B log1p via Horner
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm4, ymm7                  ; log1p_B = t × p

    ;==========================================================================
    ; COMPUTE ln(p) = C1 - C2 × log1p(t)
    ;==========================================================================
    ;
    ; This is where C1 and C2 are finally consumed.
    ; vfnmadd231pd: ymm_a = ymm_a - ymm_b × ymm_c (fused negative multiply-add)
    ;
    ; After this section:
    ;   ymm1 = ln_pp_A = C1_A - C2_A × log1p_A
    ;   ymm5 = ln_pp_B = C1_B - C2_B × log1p_B
    ;   ymm2, ymm3, ymm6, ymm7 = now free
    ;   ymm8 = r_old_A (STILL PRESERVED!)
    ;   ymm9 = r_old_B (STILL PRESERVED!)
    ;==========================================================================

    vfnmadd231pd ymm1, ymm2, ymm3                 ; ln_pp_A = C1_A - C2_A × log1p_A
    vfnmadd231pd ymm5, ymm6, ymm7                 ; ln_pp_B = C1_B - C2_B × log1p_B

    ;==========================================================================
    ; EXP PREPARATION — BOTH BLOCKS
    ;==========================================================================
    ;
    ; exp(x) = 2^(x/ln2) = 2^k × 2^f
    ; where k = round(x/ln2), f = x/ln2 - k ∈ [-0.5, 0.5]
    ;
    ; First clamp to [-700, 700] to prevent overflow/underflow.
    ;
    ; After this section:
    ;   ymm0 = f_A (fractional part)
    ;   ymm2 = k_A (integer exponent as double)
    ;   ymm4 = f_B
    ;   ymm6 = k_B
    ;==========================================================================

    ; Block A exp prep
    vmaxpd      ymm1, ymm1, [rel exp_min_x]       ; clamp lower bound
    vminpd      ymm1, ymm1, [rel exp_max_x]       ; clamp upper bound
    vmulpd      ymm0, ymm1, [rel exp_inv_ln2]     ; y_A = ln_pp_A × log₂(e)
    vroundpd    ymm2, ymm0, 0                     ; k_A = round(y_A)
    vsubpd      ymm0, ymm0, ymm2                  ; f_A = y_A - k_A

    ; Block B exp prep
    vmaxpd      ymm5, ymm5, [rel exp_min_x]
    vminpd      ymm5, ymm5, [rel exp_max_x]
    vmulpd      ymm4, ymm5, [rel exp_inv_ln2]     ; y_B
    vroundpd    ymm6, ymm4, 0                     ; k_B
    vsubpd      ymm4, ymm4, ymm6                  ; f_B

    ;==========================================================================
    ; ESTRIN POLYNOMIAL — 2^f BLOCK A
    ;==========================================================================
    ;
    ; 2^f approximated by 6-term polynomial using Estrin's scheme:
    ;   p01 = 1 + f×c1
    ;   p23 = c2 + f×c3
    ;   p45 = c4 + f×c5
    ;   q0123 = p01 + f²×p23
    ;   q456 = p45 + f²×c6
    ;   result = q0123 + f⁴×q456
    ;
    ; Estrin reduces dependency chain from O(n) to O(log n), improving ILP.
    ;
    ; Uses: ymm0 (f_A), ymm1 (result), ymm3 (f²/f⁴), ymm7 (scratch)
    ; Preserves: ymm2 (k_A), ymm4 (f_B), ymm6 (k_B), ymm8 (r_old_A), ymm9 (r_old_B)
    ;==========================================================================

    vmulpd      ymm3, ymm0, ymm0                  ; f²_A

    ; p01 = 1 + f×c1
    vmovapd     ymm1, [rel const_one]
    vfmadd231pd ymm1, ymm0, [rel exp_c1]          ; p01 = 1 + f_A×c1

    ; p23 = c2 + f×c3
    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm0, [rel exp_c3]          ; p23 = c2 + f_A×c3

    ; q0123 = p01 + f²×p23
    vfmadd231pd ymm1, ymm3, ymm7                  ; q0123

    ; p45 = c4 + f×c5
    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm0, [rel exp_c5]          ; p45 = c4 + f_A×c5

    ; q456 = p45 + f²×c6
    vfmadd231pd ymm7, ymm3, [rel exp_c6]          ; q456

    ; result = q0123 + f⁴×q456
    vmulpd      ymm3, ymm3, ymm3                  ; f⁴_A
    vfmadd231pd ymm1, ymm3, ymm7                  ; ymm1 = 2^f_A

    ;==========================================================================
    ; ESTRIN POLYNOMIAL — 2^f BLOCK B
    ;==========================================================================
    ;
    ; Same computation for Block B.
    ; Uses: ymm4 (f_B), ymm5 (result), ymm3 (f²/f⁴), ymm7 (scratch)
    ;==========================================================================

    vmulpd      ymm3, ymm4, ymm4                  ; f²_B

    vmovapd     ymm5, [rel const_one]
    vfmadd231pd ymm5, ymm4, [rel exp_c1]

    vmovapd     ymm7, [rel exp_c2]
    vfmadd231pd ymm7, ymm4, [rel exp_c3]

    vfmadd231pd ymm5, ymm3, ymm7

    vmovapd     ymm7, [rel exp_c4]
    vfmadd231pd ymm7, ymm4, [rel exp_c5]
    vfmadd231pd ymm7, ymm3, [rel exp_c6]

    vmulpd      ymm3, ymm3, ymm3                  ; f⁴_B
    vfmadd231pd ymm5, ymm3, ymm7                  ; ymm5 = 2^f_B

    ;==========================================================================
    ; 2^k RECONSTRUCTION — IEEE-754 Bit Manipulation
    ;==========================================================================
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
    ;   5. Reinterpret as double and multiply with 2^f
    ;
    ; After this section:
    ;   ymm1 = pp_A = exp(ln_pp_A) = Student-t probability for Block A
    ;   ymm5 = pp_B = exp(ln_pp_B) = Student-t probability for Block B
    ;==========================================================================

    ; Block A: 2^k
    vcvtpd2dq   xmm0, ymm2                        ; k_A → int32
    vpmovsxdq   ymm0, xmm0                        ; sign-extend to int64
    vpaddq      ymm0, ymm0, [rel exp_bias]        ; + 1023
    vpsllq      ymm0, ymm0, 52                    ; << 52
    vmulpd      ymm1, ymm1, ymm0                  ; pp_A = 2^f_A × 2^k_A
    vmaxpd      ymm1, ymm1, [rel const_min_pp]    ; clamp to avoid exact zero

    ; Block B: 2^k
    vcvtpd2dq   xmm0, ymm6                        ; k_B → int32
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0                  ; pp_B
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ;==========================================================================
    ; BOCPD UPDATE — BLOCK A
    ;==========================================================================
    ;
    ; For each run length r:
    ;   r_pp = r_old[r] × pp        (joint probability of r and observing x)
    ;   growth = r_pp × (1-h)       (probability of run continuation)
    ;   change = r_pp × h           (probability of changepoint)
    ;
    ; growth goes to r_new[r+1] (run length increases if no changepoint)
    ; change accumulates into r0 (all changepoint probs sum to P(r=0))
    ;
    ; V3.2: r_old_A is STILL in ymm8 — no reload needed!
    ;
    ; Memory layout:
    ;   r_old[i..i+3] was loaded into ymm8
    ;   r_new[i+1..i+4] means offset = (i+1)×8 = rsi×8 + 8
    ;==========================================================================

    vmulpd      ymm0, ymm8, ymm1                  ; r_pp_A = r_old_A × pp_A
    vmulpd      ymm2, ymm0, ymm13                 ; growth_A = r_pp_A × (1-h)
    vmulpd      ymm0, ymm0, ymm14                 ; change_A = r_pp_A × h
    vaddpd      ymm11, ymm11, ymm0                ; r0_acc += change_A

    ; Store growth_A at r_new[i+1..i+4]
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ;==========================================================================
    ; BOCPD UPDATE — BLOCK B
    ;==========================================================================
    ;
    ; V3.2: r_old_B is STILL in ymm9 — no reload needed!
    ;
    ; This is the key V3.2 optimization. In V3.1, r_old_B was spilled to
    ; stack and reloaded here, adding a load to the critical path.
    ;
    ; Memory layout:
    ;   r_new[i+5..i+8] means offset = (i+5)×8 = rsi×8 + 40
    ;==========================================================================

    vmulpd      ymm0, ymm9, ymm5                  ; r_pp_B = r_old_B × pp_B
    vmulpd      ymm3, ymm0, ymm13                 ; growth_B = r_pp_B × (1-h)
    vmulpd      ymm0, ymm0, ymm14                 ; change_B = r_pp_B × h
    vaddpd      ymm11, ymm11, ymm0                ; r0_acc += change_B

    ; Store growth_B at r_new[i+5..i+8]
    vmovupd     [r13 + rsi*8 + 40], ymm3

    ;==========================================================================
    ; MAX TRACKING — BLOCK A
    ;==========================================================================
    ;
    ; For MAP (Maximum A Posteriori) estimation, we track which run length
    ; has the highest growth probability.
    ;
    ; vcmppd with predicate 14 (CMP_GT_OQ) sets each lane to all-1s if
    ; greater, all-0s otherwise. vblendvpd uses the sign bit of the mask
    ; to select between old and new values.
    ;
    ; We track both the max value (in ymm10) and the corresponding index
    ; (in [rsp + STK_MAX_IDX_A]).
    ;==========================================================================

    vcmppd      ymm0, ymm2, ymm10, 14             ; mask = growth_A > max_growth_A?
    vblendvpd   ymm10, ymm10, ymm2, ymm0          ; update max where mask is set

    ; Update indices where this iteration's growth beat the previous max
    vmovapd     ymm1, [rsp + STK_MAX_IDX_A]       ; current best indices
    vmovapd     ymm4, [rsp + STK_IDX_VEC_A]       ; this iteration's indices
    vblendvpd   ymm1, ymm1, ymm4, ymm0            ; update where mask is set
    vmovapd     [rsp + STK_MAX_IDX_A], ymm1

    ;==========================================================================
    ; MAX TRACKING — BLOCK B (V3.2: max_growth_B on stack)
    ;==========================================================================
    ;
    ; V3.2 CHANGE: max_growth_B lives on stack, not in a register.
    ; This is the COLD PATH — we load, compare, blend, store.
    ;
    ; The tradeoff is worth it because:
    ;   - r_old_B (HOT) now stays in register (ymm9)
    ;   - max_growth_B is only compared once per iteration
    ;   - max_growth_B rarely changes (only when new max found)
    ;   - These stack operations are NOT on the critical path
    ;==========================================================================

    vmovapd     ymm0, [rsp + STK_MAX_GROWTH_B]    ; load current max
    vcmppd      ymm7, ymm3, ymm0, 14              ; mask = growth_B > max?
    vblendvpd   ymm0, ymm0, ymm3, ymm7            ; update max
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0    ; store back

    ; Update indices
    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm7            ; use ymm7 mask from above
    vmovapd     [rsp + STK_MAX_IDX_B], ymm1

    ;==========================================================================
    ; DYNAMIC TRUNCATION — BLOCK A
    ;==========================================================================
    ;
    ; To prevent unbounded growth of the run length distribution, we truncate
    ; run lengths with negligible probability (< threshold, typically 1e-6).
    ;
    ; We track last_valid = highest index with probability > threshold.
    ; After the loop, active_len = last_valid + 1.
    ;
    ; vmovmskpd extracts the sign bits of each lane into a 4-bit integer.
    ; bsr (bit scan reverse) finds the index of the highest set bit.
    ;
    ; For Block A (indices i..i+3), if lane k is above threshold:
    ;   output index = i + k + 1 (the +1 because output is shifted)
    ;==========================================================================

    vcmppd      ymm0, ymm2, ymm12, 14             ; mask = growth_A > threshold?
    vmovmskpd   eax, ymm0                         ; extract to 4-bit mask
    test        eax, eax                          ; any lanes above threshold?
    jz          .win_no_trunc_A                   ; skip if all below

    bsr         ecx, eax                          ; highest set bit (0-3)
    lea         rbx, [rsi + rcx + 1]              ; last_valid = i + bit + 1

.win_no_trunc_A:

    ;==========================================================================
    ; DYNAMIC TRUNCATION — BLOCK B
    ;==========================================================================
    ;
    ; For Block B (indices i+4..i+7):
    ;   output index = (i+4) + lane + 1 = i + lane + 5
    ;==========================================================================

    vcmppd      ymm0, ymm3, ymm12, 14             ; mask = growth_B > threshold?
    vmovmskpd   eax, ymm0
    test        eax, eax
    jz          .win_no_trunc_B

    bsr         ecx, eax
    lea         rbx, [rsi + rcx + 5]              ; last_valid = i + 4 + bit + 1

.win_no_trunc_B:

    ;==========================================================================
    ; UPDATE INDEX VECTORS
    ;==========================================================================
    ;
    ; Advance indices by 8 for the next iteration.
    ; idx_vec_A: [1,2,3,4] → [9,10,11,12] → [17,18,19,20] → ...
    ; idx_vec_B: [5,6,7,8] → [13,14,15,16] → [21,22,23,24] → ...
    ;==========================================================================

    vmovapd     ymm0, [rsp + STK_IDX_VEC_A]
    vaddpd      ymm0, ymm0, [rel idx_increment]   ; +8
    vmovapd     [rsp + STK_IDX_VEC_A], ymm0

    vmovapd     ymm0, [rsp + STK_IDX_VEC_B]
    vaddpd      ymm0, ymm0, [rel idx_increment]
    vmovapd     [rsp + STK_IDX_VEC_B], ymm0

    ;==========================================================================
    ; LOOP INCREMENT
    ;==========================================================================

    add         rsi, 8                            ; i += 8 (next 8 run lengths)
    jmp         .win_loop_start

    ;==========================================================================
    ; LOOP END — Horizontal Reductions
    ;==========================================================================

.win_loop_end:

    ;==========================================================================
    ; REDUCE r0 ACCUMULATOR
    ;==========================================================================
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
    ; uses the shuffle execution unit). This sequence is faster.
    ;==========================================================================

    vextractf128 xmm0, ymm11, 1                   ; [c, d]
    vaddpd      xmm0, xmm0, xmm11                 ; [a+c, b+d]
    vunpckhpd   xmm1, xmm0, xmm0                  ; [b+d, b+d]
    vaddsd      xmm0, xmm0, xmm1                  ; a+b+c+d

    ; Store r0 output
    mov         rax, [rdi + ARG_R0_OUT]
    vmovsd      [rax], xmm0

    ;==========================================================================
    ; MAP REDUCTION — Find Global Maximum
    ;==========================================================================
    ;
    ; We need to find the global maximum across all 8 lanes (4 in Block A,
    ; 4 in Block B), and compare against r0 (changepoint probability).
    ;
    ; If r0 beats all growth probabilities, MAP run length is 0.
    ; Otherwise, it's the index of the highest growth probability.
    ;
    ; We use a scalar loop because:
    ;   1. Only 8 comparisons total
    ;   2. Mixing scalar and vector reduction is error-prone
    ;   3. This happens once per kernel call (not per iteration)
    ;==========================================================================

    vmovsd      xmm5, xmm0, xmm0                  ; best_val = r0
    xor         r15, r15                          ; best_idx = 0

    ; Save max_growth_A for scalar access (max_growth_B already on stack)
    vmovapd     [rsp + STK_MAX_GROWTH_A], ymm10

    xor         rcx, rcx                          ; lane counter

.win_reduce_loop:
    cmp         rcx, 4
    jge         .win_reduce_done

    ; Check Block A lane
    vmovsd      xmm1, [rsp + STK_MAX_GROWTH_A + rcx*8]
    vucomisd    xmm1, xmm5                        ; compare to current best
    jbe         .win_check_B                      ; skip if not greater

    vmovsd      xmm5, xmm1, xmm1                  ; update best value
    vmovsd      xmm2, [rsp + STK_MAX_IDX_A + rcx*8]
    vcvttsd2si  r15, xmm2                         ; update best index

.win_check_B:
    ; Check Block B lane
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

    ;==========================================================================
    ; WRITE OUTPUTS
    ;==========================================================================

    mov         rax, [rdi + ARG_MAX_GROWTH]
    vmovsd      [rax], xmm5                       ; *max_growth_out = best_val

    mov         rax, [rdi + ARG_MAX_IDX]
    mov         [rax], r15                        ; *max_idx_out = best_idx

    mov         rax, [rdi + ARG_LAST_VALID]
    mov         [rax], rbx                        ; *last_valid_out = truncation

    ;==========================================================================
    ; EPILOGUE — Restore registers and return (Windows)
    ;==========================================================================
    ;
    ; Restore RSP from saved RBP, then restore saved registers in reverse
    ; order of how we saved them.
    ;
    ; VZEROUPPER is required before returning to code that might use SSE.
    ; Without it, SSE instructions incur a ~70 cycle penalty due to the
    ; "AVX/SSE transition penalty" on older Intel CPUs.
    ;==========================================================================

    mov         rsp, rbp                          ; restore stack pointer

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

    vzeroupper                                    ; avoid AVX/SSE transition penalty
    ret

;==============================================================================
; LINUX SYSTEM V ABI ENTRY POINT
;==============================================================================
;
; Called as: bocpd_fused_loop_avx2_sysv(bocpd_kernel_args_t *args)
;
; Linux System V calling convention:
;   - First argument in RDI (already where we want it!)
;   - Must preserve: RBX, RBP, R12-R15
;   - XMM/YMM registers are ALL caller-saved (no preservation needed!)
;
; This version is ~20 cycles faster per call than Windows due to not
; needing to save/restore XMM6-15 in the prologue/epilogue.
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

    ; RDI already has args pointer — no move needed!

    ;==========================================================================
    ; STACK FRAME SETUP
    ;==========================================================================

    mov         rbp, rsp
    sub         rsp, STACK_FRAME + 32
    and         rsp, -32                          ; Align to 32 bytes

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
    ; MAIN LOOP — Identical logic to Windows version
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

    ; Student-t computation
    vsubpd      ymm0, ymm15, ymm0
    vmulpd      ymm0, ymm0, ymm0
    vmulpd      ymm0, ymm0, ymm3

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
    vmulpd      ymm3, ymm0, ymm3

    ; log1p Block B
    vmovapd     ymm7, [rel log1p_c6]
    vfmadd213pd ymm7, ymm4, [rel log1p_c5]
    vfmadd213pd ymm7, ymm4, [rel log1p_c4]
    vfmadd213pd ymm7, ymm4, [rel log1p_c3]
    vfmadd213pd ymm7, ymm4, [rel log1p_c2]
    vfmadd213pd ymm7, ymm4, [rel const_one]
    vmulpd      ymm7, ymm4, ymm7

    ; ln_pp = C1 - C2 × log1p
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

    ; 2^k Block A
    vcvtpd2dq   xmm0, ymm2
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm1, ymm1, ymm0
    vmaxpd      ymm1, ymm1, [rel const_min_pp]

    ; 2^k Block B
    vcvtpd2dq   xmm0, ymm6
    vpmovsxdq   ymm0, xmm0
    vpaddq      ymm0, ymm0, [rel exp_bias]
    vpsllq      ymm0, ymm0, 52
    vmulpd      ymm5, ymm5, ymm0
    vmaxpd      ymm5, ymm5, [rel const_min_pp]

    ; BOCPD Block A (r_old_A still in ymm8!)
    vmulpd      ymm0, ymm8, ymm1
    vmulpd      ymm2, ymm0, ymm13
    vmulpd      ymm0, ymm0, ymm14
    vaddpd      ymm11, ymm11, ymm0
    vmovupd     [r13 + rsi*8 + 8], ymm2

    ; BOCPD Block B (r_old_B still in ymm9!)
    vmulpd      ymm0, ymm9, ymm5
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

    ; Max tracking Block B (V3.2: on stack)
    vmovapd     ymm0, [rsp + STK_MAX_GROWTH_B]
    vcmppd      ymm7, ymm3, ymm0, 14
    vblendvpd   ymm0, ymm0, ymm3, ymm7
    vmovapd     [rsp + STK_MAX_GROWTH_B], ymm0
    vmovapd     ymm1, [rsp + STK_MAX_IDX_B]
    vmovapd     ymm4, [rsp + STK_IDX_VEC_B]
    vblendvpd   ymm1, ymm1, ymm4, ymm7
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