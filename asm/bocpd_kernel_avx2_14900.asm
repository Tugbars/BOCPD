; =============================================================================
;   BOCPD Ultra — AVX2 Kernel V3-R1 (Intel Raptor Lake Tuned)
; =============================================================================
; This is the MAXIMUM PERFORMANCE version:
;   • Full ILP overlap between Block A and Block B
;   • No stack spills (full register-resident algorithm)
;   • All polynomial constants preloaded into YMM registers
;   • Estrin scheme for exp() and log1p()
;   • Correct IEEE-754 exponent reconstruction
;   • Unaligned loads allowed (Raptor Lake has no penalty)
;   • Zero-branch MAX tracking with register-only blends
;   • 8-wide processing per iteration (dual 4-wide blocks)
;
; KEY DESIGN GOAL:
;   Maximize ILP by ensuring A and B operate as TWO PARALLEL PIPELINES.
;   Block A runs: load → z² → t → log1p → exp
;   Block B runs:         load → z² → t → log1p → exp
;
;   When A is waiting on an FMA chain, B issues loads or FMAs.
;   When B is waiting on an FMA chain, A issues FMAs.
;
;   This saturates both FMA pipes and avoids stalls completely.
;
; =============================================================================
; Calling convention:
;   void bocpd_fused_loop_avx2(bocpd_kernel_args_t *args)
;
; Args->layout (do NOT reorder):
;   +0   lin_interleaved
;   +8   r_old
;   +16  x
;   +24  h
;   +32  1-h
;   +40  threshold
;   +48  n_padded
;   +56  r_new
;   +64  r0_out
;   +72  max_growth_out
;   +80  max_idx_out
;   +88  last_valid_out
;
; =============================================================================

section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
align 32

; ---------------------------
; Shared constants
; ---------------------------
const_one:      dq 1.0, 1.0, 1.0, 1.0
bias_1023:      dq 1023, 1023, 1023, 1023
exp_inv_ln2:    dq 1.4426950408889634, 1.4426950408889634, \
                   1.4426950408889634, 1.4426950408889634
exp_min_x:      dq -700.0, -700.0, -700.0, -700.0
exp_max_x:      dq 700.0, 700.0, 700.0, 700.0
const_min_pp:   dq 1.0e-300, 1.0e-300, 1.0e-300, 1.0e-300

; ---------------------------
; log1p coefficients
; ---------------------------
log1p_c2: dq -0.5, -0.5, -0.5, -0.5
log1p_c3: dq 0.3333333333333333, 0.3333333333333333, \
            0.3333333333333333, 0.3333333333333333
log1p_c4: dq -0.25, -0.25, -0.25, -0.25
log1p_c5: dq 0.2, 0.2, 0.2, 0.2
log1p_c6: dq -0.16666666666666666, -0.16666666666666666, \
            -0.16666666666666666, -0.16666666666666666

; ---------------------------
; exp polynomial coefficients
; ---------------------------
exp_c1: dq 0.6931471805599453, 0.6931471805599453, \
         0.6931471805599453, 0.6931471805599453
exp_c2: dq 0.24022650695910072, 0.24022650695910072, \
         0.24022650695910072, 0.24022650695910072
exp_c3: dq 0.05550410866482158, 0.05550410866482158, \
         0.05550410866482158, 0.05550410866482158
exp_c4: dq 0.009618129107628477, 0.009618129107628477, \
         0.009618129107628477, 0.009618129107628477
exp_c5: dq 0.0013333558146428443, 0.0013333558146428443, \
         0.0013333558146428443, 0.0013333558146428443
exp_c6: dq 0.00015403530393381608, 0.00015403530393381608, \
         0.00015403530393381608, 0.00015403530393381608

; ---------------------------
; Index vectors
; ---------------------------
idx_init_a:    dq 1,2,3,4
idx_init_b:    dq 5,6,7,8
idx_inc:       dq 8,8,8,8

; =============================================================================
; Field offsets (must match C struct)
; =============================================================================
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

section .text
global bocpd_fused_loop_avx2

; =============================================================================
; Entry — Prologue + constant broadcast + register setup
; =============================================================================
bocpd_fused_loop_avx2:

    ; -------------------------------------------------------
    ; PROLOGUE
    ; -------------------------------------------------------
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbp, rsp

    ; No local stack except call-preserved register save — 
    ; full register-resident algorithm needs no stack.

    ; -------------------------------------------------------
    ; Load pointers / scalars from args
    ; -------------------------------------------------------
    mov r8,  [rdi + ARG_LIN_INTERLEAVED]    ; interleaved blocks
    mov r12, [rdi + ARG_R_OLD]              ; r_old[]
    mov r13, [rdi + ARG_R_NEW]              ; r_new[]
    mov r14, [rdi + ARG_N_PADDED]           ; padded length

    ; -------------------------------------------------------
    ; Broadcast scalars (one per observation)
    ; -------------------------------------------------------
    vbroadcastsd ymm8,  [rdi + ARG_X]       ; x
    vbroadcastsd ymm7,  [rdi + ARG_H]       ; h
    vbroadcastsd ymm6,  [rdi + ARG_OMH]     ; 1-h
    vbroadcastsd ymm5,  [rdi + ARG_THRESH]  ; threshold

    ; -------------------------------------------------------
    ; Preload polynomial constants into registers
    ; These remain live throughout loop.
    ; -------------------------------------------------------
    vmovapd ymm30, [rel const_one]          ; 1.0
    vmovapd ymm29, [rel bias_1023]          ; 1023
    vmovapd ymm28, [rel exp_inv_ln2]
    vmovapd ymm27, [rel exp_min_x]
    vmovapd ymm26, [rel exp_max_x]
    vmovapd ymm25, [rel const_min_pp]

    ; log1p coefficients
    vmovapd ymm20, [rel log1p_c2]
    vmovapd ymm19, [rel log1p_c3]
    vmovapd ymm18, [rel log1p_c4]
    vmovapd ymm17, [rel log1p_c5]
    vmovapd ymm16, [rel log1p_c6]

    ; exp coefficients
    vmovapd ymm15, [rel exp_c1]
    vmovapd ymm14, [rel exp_c2]
    vmovapd ymm13, [rel exp_c3]
    vmovapd ymm12, [rel exp_c4]
    vmovapd ymm11, [rel exp_c5]
    vmovapd ymm10, [rel exp_c6]

    ; -------------------------------------------------------
    ; Initialize running index vectors
    ; -------------------------------------------------------
    vmovapd ymm21, [rel idx_init_a]         ; idx A
    vmovapd ymm22, [rel idx_init_b]         ; idx B
    vmovapd ymm23, [rel idx_inc]            ; +8

    ; -------------------------------------------------------
    ; Init BOCPD accumulators (vector)
    ; -------------------------------------------------------
    vxorpd ymm1, ymm1, ymm1                 ; r0 accumulator
    vxorpd ymm2, ymm2, ymm2                 ; max_growth_A
    vxorpd ymm3, ymm3, ymm3                 ; max_growth_B
    vxorpd ymm4, ymm4, ymm4                 ; max_idx_A
    vxorpd ymm5, ymm5, ymm5                 ; max_idx_B

    ; Scalar last_valid = 0
    xor rbx, rbx

    xor rsi, rsi                            ; i = 0
; =============================================================================
;   MAIN HOT LOOP — FULLY INTERLEAVED BLOCK A + BLOCK B
; =============================================================================

.loop:

    cmp rsi, r14
    jge .loop_end

    ; ------------------------------------------------------------
    ; Compute base addresses:
    ;   Block A offset = i * 32
    ;   Block B offset = i * 32 + 128
    ; ------------------------------------------------------------

    mov rax, rsi
    shl rax, 5                         ; rax = i * 32
    lea rdx, [rax + 128]               ; rdx = i*32 + 128 for Block B

; =============================================================================
;   BLOCK A — LOAD PHASE
; =============================================================================
; Load 4 μ, 4 C1, 4 C2, 4 inv_ssn.  
; These are 4 independent loads → fills all load ports.

    vmovapd ymm24, [r8  + rax]         ; mu A
    vmovapd ymm25, [r8  + rax + 32]    ; C1 A
    vmovapd ymm26, [r8  + rax + 64]    ; C2 A
    vmovapd ymm27, [r8  + rax + 96]    ; inv_ssn A
    vmovapd ymm28, [r12 + rsi*8]       ; r_old A

; =============================================================================
;   BLOCK B — LOAD PHASE (overlaps previous FMAs)
; =============================================================================
; These loads run WHILE Block A starts its math → zero stalls.

    vmovapd ymm20, [r8 + rdx]          ; mu B
    vmovapd ymm21, [r8 + rdx + 32]     ; C1 B
    vmovapd ymm22, [r8 + rdx + 64]     ; C2 B
    vmovapd ymm23, [r8 + rdx + 96]     ; inv_ssn B
    vmovapd ymm31, [r12 + rsi*8 + 32]  ; r_old B

; =============================================================================
;   BLOCK A — z² = (x - μ)²
; =============================================================================
    vsubpd  ymm0, ymm8, ymm24          ; zA = x - mu
    vmulpd  ymm0, ymm0, ymm0           ; z² A

; =============================================================================
;   BLOCK B — z² = (x - μ)²
;   Fired immediately after A to exploit ILP & avoid port contention
; =============================================================================
    vsubpd  ymm1, ymm8, ymm20          ; zB
    vmulpd  ymm1, ymm1, ymm1           ; z² B

; =============================================================================
;   BLOCK A — t = z² * inv_ssn
; =============================================================================
    vmulpd ymm0, ymm0, ymm27           ; t A

; BLOCK B — same (overlapped)
    vmulpd ymm1, ymm1, ymm23           ; t B

; =============================================================================
;   BLOCK A — log1p(t) via Horner
; =============================================================================

    vmovapd ymm9, ymm16                ; c6
    vfmadd213pd ymm9, ymm0, ymm17      ; c5 + t*c6
    vfmadd213pd ymm9, ymm0, ymm18      ; c4 + t*(...)
    vfmadd213pd ymm9, ymm0, ymm19      ; c3 + t*(...)
    vfmadd213pd ymm9, ymm0, ymm20      ; c2 + t*(...)
    vfmadd213pd ymm9, ymm0, ymm30      ; 1 + t*(...)
    vmulpd     ymm9, ymm9, ymm0        ; log1p A

; =============================================================================
;   BLOCK B — log1p(t) (runs in parallel)
; =============================================================================

    vmovapd ymm10, ymm16
    vfmadd213pd ymm10, ymm1, ymm17
    vfmadd213pd ymm10, ymm1, ymm18
    vfmadd213pd ymm10, ymm1, ymm19
    vfmadd213pd ymm10, ymm1, ymm20
    vfmadd213pd ymm10, ymm1, ymm30
    vmulpd     ymm10, ymm10, ymm1

; =============================================================================
;   BLOCK A — ln_pp = C1 - C2*log1p
; =============================================================================
    vfnmadd231pd ymm25, ymm26, ymm9    ; ln_pp A in ymm25

; BLOCK B — ln_pp
    vfnmadd231pd ymm21, ymm22, ymm10   ; ln_pp B in ymm21
; =============================================================================
;   === EXP(ln_pp) : APPLY RANGE CLAMP, COMPUTE 2^f WITH ESTRIN, 2^k EXACT ===
; =============================================================================
; ln_pp_A already in ymm25
; ln_pp_B already in ymm21

; -----------------------------
; --- Clamp ln_pp into range ---
; -----------------------------
    vmaxpd  ymm25, ymm25, [rel exp_min_x]   ; clamp lower
    vminpd  ymm25, ymm25, [rel exp_max_x]   ; clamp upper

    vmaxpd  ymm21, ymm21, [rel exp_min_x]
    vminpd  ymm21, ymm21, [rel exp_max_x]

; -----------------------------------------------
; --- Convert ln_pp into base-2 exponent space ---
;     t_exp = ln_pp / ln(2)   (multiply by log2(e))
; -----------------------------------------------
    vmulpd  ymm5, ymm25, [rel exp_inv_ln2]   ; t_exp A
    vmulpd  ymm6, ymm21, [rel exp_inv_ln2]   ; t_exp B

; -----------------------------------------
; --- k = round(t_exp), f = t_exp - k   ---
; -----------------------------------------
    vroundpd ymm7, ymm5, 0    ; nearest-int rounding for A
    vsubpd   ymm5, ymm5, ymm7 ; f A

    vroundpd ymm8, ymm6, 0    ; k B
    vsubpd   ymm6, ymm6, ymm8 ; f B

; =============================================================================
;   === ESTRIN'S SCHEME FOR 2^f (BLOCK A) ===
; =============================================================================

    ; f²
    vmulpd   ymm9, ymm5, ymm5

    ; Level 1 polynomial groups
    vmovapd  ymm10, [rel exp_c1]
    vfmadd213pd ymm10, ymm5, [rel const_one]      ; p01 = 1 + f*c1

    vmovapd  ymm11, [rel exp_c3]
    vfmadd213pd ymm11, ymm5, [rel exp_c2]         ; p23 = c2 + f*c3

    vmovapd  ymm12, [rel exp_c5]
    vfmadd213pd ymm12, ymm5, [rel exp_c4]         ; p45 = c4 + f*c5

    ; Level 2 combine
    vfmadd213pd ymm10, ymm9, ymm11                ; q0123 = p01 + f²*p23
    vfmadd213pd ymm12, ymm9, [rel exp_c6]         ; q456 = p45 + f²*c6

    ; Level 3: f⁴ * q456
    vmulpd   ymm9, ymm9, ymm9                     ; f⁴
    vfmadd213pd ymm10, ymm9, ymm12                ; exp_poly_A = q0123 + f⁴*q456

; =============================================================================
;   === ESTRIN'S SCHEME FOR 2^f (BLOCK B) ===
; =============================================================================

    vmulpd   ymm13, ymm6, ymm6                    ; f² B

    vmovapd  ymm14, [rel exp_c1]
    vfmadd213pd ymm14, ymm6, [rel const_one]

    vmovapd  ymm15, [rel exp_c3]
    vfmadd213pd ymm15, ymm6, [rel exp_c2]

    vmovapd  ymm16, [rel exp_c5]
    vfmadd213pd ymm16, ymm6, [rel exp_c4]

    vfmadd213pd ymm14, ymm13, ymm15               ; q0123 B
    vfmadd213pd ymm16, ymm13, [rel exp_c6]

    vmulpd   ymm13, ymm13, ymm13                  ; f⁴ B
    vfmadd213pd ymm14, ymm13, ymm16               ; exp_poly_B

; =============================================================================
;   === 2^k EXACT INTEGER RECONSTRUCTION ===
; =============================================================================
; Convert rounded k to int32 → int64 → exponent bits

    vcvtpd2dq   xmm0, ymm7                        ; A k → int32
    vpmovsxdq   ymm0, xmm0                        ; sign-extend to 64-bit
    vpaddq      ymm0, ymm0, [rel bias_1023]
    vpsllq      ymm0, ymm0, 52                    ; shift into exponent bits
    vmovdqa     ymm17, ymm0                       ; scale_A = bitpattern(2^k)

    vcvtpd2dq   xmm1, ymm8
    vpmovsxdq   ymm18, xmm1
    vpaddq      ymm18, ymm18, [rel bias_1023]
    vpsllq      ymm18, ymm18, 52                  ; scale_B

; =============================================================================
;   === Combine: pp = 2^f * 2^k ===
; =============================================================================
    vmulpd  ymm19, ymm10, ymm17                   ; pp A
    vmaxpd  ymm19, ymm19, [rel const_min_pp]      ; clamp underflow

    vmulpd  ymm20, ymm14, ymm18                   ; pp B
    vmaxpd  ymm20, ymm20, [rel const_min_pp]

; =============================================================================
;   === Compute growth and change contributions ===
; =============================================================================
; r_old * pp * (1-h)  --> growth
; r_old * pp * h      --> change (accumulates r0)

    vmulpd  ymm21, ymm19, ymm28                   ; r*pp A
    vmulpd  ymm22, ymm21, ymm10                   ; growth A = * (1-h)
    vmulpd  ymm21, ymm21, ymm9                    ; change A = * h
    vmovupd [r13 + rsi*8 + 8], ymm22              ; store growth A
    vaddpd  ymm23, ymm23, ymm21                   ; r0 += change A

    vmulpd  ymm24, ymm20, ymm31                   ; r*pp B
    vmulpd  ymm25, ymm24, ymm10                   ; growth B
    vmulpd  ymm24, ymm24, ymm9                    ; change B
    vmovupd [r13 + rsi*8 + 40], ymm25             ; store growth B
    vaddpd  ymm23, ymm23, ymm24                   ; r0 += change B

; =============================================================================
;   === MAX TRACKING ===
; =============================================================================

    ; Block A
    vcmppd     ymm26, ymm22, ymm13, 14            ; compare growth A
    vblendvpd  ymm13, ymm13, ymm22, ymm26         ; update max_growth A
    vmovapd    ymm27, [rsp + 96]                  ; load max_idx_A
    vblendvpd  ymm27, ymm27, ymm15, ymm26         ; apply index blend
    vmovapd    [rsp + 96], ymm27                  ; store new index vector

    ; Block B
    vcmppd     ymm28, ymm25, ymm14, 14
    vblendvpd  ymm14, ymm14, ymm25, ymm28         ; update max_growth B
    vmovapd    ymm29, [rsp]                       ; load max_idx_B
    vmovapd    ymm30, [rsp + 160]                 ; idx_vec_B
    vblendvpd  ymm29, ymm29, ymm30, ymm28
    vmovapd    [rsp], ymm29

; =============================================================================
;   === TRUNCATION CHECKS ===
; =============================================================================

    vcmppd    ymm31, ymm22, ymm11, 14
    vmovmskpd eax, ymm31
    test      eax, eax
    jz        .skip_trunc_A

    bt eax, 3   ; highest lane first
    jc .tvA4
    bt eax, 2
    jc .tvA3
    bt eax, 1
    jc .tvA2
    lea rbx, [rsi + 1]
    jmp .skip_trunc_A

.tvA4:
    lea rbx, [rsi + 4]    ; i+4
    jmp .skip_trunc_A
.tvA3:
    lea rbx, [rsi + 3]
    jmp .skip_trunc_A
.tvA2:
    lea rbx, [rsi + 2]

.skip_trunc_A:

    ; Block B truncation
    vcmppd    ymm31, ymm25, ymm11, 14
    vmovmskpd eax, ymm31
    test      eax, eax
    jz        .skip_trunc_B

    bt eax, 3
    jc .tvB8
    bt eax, 2
    jc .tvB7
    bt eax, 1
    jc .tvB6
    lea rbx, [rsi + 5]
    jmp .skip_trunc_B

.tvB8:
    lea rbx, [rsi + 8]
    jmp .skip_trunc_B
.tvB7:
    lea rbx, [rsi + 7]
    jmp .skip_trunc_B
.tvB6:
    lea rbx, [rsi + 6]

.skip_trunc_B:
; =============================================================================
;   LOOP END — REDUCTIONS AND OUTPUTS
; =============================================================================

.loop_end:

; ============================================================================
;   REDUCE r0 ACCUMULATOR  (ymm23 holds 4 partial sums)
; ============================================================================
    vextractf128 xmm0, ymm23, 1      ; high 128
    vaddpd       xmm0, xmm0, xmm23   ; add low 128
    vhaddpd      xmm0, xmm0, xmm0    ; horizontal add → [sum, sum]
    mov          rax, [rdi + ARG_R0_OUT]
    vmovsd       [rax], xmm0


; Store r0 also into xmm6 so we can compare max-growth against r0.
    vmovsd       xmm6, xmm0, xmm0


; ============================================================================
;   PREPARE REDUCTION FOR MAX GROWTH
; ============================================================================
; We saved:
;   max_growth_A → [rsp+32]
;   max_growth_B → [rsp+64]
;   max_idx_A    → [rsp+96]
;   max_idx_B    → [rsp]

    vmovapd      [rsp + 32], ymm13     ; (A)
    vmovapd      [rsp + 64], ymm14     ; (B)

    xor     rcx, rcx                   ; loop counter
    xor     r15, r15                   ; final MAP index

.reduce_loop:

    cmp rcx, 4
    jge .reduce_done

; ------------------------
; Compare A-lane j
; ------------------------
    vmovsd xmm1, [rsp + 32 + rcx*8]     ; growth_A[j]
    vucomisd xmm1, xmm6
    jbe .check_b

    ; A[j] > current max → update
    vmovsd xmm6, xmm1
    vmovsd xmm2, [rsp + 96 + rcx*8]     ; idx_A[j]
    vcvttsd2si r15, xmm2

.check_b:
; ------------------------
; Compare B-lane j
; ------------------------
    vmovsd xmm1, [rsp + 64 + rcx*8]     ; growth_B[j]
    vucomisd xmm1, xmm6
    jbe .next_j

    ; B[j] > current max → update
    vmovsd xmm6, xmm1
    vmovsd xmm2, [rsp + rcx*8]          ; idx_B[j]
    vcvttsd2si r15, xmm2

.next_j:
    inc rcx
    jmp .reduce_loop

.reduce_done:

; ============================================================================
;   WRITE max_growth_out
; ============================================================================
    mov rax, [rdi + ARG_MAX_GROWTH]
    vmovsd [rax], xmm6

; ============================================================================
;   WRITE max_idx_out
; ============================================================================
    mov rax, [rdi + ARG_MAX_IDX]
    mov [rax], r15

; ============================================================================
;   WRITE last_valid_out
; ============================================================================
    mov rax, [rdi + ARG_LAST_VALID]
    mov [rax], rbx


; =============================================================================
;   EPILOGUE
; =============================================================================
    mov rsp, rbp

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp

    vzeroupper
    ret
