import torch
import triton
import triton.language as tl
import math

# ==========================================
# Part 1: Forward Triton Kernel
# ==========================================

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O, L,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    n_heads, seq_len_q, seq_len_k, 
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_h = off_hz % n_heads
    off_b = off_hz // n_heads

    # 1. Initialize Block Pointers
    q_offset = off_b * stride_qb + off_h * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seq_len_q, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    k_offset = off_b * stride_kb + off_h * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, seq_len_k), 
        strides=(stride_kd, stride_kn), 
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )

    v_offset = off_b * stride_vb + off_h * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seq_len_k, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )

    o_offset = off_b * stride_ob + off_h * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(seq_len_q, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    lo = 0
    hi = seq_len_k
    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M 

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.dot(q, k)
        qk *= scale

        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p_scale = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = l_i * alpha + tl.sum(p_scale, 1)
        
        p = p_scale.to(v.type.element_ty)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(0,))

    l_offset = off_b * stride_lb + off_h * stride_lh + start_m * BLOCK_M
    L_ptr = L + l_offset + tl.arange(0, BLOCK_M)
    l_val = m_i + tl.log(l_i)
    mask_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_len_q
    tl.store(L_ptr, l_val, mask=mask_m)


# ==========================================
# Part 2: Backward Triton Kernel
# ==========================================

@triton.jit
def flash_attn_bwd_kernel(
    Q, K, V, O, DO,
    DQ, DK, DV,
    L, D,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    n_heads, seq_len_q, seq_len_k,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0) # j (Key Block)
    pid_bh = tl.program_id(1)
    
    off_h = pid_bh % n_heads
    off_b = pid_bh // n_heads

    # Load K_j, V_j
    k_offset = off_b * stride_kb + off_h * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, seq_len_k),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_b * stride_vb + off_h * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seq_len_k, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    
    k_j = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    v_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

    dk_j = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Base pointers for loop
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    dq_base = DQ + off_b * stride_dqb + off_h * stride_dqh
    
    l_base = L + pid_bh * seq_len_q
    d_base = D + pid_bh * seq_len_q

    start_m_min = 0
    if IS_CAUSAL:
        start_m_min = (pid_n * BLOCK_N) // BLOCK_M

    num_q_blocks = tl.cdiv(seq_len_q, BLOCK_M)
    
    for start_m in range(start_m_min, num_q_blocks):
        # Load Q_i, dO_i
        Q_ptr = tl.make_block_ptr(
            base=q_base, shape=(seq_len_q, HEAD_DIM), strides=(stride_qn, stride_qd),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
        )
        q_i = tl.load(Q_ptr, boundary_check=(0,), padding_option="zero")

        DO_ptr = tl.make_block_ptr(
            base=do_base, shape=(seq_len_q, HEAD_DIM), strides=(stride_don, stride_dod),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
        )
        do_i = tl.load(DO_ptr, boundary_check=(0,), padding_option="zero")

        # Load L_i, D_i
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        l_i = tl.load(l_base + offs_m, mask=mask_m, other=0.0)
        d_i = tl.load(d_base + offs_m, mask=mask_m, other=0.0)

        # Recompute Attention
        qk = tl.dot(q_i, k_j)
        qk *= scale

        if IS_CAUSAL:
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        p = tl.exp(qk - l_i[:, None])
        
        # Compute dV_j
        p_curr = p.to(do_i.type.element_ty)
        dv_j += tl.dot(tl.trans(p_curr), do_i)

        # Compute dP_ij
        dp = tl.dot(do_i, tl.trans(v_j))

        # Compute dS_ij
        ds = p * (dp - d_i[:, None])
        ds = ds * scale
        ds = ds.to(q_i.type.element_ty)

        # Compute dK_j
        dk_j += tl.dot(tl.trans(ds), q_i)

        # Compute dQ_i (Atomic Add)
        dq_update = tl.dot(ds, tl.trans(k_j))

        # Construct pointers for atomic add
        dq_offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dq_offs_d = tl.arange(0, HEAD_DIM)
        dq_ptrs = dq_base + dq_offs_m[:, None] * stride_dqn + dq_offs_d[None, :] * stride_dqd
        dq_mask = (dq_offs_m[:, None] < seq_len_q)
        
        tl.atomic_add(dq_ptrs, dq_update, mask=dq_mask)

    # Store dK_j, dV_j
    dk_base = DK + off_b * stride_dkb + off_h * stride_dkh
    dv_base = DV + off_b * stride_dvb + off_h * stride_dvh

    DK_ptr = tl.make_block_ptr(
        base=dk_base, shape=(seq_len_k, HEAD_DIM), strides=(stride_dkn, stride_dkd),
        offsets=(pid_n * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    DV_ptr = tl.make_block_ptr(
        base=dv_base, shape=(seq_len_k, HEAD_DIM), strides=(stride_dvn, stride_dvd),
        offsets=(pid_n * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    
    tl.store(DK_ptr, dk_j.to(DK.type.element_ty), boundary_check=(0,))
    tl.store(DV_ptr, dv_j.to(DV.type.element_ty), boundary_check=(0,))


# ==========================================
# Part 3: Autograd Wrapper
# ==========================================

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        q_in_dim = q.dim()
        if q_in_dim == 3:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            
        BATCH, HEADS, N_CTX, HEAD_DIM = q.shape
        
        o = torch.empty_like(q)
        L = torch.empty((BATCH, HEADS, N_CTX), device=q.device, dtype=torch.float32)

        # [精细化 Tuning]: 仅当 FP32 且 HeadDim 很大时才减小 Block Size
        # BF16 (2 bytes) 即使在 128 Dim 下也能跑 BLOCK_M=128
        is_fp32 = (q.dtype == torch.float32)
        
        BLOCK_M = 128
        BLOCK_N = 64 
        if is_fp32 and HEAD_DIM >= 128:
             BLOCK_M = 64 

        grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * HEADS)
        
        flash_attn_fwd_kernel[grid](
            q, k, v, o, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            HEADS, N_CTX, N_CTX, 
            scale=1.0 / math.sqrt(HEAD_DIM),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=is_causal,
            num_warps=4, 
            num_stages=2
        )

        L_saved = L
        if q_in_dim == 3:
            o = o.squeeze(1)
            L_saved = L.squeeze(1)

        ctx.save_for_backward(q, k, v, o, L_saved)
        ctx.is_causal = is_causal
        ctx.scale = 1.0 / math.sqrt(HEAD_DIM)
        ctx.q_in_dim = q_in_dim
        
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        q_in_dim = ctx.q_in_dim

        if q_in_dim == 3:
            if o.dim() == 3: o = o.unsqueeze(1)
            if l.dim() == 2: l = l.unsqueeze(1)
            if do.dim() == 3: do = do.unsqueeze(1)
        
        D = torch.sum(do * o, dim=-1)

        dq = torch.zeros_like(q, dtype=torch.float32) 
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, HEADS, N_CTX_Q, HEAD_DIM = q.shape
        _, _, N_CTX_K, _ = k.shape
        
        # [精细化 Tuning]
        is_fp32 = (q.dtype == torch.float32)
        BLOCK_M = 128
        BLOCK_N = 64
        if is_fp32 and HEAD_DIM >= 128:
             BLOCK_M = 64
        
        grid = (triton.cdiv(N_CTX_K, BLOCK_N), BATCH * HEADS)

        flash_attn_bwd_kernel[grid](
            q, k, v, o, do,
            dq, dk, dv,
            l, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            HEADS, N_CTX_Q, N_CTX_K,
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=1
        )

        dq = dq.to(q.dtype)

        if q_in_dim == 3:
            dq = dq.squeeze(1)
            dk = dk.squeeze(1)
            dv = dv.squeeze(1)

        return dq, dk, dv, None