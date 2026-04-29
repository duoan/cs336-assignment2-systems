import math
import torch
import triton
import triton.language as tl
from jaxtyping import Float
from torch import Tensor
from einops import rearrange


# fmt: off
@triton.jit
def _flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr,  # pointers to input Q, K, V
    O_ptr, L_ptr,  # pointers to outputs O, L
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_km, stride_kd,
    stride_vb, stride_vm, stride_vd,
    stride_ob, stride_om, stride_od,
    stride_lb, stride_lm,
    L_Q, L_K,
    scale,
    # constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    
):
# fmt: on
    pid_q = tl.program_id(0) # Q tile (i)
    pid_b = tl.program_id(1) # batch*head

    # tl.device_print("pid_q", pid_q)
    # tl.device_print("pid_b", pid_b)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_b * stride_qb,
        shape=(L_Q, D_K),
        strides=(stride_qm, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_K),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_b * stride_kb,
        shape=(L_K, D_K),
        strides=(stride_km, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D_K),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_b * stride_vb,
        shape=(L_K, D_V),
        strides=(stride_vm, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D_V),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + pid_b * stride_ob,
        shape=(L_Q, D_V),
        strides=(stride_om, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_V),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + pid_b * stride_lb,
        shape=(L_Q,),
        strides=(stride_lm,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_i = tl.zeros((BLOCK_Q, D_V), dtype=tl.float32)

    offs_m = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)

    if IS_CAUSAL:
        q_end = (pid_q + 1) * BLOCK_Q
        n_kv_tiles = tl.cdiv(tl.minimum(q_end, L_K), BLOCK_K)
    else:
        n_kv_tiles = tl.cdiv(L_K, BLOCK_K)

    for j in range(0, n_kv_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        Sij = tl.dot(Qi, tl.trans(Kj)).to(tl.float32) * scale

        offs_n = j * BLOCK_K + tl.arange(0, BLOCK_K)
        Sij = tl.where(offs_n[None, :] < L_K, Sij, -float("inf"))

        if IS_CAUSAL:
            Sij = tl.where(offs_m[:, None] >= offs_n[None, :], Sij, -float("inf"))
        
        # online softmax
        m_ij = tl.maximum(m_i, tl.max(Sij, axis=1))
        alpha = tl.exp(m_i - m_ij)
        Pij = tl.exp(Sij - m_ij[:, None])

        l_i = alpha * l_i + tl.sum(Pij, axis=1)
        O_i = O_i * alpha[:, None] + tl.dot(Pij.to(Vj.dtype), Vj).to(tl.float32)
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_K, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_K, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


# =============================================================================
# Backward kernel #1: dQ —— outer parallel over Q tiles
# =============================================================================
@triton.jit
def _flash_attention_bwd_dq(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dQ_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_km, stride_kd,
    stride_vb, stride_vm, stride_vd,
    stride_dob, stride_dom, stride_dod,
    stride_dqb, stride_dqm, stride_dqd,
    stride_lb, stride_lm,
    stride_db, stride_dm,
    L_Q, L_K,
    scale,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # ---- block pointers ----
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_b * stride_qb,
        shape=(L_Q, D_K), strides=(stride_qm, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_K), order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_b * stride_dob,
        shape=(L_Q, D_V), strides=(stride_dom, stride_dod),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_V), order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + pid_b * stride_dqb,
        shape=(L_Q, D_K), strides=(stride_dqm, stride_dqd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_K), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_b * stride_kb,
        shape=(L_K, D_K), strides=(stride_km, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D_K), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_b * stride_vb,
        shape=(L_K, D_V), strides=(stride_vm, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D_V), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + pid_b * stride_lb,
        shape=(L_Q,), strides=(stride_lm,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,), order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + pid_b * stride_db,
        shape=(L_Q,), strides=(stride_dm,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,), order=(0,),
    )

    # ---- load tile-resident tensors ----
    Qi  = tl.load(Q_block_ptr,  boundary_check=(0,), padding_option="zero")  # (BQ, D_K)
    dOi = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")  # (BQ, D_V)
    Li  = tl.load(L_block_ptr,  boundary_check=(0,), padding_option="zero")  # (BQ,)
    Di  = tl.load(D_block_ptr,  boundary_check=(0,), padding_option="zero")  # (BQ,)

    dQi = tl.zeros((BLOCK_Q, D_K), dtype=tl.float32)
    offs_m = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)

    # causal: 跳过对角线之上的整块 KV
    if IS_CAUSAL:
        q_end = (pid_q + 1) * BLOCK_Q
        n_kv_tiles = tl.cdiv(tl.minimum(q_end, L_K), BLOCK_K)
    else:
        n_kv_tiles = tl.cdiv(L_K, BLOCK_K)

    for j in range(0, n_kv_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")  # (BK, D_K)
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")  # (BK, D_V)

        # S = Qi Kj^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)).to(tl.float32) * scale                  # (BQ, BK)

        offs_n = j * BLOCK_K + tl.arange(0, BLOCK_K)
        Sij = tl.where(offs_n[None, :] < L_K, Sij, -float("inf"))
        if IS_CAUSAL:
            Sij = tl.where(offs_m[:, None] >= offs_n[None, :], Sij, -float("inf"))

        # P = exp(S - L)
        Pij = tl.exp(Sij - Li[:, None])                                        # (BQ, BK)

        # dP = dOi Vj^T
        dPij = tl.dot(dOi, tl.trans(Vj)).to(tl.float32)                        # (BQ, BK)

        # dS = P * (dP - D)
        dSij = (Pij * (dPij - Di[:, None])).to(Kj.dtype)                       # (BQ, BK), cast 回 input dtype 走 TC

        # dQi += dS Kj * scale
        dQi += tl.dot(dSij, Kj).to(tl.float32) * scale

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_K, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_K, 0))

    tl.store(dQ_block_ptr, dQi.to(dQ_ptr.dtype.element_ty), boundary_check=(0,))

# =============================================================================
# Backward kernel #2: dK, dV —— outer parallel over K/V tiles
# =============================================================================
@triton.jit
def _flash_attention_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_km, stride_kd,
    stride_vb, stride_vm, stride_vd,
    stride_dob, stride_dom, stride_dod,
    stride_dkb, stride_dkm, stride_dkd,
    stride_dvb, stride_dvm, stride_dvd,
    stride_lb, stride_lm,
    stride_db, stride_dm,
    L_Q, L_K,
    scale,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_k = tl.program_id(0)        # 哪个 K/V tile (j)
    pid_b = tl.program_id(1)

    # ---- KV-side block pointers (固定不动) ----
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_b * stride_kb,
        shape=(L_K, D_K), strides=(stride_km, stride_kd),
        offsets=(pid_k * BLOCK_K, 0),
        block_shape=(BLOCK_K, D_K), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_b * stride_vb,
        shape=(L_K, D_V), strides=(stride_vm, stride_vd),
        offsets=(pid_k * BLOCK_K, 0),
        block_shape=(BLOCK_K, D_V), order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + pid_b * stride_dkb,
        shape=(L_K, D_K), strides=(stride_dkm, stride_dkd),
        offsets=(pid_k * BLOCK_K, 0),
        block_shape=(BLOCK_K, D_K), order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + pid_b * stride_dvb,
        shape=(L_K, D_V), strides=(stride_dvm, stride_dvd),
        offsets=(pid_k * BLOCK_K, 0),
        block_shape=(BLOCK_K, D_V), order=(1, 0),
    )

    # ---- Q-side block pointers (沿 L_Q 滑动) ----
    # causal 时只需要扫过对角线及以下的 Q tile
    if IS_CAUSAL:
        # 第一个能 attend 到本 K tile 的 Q 行：q >= k_start
        q_start_tile = (pid_k * BLOCK_K) // BLOCK_Q
    else:
        q_start_tile = 0
    n_q_tiles = tl.cdiv(L_Q, BLOCK_Q)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_b * stride_qb,
        shape=(L_Q, D_K), strides=(stride_qm, stride_qd),
        offsets=(q_start_tile * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_K), order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_b * stride_dob,
        shape=(L_Q, D_V), strides=(stride_dom, stride_dod),
        offsets=(q_start_tile * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D_V), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + pid_b * stride_lb,
        shape=(L_Q,), strides=(stride_lm,),
        offsets=(q_start_tile * BLOCK_Q,),
        block_shape=(BLOCK_Q,), order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + pid_b * stride_db,
        shape=(L_Q,), strides=(stride_dm,),
        offsets=(q_start_tile * BLOCK_Q,),
        block_shape=(BLOCK_Q,), order=(0,),
    )

    # ---- load Kj, Vj 一次，放在寄存器里 ----
    Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")     # (BK, D_K)
    Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")     # (BK, D_V)

    dKj = tl.zeros((BLOCK_K, D_K), dtype=tl.float32)
    dVj = tl.zeros((BLOCK_K, D_V), dtype=tl.float32)
    offs_n = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    for i in range(q_start_tile, n_q_tiles):
        Qi  = tl.load(Q_block_ptr,  boundary_check=(0,), padding_option="zero")   # (BQ, D_K)
        dOi = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")   # (BQ, D_V)
        Li  = tl.load(L_block_ptr,  boundary_check=(0,), padding_option="zero")   # (BQ,)
        Di  = tl.load(D_block_ptr,  boundary_check=(0,), padding_option="zero")   # (BQ,)

        # S = Qi Kj^T * scale  (BQ, BK)
        Sij = tl.dot(Qi, tl.trans(Kj)).to(tl.float32) * scale

        offs_m = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
        Sij = tl.where(offs_n[None, :] < L_K, Sij, -float("inf"))
        if IS_CAUSAL:
            Sij = tl.where(offs_m[:, None] >= offs_n[None, :], Sij, -float("inf"))

        # P = exp(S - L)
        Pij = tl.exp(Sij - Li[:, None])                                            # (BQ, BK)

        # dV += P^T dO
        dVj += tl.dot(tl.trans(Pij).to(dOi.dtype), dOi).to(tl.float32)            # (BK, D_V)

        # dP = dO V^T
        dPij = tl.dot(dOi, tl.trans(Vj)).to(tl.float32)                            # (BQ, BK)

        # dS = P * (dP - D)
        dSij = (Pij * (dPij - Di[:, None])).to(Qi.dtype)                           # (BQ, BK)

        # dK += dS^T Q * scale
        dKj += tl.dot(tl.trans(dSij), Qi).to(tl.float32) * scale                  # (BK, D_K)

        Q_block_ptr  = tl.advance(Q_block_ptr,  (BLOCK_Q, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (BLOCK_Q, 0))
        L_block_ptr  = tl.advance(L_block_ptr,  (BLOCK_Q,))
        D_block_ptr  = tl.advance(D_block_ptr,  (BLOCK_Q,))

    tl.store(dK_block_ptr, dKj.to(dK_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(dV_block_ptr, dVj.to(dV_ptr.dtype.element_ty), boundary_check=(0,))



class TritonFlashAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[Tensor, "... L_q d_k"],
        K: Float[Tensor, "... L_k d_k"],
        V: Float[Tensor, "... L_k d_v"],
        is_causal=False,
        block_q: int = 16,
        block_k: int = 16,
    ):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel only support CUDA"
        batch_shape = Q.shape[:-2]
        L_q, d_k = Q.shape[-2], Q.shape[-1]
        L_k, d_v = K.shape[-2], V.shape[-1]

        # split into tiles
        Qf = rearrange(Q, "... L_q d_k -> (...) L_q d_k").contiguous()  # (B, L_q, d_k)
        Kf = rearrange(K, "... L_k d_k -> (...) L_k d_k").contiguous()  # (B, L_k, d_k)
        Vf = rearrange(V, "... L_k d_v -> (...) L_k d_v").contiguous()  # (B, L_k, d_v)

        B = Qf.shape[0]

        Ot = torch.empty((B, L_q, d_v), dtype=Q.dtype, device=Q.device)
        Lt = torch.empty((B, L_q), dtype=torch.float32, device=Q.device)
        scale = 1.0 / math.sqrt(d_k)

        grid = (triton.cdiv(L_q, block_q), B)
    
        # fmt: off
        _flash_attention_fwd[grid](
            Qf, Kf, Vf,
            Ot, Lt,
            Qf.stride(0), Qf.stride(1), Qf.stride(2),
            Kf.stride(0), Kf.stride(1), Kf.stride(2),
            Vf.stride(0), Vf.stride(1), Vf.stride(2),
            Ot.stride(0), Ot.stride(1), Ot.stride(2),
            Lt.stride(0), Lt.stride(1),
            L_q, L_k,
            scale,
            D_K=d_k, D_V=d_v,
            BLOCK_Q=block_q, BLOCK_K=block_k,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=2 if d_k >= 128 else 3,
        )
        # fmt: on

        O = Ot.reshape(*batch_shape, L_q, d_v)
        L = Lt.reshape(*batch_shape, L_q)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.block_q = block_q
        ctx.block_k = block_k

        return O

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: Float[Tensor, "... L_q d_v"],
    ):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        block_q   = ctx.block_q
        block_k   = ctx.block_k

        batch_shape = Q.shape[:-2]
        L_q, d_k = Q.shape[-2], Q.shape[-1]
        L_k, d_v = K.shape[-2], V.shape[-1]

        # 把 leading dims 压成 b，跟 forward 对齐
        Qf  = Q.reshape(-1, L_q, d_k).contiguous()
        Kf  = K.reshape(-1, L_k, d_k).contiguous()
        Vf  = V.reshape(-1, L_k, d_v).contiguous()
        Of  = O.reshape(-1, L_q, d_v).contiguous()
        dOf = grad_out.reshape(-1, L_q, d_v).contiguous()
        Lf  = L.reshape(-1, L_q).contiguous()
        B   = Qf.shape[0]

        # ---- 预算 D = rowsum(dO ⊙ O), 用 PyTorch 即可，cheap ----
        Df = (dOf.float() * Of.float()).sum(dim=-1)                           # (B, L_q), fp32

        dQf = torch.zeros_like(Qf)
        dKf = torch.zeros_like(Kf)
        dVf = torch.zeros_like(Vf)

        scale = 1.0 / math.sqrt(d_k)

        # ---- kernel 1: dQ ----
        grid_q = (triton.cdiv(L_q, block_q), B)
        _flash_attention_bwd_dq[grid_q](
            Qf, Kf, Vf,
            dOf, dQf,
            Lf, Df,
            Qf.stride(0),  Qf.stride(1),  Qf.stride(2),
            Kf.stride(0),  Kf.stride(1),  Kf.stride(2),
            Vf.stride(0),  Vf.stride(1),  Vf.stride(2),
            dOf.stride(0), dOf.stride(1), dOf.stride(2),
            dQf.stride(0), dQf.stride(1), dQf.stride(2),
            Lf.stride(0),  Lf.stride(1),
            Df.stride(0),  Df.stride(1),
            L_q, L_k,
            scale,
            D_K=d_k, D_V=d_v,
            BLOCK_Q=block_q, BLOCK_K=block_k,
            IS_CAUSAL=is_causal,
            num_warps=4, num_stages=3,
        )

        # ---- kernel 2: dK, dV ----
        grid_k = (triton.cdiv(L_k, block_k), B)
        _flash_attention_bwd_dkdv[grid_k](
            Qf, Kf, Vf,
            dOf, dKf, dVf,
            Lf, Df,
            Qf.stride(0),  Qf.stride(1),  Qf.stride(2),
            Kf.stride(0),  Kf.stride(1),  Kf.stride(2),
            Vf.stride(0),  Vf.stride(1),  Vf.stride(2),
            dOf.stride(0), dOf.stride(1), dOf.stride(2),
            dKf.stride(0), dKf.stride(1), dKf.stride(2),
            dVf.stride(0), dVf.stride(1), dVf.stride(2),
            Lf.stride(0),  Lf.stride(1),
            Df.stride(0),  Df.stride(1),
            L_q, L_k,
            scale,
            D_K=d_k, D_V=d_v,
            BLOCK_Q=block_q, BLOCK_K=block_k,
            IS_CAUSAL=is_causal,
            num_warps=4, num_stages=3,
        )

        dQ = dQf.reshape(*batch_shape, L_q, d_k)
        dK = dKf.reshape(*batch_shape, L_k, d_k)
        dV = dVf.reshape(*batch_shape, L_k, d_v)

        # forward 签名是 (Q, K, V, is_causal, block_q, block_k) → 6 个返回
        return dQ, dK, dV, None, None, None
