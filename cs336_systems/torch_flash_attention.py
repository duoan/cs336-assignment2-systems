import math

import torch

from torch import Tensor
from jaxtyping import Float
from einops import rearrange

_TILE_SIZE = 16


# ---------------------------------------------------------------------------
# 用 torch.compile 包住"单个 (Qi, Kj, Vj) tile 的核心计算"
#   - dQ pass 用：返回该 (i,j) tile 对 dQ_i 的贡献
#   - dKV pass 用：返回该 (i,j) tile 对 dK_j, dV_j 的贡献
# 把 D_i 提前算好传进来，避免 backward 内部再 reduce
# ---------------------------------------------------------------------------
@torch.compile(fullgraph=True, dynamic=False)
def _bwd_tile_dq(Qi, Kj, Vj, dOi, Li, Di, scale: float, causal_mask):
    """
    返回 dQ_i 在 (i, j) tile 上的增量 (b, B_q, d_k)
    causal_mask: None 或 (B_q, B_k) bool，True 表示要 mask 掉
    """
    Sij = torch.einsum("b q d, b k d -> b q k", Qi, Kj) * scale
    if causal_mask is not None:
        Sij = Sij.masked_fill(causal_mask, float("-inf"))
    Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (b, B_q, B_k)
    dPij = torch.einsum("b q d, b k d -> b q k", dOi, Vj)  # (b, B_q, B_k)
    dSij = Pij * (dPij - Di.unsqueeze(-1))  # (b, B_q, B_k)
    return torch.einsum("b q k, b k d -> b q d", dSij, Kj) * scale  # (b, B_q, d_k)


@torch.compile(fullgraph=True, dynamic=False)
def _bwd_tile_dkdv(Qi, Kj, Vj, dOi, Li, Di, scale: float, causal_mask):
    """
    返回 (dKj_inc, dVj_inc) 这一 (i, j) tile 的增量
    """
    Sij = torch.einsum("b q d, b k d -> b q k", Qi, Kj) * scale
    if causal_mask is not None:
        Sij = Sij.masked_fill(causal_mask, float("-inf"))
    Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (b, B_q, B_k)
    # dV_j += P^T dO
    dVj_inc = torch.einsum("b q k, b q d -> b k d", Pij, dOi)  # (b, B_k, d_v)
    dPij = torch.einsum("b q d, b k d -> b q k", dOi, Vj)  # (b, B_q, B_k)
    dSij = Pij * (dPij - Di.unsqueeze(-1))  # (b, B_q, B_k)
    # dK_j += dS^T Q * scale
    dKj_inc = torch.einsum("b q k, b q d -> b k d", dSij, Qi) * scale  # (b, B_k, d_k)
    return dKj_inc, dVj_inc


class TorchFlashAttentionAutograd(torch.autograd.Function):
    """
    PyTorch-only FlashAttention v2 style implementation with autograd
    """

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
        """
        save L, Q, K, V, O for the backward pass
        and return O
        """

        batch_shape = Q.shape[:-2]
        L_q, d_k = Q.shape[-2], Q.shape[-1]
        L_k, d_v = K.shape[-2], V.shape[-1]

        assert L_q % block_q == 0
        assert L_k % block_k == 0

        # split into tiles
        Qt = rearrange(Q, "... (tq bq) d -> (...) tq bq d", bq=block_q)  # (b, T_q, B_q, d_k)
        Kt = rearrange(K, "... (tk bk) d -> (...) tk bk d", bk=block_k)  # (b, T_k, B_k, d_k)
        Vt = rearrange(V, "... (tk bk) d -> (...) tk bk d", bk=block_k)  # (b, T_k, B_k, d_v)

        B, T_q, T_k = Qt.shape[0], Qt.shape[1], Kt.shape[1]
        scale = 1.0 / math.sqrt(d_k)
        device, dtype = Q.device, Q.dtype
        NEG_INF = float("-inf")

        O_tiles = torch.zeros(B, T_q, block_q, d_v, dtype=torch.float32, device=device)
        L_tiles = torch.zeros(B, T_q, block_q, dtype=torch.float32, device=device)

        for i in range(T_q):
            Qi = Qt[:, i].to(torch.float32)

            Oi = torch.zeros(B, block_q, d_v, dtype=torch.float32, device=device)
            li = torch.zeros(B, block_q, dtype=torch.float32, device=device)
            mi = torch.full((B, block_q), NEG_INF, dtype=torch.float32, device=device)

            j_max = T_k
            if is_causal:
                q_end = (i + 1) * block_q
                j_max = min(T_k, (q_end - 1) // block_k + 1)

            for j in range(j_max):
                Kj = Kt[:, j].to(torch.float32)
                Vj = Vt[:, j].to(torch.float32)

                Sij = torch.einsum("b q d, b k d -> b q k", Qi, Kj) * scale

                if is_causal:
                    q_idx = torch.arange(i * block_q, i * block_q + block_q, device=device)
                    k_idx = torch.arange(j * block_k, j * block_k + block_k, device=device)

                    mask = q_idx[:, None] < k_idx[None, :]
                    Sij = Sij.masked_fill(mask, NEG_INF)

                # online softmax
                row_max = Sij.amax(dim=-1)
                mi_new = torch.maximum(mi, row_max)
                Pij = torch.exp(Sij - mi_new.unsqueeze(-1))
                alpha = torch.exp(mi - mi_new)

                li = alpha * li + Pij.sum(dim=-1)
                Oi = alpha.unsqueeze(-1) * Oi + torch.einsum("b q k, b k d -> b q d", Pij, Vj)
                mi = mi_new

            O_tiles[:, i] = Oi / li.unsqueeze(-1)
            L_tiles[:, i] = mi + torch.log(li)

        O = rearrange(O_tiles, "b tq bq d -> b (tq bq) d").to(dtype).reshape(*batch_shape, L_q, d_v)
        L = rearrange(L_tiles, "b tq bq ->  b (tq bq)").reshape(*batch_shape, L_q)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.block_q = block_q
        ctx.block_k = block_k

        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        block_q = ctx.block_q
        block_k = ctx.block_k

        batch_shape = Q.shape[:-2]
        L_q, d_k = Q.shape[-2], Q.shape[-1]
        L_k, d_v = K.shape[-2], V.shape[-1]
        device, orig_dtype = Q.device, Q.dtype
        scale = 1.0 / math.sqrt(d_k)

        # ---- 把 leading dims 压成 b，并 cast 到 fp32 ----
        Qf = Q.reshape(-1, L_q, d_k).to(torch.float32)
        Kf = K.reshape(-1, L_k, d_k).to(torch.float32)
        Vf = V.reshape(-1, L_k, d_v).to(torch.float32)
        Of = O.reshape(-1, L_q, d_v).to(torch.float32)
        dOf = dO.reshape(-1, L_q, d_v).to(torch.float32)
        Lf = L.reshape(-1, L_q).to(torch.float32)
        B = Qf.shape[0]

        # ---- 提前算 D = rowsum(dO ⊙ O) ----
        Df = (dOf * Of).sum(dim=-1)  # (B, L_q)

        # ---- 切 tile ----
        Qt = rearrange(Qf, "b (tq bq) d -> b tq bq d", bq=block_q)  # (B, T_q, B_q, d_k)
        Kt = rearrange(Kf, "b (tk bk) d -> b tk bk d", bk=block_k)  # (B, T_k, B_k, d_k)
        Vt = rearrange(Vf, "b (tk bk) d -> b tk bk d", bk=block_k)  # (B, T_k, B_k, d_v)
        dOt = rearrange(dOf, "b (tq bq) d -> b tq bq d", bq=block_q)  # (B, T_q, B_q, d_v)
        Lt = rearrange(Lf, "b (tq bq)   -> b tq bq", bq=block_q)  # (B, T_q, B_q)
        Dt = rearrange(Df, "b (tq bq)   -> b tq bq", bq=block_q)  # (B, T_q, B_q)
        T_q, T_k = Qt.shape[1], Kt.shape[1]

        # ---- 输出 buffer（fp32 累加）----
        dQ_tiles = torch.zeros(B, T_q, block_q, d_k, dtype=torch.float32, device=device)
        dK_tiles = torch.zeros(B, T_k, block_k, d_k, dtype=torch.float32, device=device)
        dV_tiles = torch.zeros(B, T_k, block_k, d_v, dtype=torch.float32, device=device)

        # =====================================================================
        # Pass 1: 外层 Q tile，内层 K tile，累加 dQ
        # =====================================================================
        for i in range(T_q):
            Qi = Qt[:, i]  # (B, B_q, d_k)
            dOi = dOt[:, i]  # (B, B_q, d_v)
            Li = Lt[:, i]  # (B, B_q)
            Di = Dt[:, i]  # (B, B_q)

            j_max = T_k
            if is_causal:
                q_end = (i + 1) * block_q
                j_max = min(T_k, (q_end - 1) // block_k + 1)

            dQi_acc = torch.zeros(B, block_q, d_k, dtype=torch.float32, device=device)

            q_idx = torch.arange(i * block_q, (i + 1) * block_q, device=device)  # (B_q,)
            for j in range(j_max):
                Kj = Kt[:, j]
                Vj = Vt[:, j]

                causal_mask = None
                if is_causal:
                    k_idx = torch.arange(j * block_k, (j + 1) * block_k, device=device)
                    causal_mask = q_idx[:, None] < k_idx[None, :]  # (B_q, B_k)

                dQi_acc = dQi_acc + _bwd_tile_dq(Qi, Kj, Vj, dOi, Li, Di, scale, causal_mask)

            dQ_tiles[:, i] = dQi_acc

        # =====================================================================
        # Pass 2: 外层 K tile，内层 Q tile，累加 dK, dV
        # =====================================================================
        for j in range(T_k):
            Kj = Kt[:, j]  # (B, B_k, d_k)
            Vj = Vt[:, j]  # (B, B_k, d_v)

            # causal: 只需要 i 满足 (i+1)*B_q > j*B_k，即 i >= floor(j*B_k / B_q)
            i_start = 0
            if is_causal:
                i_start = (j * block_k) // block_q

            dKj_acc = torch.zeros(B, block_k, d_k, dtype=torch.float32, device=device)
            dVj_acc = torch.zeros(B, block_k, d_v, dtype=torch.float32, device=device)

            k_idx = torch.arange(j * block_k, (j + 1) * block_k, device=device)
            for i in range(i_start, T_q):
                Qi = Qt[:, i]
                dOi = dOt[:, i]
                Li = Lt[:, i]
                Di = Dt[:, i]

                causal_mask = None
                if is_causal:
                    q_idx = torch.arange(i * block_q, (i + 1) * block_q, device=device)
                    causal_mask = q_idx[:, None] < k_idx[None, :]

                dKj_inc, dVj_inc = _bwd_tile_dkdv(Qi, Kj, Vj, dOi, Li, Di, scale, causal_mask)
                dKj_acc = dKj_acc + dKj_inc
                dVj_acc = dVj_acc + dVj_inc

            dK_tiles[:, j] = dKj_acc
            dV_tiles[:, j] = dVj_acc

        # ---- 拼回 + reshape + cast ----
        dQ = rearrange(dQ_tiles, "b tq bq d -> b (tq bq) d").to(orig_dtype).reshape(*batch_shape, L_q, d_k)
        dK = rearrange(dK_tiles, "b tk bk d -> b (tk bk) d").to(orig_dtype).reshape(*batch_shape, L_k, d_k)
        dV = rearrange(dV_tiles, "b tk bk d -> b (tk bk) d").to(orig_dtype).reshape(*batch_shape, L_k, d_v)

        # forward 签名是 (Q, K, V, is_causal, block_q, block_k) → 6 个返回
        return dQ, dK, dV, None, None, None
