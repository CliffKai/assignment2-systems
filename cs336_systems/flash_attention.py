# CUDA_VISIBLE_DEVICES=4 uv run pytest -v -k test_flash_forward_pass_pytorch

import torch
import math

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        q_input_shape = q.shape
        if q.dim() == 3:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        
        batch_size, n_heads, seq_len_q, d_head = q.shape
        _, _, seq_len_k, _ = k.shape

        Br = 32
        Bc = 32
        
        o = torch.zeros_like(q)
        l_final = torch.empty((batch_size, n_heads, seq_len_q), device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d_head)

        for b in range(batch_size):
            for h in range(n_heads):
                
                num_block_q = (seq_len_q + Br - 1) // Br

                for i in range(num_block_q):
                    q_start = i * Br
                    q_end = min(q_start + Br, seq_len_q)
                    qi = q[b, h, q_start:q_end, :]
                    
                    mi = torch.full((q_end - q_start,), float('-inf'), device=q.device)
                    li = torch.zeros((q_end - q_start,), device=q.device)
                    oi = torch.zeros_like(qi)

                    num_block_k = (seq_len_k + Bc - 1) // Bc
                    for j in range(num_block_k):
                        k_start = j * Bc
                        k_end = min(k_start + Bc, seq_len_k)
                        
                        kj = k[b, h, k_start:k_end, :]
                        vj = v[b, h, k_start:k_end, :]

                        s_ij = torch.matmul(qi, kj.transpose(-2, -1)) * scale

                        if is_causal:
                            q_idxs = torch.arange(q_start, q_end, device=q.device)[:, None]
                            k_idxs = torch.arange(k_start, k_end, device=q.device)[None, :]
                            mask = q_idxs < k_idxs
                            s_ij = s_ij.masked_fill(mask, float('-inf'))

                        m_ij_block_max = torch.max(s_ij, dim=-1).values 
                        mi_new = torch.maximum(mi, m_ij_block_max)

                        p_tilde = torch.exp(s_ij - mi_new.unsqueeze(-1))

                        scale_factor = torch.exp(mi - mi_new)
                        li_new = scale_factor * li + torch.sum(p_tilde, dim=-1)

                        oi = scale_factor.unsqueeze(-1) * oi + torch.matmul(p_tilde, vj)

                        mi = mi_new
                        li = li_new

                    oi = oi / (li.unsqueeze(-1) + 1e-6)
                    
                    o[b, h, q_start:q_end, :] = oi
                    
                    l_final[b, h, q_start:q_end] = mi + torch.log(li + 1e-6)

        if len(q_input_shape) == 3:
            o = o.squeeze(1)
            l_final = l_final.squeeze(1)

        ctx.save_for_backward(q, k, v, o, l_final)
        ctx.is_causal = is_causal 
        
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet")