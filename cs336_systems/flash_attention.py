# CUDA_VISIBLE_DEVICES=4 uv run pytest -v -k test_flash_forward_pass_pytorch

import torch
import math

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        FlashAttention-2 forward pass implementation in pure PyTorch.
        References Algorithm 1 in the assignment PDF.
        """
        # Check input dimension and normalize to 4D if necessary (B, H, N, D)
        # Test suite uses (B, N, D) which implies H=1
        q_input_shape = q.shape
        if q.dim() == 3:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        
        # Shapes: (batch_size, n_heads, seq_len, d_head)
        batch_size, n_heads, seq_len_q, d_head = q.shape
        _, _, seq_len_k, _ = k.shape

        # Tile sizes (Br, Bc) - as suggested in PDF, at least 16x16
        Br = 32
        Bc = 32
        
        # Initialize Output O and LogSumExp L
        # O: (B, H, N, d)
        o = torch.zeros_like(q)
        # L: (B, H, N) - logsumexp for backward pass
        l_final = torch.empty((batch_size, n_heads, seq_len_q), device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d_head)

        # Grid loops corresponding to Algorithm 1
        for b in range(batch_size):
            for h in range(n_heads):
                
                # Split Q into Tq tiles
                num_block_q = (seq_len_q + Br - 1) // Br
                # num_block_k = (seq_len_k + Bc - 1) // Bc

                for i in range(num_block_q):
                    # Load Qi
                    q_start = i * Br
                    q_end = min(q_start + Br, seq_len_q)
                    # shape: (curr_Br, d)
                    qi = q[b, h, q_start:q_end, :]
                    
                    # Initialize row statistics for online softmax
                    # mi: running max, initialized to -inf
                    mi = torch.full((q_end - q_start,), float('-inf'), device=q.device)
                    # li: running sum of exp, initialized to 0
                    li = torch.zeros((q_end - q_start,), device=q.device)
                    # oi: accumulated output, initialized to 0
                    oi = torch.zeros_like(qi)

                    # Block K loop
                    num_block_k = (seq_len_k + Bc - 1) // Bc
                    for j in range(num_block_k):
                        # Load Kj, Vj
                        k_start = j * Bc
                        k_end = min(k_start + Bc, seq_len_k)
                        
                        # shape: (curr_Bc, d)
                        kj = k[b, h, k_start:k_end, :]
                        vj = v[b, h, k_start:k_end, :]

                        # 1. Compute tile of pre-softmax scores S_ij
                        # shape: (curr_Br, curr_Bc)
                        s_ij = torch.matmul(qi, kj.transpose(-2, -1)) * scale

                        # Causal Masking (Simple implementation for reference)
                        if is_causal:
                            q_idxs = torch.arange(q_start, q_end, device=q.device)[:, None]
                            k_idxs = torch.arange(k_start, k_end, device=q.device)[None, :]
                            mask = q_idxs < k_idxs
                            s_ij = s_ij.masked_fill(mask, float('-inf'))

                        # 2. Compute mi_new (current max)
                        m_ij_block_max = torch.max(s_ij, dim=-1).values 
                        mi_new = torch.maximum(mi, m_ij_block_max)

                        # 3. Compute P_tilde (unnormalized probs)
                        p_tilde = torch.exp(s_ij - mi_new.unsqueeze(-1))

                        # 4. Compute li_new (running denominator)
                        # Update rule: l_new = e^(m_old - m_new) * l_old + rowsum(P_tilde)
                        scale_factor = torch.exp(mi - mi_new)
                        li_new = scale_factor * li + torch.sum(p_tilde, dim=-1)

                        # 5. Compute oi_new (accumulated output)
                        # Update rule: o_new = diag(scale_factor) * o_old + P_tilde @ Vj
                        oi = scale_factor.unsqueeze(-1) * oi + torch.matmul(p_tilde, vj)

                        # Update running statistics
                        mi = mi_new
                        li = li_new

                    # 6. Finalize Output for this Q tile
                    # O_i = diag(l_i)^{-1} * O_i_accum
                    oi = oi / (li.unsqueeze(-1) + 1e-6)
                    
                    # Write back to Global Memory
                    o[b, h, q_start:q_end, :] = oi
                    
                    # 7. Compute and store L
                    l_final[b, h, q_start:q_end] = mi + torch.log(li + 1e-6)

        # Restore shapes if input was 3D
        if len(q_input_shape) == 3:
            o = o.squeeze(1)
            l_final = l_final.squeeze(1)

        # Save tensors for backward
        ctx.save_for_backward(q, k, v, o, l_final)
        ctx.is_causal = is_causal 
        
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet")