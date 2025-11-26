import torch
import time
import pandas as pd
import argparse
import sys
from cs336_systems.flash_attention_triton import FlashAttentionTriton

# ==========================================
# 1. 定义 Naive Attention (支持 Causal)
# ==========================================
def naive_attention(q, k, v, scale=None, is_causal=False):
    """
    Standard Attention implementation: Softmax(QK^T)V
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # q, k, v: (B, H, N, D)
    # Transpose K: (B, H, D, N) -> S: (B, H, N, N)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        N = q.shape[-2]
        # 创建上三角 Mask (Mask out upper triangle)
        mask = torch.triu(torch.ones((N, N), device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float('-inf'))
        
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o

# ==========================================
# 2. 通用 Benchmark 函数
# ==========================================
def benchmark_run(func, args, do, n_iters=100, is_backward=False):
    # args 是一个元组，解包传递给 func(*args)
    
    # Warmup
    for _ in range(5):
        o = func(*args)
        if is_backward:
            o.backward(do, retain_graph=True)
            # Reset grads
            for t in args:
                if isinstance(t, torch.Tensor) and t.grad is not None:
                    t.grad = None
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        o = func(*args)
        if is_backward:
            o.backward(do, retain_graph=True)
            # Reset grads
            for t in args:
                if isinstance(t, torch.Tensor) and t.grad is not None:
                    t.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / n_iters * 1000  # ms

# ==========================================
# 3. 主测试循环
# ==========================================
def run_benchmark_suite(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据参数选择精度 (推荐 bfloat16 以获得最佳性能)
    dtype = getattr(torch, args.dtype)
    
    BATCH_SIZE = args.batch_size
    d_heads = args.heads
    seq_lens = args.seq_lens
    is_causal = args.causal
    
    results = []

    print(f"Benchmarking with Batch={BATCH_SIZE}, Iters={args.n_iters}, Dtype={dtype}, Causal={is_causal}")
    print(f"{'SeqLen':<8} | {'Dim':<4} | {'Impl':<15} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10}")
    print("-" * 65)

    for N in seq_lens:
        for D in d_heads:
            # 准备数据 (每次循环重置种子，保证公平)
            try:
                torch.manual_seed(0)
                # Shape: (B, 1, N, D) -> Heads=1 (作业要求单 Head 测试)
                q = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                do = torch.randn_like(q)
                scale = 1.0 / (D ** 0.5)
            except torch.cuda.OutOfMemoryError:
                print(f"{N:<8} | {D:<4} | {'ALL':<15} | {'OOM':<10} | {'OOM':<10}")
                torch.cuda.empty_cache()
                continue

            # -------------------------------------------------
            # 1. Naive PyTorch
            # -------------------------------------------------
            if not args.skip_naive:
                try:
                    torch.cuda.empty_cache()
                    # 打包参数
                    func_args = (q, k, v, scale, is_causal)
                    
                    fwd_naive = benchmark_run(naive_attention, func_args, do, n_iters=args.n_iters, is_backward=False)
                    
                    try:
                        bwd_naive_total = benchmark_run(naive_attention, func_args, do, n_iters=args.n_iters, is_backward=True)
                        bwd_only_naive = max(0, bwd_naive_total - fwd_naive)
                    except torch.cuda.OutOfMemoryError:
                        bwd_only_naive = "OOM"

                    results.append({"SeqLen": N, "Dim": D, "Impl": "Naive", "Fwd": fwd_naive, "Bwd": bwd_only_naive})
                    print(f"{N:<8} | {D:<4} | {'Naive':<15} | {fwd_naive:<10.3f} | {bwd_only_naive if isinstance(bwd_only_naive, str) else f'{bwd_only_naive:<10.3f}'}")
                
                except torch.cuda.OutOfMemoryError:
                     print(f"{N:<8} | {D:<4} | {'Naive':<15} | {'OOM':<10} | {'OOM':<10}")
                except Exception as e:
                    print(f"{N:<8} | {D:<4} | {'Naive':<15} | ERROR: {str(e)[:20]}")

            # -------------------------------------------------
            # 2. Compiled PyTorch
            # -------------------------------------------------
            if not args.skip_compiled:
                try:
                    torch.cuda.empty_cache()
                    torch.compiler.reset() # 强制重置编译器
                    q.grad = None; k.grad = None; v.grad = None
                    
                    compiled_fn = torch.compile(naive_attention)
                    func_args = (q, k, v, scale, is_causal)

                    fwd_comp = benchmark_run(compiled_fn, func_args, do, n_iters=args.n_iters, is_backward=False)
                    
                    try:
                        bwd_comp_total = benchmark_run(compiled_fn, func_args, do, n_iters=args.n_iters, is_backward=True)
                        bwd_only_comp = max(0, bwd_comp_total - fwd_comp)
                    except torch.cuda.OutOfMemoryError:
                        bwd_only_comp = "OOM"

                    results.append({"SeqLen": N, "Dim": D, "Impl": "Compiled", "Fwd": fwd_comp, "Bwd": bwd_only_comp})
                    print(f"{N:<8} | {D:<4} | {'Compiled':<15} | {fwd_comp:<10.3f} | {bwd_only_comp if isinstance(bwd_only_comp, str) else f'{bwd_only_comp:<10.3f}'}")

                except torch.cuda.OutOfMemoryError:
                     print(f"{N:<8} | {D:<4} | {'Compiled':<15} | {'OOM':<10} | {'OOM':<10}")
                except Exception as e:
                     print(f"{N:<8} | {D:<4} | {'Compiled':<15} | ERROR: {str(e)[:20]}")

            # -------------------------------------------------
            # 3. Triton FlashAttention
            # -------------------------------------------------
            if not args.skip_triton:
                try:
                    torch.cuda.empty_cache()
                    q.grad = None; k.grad = None; v.grad = None
                    
                    # Triton Wrapper Apply (Signature: q, k, v, is_causal)
                    triton_args = (q, k, v, is_causal)
                    
                    fwd_triton = benchmark_run(FlashAttentionTriton.apply, triton_args, do, n_iters=args.n_iters, is_backward=False)
                    
                    bwd_triton_total = benchmark_run(FlashAttentionTriton.apply, triton_args, do, n_iters=args.n_iters, is_backward=True)
                    bwd_only_triton = max(0, bwd_triton_total - fwd_triton)

                    results.append({"SeqLen": N, "Dim": D, "Impl": "Triton", "Fwd": fwd_triton, "Bwd": bwd_only_triton})
                    print(f"{N:<8} | {D:<4} | {'Triton':<15} | {fwd_triton:<10.3f} | {bwd_only_triton:<10.3f}")

                except torch.cuda.OutOfMemoryError:
                     print(f"{N:<8} | {D:<4} | {'Triton':<15} | {'OOM':<10} | {'OOM':<10}")
                except Exception as e:
                     print(f"{N:<8} | {D:<4} | {'Triton':<15} | ERROR: {str(e)}")


    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Attention Implementations Combined")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--heads", type=int, nargs="+", default=[16, 32, 64, 128])
    # 默认包含了 PDF 要求的全范围
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--n_iters", type=int, default=100)
    # H100 建议默认使用 bfloat16
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32 or bfloat16")
    parser.add_argument("--causal", action="store_true", help="Enable causal masking (default False, use flag to enable)")
    parser.add_argument("--skip_naive", action="store_true")
    parser.add_argument("--skip_compiled", action="store_true")
    parser.add_argument("--skip_triton", action="store_true")
    parser.add_argument("--output_file", type=str, default="attention_benchmark_combined.csv")

    args = parser.parse_args()
    run_benchmark_suite(args)