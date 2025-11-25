import torch
import time
import pandas as pd
import argparse
from cs336_systems.flash_attention import FlashAttentionFunction

def naive_attention(q, k, v):
    """Standard Attention implementation"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o

def benchmark_run(name, func, q, k, v, do, n_iters=100, is_backward=False):
    # Warmup
    for _ in range(5):
        o = func(q, k, v)
        if is_backward:
            o.backward(do, retain_graph=True)
            q.grad = None; k.grad = None; v.grad = None
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        o = func(q, k, v)
        if is_backward:
            o.backward(do, retain_graph=True)
            # Reset grads to avoid accumulation overhead affecting timing
            q.grad = None; k.grad = None; v.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / n_iters * 1000  # ms

def run_benchmark_suite(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 
    
    BATCH_SIZE = args.batch_size
    d_heads = args.heads
    seq_lens = args.seq_lens
    
    results = []

    print(f"Benchmarking with BatchSize={BATCH_SIZE}, Iters={args.n_iters}")
    print(f"{'SeqLen':<8} | {'Dim':<4} | {'Mode':<15} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10}")
    print("-" * 65)

    for N in seq_lens:
        for D in d_heads:
            try:
                torch.manual_seed(0)
                # Shape: (B, 1, N, D) implies H=1 as per assignment
                q = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(BATCH_SIZE, 1, N, D, device=device, dtype=dtype, requires_grad=True)
                do = torch.randn_like(q)

                # 1. Naive PyTorch
                fwd_naive = benchmark_run("Naive", naive_attention, q, k, v, do, n_iters=args.n_iters, is_backward=False)
                
                # Check for OOM before running backward if possible, or rely on try-catch
                try:
                    bwd_naive_total = benchmark_run("Naive", naive_attention, q, k, v, do, n_iters=args.n_iters, is_backward=True)
                    bwd_only_naive = max(0, bwd_naive_total - fwd_naive)
                except torch.cuda.OutOfMemoryError:
                    bwd_only_naive = "OOM"

                results.append({"SeqLen": N, "Dim": D, "Impl": "Naive", "Fwd": fwd_naive, "Bwd": bwd_only_naive})
                print(f"{N:<8} | {D:<4} | {'Naive':<15} | {fwd_naive:<10.3f} | {bwd_only_naive if isinstance(bwd_only_naive, str) else f'{bwd_only_naive:<10.3f}'}")

                # 2. Compiled PyTorch
                torch.cuda.empty_cache()
                q.grad = None; k.grad = None; v.grad = None
                compiled_fn = torch.compile(naive_attention)
                
                fwd_comp = benchmark_run("Compiled", compiled_fn, q, k, v, do, n_iters=args.n_iters, is_backward=False)
                
                try:
                    bwd_comp_total = benchmark_run("Compiled", compiled_fn, q, k, v, do, n_iters=args.n_iters, is_backward=True)
                    bwd_only_comp = max(0, bwd_comp_total - fwd_comp)
                except torch.cuda.OutOfMemoryError:
                    bwd_only_comp = "OOM"

                results.append({"SeqLen": N, "Dim": D, "Impl": "Compiled", "Fwd": fwd_comp, "Bwd": bwd_only_comp})
                print(f"{N:<8} | {D:<4} | {'Compiled':<15} | {fwd_comp:<10.3f} | {bwd_only_comp if isinstance(bwd_only_comp, str) else f'{bwd_only_comp:<10.3f}'}")
                
                # Optional: Tiled Implementation (Slow)
                if args.run_tiled:
                    torch.cuda.empty_cache()
                    q.grad = None; k.grad = None; v.grad = None
                    fwd_tiled = benchmark_run("Tiled(Py)", FlashAttentionFunction.apply, q, k, v, do, n_iters=5, is_backward=False) # fewer iters for slow code
                    print(f"{N:<8} | {D:<4} | {'Tiled(Py)':<15} | {fwd_tiled:<10.3f} | {'N/A':<10}")

            except torch.cuda.OutOfMemoryError:
                print(f"{N:<8} | {D:<4} | {'ALL':<15} | {'OOM':<10} | {'OOM':<10}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error at N={N}, D={D}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Attention Implementations")
    
    # PDF 要求默认遍历这些参数 
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (fixed to 8 per PDF)")
    parser.add_argument("--heads", type=int, nargs="+", default=[16, 32, 64, 128], help="List of head dimensions to sweep")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384], help="List of sequence lengths to sweep")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations for timing (PDF requires 100)")
    parser.add_argument("--output_file", type=str, default="attention_benchmark_results.csv", help="Output CSV filename")
    parser.add_argument("--run_tiled", action="store_true", help="Run the slow Python Tiled implementation for comparison")

    args = parser.parse_args()
    run_benchmark_suite(args)