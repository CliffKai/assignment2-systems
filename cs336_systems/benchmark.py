import torch
import time
import argparse
import contextlib
import torch.cuda.nvtx as nvtx
import numpy as np

from cs336_basics.model import BasicsTransformerLM 

# -----------------------------------------------------------------------------
# 1. 模型配置
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

# -----------------------------------------------------------------------------
# 2. 内存分析辅助函数
# -----------------------------------------------------------------------------
def profile_memory(args, model, x, y, optimizer, loss_fn, ctx):
    print(f"Starting memory recording for model: {args.model_size}...")
    torch.cuda.memory._record_memory_history(max_entries=100000)

    try:
        with ctx:
            logits = model(x)
            if args.mode == 'bwd':
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"Error during memory profiling (likely OOM): {e}")
        return

    filename = f"memory_snapshot_{args.model_size}_{args.mode}.pickle"
    torch.cuda.memory_dump_snapshot(filename)
    print(f"Snapshot saved to {filename}")
    torch.cuda.memory._record_memory_history(enabled=None)

# -----------------------------------------------------------------------------
# 3. 主运行逻辑
# -----------------------------------------------------------------------------
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_size not in MODEL_CONFIGS:
        raise ValueError(f"Model size {args.model_size} not found.")
        
    config = MODEL_CONFIGS[args.model_size]
    
    print(f"Initializing model: {args.model_size} on {device}...")
    model = BasicsTransformerLM(
        vocab_size=10000, 
        context_length=args.context_length,
        rope_theta=10000.0,
        **config
    ).to(device)
    
    x = torch.randint(0, 10000, (4, args.context_length)).to(device)
    y = torch.randint(0, 10000, (4, args.context_length)).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.mixed_precision else contextlib.nullcontext()

    if args.profile_memory:
        profile_memory(args, model, x, y, optimizer, loss_fn, ctx)
        return

    # --- Warmup ---
    print(f"Warming up for {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        with ctx:
            logits = model(x)
            if args.mode == 'bwd':
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # --- 正式计时 ---
    print(f"Benchmarking for {args.steps} steps...")
    timings = []

    for i in range(args.steps):
        # 1. 确保上一步彻底完成
        torch.cuda.synchronize()
        
        # 2. 开始计时 (使用 perf_counter)
        start_t = time.perf_counter()
        
        # 3. 运行 Step (带详细 NVTX 标记)
        with nvtx.range(f"Step {i}"):
            with ctx:
                # Forward Pass
                with nvtx.range("Forward"):
                    logits = model(x)
                
                if args.mode == 'bwd':
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Backward Pass
                    with nvtx.range("Backward"):
                        loss.backward()
                    
                    # Optimizer Step
                    with nvtx.range("Optimizer"):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True) # 性能优化

        # 4. 再次同步，确保当前 Step 彻底完成
        torch.cuda.synchronize()
        
        # 5. 结束计时
        end_t = time.perf_counter()
        timings.append(end_t - start_t)

    # --- 统计结果 ---
    timings = np.array(timings) * 1000 # 转换为毫秒
    mean_time = np.mean(timings)
    std_time = np.std(timings)

    print("-" * 60)
    print(f"Model: {args.model_size}")
    print(f"Mode: {args.mode}")
    print(f"Context: {args.context_length}")
    print(f"Precision: {'BF16' if args.mixed_precision else 'FP32'}")
    print(f"Steps: {args.steps}")
    print("-" * 60)
    print(f"Avg Time: {mean_time:.2f} ms")
    print(f"Std Dev:  {std_time:.2f} ms")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "bwd"])
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    
    args = parser.parse_args()
    run_benchmark(args)