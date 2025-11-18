`benchmark.py` 是作业 a2:system 的核心工具，主要作用是准确测量模型在 GPU 上的运行速度和现存占用情况。

### 1\. 导入与模型配置部分

```python
import torch
import time
import argparse
import contextlib
import torch.cuda.nvtx as nvtx

# 这部分 clone 下来就是直接有的
from cs336_basics.model import BasicsTransformerLM 
```

  * **`contextlib`**: 用于处理上下文管理器。用于处理“是否开启混合精度”的逻辑（`nullcontext`）。
  * **`torch.cuda.nvtx`**: **NVIDIA Tools Extension**。这是配合 `nsys` (Nsight Systems) 分析器使用的工具。它允许我们在 Python 代码中打“标签”或“范围”，这样当你查看性能分析图表时，能清楚地看到哪段 GPU 时间对应的是代码中的 "Step i"。
  * **`BasicsTransformerLM`**: 从作业 1 继承过来的 Transformer 模型。

```python
# -----------------------------------------------------------------------------
# 1. 模型配置 (Step 1)
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    # ...
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
```

  * 对应了 PDF 中 **Table 1** 的参数。将这些配置写死在字典里，可以通过命令行参数 `--model_size small` 快速切换配置，不用每次手动修改参数。

### 2\. 显存分析函数 `profile_memory`

这个函数对应 PDF **1.1.6 Profiling Memory** 的要求。

```python
def profile_memory(args, model, x, y, optimizer, loss_fn, ctx):
    print(f"Starting memory recording for model: {args.model_size}...")
    
    # 1. 开始记录显存历史
    # PyTorch 会在后台记录所有的 malloc 和 free 操作，以及对应的堆栈跟踪。
    # max_entries=100000 限制记录条数，防止内存爆炸。
    torch.cuda.memory._record_memory_history(max_entries=100000)

    try:
        # 2. 运行单次完整的 Step
        # 我们只需要跑一次就可以抓到显存分配的峰值和生命周期。
        with ctx:  # 应用混合精度上下文（如果是 FP32，ctx 就是个空操作）
            logits = model(x)
            if args.mode == 'bwd':
                # 如果是 backward 模式，需要计算 Loss 并反向传播
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # 3. 强制同步
        # 这是一个非常好的习惯。虽然记录历史是在 CPU 侧控制的，但确保 GPU 完成工作再 Dump 可以保证状态一致。
        torch.cuda.synchronize()
        
    except RuntimeError as e:
        # 如果显存爆了 (OOM)，捕获异常并打印，避免程序直接崩溃看不到报错。
        print(f"Error during memory profiling (likely OOM): {e}")
        return

    # 4. 导出快照
    # 这个文件 (.pickle) 就是后面需要拖进 https://pytorch.org/memory_viz 进行可视化的文件。
    filename = f"memory_snapshot_{args.model_size}_{args.mode}.pickle"
    torch.cuda.memory_dump_snapshot(filename)
    print(f"Snapshot saved to {filename}")
    
    # 5. 停止记录
    # 关闭记录器，释放开销。
    torch.cuda.memory._record_memory_history(enabled=None)
```

### 3\. 主基准测试逻辑 `run_benchmark`

这是代码的核心部分，对应 PDF **1.1.3 End-to-End Benchmarking**。

#### 初始化与数据准备

```python
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ... 获取 config ...
    
    print(f"Initializing model: {args.model_size} on {device}...")
    model = BasicsTransformerLM(
            vocab_size=10000, 
            context_length=args.context_length,
            rope_theta=10000.0,
            **config
        ).to(device)             # 关键：将模型搬运到 GPU
    
    # 模拟数据 (Batch Size = 4 固定值，PDF 要求)
    # 只要形状对即可，数值无所谓，因为我们测的是速度。
    x = torch.randint(0, 10000, (4, args.context_length)).to(device)
    y = torch.randint(0, 10000, (4, args.context_length)).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
```

#### 混合精度上下文设置 (Step 2 涉及)

```python
    # 设置混合精度上下文
    # 如果命令行开启 --mixed_precision，则使用 bfloat16
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    
    # torch.autocast 是 PyTorch 自动混合精度的方法。
    # 如果 args.mixed_precision 为 False，使用 contextlib.nullcontext()。
    # nullcontext 是一个空管理器，进入和退出什么都不做，
    # 这样我们就不用写两套代码（一套带 with autocast，一套不带）。
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.mixed_precision else contextlib.nullcontext()
```

#### 分支：跑显存分析还是跑速度测试？

```python
    # 如果是内存分析模式，直接跳转到上面的函数，跑完就退出。
    if args.profile_memory:
        profile_memory(args, model, x, y, optimizer, loss_fn, ctx)
        return
```

#### 预热 (Warm-up) - 极其重要

```python
    # --- Warmup (预热) ---
    print(f"Warming up for {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        with ctx:
            # 这里只是跑代码，不计时
            logits = model(x)
            if args.mode == 'bwd':
                # .view(-1, ...) 是为了展平 tensor 适配 CrossEntropyLoss
                # [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    # 关键点：torch.cuda.synchronize()
    # GPU 是异步执行的。Python 发送命令后会立即往下走。
    # 如果不在这里同步，正式计时的 start_time 可能会在预热还没真正结束时就开始了，导致计时偏大。
    torch.cuda.synchronize() 
```

#### 正式计时循环

```python
    # --- 正式计时 ---
    print(f"Benchmarking for {args.steps} steps...")
    start_time = time.time() # 记录 CPU 时间戳
    
    for i in range(args.steps):
        # NVTX 标记范围
        # 在 Nsight Systems 的时间轴上，会看到一个个叫 "Step 0", "Step 1" 的方块。
        # 这样就可以清楚地知道哪些 CUDA Kernel 属于哪一步训练。
        with nvtx.range(f"Step {i}"):
            with ctx:
                logits = model(x) # Forward
                if args.mode == 'bwd':
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
        # 确保每一步都由 CPU 等待 GPU 完成
        # 这里的 synchronize 实际上不是为了计时准确（因为总时间是在循环外算的），
        # 而是为了防止 CUDA 队列堆积过长，导致 OOM 或者触发 CPU/GPU 之间的瓶颈。
        # 对于 Benchmark 脚本，通常希望每一步都是干净的，所以加上比较好。
        # 此外，如果在循环内没有同步，end_time = time.time() 记录的时间仅仅是
        # CPU 传输完所有指令的时间，而不是 GPU "执行" 完的时间。
        # 所以，这里或者循环结束后，必须有至少一次 synchronize。
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # 计算平均时间
    avg_time = (end_time - start_time) / args.steps
```

**关于 `torch.cuda.synchronize()` 的理解：**
PyTorch 的 CUDA 操作是**异步**的。当我们执行 `output = model(input)` 时，Python 代码几乎瞬间完成，它只是把计算任务扔进了一个队列（CUDA Stream）。GPU 随后会在后台慢慢处理这个队列。
如果不写 `synchronize()`，`time.time()` 测出来的只是 CPU 把任务扔进队列的时间，而不是 GPU 真正运算矩阵乘法的时间。

### 4\. 命令行入口

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 选择模型大小，必须是 MODEL_CONFIGS 中的 key
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    
    # 上下文长度（输入序列长度），会影响显存和 Attention 速度
    parser.add_argument("--context_length", type=int, default=128)
    
    # 测量的步数，越多越稳定
    parser.add_argument("--steps", type=int, default=10)
    
    # 预热步数
    parser.add_argument("--warmup_steps", type=int, default=5)
    
    # 模式：只跑前向 (fwd) 还是 前向+反向+优化 (bwd)
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "bwd"])
    
    # 混合精度，不加这个参数，就是 False (FP32)；加了就是 True (BF16)。
    parser.add_argument("--mixed_precision", action="store_true", help="Use BF16 mixed precision")
    
    # 加了就只跑显存分析，不跑速度测试。
    parser.add_argument("--profile_memory", action="store_true", help="Generate memory snapshot instead of timing")
    
    args = parser.parse_args()
    run_benchmark(args)
```