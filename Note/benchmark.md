
### 1\. Imports

```python
import torch
import time
import argparse
import contextlib
import torch.cuda.nvtx as nvtx  # 关键库
import numpy as np

from cs336_basics.model import BasicsTransformerLM 
```

  * **`torch.cuda.nvtx`**: NVIDIA Tools Extension，使用该库我们可以在 Nsight Systems 的时间轴图表上打上“标签”，然后就能看到 "Forward", "Backward" 这样的高层逻辑块。
  * **`contextlib`**: 用于 `nullcontext`。当不需要混合精度（Mixed Precision）时，保持代码结构一致的小 trick。
  * **`cs336_basics.model`**: 这是 clone 下来直接就有的 Transformer 模型实现。

### 2\. 模型配置 (Model Configurations)

```python
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    # ... 其他尺寸 ...
}
```

  * **对应 PDF 1.1.2 Table 1**: 直接把 PDF 中的表格硬编码成了字典，可以通过命令行参数 `--model_size xl` 直接调用，方便自动化测试。

### 3\. 内存分析辅助函数 (`profile_memory`)

这个函数对应 PDF **1.1.6 Profiling Memory** 的要求。

```python
def profile_memory(args, model, x, y, optimizer, loss_fn, ctx):
    print(f"Starting memory recording for model: {args.model_size}...")
    # 1. 开始记录显存历史
    # max_entries 限制记录的条目数，防止记录过程本身耗尽内存
    torch.cuda.memory._record_memory_history(max_entries=100000)

    try:
        with ctx:  # 进入混合精度上下文（如果开启）
            logits = model(x)
            if args.mode == 'bwd':
                # 如果是反向传播模式，需要计算 Loss 并 Backward
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                # set_to_none=True 比默认的 set_to_0 更快且更省显存
                optimizer.zero_grad(set_to_none=True)
        
        # 等待 GPU 完成所有操作，确保内存状态稳定
        torch.cuda.synchronize()
    except RuntimeError as e:
        # 捕获 OOM 错误
        print(f"Error during memory profiling (likely OOM): {e}")
        return

    # 2. 导出快照文件
    filename = f"memory_snapshot_{args.model_size}_{args.mode}.pickle"
    torch.cuda.memory_dump_snapshot(filename)
    print(f"Snapshot saved to {filename}")
    
    # 3. 停止记录
    torch.cuda.memory._record_memory_history(enabled=None)
```

  * **作用**: 这段代码不关心运行速度，只关心内存分配。它生成的 `.pickle` 文件可以拖拽到 `pytorch.org/memory_viz` 进行可视化分析，找出显存峰值（Peak Memory）。

### 4\. 主运行逻辑 (`run_benchmark`)

核心部分。

#### 4.1 初始化阶段

```python
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 检查 model_size 是否存在 
        
    config = MODEL_CONFIGS[args.model_size]
    
    print(f"Initializing model: {args.model_size} on {device}...")
    # 初始化模型
    # **config 将字典解包为关键字参数
    model = BasicsTransformerLM(
        vocab_size=10000, 
        context_length=args.context_length,
        rope_theta=10000.0,  # 位置编码参数
        **config
    ).to(device)  # 关键：将模型移动到 GPU
    
    # 生成随机数据，模拟输入
    # 对于速度测试，数据内容不重要，只要形状（Shape）对就行
    x = torch.randint(0, 10000, (4, args.context_length)).to(device)
    y = torch.randint(0, 10000, (4, args.context_length)).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
```

#### 4.2 混合精度上下文设置 (PDF 1.1.5)

```python
    # 如果 args.mixed_precision 为真，使用 bfloat16，否则 float32
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    
    # 创建上下文管理器
    # 如果启用了混合精度，ctx 就是 torch.autocast
    # 如果没启用，ctx 就是 nullcontext()，意味着什么都不做
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.mixed_precision else contextlib.nullcontext()
```

  * **Context Manager**: 小 trick，避免了在主循环里写 `if args.mixed_precision:` 判断。

#### 4.3 预热阶段 (Warm-up) (PDF 1.1.3 a/c)

```python
    # ... 如果是 profile_memory 模式，前面就返回了 ...

    # --- Warmup ---
    print(f"Warming up for {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        with ctx:
            # 标准的 Training Step
            logits = model(x)
            if args.mode == 'bwd':
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
    # 必须同步
    # 预热不仅是为了让缓存热起来，也是为了让 PyTorch JIT 编译器完成内核融合等优化。
    torch.cuda.synchronize()
```

#### 4.4 正式计时阶段 (Benchmarking Phase)

这部分对应 PDF **1.1.3 End-to-End Benchmarking** 和 **1.1.4 Nsight Systems Profiler**。

```python
    # --- 正式计时 ---
    print(f"Benchmarking for {args.steps} steps...")
    timings = []

    for i in range(args.steps):
        # 1. 确保上一步彻底完成 (习惯)
        torch.cuda.synchronize()
        
        # 2. 开始计时
        # perf_counter 是 Python 中测量短时间间隔最高精度的时钟
        start_t = time.perf_counter()
        
        # 3. 运行 Step (带 NVTX 标记)
        # 在 Nsight 视图中，会看到一个大的 "Step X" 块
        with nvtx.range(f"Step {i}"):
            with ctx:
                # Forward Pass
                # 在 Nsight 中，这部分会被标记为 "Forward"
                with nvtx.range("Forward"):
                    logits = model(x)
                
                if args.mode == 'bwd':
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Backward Pass
                    # 反向传播
                    with nvtx.range("Backward"):
                        loss.backward()
                    
                    # Optimizer Step
                    with nvtx.range("Optimizer"):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True) 

        # 4. 再次同步
        # 解释：CUDA 调用是异步的。当调用 model(x) 时，Python 代码瞬间就执行完了，
        # 但 GPU 可能还在排队计算。如果不加 synchronize，测量到的只是 CPU "把任务发射给 GPU 的时间"，而不是 "GPU 计算的时间"。
        torch.cuda.synchronize()
        
        # 5. 结束计时
        end_t = time.perf_counter()
        timings.append(end_t - start_t)
```

#### 4.5 统计与输出

```python
    # --- 统计结果 ---
    timings = np.array(timings) * 1000 # 将秒转换为毫秒 (ms)
    mean_time = np.mean(timings)
    std_time = np.std(timings)

    # 打印报表
    print("-" * 60)
    # ... 打印配置 ...
    print("-" * 60)
    print(f"Avg Time: {mean_time:.2f} ms")
    print(f"Std Dev:  {std_time:.2f} ms") # 标准差越小说明测试越稳定
    print("-" * 60)
```

### 5\. 入口

```python
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
```

- model_size：模型大小
- context_length：控制上下文长度大小
- steps：测量步数（不算 warmup 的部分）
- warmup_steps：预热，初始化开销，进入稳定状态，保证测量的是“稳定状态”的速度
- mode：是否计算反向传播的速度
- mixed_precision：是否启用混合精度（FP32 与 BF16 的对比）
- profile_memory：是否测显存