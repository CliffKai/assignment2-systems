
### 第一阶段：基准测试与性能分析 (Benchmarking & Profiling)
**目标：** 建立基线 (Baseline)，学会使用工具分析性能瓶颈。这部分不涉及复杂的算子实现，但对后续优化至关重要。

1.  **环境准备 & 模型导入**
    * 导入 `cs336_basics` 中的 Transformer 模型（原本就有）。
    * 在 `cs336_systems` 中创建一个 `benchmark.py` 脚本。

2.  **端到端基准测试**
    * **实现：** 编写脚本测量 Forward 和 Backward 的运行时间。
    * **关键点：** 必须使用 `torch.cuda.synchronize()` 确保计时准确；实现 Warm-up 机制。
    * **产出：** 记录不同模型尺寸（small 到 2.7B）的运行时间。

3.  **Nsight Systems 性能分析**
    * **实现：** 使用 `nsys profile` 运行脚本。
    * **代码修改：** 使用 `torch.cuda.nvtx.range` 在代码中标记 "Attention", "FFN" 等区域，以便在视图中定位。
    * **分析：** 找出耗时最长的 CUDA Kernel（通常是 GEMM），观察 Softmax 和 Attention 的开销。

4.  **混合精度训练**
    * **实现：** 修改基准脚本，使用 `torch.autocast(dtype=torch.bfloat16)`。
    * **思考：** LayerNorm 为什么通常保持 FP32？(数值稳定性)。
    * **产出：** 对比 FP32 和 BF16 的速度差异。

5.  **显存分析**
    * **实现：** 使用 `torch.cuda.memory._record_memory_history` 抓取显存快照。
    * **工具：** 使用 [https://pytorch.org/memory_viz](https://pytorch.org/memory_viz) 可视化。
    * **产出：** 确定 Activation Memory（中间激活值）是显存占用的主要来源。

### 第二阶段：FlashAttention-2 实现 (核心)
**目标：** 优化 Transformer 中最耗显存的 Attention 层。这部分是作业中最硬核的编码部分。

1.  **PyTorch 原生 FlashAttention**
    * **任务：** 为了理解分块 (Tiling) 和在线 Softmax (Online Softmax) 算法，先用纯 PyTorch 实现 FlashAttention-2 的逻辑。
    * **位置：** 实现 `adapters.get_flashattention_autograd_function_pytorch`。
    * **逻辑：**
        * 按照 PDF Algorithm 1 实现。
        * 手动对 Query (Q), Key (K), Value (V) 进行分块循环。
        * 实现 Online Softmax 更新公式 ($m, l$ 的更新)。
    * **测试：** `pytest -k test_flash_forward_pass_pytorch`

2.  **编译优化**
    * **任务：** 对上述 PyTorch 实现使用 `torch.compile`。
    * **产出：** 对比编译前后的速度。

3.  **Triton Kernel 实现**
    * **任务：** 用 Triton 语言编写 Fused Kernel。这是作业最难的部分。
    * **位置：** 实现 `adapters.get_flashattention_autograd_function_triton`。
    * **步骤：**
        * **Block Pointers:** 使用 `tl.make_block_ptr` 处理指针运算（这是 Triton 新特性，比旧版指针运算简单）。
        * **Forward Loop:** 实现 Q 循环（外层，Grid维度）和 K 循环（内层）。
        * **计算逻辑：** 在 Kernel 内完成 $QK^T$ -> Mask -> Softmax -> $\times V$ 的全过程。
        * **Causal Masking:** 添加 `is_causal` 标志，在 Kernel 内利用 `tl.program_id` 和 `tl.arange` 构造掩码。
    * **测试：** `pytest -k test_flash_forward_pass_triton`。

4.  **FlashAttention Backward**
    * **任务：** 实现反向传播。
    * **简化策略：** 此处可以**不写 Triton Kernel**，而是使用 PyTorch 实现，然后用 `torch.compile` 加速。
    * **位置：** 完善 Autograd Function 的 `backward` 方法。
    * **测试：** `pytest -k test_flash_backward`

---

### 第三阶段：分布式数据并行 (DDP)
**目标：** 实现多 GPU 训练的基础设施。从朴素实现到重叠通信的高效实现。

1.  **通信原语基准测试**
    * **任务：** 编写脚本测试 `dist.all_reduce` 在不同数据量下的耗时。
    * **目的：** 理解通信带宽和延迟。

2.  **朴素 DDP 实现**
    * **任务：** 在 Backward 结束后，对每个参数分别调用 `all_reduce`。
    * **位置：** `adapters.get_ddp_individual_parameters` (初始版本)。
    * **测试：** 验证多卡训练后的权重是否一致。

3.  **通信与计算重叠 (Overlap)**
    * **任务：** 利用 PyTorch 的 Hook (`register_post_accumulate_grad_hook`)。
    * **逻辑：** 当某层梯度假设计算完毕，立即触发异步 `all_reduce`，而不是等所有层算完。
    * **同步：** 在 `optimizer.step()` 前必须确保所有异步通信完成 (`finish_gradient_synchronization`)。
    * **位置：** `adapters.ddp_individual_parameters_on_after_backward`。

4.  **梯度分桶 (Bucketing)**
    * **任务：** 将多个小的梯度 Tensor 拼成一个大 Bucket 进行通信，减少通信握手开销。
    * **位置：** 实现 `adapters.get_ddp_bucketed`。
    * **逻辑：**
        * 在 Hook 中将梯度拷贝到预分配的 Bucket Buffer。
        * 当 Bucket 满时，触发一次 `all_reduce`。
    * **测试：** `pytest tests/test_ddp.py`

### 第四阶段：优化器状态分片 (Optimizer State Sharding)
**目标：** 解决 DDP 显存占用翻倍的问题（类似 DeepSpeed ZeRO-1）。

1.  **实现 Sharded Optimizer**
    * **原理：** 每个 GPU 只维护 1/N 的参数的优化器状态（如 Adam 的 momentum 和 variance）。
    * **位置：** 实现 `adapters.get_sharded_optimizer`。
    * **步骤：**
        * **初始化 (`__init__`)：** 将模型参数平分给各个 Rank。Rank 0 负责参数 0-10，Rank 1 负责参数 11-20。
        * **Step (`step`)：**
            1.  每个 Rank 只更新它负责的那部分参数。
            2.  更新后，使用 `dist.broadcast` 或 `dist.all_gather` 将更新后的参数同步给所有其他 Rank。
    * **测试：** `pytest tests/test_sharded_optimizer.py`

### 第五阶段：总结与报告 (Finalizing)
**目标：** 完成 PDF 中的 Written Questions。

1.  **填充 Writeup：**
    * 回答所有关于 Profile 结果、显存计算、4D 并行理论的问题。

2.  **检查：**
    * 运行 `./test_and_make_submission.sh`。
    * 确保所有测试通过（因为使用了 Triton，基本是强绑定 GPU 架构的，所以必须在 H100 环境下测试）。