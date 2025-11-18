import subprocess
import csv
import os
import re
import argparse
import datetime
import sys

# ================= 配置区域 =================
# 你可以在这里调整要测试的参数范围
MODEL_SIZES = ["small", "medium", "large", "xl", "2.7B"]
MODES = ["fwd", "bwd"]
PRECISIONS = ["fp32", "bf16"] # bf16 对应 --mixed_precision
CONTEXT_LENGTH = 128
# ===========================================

def parse_output(output_str):
    """从 benchmark.py 的输出中提取 Avg Time 和 Std Dev"""
    avg_match = re.search(r"Avg Time:\s+([\d\.]+)\s+ms", output_str)
    std_match = re.search(r"Std Dev:\s+([\d\.]+)\s+ms", output_str)
    
    avg_time = float(avg_match.group(1)) if avg_match else None
    std_dev = float(std_match.group(1)) if std_match else None
    return avg_time, std_dev

def main():
    parser = argparse.ArgumentParser(description="Batch runner for CS336 benchmarks")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to run each config")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps inside benchmark.py")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps inside benchmark.py")
    args = parser.parse_args()

    # 1. 准备结果目录和文件
    # 当前在 cs336_systems, 目标是 ../result
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(os.path.dirname(current_dir), "result")
    os.makedirs(result_dir, exist_ok=True)
    
    csv_file = os.path.join(
        result_dir,
        f"benchmark_w{args.warmup_steps}_s{args.steps}_r{args.repeats}.csv"
    )
    file_exists = os.path.isfile(csv_file)

    print(f"🚀 Starting Batch Benchmark on GPU {args.device}...")
    print(f"📂 Results will be saved to: {csv_file}")
    print("-" * 60)

    # 2. 打开 CSV 文件 (追加模式 'a')
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入表头
        if not file_exists:
            header = [
                "Timestamp",
                "Model_Size",
                "Mode",
                "Precision",
                "Context_Len",
                "Run_Index",
                "Avg_Time_ms",
                "Std_Dev_ms",
                "Steps_Per_Run",
                "Warmup_Steps"
            ]
            writer.writerow(header)

        # 3. 循环遍历所有配置
        total_experiments = len(MODEL_SIZES) * len(MODES) * len(PRECISIONS) * args.repeats
        count = 0

        for size in MODEL_SIZES:
            for mode in MODES:
                for prec in PRECISIONS:
                    # 构建基础命令
                    cmd = [
                        "uv", "run", "python", "benchmark.py",
                        "--model_size", size,
                        "--mode", mode,
                        "--context_length", str(CONTEXT_LENGTH),
                        "--steps", str(args.steps),
                        "--warmup_steps", str(args.warmup_steps),
                    ]
                    
                    if prec == "bf16":
                        cmd.append("--mixed_precision")

                    # 重复运行 N 次
                    for i in range(args.repeats):
                        count += 1
                        print(f"[{count}/{total_experiments}] Running: {size} | {mode} | {prec} | Run {i+1}/{args.repeats} ... ", end="", flush=True)
                        
                        try:
                            # 设置环境变量指定 GPU
                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = args.device

                            # 执行命令
                            result = subprocess.run(
                                cmd, 
                                env=env, 
                                capture_output=True, 
                                text=True,
                                check=True
                            )
                            
                            # 解析结果
                            avg_time, std_dev = parse_output(result.stdout)
                            
                            if avg_time is not None:
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                writer.writerow([timestamp, size, mode, prec, CONTEXT_LENGTH, i+1, avg_time, std_dev, args.steps])
                                f.flush() # 立即写入磁盘，防止中断丢失
                                print(f"✅ {avg_time:.2f} ms")
                            else:
                                print("⚠️ Parse Error (Check benchmark.py output)")

                        except subprocess.CalledProcessError as e:
                            print(f"❌ Failed (OOM or Error)")
                            # 可以选择记录错误到 CSV，或者直接跳过
                            # writer.writerow([timestamp, size, mode, prec, CONTEXT_LENGTH, i+1, "ERROR", "ERROR", args.steps])

    print("-" * 60)
    print("🎉 All benchmarks completed!")

if __name__ == "__main__":
    main()