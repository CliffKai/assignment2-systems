# FP32
CUDA_VISIBLE_DEVICES=7 uv run python benchmark.py --model_size small --mode fwd

# BF16
CUDA_VISIBLE_DEVICES=7 uv run python benchmark.py --model_size small --mode fwd --mixed_precision

# 生成 Nsight Systems 报告
CUDA_VISIBLE_DEVICES=7 nsys profile \
    -o profile_small_fwd \
    --force-overwrite true \
    --python-backtrace=cuda \
    uv run python benchmark.py --model_size small --mode fwd --mixed_precision

# 显存分析
CUDA_VISIBLE_DEVICES=7 uv run python benchmark.py --model_size small --mode fwd --profile_memory
