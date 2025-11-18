# 运行时间测试
bash run_all.sh

# 生成 Nsight Systems 报告
CUDA_VISIBLE_DEVICES=3 nsys profile \
    -o profile_2.7B_fwd \
    --force-overwrite true \
    --python-backtrace=cuda \
    uv run python benchmark.py \
    --model_size 2.7B \
    --mode fwd \
    --mixed_precision \
    --context_length 128 \
    --steps 10 \
    --warmup_steps 10

# 显存分析
CUDA_VISIBLE_DEVICES=2 uv run python benchmark.py --model_size small --mode fwd --profile_memory
CUDA_VISIBLE_DEVICES=2 uv run python benchmark.py --model_size 2.7B --mode bwd --profile_memory


