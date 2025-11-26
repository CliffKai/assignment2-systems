# test fwd
uv run pytest -v -k test_flash_forward_pass_triton

# test bwd
uv run pytest -v -k test_flash_backward_triton

# test all
CUDA_VISIBLE_DEVICES=1 uv run pytest -v -k triton

# BF16
CUDA_VISIBLE_DEVICES=3 uv run python benchmark_flash_attention.py --causal

# FP32
CUDA_VISIBLE_DEVICES=3 uv run python benchmark_flash_attention.py --causal --dtype float32