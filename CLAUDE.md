# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CS336 Assignment 2: Systems - a course assignment focused on implementing optimized Transformer components and distributed training. The project builds on Assignment 1 (cs336-basics) which provides a baseline Transformer implementation.

## Build and Test Commands

```bash
# Run all tests
uv run pytest -v ./tests

# Run a single test file
uv run pytest -v ./tests/test_attention.py

# Run a specific test
uv run pytest -v -k test_flash_forward_pass_pytorch

# Run tests and create submission
./test_and_make_submission.sh

# Run Python with dependencies
uv run python
```

## Architecture

### Module Structure

- **cs336_systems/**: Main implementation module for this assignment
  - `flash_attention.py`: FlashAttention2 implementation using PyTorch autograd
  - `flash_attention_triton.py`: FlashAttention2 using Triton kernels (GPU)
  - `benchmark.py`: Benchmarking utilities

- **cs336-basics/cs336_basics/**: Staff implementation from Assignment 1 (can be replaced with your own)
  - `model.py`: Transformer LM with RoPE, RMSNorm, SwiGLU, multi-head attention
  - `nn_utils.py`: Neural network utilities (softmax, etc.)
  - `optimizer.py`: Optimizer implementations
  - `data.py`: Data loading utilities

- **tests/adapters.py**: Adapter functions that tests use to access your implementations. Implement the functions here to expose your code to the test suite:
  - `get_flashattention_autograd_function_pytorch()`: Return FlashAttention2 autograd class
  - `get_flashattention_autograd_function_triton()`: Return Triton-based FlashAttention2
  - `get_ddp_individual_parameters()`: Return DDP wrapper with per-parameter gradient sync
  - `get_ddp_bucketed()`: Return DDP wrapper with bucketed gradient sync
  - `get_sharded_optimizer()`: Return optimizer with sharded state

### Key Implementation Tasks

1. **FlashAttention2**: Memory-efficient attention using tiled computation
   - Forward pass saves log-sum-exp (L) tensor for backward
   - Block sizes: Br=32, Bc=32 for tiled computation
   - Supports causal masking via `is_causal` parameter

2. **Distributed Data Parallel (DDP)**: Two variants
   - Individual parameters: Async gradient sync per parameter
   - Bucketed: Gradient sync in configurable bucket sizes (MB)

3. **Sharded Optimizer**: ZeRO-style optimizer state sharding across ranks

### Test Fixtures

Tests use `torch.multiprocessing.spawn` for distributed tests with gloo backend. Test data is stored in `tests/_fixtures/` and snapshots in `tests/_snapshots/`.

## Dependencies

Managed via `uv`. Key dependencies: PyTorch 2.6, einops, einx, jaxtyping. The cs336-basics package is installed as an editable local dependency.
