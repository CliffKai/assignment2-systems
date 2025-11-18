for w in 50 100; do
  for s in 10 100 200; do
    for r in 5 10; do
      uv run python run_batch_benchmarks.py --device 7 --warmup_steps $w --steps $s --repeats $r
    done
  done
done
