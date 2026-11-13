# Results: 218 cuDNN Convolution Microbenchmark

**Main goal:** Measure cuDNN convolution solver speed  
**Tools:** Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via collect_cudnn_conv.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 NVL (95830 MiB) · AMD EPYC 9374F 32-Core Processor · Ubuntu 24.04.5 LTS · kernel 6.8.0-138-generic · driver 595.91.07
- **Run:** 2026-11-12T23:06:05Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 00:38 · benchmark 00:03 · whole run 00:48
- **Source:** run 218_20261112_230605
- **Command:** `bash run_benchmark.sh --smoke --dtype FP16 --conv-direction all --batch-size 1 --input-channels 8 --input-height 16 --input-width 16 --output-channels 8 --kernel-height 3 --kernel-width 3 --stride-h 1 --stride-w 1 --group-count 1 --search-strategy find --warmup-iters 1 --num-iterations 2`

| # | Metric | Result |
|---|---|---|
| 1 | Convolution kernel execution time, forward (kernel_time_msec) | 0.019192 |
| 2 | Achieved convolution throughput, forward, TFLOP/s (tflops_fwd) | 0.015369500000000001 |
| 3 | Minimum forward operand traffic, GB/s (min_operand_traffic_gb_s) | 0.4869735 |
| 4 | Solver search time (solver_search_msec), sum of cudnnFind wall time across the directions run | 189.405506 |
<!-- /profile:smoke -->
