# vLLM contribution validation results (A800-SXM4-80GB, 2026-10-20)

## #61998 top-k pivot order statistic (branch fix/topk-order-statistic-pivot)
Benchmark (issue repro, CUDA graph, median of 50 us):
- two modes B=64 k=50: stock 923.6 -> fixed 182.3 (5.1x)
- two modes B=64 k=200: stock 1062.9 -> fixed 188.4 (5.6x)
- zipf sorted ids B=64 k=200: stock 1203.2 -> fixed 189.4 (6.4x)
- 63 healthy + 1 failing row, B=64 k=64: 620.5 -> 189.4 (3.3x)
- healthy zipf permuted rows: within ±3% of stock
Tests: existing sampler suite 154 passed; new exact-mask suite 8 passed;
5-distribution x 3-batch x 6-k fuzz: 0 mismatches.

## #62336 sliding-window probe scan (branch perf/sliding-window-probe-scan)
- cold cache 128K/4K example: 0.958 ms (8192 lookups) -> 0.032 ms (33 lookups), ~30x
- 1% random cached: 0.037 ms; full-cache tail hit: 0.202 ms (unchanged)
Equivalence: 49,128 real-class cases 0 differences;
114,681 pure-algorithm cases 0 differences.
Tests: existing manager suite 49 passed; new probe-scan suite 3 passed.
