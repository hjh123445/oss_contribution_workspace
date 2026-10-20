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

## Real-model results (A800-SXM4-80GB, 2026-10-20, wheel 0.31.0 A/B)

Stock zeroth-pass fallback rate on ~250 teacher-forced decode rows per model:
- Qwen3-0.6B (V=151936): k=20 0.0% | k=200 10.0% | k=500 81.2%
- Qwen3-8B   (V=151936): k=20 0.0% | k=200  7.2% | k=500 73.6%
- MiMo-V2.6-Distill-Qwen-9B (V=248320): k=20 0.0% | k=200 3.4% | k=500 72.0%

Kernel on real rows (B=64, top-k + top-p=0.95, CUDA graph, median of 50):
- Qwen3-0.6B: k=200 1010.7 -> 346.6 us (2.9x); k=500 1095.7 -> 354.3 us (3.1x); k=1000 1125.4 -> 401.4 us (2.8x)
- Qwen3-8B:   k=200  931.8 -> 349.2 us (2.7x); k=500 1018.9 -> 352.3 us (2.9x)
- MiMo-9B:    k=200 1570.8 -> 556.0 us (2.8x); k=500 1762.3 -> 558.1 us (3.2x)
- healthy rows (k=20): +3-9%

Masks: stock vs fixed bit-identical (torch.equal) on every model, every k.

E2E (Qwen3-8B, 64 concurrent x 192 tokens, VLLM_USE_FLASHINFER_SAMPLER=0):
- k=200: sampler 606.7 -> 378.6 us/call, 4147 -> 4225 tok/s (+1.9%)
- k=500: sampler 990.6 -> 395.9 us/call, 4059 -> 4198 tok/s (+3.4%)
