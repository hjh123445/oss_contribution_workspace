# vLLM top-k order-statistic pivot (#61998)

Fix for [vllm-project/vllm#61998](https://github.com/vllm-project/vllm/issues/61998),
up for review as [PR #62596](https://github.com/vllm-project/vllm/pull/62596):
the top-k branch of `_topk_topp_kernel` picked its gather pivot from a Gaussian
fit to the first 8192 logits only. On real logits the head of the row carries
systematically higher values and skewed rows break the Gaussian fit, so the
gather under-covers `k` and the row falls back to a full-vocab ternary search
(3-6x slower per row; one such row stalls the whole batch). This replaces the
Gaussian pivot with an order statistic of a strided sample across the whole row.

## Contents

- `kernel/` — the modified `topk_topp_triton.py` and the diff as a patch
- `tests/` — the new exact-mask test file (also landed in the PR)
- `PR_DESCRIPTION.md` — the PR text
- `validation/` — A800 benchmark numbers, synthetic + real-model results,
  and the correctness harness
- `bench/` — repro scripts for every table posted on the PR (see its README)

## Key results (A800-SXM4-80GB, CUDA graph, median of 50)

Synthetic (the issue's own repro script):
- two modes B=64 k=200: 1127 -> 215 us (5.2x)
- zipf, sorted ids B=64 k=200: 1266 -> 216 us (5.9x)
- 63 healthy + 1 failing row, B=64 k=64: 647 -> 214 us (3.0x)
- healthy rows: +4-7%

Real decode logits (teacher-forced, ~250 rows/model):
- Qwen3-0.6B / 8B, MiMo-V2.6-Distill-Qwen-9B (V=248320)
- fallback rate at k=500: 72-81% (stock) -> 0% (fixed)
- k=200: 2.7-2.9x; k=500: 2.9-3.2x; k=1000: 2.8x
- output masks bit-identical to stock on every row and every k tested

End-to-end (Qwen3-8B, 64 concurrent requests, `VLLM_USE_FLASHINFER_SAMPLER=0`):
- sampler time per step: 607 -> 379 us at k=200, 991 -> 396 us at k=500
- throughput +1.9% / +3.4%

## Branch

[hjh123445/vllm@fix/topk-order-statistic-pivot](https://github.com/hjh123445/vllm/tree/fix/topk-order-statistic-pivot)
