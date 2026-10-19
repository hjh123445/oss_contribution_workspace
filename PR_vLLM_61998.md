# PR 1: [Kernel][Sampler] Replace Gaussian pivot with order statistic in combined top-k/top-p Triton kernel

Fixes #61998

## Overview

The zeroth pass of `_topk_topp_kernel` picks the gather pivot with a Gaussian
quantile (`avg + std * sigma`) fitted to the **first 8192 logits only**. On
real logits (MiMo-V2.6 measurements in the issue) the head of the row carries
systematically higher values, and skewed rows break the Gaussian fit entirely;
the gather then returns fewer than `k` candidates and the row falls back to a
full-vocab ternary search, 3-6x slower per row, stalling every other row in
the batch.

This PR replaces the Gaussian estimate with an **order statistic of a strided
sample taken across the whole row**:

- Sample `S = BLOCK_SIZE / 4` logits uniformly across the row (one lane every
  `NUM_TILES * 4` vocab entries), so head bias cannot skew the estimate.
- Target `m = max(8, ceil(4 * k * S / V))` — the m-th largest sample value,
  found with a 12-step in-register binary search (no extra global memory
  traffic). The 4x margin over `k` keeps the gathered set small (a few
  hundred entries at V=152576) while reliably covering `k`.
- When the sample holds no finite logits (grammar bitmask hiding most of the
  vocab), the pivot degenerates to `+inf` (gather nothing) and the existing
  full-row fallback search runs with its finite, well-scaled bounds. A
  sentinel like `-1e30` would instead poison the buffer-search range and never
  converge (found via a fuzz case, fixed in `test_topk_pivot_all_masked_sample`).
- The Gaussian path, both sigma tables and the table upload, are removed.

## Claims

- Rows whose gather previously failed no longer take the full-vocab fallback:
  **3-6x lower latency per affected row**; batches containing one such row
  stop stalling.
- Gather-healthy rows are within noise of the previous numbers (the strided
  sample is a quarter of the old first-tile read).

## Validation

A800-SXM4-80GB, V=152576, CUDA graph, median of 50 runs, issue's repro script
(`zipf, sorted ids` and `two modes` are the distributions whose rows failed
the old gather; the `gather` column in the repro is the *old* diagnostic):

| rows | B | k | main top-k | this PR top-k |
|---|---|---|---:|---:|
| two modes | 64 | 50 | 923.6 us | 182.3 us |
| two modes | 64 | 200 | 1062.9 us | 188.4 us |
| zipf, sorted ids | 64 | 200 | 1203.2 us | 189.4 us |
| 63 zipf + 1 two-modes | 64 | 64 | 620.5 us | 189.4 us |
| zipf, permuted (healthy) | 64 | 200 | 232.4 us | 237.6 us |
| zipf, permuted (healthy) | 1 | 50 | 176.1 us | 194.6 us |

Correctness:
- Existing `tests/v1/sample/test_topk_topp_sampler.py`: **154 passed** (one
  grammar-mask case initially caught a real degenerate-pivot bug in this
  change; fixed by the `+inf` sentinel).
- New `tests/v1/sample/test_topk_pivot_order_statistic.py`: 8 passed —
  exact-mask comparison against a PyTorch reference over the failing
  distributions (zipf-sorted, two-modes, uniform, sparse grammar masks).
- Fuzz over 5 distributions x 3 batch sizes x 6 k values: 0 mismatches.

## Details

- The buffer search and the top-p stage are untouched; only the pivot choice
  in the zeroth pass changes, so the fallback path remains as a safety net.
- Limitation: like the old code, the sample is a lattice, so adversarial
  distributions can still miss `k` by a few tokens; the fallback search then
  runs exactly as before (correct, slower). The 4x margin makes this
  measure-by-measure rare on real logits.
