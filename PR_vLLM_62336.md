# PR 2: [Core] Skip gaps in sliding-window prefix-cache hit search

Fixes #62336

## Overview

`SlidingWindowManager.find_longest_cache_hit` scanned every block from right
to left — up to `max_num_blocks` cache lookups even on a cold cache, where a
128K-token request with a 4K window does ~8192 lookups although a hit only
needs `cdiv(window-1, block_size) = 256` contiguous blocks. The function
carried a TODO for exactly this optimization.

This PR implements the probe-lattice scan:

- Probe one block per `stride = sliding_window_contiguous_blocks` from the
  right. Every candidate run of `stride` consecutive cached blocks contains
  exactly one probe position, so a probe miss rules out all runs through it
  and the scan skips a whole stride.
- On a probe hit, the run is reconstructed by expanding **right** (bounded:
  the next probe to the right missed or is out of range) and **left**
  (unbounded: the run may span many strides).
- With no full-window match, the fallback reports the **leftmost run** —
  exactly the residual `num_contiguous_blocks` the old scan left behind —
  walked directly from block 0 (the probe lattice does not necessarily land
  on it).
- When blocks must stay aligned to a finer hybrid-cache grid
  (`block_size != alignment_tokens`), the original per-block scan is kept:
  its alignment-continue path does not break runs, which the probe lattice
  cannot express. `sliding_window=1` with EAGLE disabled yields stride 0 and
  is clamped to 1.

## Claims

- Low-hit-rate lookups drop from `O(max_num_blocks)` to
  `O(max_num_blocks / stride + run lengths)`.
- Hit results are bit-identical to the previous implementation.

## Validation

Function-level (128K tokens, block_size=16, window=4096, alignment=16, single
CPU thread):

| pattern | main | this PR |
|---|---:|---:|
| cold cache | 4.49 ms / 8192 lookups | 0.032 ms / 33 lookups |
| 1% random cached | 4.61 ms | 0.037 ms |
| tail hit / full cache | 0.23 ms | 0.202 ms |

(Note: measured on A800 host CPU; the issue's PR reports 0.021 ms on the
author's machine — same magnitude.)

Equivalence, two independent ways:
- Offline exhaustive comparison against the previous implementation using the
  real `BlockPool`/`SlidingWindowManager`: every cached-block pattern up to 10
  blocks over a grid of block sizes (2/4), windows (2/4/8), EAGLE on/off and
  alignment (same/finer): **49,128 cases, 0 differences**.
- Pure-algorithm exhaustive check of the scan semantics (kept length + match
  flag) over all patterns up to 13 blocks x K in 1..7: **114,681 cases,
  0 differences**.

Tests:
- Existing `tests/v1/core/test_single_type_kv_cache_manager.py`: 49 passed.
- New `tests/v1/core/test_sliding_window_probe_scan.py`: 3 passed —
  randomized-pattern equivalence, a lookup-count budget on cold cache, and a
  run straddling probe boundaries.

## Details

- Scheduler-side CPU optimization only: no change to hit results, model
  output, or GPU execution.
- Worst case remains `O(max_num_blocks)` (all blocks cached in one long run
  with no full window), but then the old scan was equally slow and the case
  is a cache hit, not the cold path this targets.
