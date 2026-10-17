# PR: [Fix] Share target embed/lm_head with EAGLE/MTP drafts before KV pool sizing

Fixes #44145

## Problem

On untied-embedding checkpoints (`tie_word_embeddings: false`), the EAGLE/MTP
draft worker holds a **duplicate** `embed_tokens` (and `lm_head`) in GPU memory
from the moment its weights are loaded. The scheduler then profiles free GPU
memory and sizes the **target KV pool while those duplicates are still
resident**, because the sharing in `EagleDraftWorker.alloc_memory_pool()` /
`init_lm_head()` — which frees them by rebinding the draft onto the target's
tensors — runs only *after* `init_target_memory_pool()`:

```
Scheduler.init_model_worker()
  ├─ init_tp_model_worker()          # target weights
  ├─ maybe_init_draft_worker()       # draft weights — duplicate embed/lm_head resident
  └─ init_memory_pools()
       ├─ init_target_memory_pool()  # ← profiles free VRAM, KV pool size frozen here
       └─ draft_worker.alloc_memory_pool()   # ← duplicates freed here (too late)
```

The duplicate buffers are dropped moments later, but the KV pool was already
sized against the starved footprint, and post-capture resize can only shrink
(`compute_post_capture_kv_resize`). On a 32 GB card with a 248k-vocab Qwen
model the issue reports >90 % of KV capacity locked out (225,211 → 15,363
tokens).

## Fix

Bind the target's embed/lm_head into the draft **eagerly at the end of
`EagleDraftWorker.__init__`** (and `MultiLayerEagleDraftWorker.__init__`),
i.e. still before `init_memory_pools()`. Draft weights are fully loaded by
`TpModelWorker.__init__` at that point (the startup-overlap path only defers
page-cache prefetch, not the load), so the duplicate buffers can be freed
immediately and are no longer counted by the profile.

- `alloc_memory_pool()` keeps its deferred bind as fallback: the eager path is
  skipped when the draft model has no `set_embed_and_head`, or when the eager
  bind raises (logged, non-fatal) — the original deferred path still runs.
- This mirrors the pattern already used by `FrozenKVMTPWorkerV2.__init__`
  (see #29021).
- `StandaloneWorkerV2` deliberately does NOT share embeddings; it composes its
  own draft worker and never runs the modified `__init__`, so it is unaffected.

## Measurement (A800-80GB, Llama-3.1-8B-Instruct + lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B, EAGLE3, fp16, triton, mem-fraction-static=0.28)

| build | KV cache tokens |
|---|---|
| main | 41,128 |
| this PR | **48,901 (+18.9 %)** |

E2E quality unchanged: greedy outputs identical before/after; 5/5 factual
completions correct with depth-3/topk-4 speculation on the fixed build.

## Tests

- New: `test/registered/unit/speculative/test_eagle_embed_share_before_kv_sizing.py`
  - eager bind shares the target tensors (identity),
  - `alloc_memory_pool()` skips the re-bind when the eager bind ran,
  - no-setter draft model and failing eager bind fall back to the deferred path.
- Existing `test/registered/unit/speculative/` suite passes (8/8).
