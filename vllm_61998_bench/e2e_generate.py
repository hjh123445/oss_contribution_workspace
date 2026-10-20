"""True end-to-end: vLLM offline generation with the Triton sampler path active.

VLLM_USE_FLASHINFER_SAMPLER=0 routes sampling through apply_top_k_top_p_triton.
Batch of 64 concurrent requests, k=200 (where real-logit fallbacks occur).
Reports generate throughput and per-step sampler time.

Usage: VLLM_USE_FLASHINFER_SAMPLER=0 python e2e_generate.py <model> <stock|fixed>
"""

import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch


def main():
    model_path, tag = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    import vllm.v1.sample.ops.topk_topp_triton as tt
    orig = tt.apply_top_k_top_p_triton
    calls = {"n": 0, "us": 0.0}

    def timed(logits, kk, pp, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig(logits, kk, pp, *a, **kw)
        torch.cuda.synchronize()
        calls["n"] += 1
        calls["us"] += (time.perf_counter() - t0) * 1e6
        return r

    import vllm.v1.sample.ops.topk_topp_sampler as tts
    tts.apply_top_k_top_p_triton = timed
    tt.apply_top_k_top_p_triton = timed

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=False)

    prompts = [
        "Write a Python function to merge two sorted lists.",
        "Explain the difference between TCP and UDP.",
        "Summarize the plot of Hamlet in three sentences.",
        "What are the tradeoffs of B-trees versus hash tables?",
        "Design a rate limiter API. What are the core methods?",
        "Compare Rust and Go for a backend service.",
        "Why is the sky blue at noon and red at sunset?",
        "Refactor this loop: for i in xs: if i > 0: ys.append(i*i)",
    ] * 8  # 64 concurrent requests

    sp = SamplingParams(temperature=0.6, top_k=k, top_p=0.95, max_tokens=192)
    llm.generate(prompts, sp)  # warmup (full batch to cover batch-size specializations)
    calls["n"] = calls["us"] = 0

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    # per decode step: each step samples once for the batch
    print(f"[{tag}] k={k} e2e={dt:.2f}s tokens={ntok} "
          f"tok/s={ntok/dt:.1f} sampler_calls={calls['n']} "
          f"sampler_us_total={calls['us']:.0f} "
          f"sampler_us_per_call={calls['us']/max(calls['n'],1):.1f}")


if __name__ == "__main__":
    main()
