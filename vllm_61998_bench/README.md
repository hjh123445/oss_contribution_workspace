# vLLM #61998 / PR #62596 benchmark harness

Repro scripts for the measurements posted on
[vllm-project/vllm#62596](https://github.com/vllm-project/vllm/pull/62596)
(issue [#61998](https://github.com/vllm-project/vllm/issues/61998)).

Every table in the PR comments maps to one command here. Timing uses CUDA graph
replay, median of 50, matching the issue's methodology.

## Setup

```bash
pip install vllm  # stock kernel = the wheel; for the fixed kernel, check out
                  # the PR branch and point scripts at its file, or pip install -e it
```

## 1. Synthetic matrix (the issue's repro script, unmodified)

Runs the issue's exact script against any kernel file:

```bash
python repro_issue.py $(python -c "import vllm.v1.sample.ops.topk_topp_triton as m; print(m.__file__)")  # stock
python repro_issue.py path/to/pr/topk_topp_triton.py                                                        # fixed
```

Covers: zipf-permuted / zipf-sorted / two-modes rows, B in {1,16,64}, k in
{50,64,200}, top-k and top-k+top-p, the 63+1 mixed batch, plus the exact-k
assertions.

## 2. Real decode logits

Capture ~250 teacher-forced decode rows from any HF model:

```bash
python capture_logits2.py <model_path> --out rows.pt --rows 2048
```

Models used in the PR: Qwen/Qwen3-0.6B, Qwen/Qwen3-8B,
XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (V=248320).

## 3. Kernel timing on captured rows

CUDA-graph replay timing of `apply_top_k_top_p_triton` on batches drawn from
the captured rows; also emulates the stock zeroth pass to count fallback rows:

```bash
python time_kernel.py rows.pt --k 200 --bs 64
```

For mask-equality A/B, run the kernel on all rows under both kernels and
`torch.equal` the outputs (see mask_check section of the PR).

## 4. End-to-end generation

Full vLLM offline generation with the Triton sampler path active and per-call
sampler timing. Needs `VLLM_USE_FLASHINFER_SAMPLER=0` (route to the Triton
kernel) and `VLLM_ENABLE_V1_MULTIPROCESSING=0` (so the timing hook can patch
in-process):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python e2e_generate.py <model_path> stock 200
```

Warmup runs a full batch first so all batch-size Triton specializations are
compiled before measurement.
