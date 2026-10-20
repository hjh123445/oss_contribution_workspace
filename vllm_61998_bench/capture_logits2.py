"""Capture many decode-step logits from a model with teacher forcing on long contexts.

For each prompt, run the full forward and collect the decode distribution at
every position (each position's next-token distribution is a decode step under
teacher forcing). This gives hundreds of real rows per prompt.

Usage: python capture_logits2.py <model_path> --out rows.pt --rows 2048
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    prompts = [
        # long-context code prompts (the issue's capture used long code contexts)
        """def merge_sorted_lists(a, b):
    \"\"\"Merge two sorted lists into one sorted list.\"\"\"
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

# Test cases
assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted_lists([], [1, 2]) == [1, 2]
""",
        """The Transformer architecture, introduced in 2017, replaced recurrence with
multi-head self-attention. Each layer computes:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The computational complexity is O(n^2 * d) in sequence length n, which motivated
linear attention variants like Linformer and Performer. KV caching during
autoregressive decoding stores key and value tensors for all previous positions,
trading memory for compute. Paged attention (vLLM) partitions the KV cache into
fixed-size blocks to reduce fragmentation and enable prefix sharing across
requests sharing a common system prompt.""",
        """import asyncio
import json
from dataclasses import dataclass

@dataclass
class Task:
    id: int
    payload: dict
    retries: int = 0

class Worker:
    def __init__(self, concurrency: int = 8):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.results = []

    async def process(self, task: Task) -> dict:
        async with self.semaphore:
            await asyncio.sleep(0.1)
            return {"id": task.id, "ok": True}
""",
        "Write a detailed explanation of how the Linux kernel schedules processes, "
        "covering CFS, nice values, and how cgroups interact with the scheduler. "
        "Then compare with the real-time scheduling classes.",
        "Explain the mathematics of attention is all you need, including the "
        "derivation of scaled dot-product attention, multi-head projection, "
        "positional encoding sinusoids, and why layer normalization is applied "
        "before the residual addition.",
    ]

    rows = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to("cuda")
            out = model(**ids)
            # sample positions spread across the sequence as decode steps
            L = out.logits.shape[1]
            positions = list(range(0, L, 2))  # every other position
            for pos in positions:
                rows.append(out.logits[0, pos].float().clone())
                if len(rows) >= args.rows:
                    break
            if len(rows) >= args.rows:
                break
            del out

    x = torch.stack(rows)
    torch.save({"logits": x, "model": args.model_path.rstrip("/").split("/")[-1]}, args.out)
    print(f"saved {tuple(x.shape)} to {args.out}")


if __name__ == "__main__":
    main()
