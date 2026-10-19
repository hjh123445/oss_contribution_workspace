import importlib.util
import sys

import torch

# Usage: python repro.py [path/to/topk_topp_triton.py]  (default: installed vLLM)
path = sys.argv[1] if len(sys.argv) > 1 else (
    importlib.util.find_spec("vllm.v1.sample.ops.topk_topp_triton").origin)
spec = importlib.util.spec_from_file_location("tt", path)
tt = importlib.util.module_from_spec(spec)
sys.modules["tt"] = tt
spec.loader.exec_module(tt)

V, dev = 152576, "cuda"
g = torch.Generator(device=dev).manual_seed(0)
TABLE = torch.tensor(tt._PERCENTILE_TO_STD_TABLE, device=dev)


def make(kind, B):
    if kind.startswith("zipf"):  # Zipf-like: high logits at low token ids
        x = -torch.log1p(torch.arange(V, device=dev).float())
        x = x + 2.0 * torch.randn(B, V, device=dev, generator=g)
        if kind == "zipf, permuted":  # same values, random token order
            perm = torch.argsort(torch.rand(B, V, device=dev, generator=g), -1)
            x = x.gather(1, perm)
        return x
    # two modes: 20% of the vocab (random ids) suppressed by 10
    x = torch.randn(B, V, device=dev, generator=g)
    return x - 10.0 * (torch.rand(B, V, device=dev, generator=g) < 0.2)


def n_outliers(x, k):
    """Tokens above the kernel's gather pivot (zeroth pass, L143-171)."""
    t0 = x[:, :8192]  # stats from the first BLOCK_SIZE entries only
    avg, std = t0.mean(-1), t0.std(-1, correction=0)
    sigma = TABLE[min(int(k / V * 200), 199)] * 0.85
    return (x > (avg + std * sigma)[:, None]).sum(-1)


def bench(x, k, p, iters=50):
    buf = x.clone()
    kt = torch.full((x.shape[0],), k, device=dev, dtype=torch.int32)
    pt = None if p is None else torch.full((x.shape[0],), p, device=dev)
    for _ in range(3):
        buf.copy_(x)
        tt.apply_top_k_top_p_triton(buf, kt, pt)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tt.apply_top_k_top_p_triton(buf, kt, pt)
    ev = []
    for _ in range(iters):
        buf.copy_(x)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in ev)
    return ts[iters // 2], buf


print(f"{torch.cuda.get_device_name()}, V={V}, CUDA graph, median of 50 (us)")
print(f"{'rows':18s} {'B':>3s} {'k':>4s} {'n_out min':>9s} {'gather':>6s} "
      f"{'top-k':>7s} {'top-k+top-p':>11s}")
for kind in ("zipf, permuted", "zipf, sorted ids", "two modes"):
    for B in (1, 16, 64):
        x = make(kind, B)
        for k in (50, 64, 200):
            n_out = int(n_outliers(x, k).min())
            t_k, out = bench(x, k, None)
            assert ((out > float("-inf")).sum(-1) == k).all()
            t_kp, _ = bench(x, k, 0.95)
            print(f"{kind:18s} {B:3d} {k:4d} {n_out:9d} "
                  f"{'ok' if n_out > k else 'FAIL':>6s} {t_k:7.1f} {t_kp:11.1f}")
# One slow row stalls the batch: 63 "zipf, permuted" rows + 1 "two modes" row.
x = torch.cat([make("zipf, permuted", 63), make("two modes", 1)])
n_out = n_outliers(x, 64)
print(f"{'63 zipf-perm + 1':18s} {64:3d} {64:4d} {int(n_out.min()):9d} "
      f"{'FAIL' if (n_out <= 64).any() else 'ok':>6s} "
      f"{bench(x, 64, None)[0]:7.1f} {bench(x, 64, 0.95)[0]:11.1f}")
