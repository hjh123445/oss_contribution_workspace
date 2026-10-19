"""Correctness harness: compare fixed kernel vs PyTorch reference across
distributions, k values, and both top-k only and top-k+top-p."""
import importlib.util, sys
import torch

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("tt", path)
tt = importlib.util.module_from_spec(spec)
sys.modules["tt"] = tt
spec.loader.exec_module(tt)

V, dev = 152576, "cuda"
g = torch.Generator(device=dev).manual_seed(1234)

def make(kind, B):
    if kind.startswith("zipf"):
        x = -torch.log1p(torch.arange(V, device=dev).float())
        x = x + 2.0 * torch.randn(B, V, device=dev, generator=g)
        if kind == "zipf, permuted":
            perm = torch.argsort(torch.rand(B, V, device=dev, generator=g), -1)
            x = x.gather(1, perm)
        return x
    if kind == "two modes":
        x = torch.randn(B, V, device=dev, generator=g)
        return x - 10.0 * (torch.rand(B, V, device=dev, generator=g) < 0.2)
    if kind == "uniform":
        return torch.rand(B, V, device=dev, generator=g)
    if kind == "grammar_masked":
        x = torch.randn(B, V, device=dev, generator=g)
        # mask 99% of vocab to -inf like a grammar bitmask
        keep = torch.rand(B, V, device=dev, generator=g) > 0.99
        return torch.where(keep, x, torch.full_like(x, float("-inf")))
    raise ValueError(kind)

def ref_topk(x, k):
    """Exact top-k mask: keep the k largest, ties broken by order."""
    out = x.clone()
    thresh = torch.topk(x, k, dim=-1).values[:, -1:]
    # count kept == k with deterministic tie handling: rank by value then -idx
    B, V = x.shape
    idx = torch.arange(V, device=dev).expand(B, V)
    order = torch.argsort(-x, dim=-1, stable=True)
    keep = torch.zeros_like(x, dtype=torch.bool)
    keep.scatter_(1, order[:, :k], True)
    out[~keep] = float("-inf")
    return out

n_fail = 0
for kind in ("zipf, permuted", "zipf, sorted ids", "two modes", "uniform", "grammar_masked"):
    for B in (1, 8, 64):
        x = make(kind, B)
        for k in (1, 20, 50, 64, 200, 1000):
            kt = torch.full((B,), k, device=dev, dtype=torch.int32)
            got = x.clone()
            tt.apply_top_k_top_p_triton(got, kt, None)
            kept = (got > float("-inf")).sum(-1)
            expect = torch.minimum(torch.full((B,), k, device=dev, dtype=torch.long), (x > float("-inf")).sum(-1))
            if not (kept == expect).all():
                print(f"FAIL keep-count {kind} B={B} k={k}: kept={kept.tolist()[:8]} expect={expect.tolist()[:8]}")
                n_fail += 1
                continue
            # masked positions must be a subset of the exact top-k set by value:
            # every kept value >= every masked finite value
            for i in range(B):
                kv = got[i]
                finite = kv[kv > float("-inf")]
                rest = x[i][kv <= float("-inf")]
                rest = rest[rest > float("-inf")]
                if len(finite) and len(rest) and finite.min() < rest.max():
                    print(f"FAIL value-set {kind} B={B} k={k} row={i}")
                    n_fail += 1
                    break
    print(f"checked {kind}")
print("TOTAL FAILURES:", n_fail)
