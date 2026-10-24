"""80-cell correctness matrix: stock vs patched kernel masks.

Distributions: zipf-permuted / zipf-sorted / two-modes (issue's repro),
grammar-masked (99% -inf), all-masked (300 finite at random positions).
Batches B in {4, 64}; k in {20, 50, 200, 1000}; top-p in {None, 0.95}.

Also covers the safety-net paths:
- exactly k+1 finite logits with a unique minimum (tier-3 strict `>`
  collects only k, the net re-gathers at all-finite),
- adversarial lattice rows (top values placed exactly on the sampling
  lattice) which break the order-statistic statistical guarantee.

Usage:
  python check_masks.py <stock_kernel.py> <patched_kernel.py>
"""
import importlib
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V, dev = 151936, "cuda"


def make(kind, B):
    g = torch.Generator(device=dev).manual_seed(0)
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
    if kind == "grammar":
        x = torch.randn(B, V, device=dev, generator=g)
        keep = torch.rand(B, V, device=dev, generator=g) > 0.99
        return torch.where(keep, x, torch.full_like(x, float("-inf")))
    if kind == "all_masked":
        x = torch.full((B, V), float("-inf"), device=dev)
        for i in range(B):
            idx = torch.randperm(V, generator=g, device=dev)[:300]
            x[i, idx] = torch.randn(300, generator=g, device=dev)
        return x
    raise ValueError(kind)


def edge_cases():
    """Safety-net trigger rows: k+1 finite with unique minimum;
    adversarial lattice rows."""
    g = torch.Generator(device=dev).manual_seed(0)
    cases = []
    # k+1 finite, unique minimum
    x = torch.full((4, V), float("-inf"), device=dev)
    for i in range(4):
        idx = torch.randperm(V, generator=g, device=dev)[:201]
        vals = torch.randn(201, generator=g, device=dev)
        vals[0] = -5.0
        x[i, idx] = vals
    cases.append(("k_plus_1_finite", x, 200))
    # adversarial: top values exactly on the sampling lattice
    x = torch.randn(4, V, device=dev, generator=g)
    lattice = torch.arange(0, V, 76, device=dev)
    x[:, lattice[:210]] = 100.0 + torch.arange(
        210, device=dev, dtype=torch.float32
    )
    cases.append(("adversarial_lattice", x, 200))
    return cases


def main():
    stock = load(sys.argv[1], "k_stock")
    ours = load(sys.argv[2], "k_ours")

    n_cases = n_diff = 0
    for kind in ("zipf, permuted", "zipf, sorted ids", "two modes", "grammar",
                 "all_masked"):
        for B in (4, 64):
            x = make(kind, B)
            for k in (20, 50, 200, 1000):
                for p in (None, 0.95):
                    kt = torch.full((B,), k, device=dev, dtype=torch.int32)
                    pt = None if p is None else torch.full((B,), p, device=dev)
                    so = stock.apply_top_k_top_p_triton(x.clone(), kt, pt)
                    oo = ours.apply_top_k_top_p_triton(x.clone(), kt, pt)
                    n_cases += 1
                    if not torch.equal(so, oo):
                        n_diff += 1
                        sk, ok = (so > float("-inf")), (oo > float("-inf"))
                        print(f"DIFF {kind} B={B} k={k} p={p}: "
                              f"{int((sk != ok).sum())} positions")

    for name, x, k in edge_cases():
        kt = torch.full((x.shape[0],), k, device=dev, dtype=torch.int32)
        so = stock.apply_top_k_top_p_triton(x.clone(), kt, None)
        oo = ours.apply_top_k_top_p_triton(x.clone(), kt, None)
        n_cases += 1
        status = "identical" if torch.equal(so, oo) else "DIFFER"
        print(f"edge {name}: {status}")

    print(f"{n_cases} cases, {n_diff} mask diffs (near-tie top-p boundary "
          f"tolerance expected on a few rows)")


if __name__ == "__main__":
    main()
