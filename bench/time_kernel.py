import argparse, statistics, torch
import vllm.v1.sample.ops.topk_topp_triton as tt

ap = argparse.ArgumentParser()
ap.add_argument("rows_file"); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--p", type=float, default=0.95)
args = ap.parse_args()

d = torch.load(args.rows_file); x = d["logits"].cuda()
g = torch.Generator(device="cuda").manual_seed(0)
idx = torch.randint(0, len(x), (args.bs,), device="cuda", generator=g)
xb = x[idx]
kt = torch.full((args.bs,), args.k, device="cuda", dtype=torch.int32)
pt = torch.full((args.bs,), args.p, device="cuda")
buf = xb.clone()

tt.apply_top_k_top_p_triton(buf, kt, pt)  # compile
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    tt.apply_top_k_top_p_triton(buf, kt, pt)

times = []
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
for _ in range(50):
    buf.copy_(xb)
    torch.cuda.synchronize()
    s.record(); graph.replay(); e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e) * 1e3)
times.sort()
is_fixed = "SAMPLE_STRIDE" in open(tt.__file__).read()
print(f"{d['model']} kernel={'fixed' if is_fixed else 'stock'} B={args.bs} k={args.k} p={args.p} "
      f"p50={statistics.median(times):.1f}us min={times[0]:.1f} max={times[-1]:.1f}")
