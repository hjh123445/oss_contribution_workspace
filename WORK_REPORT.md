# vLLM #61998 工作报告 — top-k 采样内核哨兵修复（PR #62596）

**日期**：2026-10-24
**贡献者**：He JiaHua (hjh123445)
**PR**：[vllm-project/vllm#62596](https://github.com/vllm-project/vllm/pull/62596)
**Issue**：[#61998](https://github.com/vllm-project/vllm/issues/61998)
**最终 commit**：`ed2b543b`（基于 main `7a64fb8fc9` 的 `topk_topp_triton.py`）

---

## 一、问题与方案概述

**问题**：`_topk_topp_kernel` 的 top-k 零遍用"前 8192 个 logit 的高斯分位数"选取 gather 哨兵。真实模型 logit 头部系统性偏高（布局偏差）且分布偏斜（高斯假设失效），导致 gather 凑不齐 k 个候选，行退回到全词表三元搜索（慢 3-6 倍），而 batch 等待最慢行，整批延迟被拖垮（issue 实测 MiMo logit：k=200 时 43% 行退回，k=500 时 86%）。

**最终方案**（三级架构，融合两个独立贡献）：

```
第一级：高斯分位数 + 四段合并采样（采纳自 #62760 的布局修复）
        采样 = 行首/1/4/1/2/3/4 处各取 2048 个连续元素
  ├─ gather 覆盖 k（真实行 >99%）→ buffer 搜索（原快路径，未改动）
  └─ gather 不足
       第二级：顺序统计量重采（#62596 原创）——全行跨步采样 2048 点，
               取第 m = max(8, ⌈4kS/V⌉) 大的值作哨兵（无分布假设，
               4 倍余量；偏斜分布天然免疫）
       第三级：行最小值兜底（稀疏行直接走；稠密行由安全网降级触发）
               收集全部有限值，上游 num_finite_total <= k 短路保证必然 > k
       → 重采后 if num_outliers <= k → 安全网再采（确定性闭合）
       → 所有搜索迭代在 L2 buffer 内（采纳 #62760 的退回框架）
```

**与竞争方案 #62760 的分工**：#62760 的布局修复（四段采样）和 L2 退回框架被采纳；其 sigma 0.5 大余量方案被证明在偏斜分布上仍退回且均值重采会爆炸（two-modes 上比 main 更慢），由顺序统计量替代。

## 二、正确性验证

| 验证项 | 方法 | 结果 |
|---|---|---|
| 官方测试套件 | `tests/v1/sample/test_topk_topp_sampler.py`（vLLM 全量采样器测试，覆盖纯 topk / 纯 topp / 混合 batch / fp64 gumbel / 分布匹配统计检验） | **132 passed, 1 skipped** |
| 分布矩阵 | 5 种分布（zipf-permuted / zipf-sorted / two-modes / grammar-mask / all-masked）× 2 batch × 4 k 值 × topp 有无 = 80 组合，掩码与 main 逐位比对 | 4 个差异，全部是 top-p 近平局容差类（prob ~1e-4 的边界 token，stock 与两个 PR 之间同样存在，属已文档化行为） |
| k+1 有限值边界 | 行内恰好 k+1 个有限值且最小值唯一 → 第三层 `>` 语义只收 k 个 → 安全网触发 | 正确保留 k，与 main 一致 |
| 对抗性格点构造 | 稠密行 + 高值精确放在采样格点上（统计保证失效的构造）| 安全网触发，正确保留 k，与 main 一致 |
| 真实 logit 掩码 | Qwen3-0.6B/8B、Qwen2.5-1.5B、MiMo-9B 捕获行全量对比 | 与 main 一致（1 个近平局行除外，且该行 v6 匹配精确参考、main 多保留平局 token） |
| 第三方独立验证 | MarinaMackay 在 H200 上对比 #62596 与 #62760（见 issue #61998）| 4855 行全部匹配 torch.topk；top-p 精度与 main 一致 |

## 三、性能验证（A800-SXM4-80GB，CUDA graph，中位数/50）

### 方法论（所有对比同一协议）

- 内核加载：单文件替换——stock 用 wheel 0.31.0 原文件，#62760 用其 PR head 文件，v6 用 PR 分支文件，其余代码完全相同；
- 计时：CUDA graph 捕获重放，`torch.cuda.Event`，50 次取中位数；每次重放前 `buf.copy_(x)` 恢复输入（避免就地掩码污染）；
- 合成行：issue 原版 repro 脚本（同种子、同三种行类型，V=152576）；
- 真实行：transformers 加载模型，teacher forcing 长上下文，每个位置取一行解码 logit（~250 行/模型）；
- E2E：vLLM 离线生成，`VLLM_USE_FLASHINFER_SAMPLER=0` 强制走 Triton 采样路径，`VLLM_ENABLE_V1_MULTIPROCESSING=0` 使计时 hook 落在进程内，先全 batch warmup 覆盖所有 batch 尺寸特化。

### 合成分布（B=64，top-k + top-p=0.95，us）

| 行类型 | k | main | #62760 | v6 |
|---|---:|---:|---:|---:|
| zipf-permuted（健康）| 50 | 272 | 434 | 295 |
| zipf-permuted（健康）| 200 | 345 | 524 | 372 |
| zipf-sorted（布局偏差）| 200 | 1280 | **242** | 313 |
| two-modes（偏斜）| 200 | 1349 | 1402 | **371** |
| two-modes（偏斜）| 1000 | 1500 | 1491 | **488** |

### 真实解码 logit（B=64，top-k + top-p=0.95，us）

| 模型 | V | k | main | #62760 | v6 |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 151936 | 200 | 1767 | 961 | **960** |
| Qwen3-0.6B | 151936 | 1000 | 3267 | 1061 | **869** |
| Qwen2.5-1.5B | 151936 | 200 | 1824 | 941 | **810** |
| Qwen3-8B | 151936 | 1000 | 2501 | 910 | **922** |
| MiMo-V2.6-9B | 248320 | 200 | 2214 | 1471 | **1316** |
| MiMo-V2.6-9B | 248320 | 1000 | 4355 | 1490 | **930** |

### 端到端（vLLM 生成，64 并发 × 192 token，采样器 us/步）

| 模型 | k | main | #62760 | v6 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 200 | 720 | 409 | **387** |
| Qwen3-8B | 200 | 607 | 399 | **391** |
| Qwen3-8B | 20（默认）| ~384 | 411 | 384 |

## 四、全部测试脚本（简历用）

### 1. `repro_issue.py` — issue 原版复现脚本（逐字保留）

用途：三种合成行类型 × B×k 矩阵 + 63+1 混合批，验证退化行为与 issue 在 B300 上同型。断言每行恰好保留 k 个 token。

```python
import importlib.util, sys, torch
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("tt", path)
tt = importlib.util.module_from_spec(spec); sys.modules["tt"] = tt
spec.loader.exec_module(tt)

V, dev = 152576, "cuda"
g = torch.Generator(device=dev).manual_seed(0)
TABLE = torch.tensor(tt._PERCENTILE_TO_STD_TABLE, device=dev)

def make(kind, B):
    if kind.startswith("zipf"):
        x = -torch.log1p(torch.arange(V, device=dev).float())
        x = x + 2.0 * torch.randn(B, V, device=dev, generator=g)
        if kind == "zipf, permuted":
            perm = torch.argsort(torch.rand(B, V, device=dev, generator=g), -1)
            x = x.gather(1, perm)
        return x
    x = torch.randn(B, V, device=dev, generator=g)
    return x - 10.0 * (torch.rand(B, V, device=dev, generator=g) < 0.2)

def n_outliers(x, k):
    t0 = x[:, :8192]
    avg, std = t0.mean(-1), t0.std(-1, correction=0)
    sigma = TABLE[min(int(k / V * 200), 199)] * 0.85
    return (x > (avg + std * sigma)[:, None]).sum(-1)

def bench(x, k, p, iters=50):
    buf = x.clone()
    kt = torch.full((x.shape[0],), k, device=dev, dtype=torch.int32)
    pt = None if p is None else torch.full((x.shape[0],), p, device=dev)
    for _ in range(3):
        buf.copy_(x); tt.apply_top_k_top_p_triton(buf, kt, pt)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tt.apply_top_k_top_p_triton(buf, kt, pt)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for s, e in ev:
        buf.copy_(x); s.record(); graph.replay(); e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in ev)
    return ts[iters // 2], buf

for kind in ("zipf, permuted", "zipf, sorted ids", "two modes"):
    for B in (1, 16, 64):
        x = make(kind, B)
        for k in (50, 64, 200):
            n_out = int(n_outliers(x, k).min())
            t_k, out = bench(x, k, None)
            assert ((out > float("-inf")).sum(-1) == k).all()
            t_kp, _ = bench(x, k, 0.95)
            print(f"{kind} B={B} k={k} n_out={n_out} topk={t_k:.1f} k+p={t_kp:.1f}")
```

### 2. `capture_logits.py` — 真实模型解码 logit 捕获

用途：从任意 HF 模型抓 teacher-forced 解码分布，供离线内核 A/B（消除模型加载变量，同一批行可反复测试三个内核）。

```python
import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path"); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2048)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    prompts = [  # 长代码/说明文提示词 × 5（见 PR 仓库 bench/capture_logits2.py）... ]
    rows = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to("cuda")
            out = model(**ids)
            L = out.logits.shape[1]
            for pos in range(0, L, 2):          # 每隔一个位置采一个解码步
                rows.append(out.logits[0, pos].float().clone())
                if len(rows) >= args.rows: break
            if len(rows) >= args.rows: break
    x = torch.stack(rows)
    torch.save({"logits": x, "model": args.model_path}, args.out)
```

### 3. `time_kernel.py` — CUDA graph 内核计时

用途：对捕获行做三方（stock/his/ours）同数据计时。

```python
import argparse, statistics, torch

ap = argparse.ArgumentParser()
ap.add_argument("rows_file"); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--bs", type=int, default=64)
args = ap.parse_args()

d = torch.load(args.rows_file); x = d["logits"].cuda()
g = torch.Generator(device="cuda").manual_seed(0)
idx = torch.randint(0, len(x), (args.bs,), device="cuda", generator=g)
xb = x[idx]
kt = torch.full((args.bs,), args.k, device="cuda", dtype=torch.int32)
pt = torch.full((args.bs,), 0.95, device="cuda")

tt.apply_top_k_top_p_triton(xb.clone(), kt, pt)   # 编译
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    tt.apply_top_k_top_p_triton(xb, kt, pt)
times = []
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
for _ in range(50):
    torch.cuda.synchronize(); s.record(); graph.replay(); e.record()
    torch.cuda.synchronize(); times.append(s.elapsed_time(e) * 1e3)
times.sort()
print(f"kernel={kernel_label} B={args.bs} k={args.k} p50={statistics.median(times):.1f}us")
```

### 4. `e2e_generate.py` — 端到端 vLLM 生成

用途：完整推理管线验证（含 CUDA graph 捕获、批动态、split 流水线交互），测采样器每步耗时与吞吐。

```python
import os, sys, time, torch
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")     # 走 Triton 采样路径
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # hook 落在进程内

import vllm.v1.sample.ops.topk_topp_sampler as tts
orig = tts.apply_top_k_top_p_triton
calls = {"n": 0, "us": 0.0}
def timed(logits, k, p, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = orig(logits, k, p, *a, **kw)
    torch.cuda.synchronize()
    calls["n"] += 1; calls["us"] += (time.perf_counter() - t0) * 1e6
    return r
tts.apply_top_k_top_p_triton = timed

from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.85, max_model_len=4096)
sp = SamplingParams(temperature=0.6, top_k=int(sys.argv[3]), top_p=0.95, max_tokens=192)
prompts = [...8 个提示词...] * 8        # 64 并发
llm.generate(prompts, sp)               # 全 batch warmup（覆盖所有批尺寸特化）
calls["n"] = calls["us"] = 0
t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
print(f"[{sys.argv[2]}] k={sys.argv[3]} e2e={dt:.2f}s tok/s={ntok/dt:.0f} "
      f"sampler_calls={calls['n']} us/call={calls['us']/calls['n']:.1f}")
```

### 5. `correctness_matrix.py` — 80 组合掩码等价验证

用途：5 分布 × 2 batch × 4 k × topp 有无，两个内核输出掩码逐位比对（含 grammar mask 全 -inf、k+1 有限值等边界）。

```python
# 核心结构：对每个 (kind, B, k, p) 组合
so = stock.apply_top_k_top_p_triton(x.clone(), kt, pt)
vo = ours.apply_top_k_top_p_triton(x.clone(), kt, pt)
assert torch.equal(so > float("-inf"), vo > float("-inf"))
# 边界：all_masked(300 有限值随机位置)、grammar(99% -inf)、
#       k+1 有限且最小值唯一（触发安全网）、对抗性格点(高值放采样点上)
```

### 6. 边界/对抗测试要点（脚本化）

```python
# 安全网触发路径验证：
# 1. 行内恰好 k+1 个有限值，最小值唯一 → 第三层严格 > 只收 k → 安全网补齐
# 2. 对抗构19    raise ValueError

```

---

## 五、复现仓库

所有脚本 + 最终内核 + 数据：[hjh123445/vllm-topk-order-statistic-pivot](https://github.com/hjh123445/vllm-topk-order-statistic-pivot)
- `kernel/kernel_v6.py` — 最终内核
- `bench/` — 脚本 1-4
- `validation/` — 正确性 harness 与结果

## 六、第三方独立验证（H200）

MarinaMackay 用共享协议在 H200 上复测（测试的是 v5 commit 3d3a8ec2；v6 与 v5 行为一致——安全网在这些负载上从不触发）：#62596 在所有失败场景消除退回（-55% ~ -80%），健康行与 main 差距 ~2%（B≥16）；#62760 在健康行慢 15-50%（sigma 0.5 的代价，独立数据确认了我们的 A800 测量）。

## 七、时间线

| 日期 | 事件 |
|---|---|
| 10-19 | issue 下报告独立实现（顺序统计量方案）|
| 10-20 | 开 PR #62596；合成 + 三模型真实 logit + E2E 数据 |
| 10-21 | 与 #62760 对比分析；采纳其四段采样与 L2 退回框架重构；共享协议数据 |
| 10-23 | DimensionSTP（#60104 作者）指出 standalone top-p 区域重叠 → 撤出该区域，双方 diff 零重叠 |
| 10-24 | 修复基底漂移（误删 main 的装饰器/61213/58231）；加确定性安全网（num_outliers <= k → 全有限值重采）；v6 = ed2b543b 推送 |
| 10-24 | MarinaMackay H200 独立复测确认 #62596 优势 |
