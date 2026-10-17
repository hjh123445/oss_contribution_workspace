# 工作报告：vLLM / SGLang 开源贡献（推理框架工程师求职准备）

日期：2026-10-17
工作目录：`/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/oss_contribution_workspace/`（独立目录，未触碰机器上已有文件）
计算资源：本机 A800-80GB × 8（GPU 3/4 空闲使用），远程 172.29.226.143 未使用（本机资源足够）

## 一、成果摘要

**选定并修复了 SGLang issue #44145：EAGLE/MTP 投机解码在 untied-embedding 模型上导致目标模型 KV cache 被永久挤占**。

- 修复效果（A800 实测，紧内存配置）：KV cache 容量 **41128 → 48901 tokens（+18.9%）**；issue 中报告的 248k 大词表 Qwen 模型场景（24GB 卡）损失 >90%，本修复同样适用。
- 修复方式：EAGLE draft worker 构建完成后、scheduler 对空闲显存 profile 之前，提前把 target 的 embed_tokens/lm_head 绑定进 draft（释放 draft 的重复副本），与 `FrozenKVMTPWorkerV2` 已有先例（#29021）一致。
- 产出物：
  - 修复 patch（2 个源文件，+62/-3 行）：`fix_eagle_embed_share_full.patch`
  - 新增单元测试 3 项（全部通过）：`sglang/test/registered/unit/speculative/test_eagle_embed_share_before_kv_sizing.py`
  - PR 描述（英文，可直接提交）：`PR_DESCRIPTION.md`

## 二、issue 选型过程（检索范围与方法）

检索了 vLLM 与 SGLang 的 open issue 共约 400+ 条，筛选标准：open、无 assignee、评论 <4、有技术深度（kernel/调度/量化/投机解码核心路径）、能在 A800 上复现、**无已关联 PR**。

关键发现：**分析清楚的 issue 通常 1-2 天内就被 PR 抢占**。以下高价值候选均在选择时发现已有 PR，主动放弃：

| 候选 | 放弃原因 |
|---|---|
| vLLM #60685 EPLB 统计窗口错位 | 已有 PR #60686 |
| vLLM #49290 MRotaryEmbedding Neox 硬编码 | 评论区已有人在做 + 已有 2 个相关 PR |
| vLLM #53107 LinearBase.load_weights 错误回退 | 已有 PR #53118 |
| vLLM #57986/#58930/#58532/#58353 等 10+ 个 | 均已有 open PR |
| SGLang #44535 OOV stop_token_ids 崩溃 | 6 个 PR 抢占 |
| SGLang #44758 KDA kernel NaN | 需要 B300（sm103），A800 无法复现 |
| vLLM #61673 cascade attention KV 串读 | main 已移除 cascade attention，修复只能进 release 分支 |

最终选定 **SGLang #44145**（0 评论、无关联 PR、2026-10-13 新开、问题分析极清晰、A800 可复现、投机解码 + 内存管理核心路径）。

## 三、根因分析

初始化时序（`scheduler.py`）：
```
init_model_worker()
  ├─ init_tp_model_worker()          # target 权重加载
  ├─ maybe_init_draft_worker()       # draft 权重加载 → 重复的 embed_tokens/lm_head 驻留显存
  └─ init_memory_pools()
       ├─ init_target_memory_pool()  # ← 在此 profile 空闲显存并固化 KV pool 大小
       └─ draft_worker.alloc_memory_pool()  # ← init_lm_head() 在此才释放重复张量（太晚）
```
untied-embedding 模型（如 Llama-3.1-8B、Qwen 系列）的 draft 自带完整词表的 embed/lm_head 副本（248k vocab × 5k hidden ≈ 4.74 GiB），在 profile 时被计入，KV pool 被永久压小；事后 `post_capture_resize` 只能缩不能涨。

已验证的安全性约束：
- draft 权重在 `TpModelWorker.__init__` 同步加载完毕（startup overlap 只延迟 page-cache prefetch），提前绑定不会与权重加载竞争；
- EAGLE3 的 `hot_token_id` 从 draft 权重读出，提前绑定时已可用；
- `StandaloneWorkerV2`（不共享 embedding）组合独立 draft worker，不经过被修改的 `__init__`，不受影响。

## 四、修复实现

1. `eagle_worker_v2.py`：`EagleDraftWorker.__init__` 末尾调用新增的 `_share_embed_and_head_before_kv_sizing()`（幂等守卫 + 无 setter 时静默跳过 + 异常时回退 deferred 路径）；`alloc_memory_pool()` 的旧绑定改为仅在未提前绑定时执行。
2. `multi_layer_eagle_worker_v2.py`：同样处理（该 worker 也有 deferred `init_lm_head()`）。

## 五、实验验证（全部在本机 GPU 3/4，A800-80GB）

环境：conda `sglang_contrib`（python 3.12 + torch 2.14.1+cu130），sglang main 源码安装；模型 Llama-3.1-8B-Instruct（modelscope 下载）+ lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B（CI 同款 draft）。

| 实验 | stock | fixed | 结论 |
|---|---|---|---|
| EAGLE3, bf16, mem_frac=0.28 | KV 41128 | KV 48901 | +18.9%，恢复 ~0.95GB（对应该 draft 的重复 embed 1.05GB + lm_head 0.26GB） |
| EAGLE3, bf16, mem_frac=0.85 | KV 397743 | KV 405517 | +2.0%，无回归 |
| CI 同款（fp16+triton, mem_frac=0.28） | KV 41128 | KV 48901 | 与 CI 路径完全一致，可复现 |
| E2E 生成质量（5 条事实性 prompt, greedy） | — | 5/5 正确 | 投机解码验证路径无回归 |
| speculative 单测套件 | — | 8/8 通过（含新增 3 项） | 无回归 |

注：issue 原场景（Qwen 248k vocab + 24GB 卡）的重复张量达 4.74GB，占比远高于本实验的 1.3GB，恢复比例会更大（issue 报告从 15,363 → 预期 ~225k tokens 量级）。

## 六、后续步骤（提交 PR 时）

1. fork sgl-project/sglang，从 main 拉 branch（建议名 `fix/eagle-draft-embed-share-before-kv-sizing`）。
2. 应用 `fix_eagle_embed_share_full.patch`，用 `PR_DESCRIPTION.md` 内容作为 PR 描述（勾选 CLA、引用 "Fixes #44145"）。
3. PR 前建议先在 issue 下留言说明方案（社区惯例，避免与其他人撞车——本工作期间 issue 仍无人认领）。
4. 可选增强：跑 CI 的 `test_basic_sanity_eagle3.py` 全量（需要下载 meta-llama 官方模型权重，受限未做；已用同架构 NousResearch 镜像权重等价验证）。

## 七、目录清单

```
oss_contribution_workspace/
├── sglang/                      # 源码（git 仓库，含未提交的修复 + 新测试）
│   ├── python/sglang/srt/speculative/eagle_worker_v2.py        [已修改]
│   ├── python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py [已修改]
│   └── test/registered/unit/speculative/test_eagle_embed_share_before_kv_sizing.py [新增]
├── fix_eagle_embed_share_full.patch   # 最终 patch（含测试）
├── PR_DESCRIPTION.md                  # PR 描述（英文）
├── scripts/                           # 复现/验证脚本 + 全部运行日志与结果 JSON
├── models/                            # 模型（ms_cache 30G Llama + 2 个 EAGLE3 draft）
├── issues/                            # issue 调研存档（JSON）
└── 本报告
```

## 八、经验与简历表述建议

- 简历可写：深入理解 SGLang 投机解码（EAGLE-2/3、MTP）与调度器内存管理初始化时序；定位并修复了 KV cache 内存预算被 draft 重复 embedding 挤占的核心 bug（A800 实测恢复 18.9% 容量，紧内存场景更多）；贡献含根因分析、修复、单元测试与 E2E 验证。
- 调研副产品：对 vLLM/SGLang 当前热点 bug 分布有全景认知（kernel 数值正确性、spec decode、KV offload、PD disagg、MoE 量化是当前最活跃的领域），面试聊社区动态时可用。
