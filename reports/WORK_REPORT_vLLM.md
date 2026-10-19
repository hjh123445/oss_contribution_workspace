# vLLM 开源贡献工作报告（第二轮）

日期：2026-10-20 凌晨
工作目录：`oss_contribution_workspace/`（延续第一轮，未触碰机器其他内容）
计算资源：本机 A800-80GB GPU 3（空闲卡）

## 一、成果摘要

本轮完成 **两个 vLLM 算法类贡献**，均已推送到 fork 并在 issue 下留言认领：

| Issue | 类型 | 效果 | 分支 |
|---|---|---|---|
| [#61998](https://github.com/vllm-project/vllm/issues/61998) | Triton 采样 kernel 算法 | 失败行延迟 **3-6x 降低**；单慢行拖垮整 batch 的场景 3.3x | `hjh123445/vllm@fix/topk-order-statistic-pivot` |
| [#62336](https://github.com/vllm-project/vllm/issues/62336) | 调度器 CPU 算法 | 冷缓存 prefix 查找 **~30x 加速**（8192 次查找→33 次） | `hjh123445/vllm@perf/sliding-window-probe-scan` |

两轮合计：SGLang 1 个（被抢注未提交）+ vLLM 2 个（分支已就绪）。

## 二、贡献 1：#61998 top-k 采样 kernel 的 pivot 算法重设计

**问题**：`_topk_topp_kernel` 零阶段用"前 8192 个 logits 的高斯分位数"选 gather 阈值。真实 logits 头部系统性偏高 + 偏态分布打破高斯假设 → gather 凑不齐 k 个候选 → 整行回退到全词表三分搜索（慢 3-6 倍），且 batch 中一个慢行拖垮全部行（单 program per row 架构）。

**算法方案**（issue 作者提出方向，我实现并调优）：
- 跨全行均匀采样 2048 点（`offs = arange(S) * NUM_TILES * 4`）
- 用 12 步寄存器内二分求样本第 m 大值（m = max(8, ceil(4kS/V))，4 倍余量）
- 零额外全局内存流量
- 边界处理：样本全 -inf（grammar 掩码稀疏场景）时 pivot 设 +inf 走原有全行 fallback——试验过 -1e30 哨兵值会污染 buffer 搜索区间永不收敛（这是被官方测试 `test_few_valid_tokens_with_neginf` 抓出的真 bug，最终修复并加回归测试）

**调优过程**：8192 采样点导致正常行 +10% 回归 → 降到 2048 点后正常行 ±3% 以内。

**验证**：
- 官方 `test_topk_topp_sampler.py` 154 项全过
- 新增 `test_topk_pivot_order_statistic.py` 8 项（vs PyTorch 精确参考的掩码比对）
- 5 分布 × 3 batch × 6 k 值 fuzz 0 失配

## 三、贡献 2：#62336 滑动窗口 prefix-cache 查找的 probe-lattice 算法

**问题**：`SlidingWindowManager.find_longest_cache_hit` 从右向左逐块扫描，冷缓存下 128K token 请求做 ~8192 次哈希查找，而命中只需 256 个连续块（代码里挂着 TODO）。

**算法方案**：
- 从右向左每 `stride=所需连续块数` 探测一块：任何长度 ≥ stride 的连续 run 必含一个探测点 → miss 可跳过整个 stride
- probe hit 后：向右扩（有界，右侧 probe 已 miss）+ 向左扩（无界，run 可跨多个 stride）
- 无完整命中时 fallback = **最左 run**（与原算法残留 `num_contiguous_blocks` 语义一致），直接从块 0 走——probe 格点不保证落在最左 run 上
- `block_size != alignment_tokens`（混合模型细粒度对齐）时保留原逐块扫描（其对齐 continue 不打断 run，probe 格点无法表达）
- `sliding_window=1` + 无 EAGLE 时 stride 钳为 1（防死循环）

**验证（关键卖点）**：
- 纯算法穷举：n≤13 全部 2^13 模式 × K 1..7 = 114,681 组合 vs 原算法 **0 差异**
- 真实类穷举：真实 BlockPool/Manager，block_size 2/4 × window 2/4/8 × EAGLE 开关 × 对齐 3 种 = 49,128 组合 **0 差异**
- 冷缓存 0.958ms→0.032ms（8192→33 次查找）
- 官方 `test_single_type_kv_cache_manager.py` 49 项全过；新增 3 项测试（随机模式等价、查找次数预算、跨 probe 边界 run）

## 四、提交流程（已完成的部分 + 剩余步骤）

已完成：
1. ✅ fork `hjh123445/vllm`，推两个分支（各自 1 commit，只含对应文件）
2. ✅ 在两个 issue 下留言说明方案与数据（防撞车、吸引 maintainer 注意）

剩余（需要你在 GitHub 网页操作，或授权我用 API 创建 PR）：
1. 开 PR：https://github.com/vllm-project/vllm/compare/main...hjh123445:vllm:fix/topk-order-statistic-pivot 用 `PR_vLLM_61998.md` 内容；同理 `perf/sliding-window-probe-scan` 用 `PR_vLLM_62336.md`
2. 按 vLLM 规范：标题 `[Kernel]`/`[Core]` 前缀已满足；**commit 需要 DCO sign-off**（`Signed-off-by: Your Name <email>`——API 提交时未加，开 PR 前需要在分支上补一个带 sign-off 的 commit，或开 PR 后 maintainer 会提醒）
3. AI 辅助披露：PR checklist 要求说明 AI 参与（加 `Co-authored-by` trailer）

## 五、与第一轮（SGLang #44145）的对照

第一轮工作方向正确但被 BESzbing 的 PR #45151 抢先（晚了 2 天）。教训已应用：
- 本轮两个 issue 均为**发布当天/次日就留言+推分支**
- #61998 是 10-16 发布的，10-20 已 4 天，风险中等（留言中给出差异化数据）
- #62336 是 10-18 发布的，2 天内行动
- 后续盯 maintainer 回复，一旦有人竞争立即开 PR 占位

## 六、文件清单（已上传 hjh123445/oss_contribution_workspace）

- `PR_vLLM_61998.md` / `PR_vLLM_62336.md` — PR 描述
- `vllm_61998_topk_pivot/` — patch + 修改的 kernel + 新测试
- `vllm_62336_sliding_window/` — patch + 修改的 manager + 新测试
- `vllm_validation/` — 复现脚本、正确性 harness、结果汇总
