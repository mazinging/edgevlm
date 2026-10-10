# M1 基线剖析 — 实测发现

> 全部数据来自 `python -m edgevlm.bench.baseline`，原始结果见 `results/baseline.json`。
> 平台：GTX 1650 Ti (SM 7.5, 4.3GB, 16 SMs) | SmolVLM-256M-Instruct | 512×512 单图 | 24 输出 token

## 1. 主结果表

| dtype | attn | vision (ms) | prefill (ms) | decode (ms/tok) | tok/s | peak (GB) |
|---|---|---|---|---|---|---|
| **fp32** | eager | 140.42 | 253.20 | **37.40** | **18.59** | 1.22 |
| fp16 | eager | 527.09 | 638.65 | 29.73 | 12.77 | 0.67 |
| bf16 | eager | 155.63 | 213.12 | 50.76 | 15.12 | 0.69 |
| * | sdpa | — | — | — | — | *不支持* |

## 2. 发现 A：FP32 最快，FP16 最慢 —— 与常识相反

**预期**：半精度应显著快于 fp32。
**实测**：`fp32 (18.59 tok/s) > bf16 (15.12) > fp16 (12.77)`。

fp16 在**每个阶段**都最慢，尤其是视觉塔：527ms vs fp32 的 140ms，**慢 3.75×**。

### 根因分析（微基准确认）
在 2048×2048 matmul 上逐 dtype 测量：

| dtype | TFLOPS | 相对 fp32 |
|---|---|---|
| fp32 | 2.79 | 1.00× |
| bf16 | 1.66 | 0.60× |
| **fp16** | **0.42** | **0.15×** |

关键证据：**fp16 吞吐在所有 K 维度上恒为 0.42 TFLOPS**（K=64/256/1024/4096 完全不变），
且所有调用形式（`mm`/`addmm`/`bmm`/`@`/`out=`）耗时**完全一致（40.5ms）**。

这种"固定速率、与形状无关"的特征，是 cuBLAS **回退到非 Tensor Core 参考 kernel** 的典型signature。
真实 TC 计算应随 K 增大而提升利用率，且应比 fp32 快数倍。

### 环境层面的可能原因
- 驱动 572.83 提供 CUDA **12.8** runtime，而 torch 2.4.1 编译针对 **cu121**
- `CUDA_HOME` 指向 **v12.0** toolkit
- 三者版本不一致 → cuBLAS 对 SM7.5 的 fp16 kernel 选择/加载异常

### 工程含义（重要）
> **在本机不能用「半精度更快」这一常识做优化决策。**
> 任何量化/精度方案都必须**实测**，不能从纸面规格推导。
> 这也说明：如果只做理论分析而不在本机验证，会得出完全错误的优化方向。

## 3. 发现 B：decode 占 69.5% —— 优化重心应在 LLM 侧

最快配置的阶段占比：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| vision（视觉编码，1 次） | 140.42 ms | 10.9% |
| prefill（含 connector） | 253.20 ms | 19.6% |
| **decode（24 token 自回归）** | **897.58 ms** | **69.5%** |
| 合计 | 1291.20 ms | 100% |

**结论**：
- decode 是绝对瓶颈（37.4 ms/token，即 26.7 tok/s 的极限单流解码速率）
- **视觉塔只有 10.9%** → 视觉 token 剪枝的端到端收益上限约 11%（单图场景）
- KV Cache / attention / 量化等 LLM 侧优化，收益上限约 70%

**这修正了一个常见误判**：VLM 优化常被默认"视觉塔是瓶颈"，
但在单图短输出场景下，**自回归解码才是主导**。

## 4. 发现 C：视觉塔不支持 SDPA

`Idefics3VisionTransformer does not support an attention implementation`
→ 视觉塔**无法通过 SDPA 后端加速**，M4 的 attention 优化需走手工路径。

## 5. 发现 D：参数量分布 ≠ 耗时分布

| 组件 | 参数量 | 占比 | 运行次数 |
|---|---|---|---|
| vision_model | 86.43 M | 33.7% | **1 次** |
| connector | 7.08 M | 2.8% | 1 次 |
| text_model | 134.59 M | 52.5% | **N 次（每 token）** |
| lm_head + others | ~28.4 M | 11.1% | N 次 |

视觉塔参数量占 1/3 但只跑一次（10.9% 耗时）；
语言塔参数量占一半但跑 N 次（主导耗时）。
→ **量化收益应按「参数量 × 调用次数」评估，而非只看参数量。**

## 6. 待验证/待修正项

- [ ] fp16 慢的根因需进一步确认：尝试 `torch.backends.cuda.preferred_linalg_library`、
      不同 cuBLAS workspace 配置、或验证是否为笔记本 GPU 的已知 cuBLAS 问题
- [ ] `decode_ms_per_tok` 当前用「总时间 - vision - prefill」估算，
      含首 token 与 generate 循环开销，M2 将改为显式逐 token 计时
- [ ] 单图 512 场景 vs 多图/高分辨率场景的最优配置可能不同，M3 需重测
