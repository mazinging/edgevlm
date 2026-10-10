# M6 多平台对比与 Jetson Orin 外推 — 实测发现

> 数据：`results/m6_platforms.json`

## 0. 重要声明：本项目没有 Jetson 真机

所有 Jetson Orin 数字均为**模型外推（PREDICTED）**，不是实测。
本文严格区分 `[MEASURED]` 与 `[PREDICTED]`，两者不得混淆。

## 1. 一次方法论的自我纠错（本模块最有价值的部分）

### v1 的错误
最初直接用 roofline 公式 `效率 = (FLOPs/峰值) / 实测时间` 反解效率，得到：

| 负载 | v1 反解效率 | 判断 |
|---|---|---|
| decode | 0.064 | 异常 |
| prefill | **0.028** | **荒谬（2.8%）** |
| vision | 0.416 | 合理 |

同样是 GEMM 主导的负载，效率差 15 倍 —— 这说明**方法有问题**，而非 prefill 真的慢。

### 排查过程（`verify_roofline.py`）

**步骤 1：用已知 FLOPs 的纯 GEMM 标定本机效率**

| N | 时间 | TFLOPS | roofline 效率 |
|---|---|---|---|
| 512 | 0.15 ms | 1.82 | 0.605 |
| 1024 | 1.30 ms | 1.65 | 0.551 |
| 2048 | 6.20 ms | 2.77 | **0.923** |
| 4096 | 77.85 ms | 1.77 | 0.589 |

→ 纯 GEMM 效率在 **0.55~0.92**，证明 0.028 不是硬件问题。

**步骤 2：用 torch profiler 实测 SmolVLM prefill 真实构成（总 295.0 ms）**

| 算子 | 时间 | 占比 | 调用次数 |
|---|---|---|---|
| `aten::addmm` | 86.42 ms | 29.3% | 72 |
| `aten::bmm` | 26.87 ms | 9.1% | 85 |
| `aten::mm` | 26.70 ms | 9.1% | 212 |
| `aten::mul` | 15.28 ms | 5.2% | 289 |
| `aten::matmul` | 13.02 ms | 4.4% | 297 |
| **`aten::as_strided`** | 12.21 ms | 4.1% | **1335** |
| `aten::reshape` | 10.34 ms | 3.5% | 567 |
| `aten::transpose` | 10.29 ms | 3.5% | 499 |
| `aten::slice` | 10.05 ms | 3.4% | 468 |

**GEMM 类合计 = 159.1 ms = 53.9%**；布局/簿记类 ≈ 45 ms = **15.3%**。

### 结论
**roofline 只对计算受限的大算子成立。**
prefill 中近一半时间花在非算术的布局变换与小算子上，
用 FLOPs 除以峰值自然得出荒谬的低效率。

### v2 的修正做法
把耗时拆成三部分，分别用不同方式外推：

| 部分 | 约束 | 外推方式 | 不确定性 |
|---|---|---|---|
| GEMM (53.9%) | 算力 | 按峰值算力比缩放，保持标定效率 0.589 | 中 |
| 布局 (15.3%) | 带宽 + 固定开销 | 按带宽比缩放 | **HIGH** |
| 其它 (30.8%) | 带宽 | 按带宽比缩放 | 中 |

修正后反解效率：prefill **0.292**、vision **0.420** —— 都回到合理区间。

## 2. [MEASURED] 本机基准

| 项目 | 实测 | 纸面规格 | 达成率 |
|---|---|---|---|
| matmul fp32 | 2.76 TFLOPS | 3.00 | 0.92× |
| matmul fp16 | **0.42 TFLOPS** | 6.00 | **0.07×** |
| matmul bf16 | 1.65 TFLOPS | — | — |
| 显存带宽 | 171.1 GB/s | 192.0 | 0.89× |

> fp16 仅达成纸面规格的 **7%** —— 即 M1 记录的 cuBLAS 回退问题。
> **这一条直接削弱了所有外推的可信度**：datasheet 与实际可达性能可能相差极大。

## 3. [PREDICTED] Orin 外推结果

| 负载 | 本机实测 | Orin Nano | Orin NX | Nano 相对 |
|---|---|---|---|---|
| prefill (seq=79) | 295.0 ms | 407.8 ms | 264.5 ms | **0.72×（更慢）** |
| vision (512px) | 140.4 ms | 77.4 ms | 46.7 ms | **1.81×（更快）** |

- **vision 受益**（计算受限，大 GEMM）→ Orin Nano 快 1.81×，NX 快 3.01×
- **prefill 受损**（53.9% GEMM + 大量小算子）→ Orin Nano 反而慢 0.72×

Orin Nano 的 prefill 预测中：GEMM 部分仅 23.9 ms，**overhead 部分高达 384.0 ms**。
即瓶颈完全不在算力，而在 1335 次 `as_strided` 这类小算子。

## 4. decode 的带宽受限分析（可靠性最高的结论）

```
decode(bs=1) 算术强度 AI = 0.50 FLOP/B
本机脊点   = 15.62 FLOP/B
Orin Nano 脊点 = 294.1 FLOP/B
=> AI 远低于两者 => decode 是带宽受限，不是计算受限
```

按带宽比外推（**不依赖 roofline，可靠性较高**）：

| | 带宽 | 本机 | Orin Nano |
|---|---|---|---|
| 显存带宽 | 171.1 GB/s | 68.0 GB/s | **0.40×** |
| decode 延迟 | — | 37.4 ms/tok | **~94 ms/tok（更慢）** |

> **反直觉但必然的结论**：Jetson Orin Nano 上 decode **比笔记本 GPU 更慢**，
> 因为其带宽只有本机的 40%。这与「Jetson 算力强」的直觉相反。

## 5. 平台能力差异（最重要的可复用结论）

| 能力 | GTX 1650 Ti (Turing SM7.5) | Orin (Ampere SM8.7) |
|---|---|---|
| FP32 | ✅ | ✅ |
| FP16 Tensor Core | ✅ | ✅ |
| BF16 Tensor Core | ❌ | ✅ |
| INT8 Tensor Core | ✅ | ✅ (40 TOPS) |
| **INT4 Tensor Core** | ❌ | ✅ (80 TOPS) |
| TF32 | ❌ | ✅ |

**因此本项目的量化结论不能直接搬到 Orin：**

| 本机结论（M2 实测） | 在 Orin 上的预期 |
|---|---|
| W4A16 变慢（0.52×）且精度崩坏 | **预期反转** —— Orin 有 INT4 TC |
| W8A8 端到端反而变慢（0.51×） | **预期反转** —— Orin INT8 TC 40 TOPS，且 prefill 无 M=1 限制 |
| W8A16 权重 1.19→0.87 GB | **直接有效** —— decode 带宽受限，减半权重即减半访存 |

## 6. 部署启示

1. **bandwidth-bound decode** → 优化重点是**减少访存**（W8/W4 权重量化），不是堆算力
2. **compute-bound vision** → 真正受益于 Orin Tensor Core，可上 FP16/BF16
3. **overhead-bound prefill** → 优化重点是**算子融合 / 减少布局变换**，
   而非量化或换硬件。本机 1335 次 `as_strided` 是首要优化目标
4. **量化方案必须按平台能力选择**：同一套 W4A16 在 Turing 上失败、在 Ampere 上可能最优

## 7. 局限（诚实声明）

1. **无 Jetson 真机**，所有 Orin 数字为外推
2. Orin 峰值取自 datasheet，而本机 fp16 仅达成纸面 7%，说明 datasheet 可信度低
3. 外推假设 compute 部分保持本机 GEMM 效率 0.589，未考虑架构/库版本差异
4. overhead 部分外推标注 **HIGH 不确定性**，仅作数量级参考
5. 未考虑 Orin 15W TDP 的持续降频 → 外推偏乐观
6. decode 外推仅基于带宽比，未考虑算子融合可能带来的改善
7. 未考虑 Orin Nano 8GB 内存对 batch size 的正面影响
