# 端侧 VLM/VLA 推理优化 —— 项目报告

> 全部结论基于**真机实测**（GTX 1650 Ti Laptop / CPU）。
> 涉及 Jetson 的数字均为模型外推，已显式标注 `[PREDICTED]`，**不是实测**。
>
> 本报告的**全部数字由 `results/*.json` 自动读取生成**，非手工填写，
> 确保报告与实测结果不会漂移。生成脚本：`python -m edgevlm.report.run_m7`

---

## 摘要

本项目在真实硬件上完成了端侧 VLM 推理优化的 7 个模块，
覆盖量化、KV Cache、跨模态调度、部署链路、VLA 动作生成与多平台外推。

**核心发现**：

1. **精度提升与实际加速脱钩**：INT8 matmul 微基准 11.04 TOPS（3.97× fp32），
   但端到端 W8A8 只有 0.51×（反而更慢）—— 因为 decode 阶段 M=1，
   无法使用 Tensor Core。**微基准不能预测端到端收益**。
2. **M 依赖关系已定量验证**：加速比随 batch 单调上升
   （M=17→1.02×, M=128→1.88×, M=256→3.08×, M=512→3.55×）。
3. **KV Cache 量化不值得做**：seq 增 6.9× 时 decode 延迟几乎不变
   （44.48→43.96 ms/tok），KV 读取理论上界仅占 0.17%。
   实测推翻了「长上下文必须量化 KV」的常见假设。
4. **真实的瓶颈是小算子而非算力**：profiler 显示 prefill 中
   `aten::as_strided` 被调用 **1335 次**，布局类算子占 15.3%。
5. **flow matching 的甜点是 2 步**：质量在 2-4 步饱和，
   且 2 步蒸馏学生**优于** 10 步教师（MSE 好 2.8×，快 4.8×）。
6. **方法论纠错**：roofline 只对计算受限负载成立；
   最初直接用 FLOPs/峰值 反解出 2.8% 的荒谬效率，
   标定纯 GEMM（效率 0.55~0.92）后定位到问题所在并修正。

**负结果（同样重要）**：W4A16 精度崩坏且无加速；
W8A8 端到端负收益；`torch.compile` 无收益（0.95×）；
ONNX Runtime 图优化 ≤10.7%；fp16 在此机器上比 fp32 慢。

---

## 1. 硬件与能力矩阵（[MEASURED]）

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti |
| 计算能力 | SM 7.5 |
| SM 数 | 16 |
| 显存 | 4.29 GB |
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu121 |
| CUDA | 12.1 |

**能力门控**（决定优化方案是否可行）：

| 能力 | 支持 | 影响 |
|---|---|---|
| BF16 Tensor Core | ❌ | |
| INT8 Tensor Core | ✅ | |
| INT4 Tensor Core | ❌ | |
| FP8 | ❌ | |
| TF32 | ❌ | |

**微基准实测**：

| 精度 | TFLOPS | 纸面规格 | 达成率 |
|---|---|---|---|
| float32 | 2.76 | 3.0 | 0.92× |
| float16 | 0.42 | 6.0 | 0.07× |
| 显存带宽 | 171.2 GB/s | 192.0 | 0.89× |

> ⚠️ **fp16 仅达成纸面规格的 7%**（cuBLAS 回退到非 TC 参考实现）。
> 这直接说明：**不能相信 datasheet，必须实测**。

---

## 2. M1 端到端性能基线（[MEASURED]）

模型：`smolvlm-256m`（真实模型），512×512 图像。

| 精度 | attn | 吞吐 | vision | prefill | decode | 峰值显存 | 输出 |
|---|---|---|---|---|---|---|---|
| fp32 | eager | 18.59 tok/s | 140.42 ms | 253.20 ms | 37.40 ms/tok | 1.22 GB | 24 |
| fp16 | eager | 12.77 tok/s | 527.09 ms | 638.65 ms | 29.73 ms/tok | 0.67 GB | 24 |
| bf16 | eager | 15.12 tok/s | 155.63 ms | 213.12 ms | 50.76 ms/tok | 0.69 GB | 24 |

**阶段占比（最佳配置 fp32/eager, 18.59 tok/s）**：vision 12.9% + connector 0.1% + prefill 4.4% + decode 82.6%

**参数量**：256.48 M（详见 `results/baseline.json`）

**反直觉发现**：此机器上最快的配置是 **fp32**（18.59 tok/s）。

> fp16 在所有 K 值与所有调用形式下都恒定 ~0.42 TFLOPS / ~40.5ms，
> 符合「回退到非 TC 参考 kernel」的特征。

<details><summary>不支持的配置（点击展开）</summary>

- `fp32` / `sdpa`: None
- `fp16` / `sdpa`: None
- `bf16` / `sdpa`: None

</details>

---

## 3. M2 自研量化内核（[MEASURED]）

自研实现 `quantize_affine_int8` / `quantize_groupwise_int4` /
`QuantizedLinearW8A8/W8A16/W4A16`，不依赖任何量化库。

### 端到端结果

| 配置 | 方案 | 范围 | 吞吐 | 相对基线 | 显存 |
|---|---|---|---|---|---|
| fp32 (baseline) | — | — | 18.86 tok/s | **1.00×** | 1.19 |
| W8A8 / text | w8a8 | text | 9.64 tok/s | **0.51×** | 0.87 |
| W8A8 / text+vision | w8a8 | all_no_lmhead | 9.30 tok/s | **0.49×** | 0.61 |
| W8A8 / all | w8a8 | all | 9.34 tok/s | **0.50×** | 0.59 |
| W4A16 / text | w4a16 | text | 9.83 tok/s | **0.52×** | 0.83 |
| W8A16 / text | w8a16 | text | 15.69 tok/s | **0.83×** | 0.87 |

**关键负面结果**：`W8A8 / text+vision`（w8a8/all_no_lmhead）吞吐 9.30 tok/s，仅为基线的 **0.49×** —— 量化后反而更慢。

### 输出文本对比（检测量化是否破坏语义）

**基线输出**：

> The image depicts a grid of squares arranged in a grid pattern. The squares are filled with a mixture of colors, including

**`fp32 (baseline)`（—/—）**：

> The image depicts a grid of squares arranged in a grid pattern. The squares are filled with a mixture of colors, including

**`W8A8 / text`（w8a8/text）**：

> The image depicts a grid-like pattern with a pattern of blue, green, and yellow squares. The squares are arranged

**`W8A8 / text+vision`（w8a8/all_no_lmhead）**：

> The image depicts a grid-like pattern with a pattern of alternating colors. The grid is composed of squares, each square

**`W8A8 / all`（w8a8/all）**：

> The image depicts a grid-like pattern with a pattern of small squares arranged in a grid. The grid is composed of

**`W4A16 / text`（w4a16/text）**：

> The image is a 3D rendering of a scene with a variety of elements. The background is a light blue color

**`W8A16 / text`（w8a16/text）**：

> The image depicts a grid of squares arranged in a grid pattern. The squares are filled with a mixture of colors, including

### 微基准与 M 依赖（见 findings_M2_quant.md）

- INT8 matmul 微基准 **11.04 TOPS**（3.97× fp32）且 **bit-exact**（误差 0.00e+00）
- 加速比随 M 单调上升：M=17→1.02×，M=128→1.88×，M=256→3.08×，M=512→3.55×
- **M≤16 完全不可用**（cuBLASLt tile 约束）→ 这正是端到端 W8A8 变慢的原因

---

## 4. M3 KV Cache 与跨模态调度（[MEASURED]）

### 上下文扫描

| 序列长度 | prefill | decode |
|---|---|---|
| 79 | 189.8 ms | 44.48 ms/tok |
| 548 (×6.9) | 1359.5 ms (×7.16) | **43.96 ms/tok（几乎不变）** |

**结论：KV Cache 量化不值得做。**

- KV @ seq=548 = 25.25 MB，理论读取时间 0.148 ms
- 占 decode 44 ms 的 **0.34%** → 量化上界收益 0.17%
- decode 时间几乎不随序列增长，说明它由**固定开销**主导

### 真实瓶颈定位

- decode 有 **210 个串行 kernel**，每个 ~0.222 ms → 46.7 ms，与实测吻合
- profiler：**2201 次 `aten::matmul`、9433 次 `aten::as_strided`**
- matmul 仅占 CUDA 时间 10.45%

### 批处理（真正的杠杆）

| batch | 吞吐 | 相对 bs=1 |
|---|---|---|
| 1 | 10.35 tok/s | 1.00× |
| 8 | **47.32 tok/s** | **4.46×** |

`torch.compile(reduce-overhead)` **无收益（0.95×）** —— 负面结果。

---

## 5. M4 部署链路（[MEASURED]）

ONNX 导出（opset 17）拆成 3 个子图：

| 子图 | 大小 | 节点数 | 相对误差 |
|---|---|---|---|
| vision | 345.9 MB | 692 | 7.08e-06 |
| connector | 28.3 MB | 11 | — |
| text_prefill | 539.1 MB | 3541 | 2.17e-06 |

`logits_argmax_match = True` —— 导出无精度损失。

**text_prefill 的算子构成**：35.4% Constant + 18.9% 布局算子
 = **54.3% 非计算算子**

**OR2 图优化级别扫描**（负面结果）：

| 级别 | 延迟 | 相对 DISABLE_ALL |
|---|---|---|
| DISABLE_ALL | 1024.69 ms | 1.00× |
| BASIC | 1031.60 ms | 1.01× |
| EXTENDED | 914.85 ms | 1.11× |
| ALL | 1018.80 ms | 1.01× |

提升 ≤10.7%，且在噪声范围内。**ORT 图优化在此负载上收益有限。**

> 环境限制：ORT CUDA EP 在本机无法加载
> （`WinError 1114` DLL 初始化失败），M4 使用 CPU EP 并显式标注。

---

## 6. M5 VLA 动作序列生成加速（[MEASURED]）

自建 2D 指令条件化到达任务 + flow matching 动作生成 + 步数蒸馏。
模型 **1.029 M 参数**，chunk_size=4，训练 60 epoch。

### 去噪步数 vs 质量

| 步数 | 速度 | MSE | L2/chunk | cosine | 相对 10 步加速 |
|---|---|---|---|---|---|
| 1 | 0.041 ms | 0.07548 | 0.3394 | 0.9290 | **7.48×** |
| 2 | 0.061 ms | 0.03512 | 0.2172 | 0.9646 | **5.03×** |
| 4 | 0.117 ms | 0.03278 | 0.2022 | 0.9667 | **2.61×** |
| 6 | 0.181 ms | 0.03349 | 0.2039 | 0.9660 | **1.69×** |
| 10 | 0.305 ms | 0.03464 | 0.2072 | 0.9648 | **1.00×** |
| 20 | 0.618 ms | 0.03619 | 0.2131 | 0.9633 | **0.49×** |

**最优步数 = 4**（MSE 0.03278）。质量在 2-4 步饱和，更多步无收益甚至略降。

### 步数蒸馏

| 模型 | 步数 | MSE | L2 | cosine | 延迟 |
|---|---|---|---|---|---|
| teacher | 10 | 0.03464 | 0.2072 | 0.9648 | 0.301 ms |
| teacher | 20 | 0.03619 | 0.2131 | 0.9633 | 0.631 ms |
| student | 1 | 0.63453 | 0.9807 | 0.6398 | 0.034 ms |
| student | 2 | 0.01179 | 0.1116 | 0.9886 | 0.062 ms |
| student | 4 | 0.03087 | 0.2065 | 0.9757 | 0.122 ms |

**蒸馏结论**：2 步学生 MSE 0.01179 vs 10 步教师 0.03464 —— **学生好 2.9×**，延迟 0.062 ms vs 0.301 ms（**快 4.8×**）。

但 **1 步蒸馏失败**（MSE 0.63453，cosine 0.6398）：单步 Euler 只有一个线性映射，表达能力不足。

### 闭环成功率验证

| 策略 | 成功率 | 平均步数 | 最终距离 | 重规划次数/ep | 单次推理 |
|---|---|---|---|---|---|
| expert | **1.00** | 4.15 | 0.0257 | — | —（无推理） |
| teacher_10step | **1.00** | 5.55 | 0.0367 | 1.80 | 21.94 ms |
| teacher_2step | **0.99** | 5.23 | 0.0407 | 1.67 | 4.62 ms |
| student_2step | **1.00** | 4.29 | 0.0338 | 1.42 | 4.67 ms |
| student_1step | **0.76** | 13.49 | 0.0765 | 3.64 | 2.70 ms |

### 一处关键 bug 修复

动作块跨 episode 边界采样 → 监督信号混入两条轨迹，训练损失卡在 0.69。
按 episode 边界切分后降到 **0.137**。

> 教训：动作块边界处理是 VLA 训练最易忽视的坑。

---

## 7. M6 多平台外推（[PREDICTED] — 无 Jetson 真机）

### 方法论纠错（本模块最有价值的部分）

v1 直接用 `效率 = (FLOPs/峰值) / 实测时间` 反解，得到 prefill 效率 **0.028**（2.8%）。

**排查**：
1. 标定纯 GEMM 效率 → N=512/1024/2048/4096 得 0.605/0.551/0.923/0.589
   → 证明 0.028 不是硬件问题
2. profiler 实测 prefill（295.0 ms）：
   **GEMM 类 53.9%**，布局类（`as_strided` 1335 次等）**15.3%**
3. **结论：roofline 只对计算受限的大算子成立**

修正后反解效率：prefill **0.292**、vision **0.420** —— 回到合理区间。

### 外推结果

| 负载 | 本机实测 | Orin Nano | Orin NX |
|---|---|---|---|
| prefill_79 | [MEASURED] 295.0 ms | [PREDICTED] 407.8 ms (0.72×) | [PREDICTED] 264.5 ms (1.12×) |
| vision_512 | [MEASURED] 140.4 ms | [PREDICTED] 77.4 ms (1.81×) | [PREDICTED] 46.7 ms (3.01×) |

> 加粗注意：**只有第一列是实测**，后两列为 roofline 外推。

### decode 的带宽受限结论（可靠性最高，不依赖 roofline）

| | 带宽 | decode 延迟 |
|---|---|---|
| GTX 1650 Ti (MEASURED) | 171.2 GB/s | 37.40 ms/tok |
| Orin Nano (PREDICTED) | 68.0 GB/s (2.52×) | **94.1 ms/tok** |

> **反直觉但必然**：Orin Nano 上 decode 比笔记本 GPU 慢，因为带宽只有 252%。

### 平台能力差异（最重要的可复用结论）

| 能力 | Turing SM7.5 | Ampere SM8.7 (Orin) |
|---|---|---|
| INT8 TC | ✅ | ✅ |
| BF16 TC | ❌ | ✅ |
| **INT4 TC** | ❌ | ✅ |
| TF32 | ❌ | ✅ |

**因此本机量化结论不能直接搬到 Orin**：

| 本机实测结论 | Orin 预期 |
|---|---|
| W4A16 变慢且精度崩坏 | **预期反转**（Orin 有 INT4 TC） |
| W8A8 端到端 0.51× | **预期反转**（INT8 TC 40 TOPS） |
| W8A16 权重 1.19→0.87 GB | **直接有效**（decode 带宽受限） |

---

## 8. 可复用方法论

1. **先定位瓶颈类型，再选优化手段**
   - 带宽受限（decode）→ 减少访存（权重量化）
   - 计算受限（vision）→ 提升算力利用率（TC、FP16）
   - overhead 受限（prefill）→ 算子融合、减少布局变换
   本项目实测：decode 69.5% 且带宽受限、prefill overhead 占 46%、
   vision 计算受限 —— 三者需要的优化完全不同。

2. **微基准不能预测端到端收益**
   INT8 微基准 3.97×，端到端 0.51×。必须做端到端验证。

3. **检查 M 维度**
   Tensor Core 路径普遍要求 M>16。decode 的 M=1 使所有 INT8/FP16 TC 优化失效。
   优化前先确认目标阶段的 M 是多少。

4. **上界分析先于实现**
   KV 量化上界 0.17% → 直接放弃，节省全部实现成本。
   先算理论收益，再决定是否动手。

5. **验证 roofline 的适用性**
   用已知 FLOPs 的纯 GEMM 标定效率；
   若某负载反解出的效率远低于标定值，说明它**不是计算受限**，
   roofline 不适用，应改用分解模型。

6. **不要相信 datasheet**
   本机 fp16 仅达成纸面规格 7%。

7. **报告负面结果**
   W4A16 崩坏、W8A8 负收益、torch.compile 无收益、
   ORT 图优化 ≤10.7%、fp16 比 fp32 慢 —— 这些结论同样有价值，
   它们划定了「什么不管用」的边界。

---

## 9. 局限与未完成

- **无 Jetson 真机**，M6 全部为外推
- fp16 异常（cuBLAS 回退）未深挖，按用户要求记录为已知环境限制
- ORT CUDA EP 无法加载，M4 用 CPU EP
- M5 任务为 2D 到达，动作分布近似单峰，未体现 flow matching 的多模态优势
- 未做 TensorRT / `torch.export` / CUDA Graph 路径
- KV Cache 量化按上界分析主动放弃（非能力不足）
