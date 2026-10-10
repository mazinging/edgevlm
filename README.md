# 端侧 VLM/VLA 推理优化项目

针对「端侧大模型（VLM/VLA）推理优化」实习岗位要求构建的完整工程实践项目。
在**真实硬件**上完成量化、KV Cache、跨模态调度、部署链路、VLA 动作生成与多平台外推共 7 个模块。

> **核心原则：所有结论基于真机实测。**
> 涉及 Jetson 的数字均为模型外推，已显式标注 `[PREDICTED]`。
> 负结果（失败的方案）同样完整记录。

---

## 快速开始

```bash
# 环境（Windows, Python 3.8）
set PYTHONPATH=E:\VLM project\src

# M0 硬件能力矩阵
python -m edgevlm.hardware

# M1 端到端基线（真实模型 SmolVLM-256M）
python -m edgevlm.bench.baseline

# M2 量化内核 + 配置扫描
python -m edgevlm.quant.run_m2

# M3 KV Cache / 调度优化
python -m edgevlm.kv.run_context_scan
python -m edgevlm.kv.run_launch_opt

# M4 ONNX 部署链路
python -m edgevlm.deploy.run_m4

# M5 VLA 动作生成 + 蒸馏
python -m edgevlm.vla.env          # 自检 + 数据集
python -m edgevlm.vla.run_m5 --epochs 60 --distill-epochs 40

# M6 多平台外推
python -m edgevlm.platforms.run_m6

# M7 报告 + 图表
python -m edgevlm.report.run_m7
```

**环境注意**：模型加载与 GPU 操作需完整文件权限（HF 缓存目录需 chmod）；
首次运行会从 `hf-mirror.com` 下载模型（489MB）。

---

## 核心成果（全部实测）

| 模块 | 关键发现 |
|---|---|
| **M1 基线** | decode 占 **82.6%** 延迟；此机器上 **fp32 最快**（fp16 仅达成纸面规格 7%） |
| **M2 量化** | INT8 matmul **3.97×**（bit-exact），但端到端仅 **0.51×** —— 因 decode 的 M=1 |
| **M3 KV/调度** | 序列增 6.9× 时 decode 几乎不变 → **KV 量化上界 0.17%，主动放弃** |
| **M4 部署** | ONNX 导出误差 <7.1e-06；ORT 图优化收益 ≤10.7% |
| **M5 VLA** | flow matching **2 步饱和**；2 步蒸馏学生**优于** 10 步教师（好 2.9×，快 4.8×） |
| **M6 外推** | **roofline 仅对计算受限负载成立**；Orin 上 decode 因带宽受限反而更慢 |

---

## 三个最有价值的技术结论

### 1. 微基准不能预测端到端收益

INT8 matmul 微基准 **11.04 TOPS**（3.97× fp32），但端到端 W8A8 只有 **0.51×**（更慢）。

**原因**：Tensor Core 的 INT8 路径要求 `M > 16`（cuBLASLt tile 约束），
而 decode 阶段 M=1，永远用不上。

通过 falsifiable 实验定量验证 M 依赖：

| M | 1–16 | 17 | 32 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 加速比 | N/A | 1.02× | 1.09× | 1.88× | 3.08× | 3.55× |

**教训**：优化前先确认目标阶段的 M 是多少。

### 2. 上界分析先于实现

「长上下文必须量化 KV Cache」是常见假设。实测：
序列从 79 增到 548（×6.9），decode 延迟 **44.48 → 43.96 ms/tok（几乎不变）**。

KV @ seq=548 仅 25.25 MB，理论读取 0.148 ms，占 decode 的 **0.34%**
→ **量化上界收益 0.17%**，直接放弃，节省全部实现成本。

转向定位真实瓶颈：**210 个串行 kernel**，**9433 次 `aten::as_strided`**，
matmul 仅占 CUDA 时间 10.45%。

**真正的杠杆是批处理**：bs=8 → **4.46×** 吞吐。

### 3. roofline 的适用性必须验证

最初直接用 `效率 = (FLOPs/峰值) / 实测时间`，得到 prefill 效率 **0.028**（2.8%）。

**排查**：
1. 用已知 FLOPs 的纯 GEMM 标定 → 效率 0.605 / 0.551 / 0.923 / 0.589
   → 证明 0.028 不是硬件问题
2. profiler 实测 prefill：**GEMM 53.9%**、布局类 **15.3%**（`as_strided` 1335 次）
3. **结论：该负载不是计算受限，roofline 不适用**

改用分解模型后效率回到 0.292（合理区间）。

**教训**：用已知 FLOPs 的基准标定；若反解效率远低于标定值，
说明该负载不是计算受限。

---

## 平台能力门控（决定优化方案是否可行）

| 能力 | GTX 1650 Ti (Turing SM7.5) | Orin (Ampere SM8.7) |
|---|---|---|
| INT8 Tensor Core | ✅ | ✅ (40 TOPS) |
| BF16 Tensor Core | ❌ | ✅ |
| **INT4 Tensor Core** | ❌ | ✅ (80 TOPS) |
| TF32 | ❌ | ✅ |

**因此本机结论不能直接搬到 Orin**：

| 本机实测 | Orin 预期 |
|---|---|
| W4A16 变慢（0.52×）且精度崩坏（7.0% + 幻觉） | **预期反转**（有 INT4 TC） |
| W8A8 端到端 0.51× | **预期反转**（INT8 TC 40 TOPS） |
| W8A16 权重 1.19→0.87 GB | **直接有效**（decode 带宽受限） |

---

## 负结果汇总（同样重要）

| 方案 | 结果 | 原因 |
|---|---|---|
| W4A16 量化 | 0.52× 且**输出幻觉** | 无 INT4 TC + group-wise 误差 7.0% |
| W8A8 端到端 | 0.51×（**更慢**） | decode M=1 用不上 TC |
| KV Cache 量化 | **主动放弃** | 上界收益仅 0.17% |
| `torch.compile` (reduce-overhead) | 0.95× | 未减少串行 kernel 数 |
| ORT 图优化 ALL | ≤10.7%（噪声内） | 图优化无法消除布局开销 |
| fp16 推理 | 比 fp32 慢 3.75×（vision 527ms vs 140ms） | cuBLAS 回退到非 TC kernel |
| 1 步蒸馏 | MSE 0.635（失败） | 单步 Euler 表达能力不足 |

---

## 目录结构

```
src/edgevlm/
├── __init__.py              # 路径解析、HF 镜像、缓存重定向
├── hardware.py              # 硬件探测、GPU 升频、微基准
├── bench/baseline.py        # M1 端到端基线（vision/prefill/decode 拆分）
├── quant/
│   ├── kernels.py           # 自研 INT8/INT4 量化内核
│   ├── model_quant.py       # 模型级量化应用
│   ├── run_m2.py            # M2 扫描
│   └── verify_m_dependency.py  # M 依赖 falsifiable 实验
├── kv/
│   ├── run_context_scan.py  # M3 上下文扫描
│   └── run_launch_opt.py    # M3 调度优化
├── deploy/
│   ├── ort_env.py           # ORT 环境（显式检测 CPU 回退）
│   └── run_m4.py            # M4 ONNX 导出
├── vla/
│   ├── env.py               # M5 自建指令条件化环境
│   ├── model.py             # flow matching VLA
│   └── run_m5.py            # M5 训练 + 蒸馏实验
├── platforms/run_m6.py      # M6 多平台外推
└── report/run_m7.py         # M7 报告生成

results/                     # 全部实测 JSON（8 个）
docs/                        # 各模块发现文档
figures/                     # 报告图表
REPORT.md                    # 自动生成的完整报告
RESUME_BULLETS.md            # 简历要点
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `REPORT.md` | **完整报告**（由 `results/*.json` 自动生成） |
| `docs/ENVIRONMENT.md` | 环境搭建与已知限制 |
| `docs/findings_M1_baseline.md` | 端到端基线 |
| `docs/findings_M2_quant.md` | 量化内核与 M 依赖 |
| `docs/findings_M3_kv_scheduling.md` | KV Cache 与调度 |
| `docs/findings_M4_deploy.md` | ONNX 部署链路 |
| `docs/findings_M5_vla.md` | VLA 动作生成与蒸馏 |
| `docs/findings_M6_platforms.md` | 多平台外推 |
| `RESUME_BULLETS.md` | 简历量化成果 |

---

## 已知限制（诚实声明）

1. **无 Jetson 真机** —— M6 全部为外推，且外推假设（效率迁移）未被验证
2. **fp16 异常未深挖** —— cuBLAS 回退，记录为已知环境限制
3. **ORT CUDA EP 无法加载**（`WinError 1114`）—— M4 使用 CPU EP 并显式标注
4. **M5 任务为 2D 到达** —— 动作分布近似单峰，未体现 flow matching 的多模态优势
5. **未做 TensorRT / CUDA Graph / torch.export** 路径
6. M5 原设计的 push 任务失败 —— 环境物理正确（已验证），但闭环控制器历经 10 版迭代仍仅 2~7% 成功率，
   失败分析完整保留在 `env.py` docstring 中，最终改为到达任务

---

## 可复用方法论

1. **先定位瓶颈类型，再选优化手段**
   - 带宽受限（decode）→ 减少访存
   - 计算受限（vision）→ 提升算力利用率
   - overhead 受限（prefill）→ 算子融合

2. **微基准不能预测端到端收益** —— 必须端到端验证

3. **检查 M 维度** —— Tensor Core 普遍要求 M>16

4. **上界分析先于实现** —— 先算理论收益再决定是否动手

5. **验证 roofline 适用性** —— 用纯 GEMM 标定效率

6. **不要相信 datasheet** —— 本机 fp16 仅达成纸面 7%

7. **报告负面结果** —— 它们划定了「什么不管用」的边界
