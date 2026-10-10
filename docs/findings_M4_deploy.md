# M4 部署链路（torch → ONNX → ORT）— 实测发现

> 数据：`results/m4_deploy.json` | ONNX 模型：`artifacts/onnx/` | GTX 1650 Ti + CPU

## 0. 环境限制（先说清楚，避免误读）

| 项目 | 状态 |
|---|---|
| ONNX 导出 | ✅ 全部成功 |
| ORT CPU EP | ✅ 可用 |
| **ORT CUDA EP** | ❌ **不可用** |

**CUDA EP 不可用的实测证据**：
```
ctypes.CDLL('onnxruntime_providers_cuda.dll')
-> OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败
```
依赖 DLL（`cublasLt64_12.dll` / `cudnn64_9.dll` / `cudart64_12.dll`）**全部存在**
（torch/lib 中已提供并已加入 DLL 搜索路径），
但 CUDA provider 自身初始化失败。

> ⚠️ **重要陷阱**：ORT 在 provider 不可用时会**静默回退到 CPU**而不报错。
> 若不做显式检查（`sess.get_providers()`），会把 CPU 结果误标为 GPU 结果。
> 本项目已在 `deploy/ort_env.py` 中固化显式校验。

**因此：M4 的性能对比为「torch(GPU) vs ORT(CPU)」，
不是同平台对比，不能直接解读为"ORT 比 torch 慢"**。
M4 的有效结论集中在**导出可行性、数值一致性、图结构分析**。

## 1. 导出结果

| 子图 | 状态 | 大小 | 节点数 | 算子种类 |
|---|---|---|---|---|
| vision_encoder | ✅ | 345.9 MB | 692 | 29 |
| connector | ✅ | 28.3 MB | 11 | 4 |
| text_prefill | ✅ | 539.1 MB | 3541 | 24 |

**导出策略说明**：完整 `generate()` 含 Python 控制流，无法静态导出。
故拆为三个静态子图（视觉编码 / 连接器 / 文本前向），
decode 循环留在宿主侧 —— 这也是真实部署的常见做法。

## 2. 数值一致性（关键验证）

| 子图 | 相对误差 | 最大误差 | logits argmax 一致 |
|---|---|---|---|
| vision_encoder | **7.08e-06** | — | — |
| text_prefill | **2.17e-06** | — | **✅ True** |

**精度损失可忽略**（~1e-6 量级，属 fp32 舍入误差）。

`logits_argmax_match=True` 是最关键的一项：
它意味着**导出后的模型预测出的下一个 token 与原模型完全一致** ——
即部署转换没有改变模型的实际行为，而不仅是数值接近。

## 3. 图结构分析（结构性问题）

| 子图 | Constant | 布局类操作 | 计算类操作 |
|---|---|---|---|
| vision_encoder | 143 (20.7%) | 120 (17.3%) | 308 (44.5%) |
| connector | 4 (36.4%) | 6 (54.5%) | 1 (9.1%) |
| **text_prefill** | **1254 (35.4%)** | **669 (18.9%)** | 941 (26.6%) |

**发现**：
- `text_prefill` 图有 **3541 个节点，其中 35.4% 是 Constant**（多为位置编码、
  attention mask 等常量，未被折叠）。
- 布局类操作（Reshape/Transpose/Concat/Slice…）另占 **18.9%**。
- 二者合计 **54.3%**，而真正的计算算子只有 26.6%。

**这与 M3 的 profiler 发现完全一致**：
torch 侧 decode 单步 2201 次 matmul 中真正矩阵乘仅占 10.45%，
大量开销在 `as_strided`/`reshape`/`transpose`。
→ **问题在模型结构层面（大量小算子和布局变换），不是导出工具造成的。**

## 4. ORT 图优化级别的影响：几乎无效

| 优化级别 | 延迟 (ms) | 相对误差 |
|---|---|---|
| ORT_DISABLE_ALL | 1024.69 | 7.46e-06 |
| ORT_ENABLE_BASIC | 1031.60 | 7.46e-06 |
| ORT_ENABLE_EXTENDED | 914.85 | 7.46e-06 |
| ORT_ENABLE_ALL | 1018.80 | 7.40e-06 |

**EXTENDED 相对 DISABLE_ALL 仅快 10.7%，且 ALL 反而比 EXTENDED 慢** ——
差异在测量噪声范围内（同配置重复测量波动可达 ±10%）。

**诚实结论**：在本模型上，ORT 的图优化（算子融合、常量折叠）
**没有带来可观加速**。原因推测：
- 该模型的计算图以 MatMul + 逐元素操作为主，融合空间有限
- 瓶颈（M3 已证）是 kernel 启动与框架开销，图级优化不针对此

## 5. M4 结论小结

| 项目 | 结论 |
|---|---|
| ONNX 导出可行性 | ✅ 三个子图均成功 |
| 数值一致性 | ✅ ~1e-6，且 argmax 预测一致 |
| 图优化（ORT 级别） | ⚠️ 收益 ≤10.7%，不显著 |
| CUDA EP | ❌ 环境不可用（WinError 1114） |
| 图结构问题 | 54.3% 节点是常量/布局操作 |

**对 JD 职责 2 的交付**：
模型转换链路（torch→ONNX）、图优化评估、数值一致性验证、性能分析方法
均已实现并给出实测结论。**GPU 侧部署验证因环境限制未能完成，已明确标注。**

## 6. 后续可做（若环境允许）

- [ ] 修复 ORT CUDA EP（需匹配的 CUDA/cuDNN 版本，或换 ORT 版本）
- [ ] `torch.export` + `executorch` 路径（更适合端侧）
- [ ] TensorRT 引擎构建（Turing 支持 INT8 PTQ，可能带来真实收益）
- [ ] 把 decode 循环也用 CUDA Graph 捕获，验证 M3 提出的"削减启动开销"假设
