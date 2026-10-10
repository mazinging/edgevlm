# EdgeVLM-Kit — 环境与约束事实表（实测，非推测）

> 本文件记录**实测验证过**的环境事实。所有后续方案设计必须与此一致。
> 最后更新：由环境探测阶段实测得出。

## 1. 硬件

| 项目 | 实测值 | 来源 |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti (Laptop) | `nvidia-smi` |
| 显存 | 4096 MiB (4 GB) | `nvidia-smi --query-gpu` |
| 计算能力 | **7.5 (Turing)** | `nvidia-smi --query-gpu=compute_cap` |
| 驱动 | 572.83, CUDA 12.8 runtime | `nvidia-smi` |
| 本机 CUDA Toolkit | v12.0 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`) | 文件系统 |
| 磁盘可用 | E: 约 314 GB | `dir` |

### 关键算力结论（直接决定量化方案）
- **Turing (SM 7.5) 支持 INT8 Tensor Core，不支持 INT4 / FP8 Tensor Core。**
  → W8A8 有理论加速收益；**W4A16 无 INT4 TC 加速，可能反而变慢**（需 dequant 开销）。
  → BF16 在 Turing 上**无原生 TC 支持**，会走较慢路径（重要！见下）。
- **Turing 不支持 BF16 Tensor Core**（BF16 TC 从 Ampere 开始）。
  当前模型原生 `torch_dtype=bfloat16`，在 1650 Ti 上可能被降级或低效执行
  → **这是一个待实测验证的重要优化点：FP16 vs BF16 在 SM7.5 上的差异。**

## 2. Python 环境

| 项目 | 值 |
|---|---|
| 可用环境 | `D:\anaconda\envs\drl_nav` (**Python 3.8.20**) |
| torch | **2.4.1+cu121**, `cuda.is_available()=True` |
| cuDNN | 90100 |
| 已装 | transformers **4.46.3**, tokenizers 0.20.3, safetensors 0.4.5, huggingface_hub 0.25.2, einops 0.8.1, psutil 5.9.8, regex 2024.9.11, numpy, PIL 10.2.0, matplotlib 3.4.3, pandas 1.3.3, cv2 4.5.3 |
| 未装 | onnx, onnxruntime(-gpu), timm, accelerate, scipy |

### 约束
- **Python 3.8 是硬约束**：`transformers` 上限约 4.46.x；新版 tokenizers/onnx 需更高版本。
- pip 安装必须 `--only-binary=:all:` 且固定版本，否则会尝试源码编译并失败。
- 其他 conda 环境（py38/rl_ppo/tt 等）为 torch CPU 版或无 torch，**不使用**。

## 3. 网络与模型源

| 端点 | 状态 | 备注 |
|---|---|---|
| pypi.tuna.tsinghua.edu.cn | ✅ 200 | 实测下载 6.25 MB/s |
| mirrors.aliyun.com / pypi.mirrors.ustc.edu.cn | ✅ 200 | 备用源 |
| pypi.org | ✅ 200 | |
| download.pytorch.org | ✅ 200 | torchvision 预训练权重可用 |
| **huggingface.co** | ❌ DNS 解析失败 | 不可用 |
| **hf-mirror.com** | ✅ **200** | **主力模型源** |
| **modelscope.cn** | ✅ 200 | 备用模型源 |

> ⚠️ PowerShell `Invoke-WebRequest` 对这些站点会报「基础连接已经关闭」，但 **Python urllib 可用**。
> 判断连通性一律用 Python，不要用 Invoke-WebRequest。

### 网络不稳定
间歇性 `getaddrinfo failed` / ConnectTimeout。pip 需 `--retries 10 --timeout 60`。

## 4. 已选定模型（已实测下载并可运行）

**`HuggingFaceTB/SmolVLM-256M-Instruct`** — 489MB safetensors，75s 下完

| 组件 | 配置 |
|---|---|
| 架构 | `Idefics3ForConditionalGeneration` |
| 视觉塔 | SigLIP-base: 768d, 12层, 12头, patch16, image 512 |
| 语言塔 | SmolLM2/Llama: 576d, **30层**, head_dim 64, vocab 49280 |
| Attention | **GQA: 9 Q heads / 3 KV heads** |
| 位置编码 | RoPE, theta=100000, max 8192 |
| 原生 dtype | bfloat16 |
| 视觉 token 压缩 | pixel_shuffle_factor = 4 |
| 总参数 | **256.5M** |

### 实测基线（GTX 1650 Ti, bf16, eager attn）
```
权重显存     0.52 GB
峰值显存     0.67 GB
输入          79 tokens (512×512 单图, 1 tile)
输出          24 tokens
延迟          3.98 s  →  6.0 tok/s
```

### 关键架构观察（影响优化设计）
1. **GQA 已经把 KV 压到 3 heads** → KV Cache 本身不大，
   KV 量化的收益主要在**长上下文/多图**场景，需实测确认而非想当然。
2. **`tie_word_embeddings=false`** → embedding 与 lm_head 独立，
   量化时词表矩阵（49280×576）是显存大头之一。
3. `do_image_splitting=true` → 大图会切成多个 512 tile，**视觉 token 数随分辨率线性增长**，
   这是「跨模态推理调度 / 视觉 token 剪枝」的天然实验场。

### 已修复的上游配置缺陷（需在代码中保留 workaround）
`preprocessor_config.json` 中 `size.longest_edge=2048` 与 `max_image_size.longest_edge=512` 冲突，
导致 `resolution_max_side cannot be larger than max_image_size` 断言失败。
**修复**：
```python
proc.image_processor.size = {"longest_edge": 512}
proc.image_processor.max_image_size = {"longest_edge": 512}
```

## 5. 沙箱与运行约束（重要）

- 文件写入限制在 `E:\VLM project` 内；装包需写 site-packages → 需提权。
- **HF 缓存必须放在工作区内**：`HF_HOME=E:\VLM project\.cache\huggingface`
  （默认 `C:\Users\...\.cache` 不可写）
- **HF 缓存内部的 chmod/rmtree 会被沙箱拒绝** → 运行模型脚本需 `danger-full-access`。
- 临时目录需重定向：`TMP`/`TEMP` → `E:\VLM project\.tmp`

## 6. 被推翻的初始假设（记录以免重蹈）

| 初始判断 | 实测结果 | 结论 |
|---|---|---|
| 「HF 下不到，只能用自建玩具模型」 | hf-mirror 通，真实模型已跑通 | ❌ 推翻，**用真实模型** |
| 「网络整体不可达」 | Invoke-WebRequest 的假象 | ❌ 推翻，用 Python 探测 |
| 「Invoke-WebRequest 失败 = 网络故障」 | Python 同 URL 全部 200 | ❌ 工具问题，非网络问题 |
