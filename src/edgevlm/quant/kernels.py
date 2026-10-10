"""M2 量化：自写量化 kernel（不依赖 bitsandbytes/GPTQ）.

设计原则：
  1. **只用 PyTorch 原语实现**，每一处数值行为都可解释、可调试。
  2. 针对 SM7.5 的算力现实设计：
     - 有 INT8 Tensor Core → W8A8 有理论收益，**必须实测验证**
     - 无 INT4 Tensor Core → W4A16 预期无 TC 加速，反量化开销可能吃掉收益
     - **本机 fp16 TC 已确认失效**（见 docs/findings_M1_baseline.md 发现 A），
       因此不能假设低精度一定更快
  3. 分层量化：按「参数量 × 调用次数」优先量化 text_model。

量化方案：
  * **W8A8-INT8**：权重 per-channel INT8，激活 per-token 动态 INT8
    → 目的是吃到 INT8 Tensor Core（int8 matmul）
  * **W4A16**：权重 per-group(128) INT4 打包，反量化回 fp16/fp32 计算
    → 目的是省显存/带宽，**不期望 TC 加速**
  * **KV-Cache INT8**：per-token per-head 量化 K/V

参考：Jacob et al. 2018 (per-channel affine quantization),
      Frantar et al. 2023 (GPTQ group-wise),
      Dettmers et al. 2022 (LLM.int8() 关键维度).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 1. 基础量化原语
# ==========================================================================
def quantize_affine_int8(w: torch.Tensor, per_channel: bool = True,
                         eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称仿射 INT8 量化.

    对称量化（zero-point=0）的理由：INT8 matmul 的 TC 路径对对称量化支持最好，
    且权重分布近似零均值，对称量化几乎无额外损失。

    Args:
        w: (*, K) 权重，最后一维为 input features（按行量化）
        per_channel: True 则每行一个 scale，False 则全局一个 scale
    Returns:
        q: INT8 tensor, scale: fp32 tensor
    """
    if per_channel:
        amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    else:
        amax = w.abs().amax().clamp(min=eps)
    scale = amax / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_affine_int8(q: torch.Tensor, scale: torch.Tensor,
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)


def quantize_groupwise_int4(w: torch.Tensor, group_size: int = 128,
                            eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """按组对称 INT4 量化，权重打包为 uint8（每字节 2 个值）.

    分组量化（group-wise）是 GPTQ/AWQ 的核心技巧：
    组内共享 scale，比 per-channel 更精细，代价是额外存 scale。
    INT4 只有 16 个量化级，per-channel 通常精度损失过大，必须分组。

    打包要求 K 同时是 group_size 和 2 的整数倍，故先补齐到
    lcm(group_size, 2)。对 group_size=128 即 128。

    Returns:
        packed: uint8 tensor, 形状 (..., Kp//2)
        scale:  fp32 tensor, 形状 (..., Kp//group_size)
        pad:    为对齐而补齐的列数
    """
    orig_shape = w.shape
    K = orig_shape[-1]

    # 需要 K 同时被 group_size 和 2 整除
    unit = group_size if group_size % 2 == 0 else group_size * 2
    pad = (unit - K % unit) % unit
    if pad:
        w = F.pad(w, (0, pad))
    Kp = w.shape[-1]
    assert Kp % group_size == 0 and Kp % 2 == 0, (Kp, group_size)

    wg = w.reshape(*w.shape[:-1], Kp // group_size, group_size)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / 7.0                       # INT4 对称范围 [-7, 7]
    q = torch.round(wg / scale).clamp(-7, 7).to(torch.int8)

    # 展平回 (..., Kp)，映射 (-8..7) -> (0..15)，两个一组塞进 uint8
    q = (q + 8).to(torch.uint8).reshape(*w.shape[:-1], Kp)
    low = q[..., 0::2]
    high = q[..., 1::2]
    packed = (low | (high << 4)).contiguous()
    return packed, scale.squeeze(-1), pad


def dequantize_groupwise_int4(packed: torch.Tensor, scale: torch.Tensor,
                              group_size: int = 128, pad: int = 0,
                              out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """INT4 解包 + 反量化."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    q = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    q = q - 8

    Kp = q.shape[-1]
    assert Kp % group_size == 0, (Kp, group_size)
    q = q.reshape(*q.shape[:-1], Kp // group_size, group_size)
    out = q.to(out_dtype) * scale.to(out_dtype).unsqueeze(-1)
    out = out.reshape(*q.shape[:-2], Kp)
    if pad:
        out = out[..., :-pad]
    return out


# ==========================================================================
# 2. 量化线性层
# ==========================================================================
@dataclass
class QuantStats:
    """单层量化误差统计."""
    name: str
    scheme: str
    mse: float
    max_abs_err: float
    rel_err: float
    weight_mb: float


class QuantizedLinearW8A8(nn.Module):
    """W8A8：权重 INT8 per-channel，激活 per-token 动态 INT8.

    激活动态量化的必要性：LLM 激活存在显著离群值（outlier），
    静态校准 scale 会被离群值拉大，导致大部分值精度损失。
    per-token 动态量化对每个 token 单独算 scale，能显著缓解。

    计算路径：
        x_int8 @ w_int8.T  →  INT32 累加  →  乘 (s_x * s_w)  →  fp32 输出
    这条路径正是 INT8 Tensor Core 的用法，故有加速潜力。
    """

    def __init__(self, linear: nn.Linear, quantize_activation: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quantize_activation = quantize_activation
        self.has_bias = linear.bias is not None

        w = linear.weight.detach().float()
        q, scale = quantize_affine_int8(w, per_channel=True)
        # 注册为 buffer（不参与梯度），保持 int8 存储
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        if self.has_bias:
            self.register_buffer("bias", linear.bias.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        w = self.weight_int8                                   # (out, in) int8
        s_w = self.weight_scale.float()                        # (out, 1)

        if self.quantize_activation:
            # per-token（最后一维）动态量化
            amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            s_x = amax / 127.0
            x_q = torch.round(xf / s_x).clamp(-127, 127).to(torch.int8)

            x2d = x_q.reshape(-1, x_q.shape[-1]).contiguous()
            M = x2d.shape[0]

            # torch._int_mm 要求 M > 16（cuBLASLt INT8 的 tile 约束）。
            # 自回归 decode 阶段 M == 1（逐 token），走不了 INT8 TC 路径。
            # 这直接决定了 INT8 的收益只在 prefill / 大 batch 场景成立 ——
            # 而 M1 已实测 decode 占端到端 69.5%，故端到端加速会被显著稀释。
            if M > 16 and hasattr(torch, "_int_mm"):
                acc = torch._int_mm(x2d, w.t().contiguous())    # int32 (M, out)
                acc = acc.reshape(*x_q.shape[:-1], self.out_features).float()
            else:
                # 回退：整数乘加用 fp32 模拟（数值等价，语义不变）
                acc = x2d.float() @ w.t().float()
                acc = acc.reshape(*x_q.shape[:-1], self.out_features)
            # 反量化：x scale 逐行(token)，w scale 逐列(channel)
            out = acc * s_x * s_w.t().reshape(1, -1)
        else:
            out = xf @ (w.float() * s_w).t()

        if self.has_bias:
            out = out + self.bias
        return out.to(orig_dtype)

    @property
    def uses_tensor_core(self) -> bool:
        return True


class QuantizedLinearW4A16(nn.Module):
    """W4A16：权重 INT4 group-wise，激活保持原精度.

    注意：本机无 INT4 Tensor Core，此方案**只省显存/带宽，不省算力**，
    且每次 forward 都要解包+反量化，有额外开销。这正是要实测验证的点。
    """

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.has_bias = linear.bias is not None

        w = linear.weight.detach().float()
        packed, scale, pad = quantize_groupwise_int4(w, group_size)
        self.pad = pad
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        if self.has_bias:
            self.register_buffer("bias", linear.bias.detach().float().clone())

    def dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        w = dequantize_groupwise_int4(
            self.weight_packed, self.weight_scale,
            self.group_size, self.pad, out_dtype=torch.float32)
        assert w.shape[-1] == self.in_features, (w.shape, self.in_features)
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        w = self.dequant_weight(orig_dtype)
        out = x @ w.t()
        if self.has_bias:
            out = out + self.bias.to(orig_dtype)
        return out.to(orig_dtype)


class QuantizedLinearW8A16(nn.Module):
    """W8A16：仅权重量化 INT8，激活保持原精度（weight-only）.

    用途：在激活量化会伤精度的层（如 lm_head、首尾层）使用。
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None

        w = linear.weight.detach().float()
        q, scale = quantize_affine_int8(w, per_channel=True)
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        if self.has_bias:
            self.register_buffer("bias", linear.bias.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        w = (self.weight_int8.float() * self.weight_scale.float()).to(orig_dtype)
        out = x @ w.t()
        if self.has_bias:
            out = out + self.bias.to(orig_dtype)
        return out.to(orig_dtype)


SCHEMES = {
    "w8a8": QuantizedLinearW8A8,
    "w8a16": QuantizedLinearW8A16,
    "w4a16": QuantizedLinearW4A16,
}


# ==========================================================================
# 3. 显存核算
# ==========================================================================
def weight_bytes(module: nn.Module) -> int:
    """量化后权重的实际字节数（buffer 存储）."""
    total = 0
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def fp_weight_bytes(linear: nn.Linear, dtype: torch.dtype = torch.float16) -> int:
    n = linear.weight.numel()
    e = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total = n * e
    if linear.bias is not None:
        total += linear.bias.numel() * e
    return total
