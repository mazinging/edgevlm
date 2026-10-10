"""M2 模型级量化：把自写量化层替换进 SmolVLM，测精度/显存/时延权衡.

分层策略（由 M1 发现 D 驱动）：
  按「参数量 × 调用次数」决定量化优先级：
    * text_model : 134.6M × N 次/token  -> 最高优先级
    * lm_head    : 词表矩阵，量化收益大但伤精度敏感 -> 单独可配
    * vision_model: 86.4M × 1 次        -> 优先级最低
    * connector  : 7.1M × 1 次          -> 一般不量化

用法:
    python -m edgevlm.quant.quantize_model --scheme w8a8 --scope text
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR
from edgevlm.quant.kernels import (
    QuantizedLinearW8A8, QuantizedLinearW8A16, QuantizedLinearW4A16,
    weight_bytes, fp_weight_bytes,
)


SCHEME_CLASSES = {
    "w8a8": QuantizedLinearW8A8,
    "w8a16": QuantizedLinearW8A16,
    "w4a16": QuantizedLinearW4A16,
}

# 参数层级作用域：决定替换哪些子模块中的 nn.Linear
SCOPES = {
    "text": ["model.text_model"],
    "vision": ["model.vision_model"],
    "connector": ["model.connector"],
    "lm_head": ["lm_head"],
    "all": ["model.text_model", "model.vision_model", "model.connector", "lm_head"],
}


def _resolve(root: nn.Module, path: str) -> Optional[nn.Module]:
    m = root
    for part in path.split("."):
        if not hasattr(m, part):
            return None
        m = getattr(m, part)
    return m


def quantize_module_linear(module: nn.Module, scheme: str,
                           min_features: int = 256) -> Tuple[int, int]:
    """递归替换 module 内的 nn.Linear 为量化层.

    Args:
        min_features: 跳过过小的层（量化收益低、开销占比高）
    Returns:
        (替换数, 跳过数)
    """
    cls = SCHEME_CLASSES[scheme]
    n_repl = n_skip = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if min(child.in_features, child.out_features) < min_features:
                n_skip += 1
                continue
            setattr(module, name, cls(child))
            n_repl += 1
        else:
            r, s = quantize_module_linear(child, scheme, min_features)
            n_repl += r
            n_skip += s
    return n_repl, n_skip


def apply_quantization(model: nn.Module, scheme: str, scope: str = "text",
                       min_features: int = 256) -> Dict:
    """对指定作用域应用量化，返回统计信息."""
    paths = SCOPES[scope]
    info = {"scheme": scheme, "scope": scope, "modules": {}}
    total_repl = total_skip = 0
    for p in paths:
        mod = _resolve(model, p)
        if mod is None:
            info["modules"][p] = "NOT_FOUND"
            continue
        r, s = quantize_module_linear(mod, scheme, min_features)
        info["modules"][p] = {"replaced": r, "skipped": s}
        total_repl += r
        total_skip += s
    info["total_replaced"] = total_repl
    info["total_skipped"] = total_skip
    return info


def measure_model_weight_bytes(model: nn.Module) -> int:
    """统计模型全部参数+buffer 字节数."""
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
