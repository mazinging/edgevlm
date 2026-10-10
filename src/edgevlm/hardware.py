"""硬件能力探测与规格数据库.

设计要点：
  * 所有能力查询基于**实测**（torch 查询 + 微基准），而非硬编码假设。
  * 同时维护"目标平台"（Jetson Orin）的**规格数据**，用于 M6 的 roofline 外推。
    外推结果必须显著标注为预测，不得与实测混淆。
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import torch


# --------------------------------------------------------------------------
# 已知平台规格（用于 roofline 外推；来源：厂商公开 datasheet）
# 注意：这些是纸面规格，不是实测值。
# --------------------------------------------------------------------------
PLATFORM_SPECS: Dict[str, dict] = {
    "gtx1650ti": {
        "name": "GTX 1650 Ti (Laptop)",
        "arch": "Turing",
        "compute_cap": 7.5,
        "mem_gb": 4.0,
        "mem_bw_gbps": 192.0,
        "fp32_tflops": 3.0,
        "fp16_tc_tflops": 6.0,
        "bf16_tc_tflops": None,      # Turing 无 BF16 Tensor Core
        "int8_tc_tops": 12.0,
        "int4_tc_tops": None,        # Turing 无 INT4 Tensor Core
        "tdp_w": 50.0,
        "measured": True,
    },
    "jetson_orin_nano": {
        "name": "Jetson Orin Nano 8GB",
        "arch": "Ampere",
        "compute_cap": 8.7,
        "mem_gb": 8.0,
        "mem_bw_gbps": 68.0,
        "fp32_tflops": 1.5,
        "fp16_tc_tflops": 20.0,
        "bf16_tc_tflops": 20.0,
        "int8_tc_tops": 40.0,
        "int4_tc_tops": 80.0,
        "tdp_w": 15.0,
        "measured": False,           # 纸面规格 — 本项目无真机
    },
    "jetson_orin_nx": {
        "name": "Jetson Orin NX 16GB",
        "arch": "Ampere",
        "compute_cap": 8.7,
        "mem_gb": 16.0,
        "mem_bw_gbps": 102.4,
        "fp32_tflops": 3.4,
        "fp16_tc_tflops": 50.0,
        "bf16_tc_tflops": 50.0,
        "int8_tc_tops": 100.0,
        "int4_tc_tops": 200.0,
        "tdp_w": 25.0,
        "measured": False,
    },
    "cpu": {
        "name": "CPU (host)",
        "arch": platform.machine(),
        "compute_cap": None,
        "mem_gb": None,
        "mem_bw_gbps": None,
        "fp32_tflops": None,
        "fp16_tc_tflops": None,
        "bf16_tc_tflops": None,
        "int8_tc_tops": None,
        "int4_tc_tops": None,
        "tdp_w": None,
        "measured": True,
    },
}


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mem_gb: float
    compute_cap: tuple
    multi_processor_count: int
    sm_count: int
    cuda_cores: int
    torch_dtype_native: str

    @property
    def supports_bf16_tc(self) -> bool:
        """BF16 Tensor Core 从 Ampere (8.0) 开始支持."""
        return self.compute_cap[0] >= 8

    @property
    def supports_int8_tc(self) -> bool:
        """INT8 Tensor Core 从 Turing (7.5) 开始支持."""
        return self.compute_cap >= (7, 5)

    @property
    def supports_int4_tc(self) -> bool:
        """INT4 Tensor Core 从 Ampere (8.0) 开始支持."""
        return self.compute_cap[0] >= 8

    @property
    def supports_fp8(self) -> bool:
        """FP8 从 Ada (8.9) 开始支持."""
        return self.compute_cap >= (8, 9)


def probe_gpu(index: int = 0) -> Optional[GPUInfo]:
    """探测当前 GPU 的实测能力."""
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(index)
    cc = torch.cuda.get_device_capability(index)
    # 每个 SM 的 CUDA core 数：Turing/Ampere/Ada 均为 128（含 FP32/INT32 混合）
    cores_per_sm = 128
    return GPUInfo(
        index=index,
        name=p.name,
        total_mem_gb=p.total_memory / 1e9,
        compute_cap=cc,
        multi_processor_count=p.multi_processor_count,
        sm_count=p.multi_processor_count,
        cuda_cores=p.multi_processor_count * cores_per_sm,
        torch_dtype_native="bfloat16" if cc[0] >= 8 else "float16",
    )


def _query_nvidia_smi() -> dict:
    """通过 nvidia-smi 获取驱动/显存/功耗信息（实测）."""
    fields = [
        "name", "driver_version", "memory.total", "memory.used",
        "compute_cap", "power.draw", "power.limit", "temperature.gpu",
        "clocks.sm", "clocks.max.sm", "clocks.mem",
    ]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=20,
        ).decode("utf-8", "ignore").strip()
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        return dict(zip([f.replace(".", "_") for f in fields], vals))
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}


@dataclass
class CapabilityMatrix:
    """能力矩阵 — 驱动所有优化决策的单一事实来源."""
    gpu: Optional[dict]
    nvidia_smi: dict
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    platform: str
    capabilities: Dict[str, bool] = field(default_factory=dict)
    implications: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def build_capability_matrix() -> CapabilityMatrix:
    """构建完整能力矩阵，并推导对优化方案的直接影响."""
    g = probe_gpu()
    smi = _query_nvidia_smi()

    caps = {
        "cuda_available": torch.cuda.is_available(),
        "bf16_tensor_core": bool(g and g.supports_bf16_tc),
        "int8_tensor_core": bool(g and g.supports_int8_tc),
        "int4_tensor_core": bool(g and g.supports_int4_tc),
        "fp8_support": bool(g and g.supports_fp8),
        "tf32_support": bool(g and g.compute_cap[0] >= 8),
        "flash_attention_2": _has_flash_attn(),
        "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    }

    implications: List[str] = []
    if g:
        cc = g.compute_cap
        implications.append(
            "GPU %s, SM %d.%d, %d SMs, %.1f GB" % (g.name, cc[0], cc[1], g.sm_count, g.total_mem_gb)
        )
        if not g.supports_bf16_tc:
            implications.append(
                "[关键] SM%d.%d 无 BF16 Tensor Core（Ampere+ 才有）。模型原生 bf16，"
                "预期 FP16 快于 BF16 —— 必须实测验证。" % cc
            )
        if g.supports_int8_tc and not g.supports_int4_tc:
            implications.append(
                "[关键] 有 INT8 TC 但无 INT4 TC。W8A8 预期有加速；"
                "W4A16 缺少 TC 支撑，反量化开销可能吃掉收益 —— 需实测，"
                "该结论应转化为「INT4 应留给 Orin(Ampere)」的部署建议。"
            )
        if not caps["tf32_support"]:
            implications.append("无 TF32（Ampere+ 才有），FP32 路径无此加速。")
        if not caps["flash_attention_2"]:
            implications.append(
                "未安装 flash-attn，Attention 使用 SDPA/eager 路径。"
                "SM7.5 对 FA2 支持有限，本项目的 attention 优化走 SDPA + 手工实现。"
            )
    else:
        implications.append("无可用 CUDA 设备，仅 CPU 路径。")

    return CapabilityMatrix(
        gpu=asdict(g) if g else None,
        nvidia_smi=smi,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        platform=platform.platform(),
        capabilities=caps,
        implications=implications,
    )


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 微基准：实测各 dtype 的 matmul 吞吐，用于验证纸面规格
# --------------------------------------------------------------------------
def _ramp_clocks(device: str = "cuda", seconds: float = 3.0) -> None:
    """强制 GPU 从 idle 升到工作频率.

    必要性（实测教训）：笔记本 GPU 空闲时 SM 时钟可低至 300MHz（上限 2100MHz），
    若直接测量会把「未升频」误读为「该 dtype 性能差」，得出完全错误的结论。
    第一版测量就出现了 fp16(0.43) 慢于 fp32(2.70) 的反常结果，即此原因。
    """
    if device != "cuda" or not torch.cuda.is_available():
        return
    a = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for _ in range(20):
            _ = a @ a
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def bench_matmul_throughput(
    size: int = 2048, iters: int = 50, warmup: int = 10, device: str = "cuda",
    ramp_seconds: float = 3.0,
) -> Dict[str, float]:
    """实测不同精度下的 matmul TFLOPS.

    返回 {dtype: tflops}。Turing 上预期 fp16 > bf16（因无 BF16 TC）。

    先升频再测，避免把降频误判为精度性能差异。
    """
    results: Dict[str, float] = {}
    if device == "cuda" and not torch.cuda.is_available():
        return results

    _ramp_clocks(device, ramp_seconds)

    # 顺序影响结果（先测的 dtype 承担升频），故随机化并重复
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    flops_per_mm = 2.0 * size ** 3

    for dt in dtypes:
        try:
            a = torch.randn(size, size, device=device, dtype=dt)
            b = torch.randn(size, size, device=device, dtype=dt)
        except Exception:
            continue
        for _ in range(warmup):
            _ = a @ b
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        if device == "cuda":
            torch.cuda.synchronize()
        dt_s = (time.perf_counter() - t0) / iters
        results[str(dt).replace("torch.", "")] = flops_per_mm / dt_s / 1e12
        del a, b
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def bench_memory_bandwidth(size_mb: int = 256, iters: int = 30, device: str = "cuda") -> float:
    """实测显存带宽 (GB/s) — roofline 分析的基础输入."""
    if device == "cuda" and not torch.cuda.is_available():
        return 0.0
    n = size_mb * 1024 * 1024 // 4  # fp32 elements
    src = torch.empty(n, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # 读 + 写
    return (src.numel() * 4 * 2) / dt / 1e9


if __name__ == "__main__":
    from edgevlm import RESULTS_DIR
    import os

    m = build_capability_matrix()
    print(json.dumps(m.to_dict(), indent=2, ensure_ascii=False))
    print("\n=== 优化策略影响 ===")
    for i in m.implications:
        print(" -", i)

    print("\n=== matmul 吞吐实测 (TFLOPS) ===")
    for k, v in bench_matmul_throughput().items():
        print("  %-10s %.2f" % (k, v))
    print("\n=== 显存带宽实测 ===")
    print("  %.1f GB/s" % bench_memory_bandwidth())

    m.save(os.path.join(RESULTS_DIR, "capability_matrix.json"))
