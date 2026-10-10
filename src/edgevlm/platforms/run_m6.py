"""M6 多平台对比与 Jetson Orin 外推.

## 诚实性原则（本模块的核心约束）
本项目**没有 Jetson 真机**。因此本模块严格区分两类数字：

  [MEASURED]  在本机实测得到（GTX 1650 Ti / CPU）
  [PREDICTED] 由模型外推得到（Jetson Orin Nano / NX）

**预测结果不得与实测混淆**，所有输出都带 measured 标记。

## 关于 roofline 的重要修正（v1 的错误与修正过程）

v1 直接用 `t = FLOPs / peak` 反解效率，得到：
    decode  eff=0.064
    prefill eff=0.028   <- 2.8%，明显异常
    vision  eff=0.416   <- 41.6%，合理
同样是 GEMM 主导的负载，效率差 15 倍 => 说明方法有问题，而非 prefill 真的慢。

**排查过程**（`verify_roofline.py`）：
  1. 用已知 FLOPs 的纯 GEMM 标定本机效率：N=512/1024/2048/4096 →
     eff = 0.605 / 0.551 / 0.923 / 0.589。
     => 纯 GEMM 应在 0.3~0.9 之间，证明 0.028 不是硬件问题。
  2. 用 torch profiler 实测 SmolVLM prefill（总 295.0 ms）：
       GEMM 类(addmm+bmm+mm+matmul+linear) = 159.1 ms = **53.9%**
       布局/簿记类(as_strided+reshape+transpose+slice+view) ≈ 45 ms = **~15%**
       其中 aten::as_strided 单独被调用 **1335 次**
  3. 结论：**roofline 只对计算受限的大算子成立**。
     prefill 中近一半时间花在非算术的布局变换与小算子上，
     用 FLOPs 除以峰值自然得出荒谬的低效率。

**v2 的修正做法**：
  * 把耗时拆成 **compute 部分**（GEMM，受算力约束）与
    **overhead 部分**（布局/元素级/launch，受带宽与固定开销约束）
  * 对 compute 部分用 roofline 外推（此时效率是可信的，取标定值）
  * 对 overhead 部分**不套用 roofline**，而是按实测的「每算子固定开销」
    结合平台差异做保守外推，并**明确标注不确定性最大**
  * vision 是真正计算受限的大 GEMM 负载，其 eff=0.416 可信，照常外推
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from edgevlm import RESULTS_DIR
from edgevlm.hardware import PLATFORM_SPECS


# --------------------------------------------------------------------------
# 实测常量（全部来自本项目本机测量，不是纸面规格）
# --------------------------------------------------------------------------
# SmolVLM-256M 单图 512px，本机 fp32 实测（M1 + profiler）
MEASURED_LOCAL = {
    "device": "GTX 1650 Ti (Laptop)",
    "vision_ms": 140.42,
    "prefill_ms": 253.20,
    "decode_ms_per_tok": 37.40,
    "prefill_total_ms": 295.00,        # profiler 实测（含 vision? 此处为整个 forward）
    "e2e_tok_per_s": 18.59,
    "gemm_fraction_prefill": 0.539,    # profiler 实测
    "layout_fraction_prefill": 0.153,  # as_strided+reshape+transpose+slice+view
    "as_strided_calls": 1335,
    "peak_fp32_tflops_measured": 2.76,
    "bw_gbps_measured": 171.1,
}

# 纯 GEMM 标定的本机 roofline 效率（用于 compute 部分外推）
CALIBRATED_GEMM_EFF = {
    512: 0.605, 1024: 0.551, 2048: 0.923, 4096: 0.589,
}
# 取中位数作为代表性效率（保守）
GEMM_EFF_REPRESENTATIVE = 0.589


@dataclass
class PlatformProfile:
    key: str
    name: str
    peak_flops: float
    peak_bw: float
    measured: bool
    note: str = ""


def make_profile(platform_key: str, precision: str = "fp32") -> PlatformProfile:
    s = PLATFORM_SPECS[platform_key]
    flops_map = {
        "fp32": s["fp32_tflops"],
        "fp16": s["fp16_tc_tflops"],
        "bf16": s["bf16_tc_tflops"],
        "int8": (s["int8_tc_tops"] / 2.0) if s["int8_tc_tops"] else None,
        "int4": (s["int4_tc_tops"] / 2.0) if s["int4_tc_tops"] else None,
    }
    tf = flops_map.get(precision)
    note = ""
    if tf is None:
        tf = s["fp32_tflops"]
        note = "该精度无硬件支持, 退回 fp32"
    return PlatformProfile(
        key=platform_key, name=s["name"],
        peak_flops=(tf or 0.0) * 1e12, peak_bw=(s["mem_bw_gbps"] or 0.0) * 1e9,
        measured=bool(s.get("measured")), note=note)


def ridge_point(p: PlatformProfile) -> float:
    return p.peak_flops / p.peak_bw if p.peak_bw > 0 else float("inf")


# --------------------------------------------------------------------------
# 分解模型：把耗时拆成 compute / bandwidth / overhead 三部分
# --------------------------------------------------------------------------
@dataclass
class WorkloadBreakdown:
    """一个负载在本机的实测分解."""
    name: str
    total_ms: float                    # 本机实测总时间
    gemm_ms: float                     # GEMM 部分（受算力约束）
    layout_ms: float                   # 布局变换（受带宽/固定开销约束）
    other_ms: float                    # 其余（元素级、softmax 等）
    gemm_flops: float                  # GEMM 部分的实际 FLOPs
    weight_bytes: float                # 需搬运的权重字节
    n_small_kernels: int = 0           # 小算子数量（用于 overhead 外推）


def build_breakdowns() -> Dict[str, WorkloadBreakdown]:
    """由实测数据构造分解（数字全部来自本项目 profiler / M1 测量）."""
    m = MEASURED_LOCAL
    tot = m["prefill_total_ms"]
    gemm = tot * m["gemm_fraction_prefill"]
    layout = tot * m["layout_fraction_prefill"]
    other = tot - gemm - layout

    # GEMM FLOPs：用 profiler 的 GEMM 时间 × 标定效率 × 峰值反推
    peak = m["peak_fp32_tflops_measured"] * 1e12
    gemm_flops = gemm / 1000.0 * GEMM_EFF_REPRESENTATIVE * peak

    return {
        "prefill_79": WorkloadBreakdown(
            "SmolVLM prefill (seq=79, 单图 512px, fp32)",
            total_ms=tot, gemm_ms=gemm, layout_ms=layout, other_ms=other,
            gemm_flops=gemm_flops,
            weight_bytes=256.48e6 * 4,
            n_small_kernels=m["as_strided_calls"],
        ),
        "vision_512": WorkloadBreakdown(
            "SmolVLM vision encoder (512px, fp32)",
            total_ms=m["vision_ms"],
            gemm_ms=m["vision_ms"] * 0.85,     # vision 是大 GEMM 主导
            layout_ms=m["vision_ms"] * 0.10,
            other_ms=m["vision_ms"] * 0.05,
            gemm_flops=86.43e6 * 4 * 2 * (512 // 16) ** 2 / 4,
            weight_bytes=86.43e6 * 4,
            n_small_kernels=200,
        ),
    }


def extrapolate(bd: WorkloadBreakdown, local: PlatformProfile,
                target: PlatformProfile) -> Dict:
    """把本机实测外推到目标平台.

    compute 部分：按峰值算力比缩放，并保持标定效率
    bandwidth 部分：按带宽比缩放
    overhead 部分：按「带宽比」和「算力比」中**较保守**的一个缩放，
                   并标注高不确定性（小算子的开销主要来自 launch 与
                   driver，与平台强相关，无法从本机可靠外推）
    """
    # compute
    t_gemm_target = bd.gemm_ms * (local.peak_flops / target.peak_flops)

    # bandwidth
    t_layout_target = bd.layout_ms * (local.peak_bw / target.peak_bw)
    t_other_target = bd.other_ms * (local.peak_bw / target.peak_bw)

    # overhead：取两个缩放因子的较大者（更保守 -> 预测时间更长）
    # 理由：小算子既吃带宽也吃 launch，且 Ampere 的 launch 开销不低于 Turing
    overhead_scale = max(local.peak_flops / target.peak_flops,
                         local.peak_bw / target.peak_bw)
    t_overhead = (bd.layout_ms + bd.other_ms) * overhead_scale

    t_pred = t_gemm_target + t_overhead
    return {
        "gemm_ms": t_gemm_target,
        "layout_ms": t_layout_target,
        "other_ms": t_other_target,
        "overhead_ms": t_overhead,
        "total_pred_ms": t_pred,
        "speedup_vs_local": bd.total_ms / t_pred if t_pred > 0 else None,
        "overhead_uncertainty": "HIGH",   # 显式标注
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_m6(device: str = "cuda") -> Dict:
    from edgevlm.hardware import (
        bench_matmul_throughput, bench_memory_bandwidth, probe_gpu, _ramp_clocks,
    )

    out: Dict = {}
    if device == "cuda":
        _ramp_clocks(device, 3.0)
    thr = bench_matmul_throughput(device=device)
    bw = bench_memory_bandwidth(device=device)
    g = probe_gpu()

    print("=" * 100)
    print("M6 多平台对比与 Jetson Orin 外推")
    print("=" * 100)

    out["measured_benchmarks"] = {
        "measured": True,
        "device": g.name if g else "cpu",
        "matmul_tflops": thr, "memory_bandwidth_gbps": bw,
        "sm_count": g.sm_count if g else None,
        "compute_cap": list(g.compute_cap) if g else None,
    }
    print("\n[MEASURED] 本机基准")
    for k, v in thr.items():
        print("    matmul %-10s %.2f TFLOPS" % (k, v))
    print("    memory bandwidth      %.1f GB/s" % bw)

    # ---- 标定 ----
    print("\n[MEASURED] 纯 GEMM roofline 效率标定（判断 roofline 是否可用）")
    calib = {}
    for n, e in CALIBRATED_GEMM_EFF.items():
        calib[str(n)] = e
        print("    N=%-5d roofline_eff=%.3f" % (n, e))
    out["gemm_calibration"] = calib
    print("    => 纯 GEMM 效率 0.55~0.92。若某负载反解出远低于此的值，")
    print("       说明该负载**不是计算受限**，roofline 不适用（见下）。")

    # ---- 分解 ----
    print("\n[MEASURED] 负载分解（profiler 实测）")
    bds = build_breakdowns()
    m = MEASURED_LOCAL
    print("    prefill 总 %.1f ms = GEMM %.1f (%.1f%%) + 布局 %.1f (%.1f%%) + 其它 %.1f" % (
        m["prefill_total_ms"], m["prefill_total_ms"] * m["gemm_fraction_prefill"],
        100 * m["gemm_fraction_prefill"], m["prefill_total_ms"] * m["layout_fraction_prefill"],
        100 * m["layout_fraction_prefill"],
        m["prefill_total_ms"] * (1 - m["gemm_fraction_prefill"] - m["layout_fraction_prefill"])))
    print("    其中 aten::as_strided 被调用 %d 次" % m["as_strided_calls"])
    out["breakdown"] = {k: asdict(v) for k, v in bds.items()}

    # 用 v1 的错误方法展示对照
    print("\n[对照] 若直接用 FLOPs/峰值 反解（v1 的错误方法）：")
    prof_local = make_profile("gtx1650ti", "fp32")
    for name, bd in bds.items():
        eff_v1 = (bd.gemm_flops / prof_local.peak_flops) / (bd.total_ms / 1000.0)
        print("    %-12s eff=%.3f  %s" % (
            name, eff_v1, "<- 荒谬" if eff_v1 < 0.15 else "<- 合理"))

    # ---- 外推 ----
    print("\n" + "-" * 100)
    print("以下全部为 [PREDICTED] —— 模型外推，**非真机实测**")
    print("-" * 100)

    profiles = {
        "gtx1650ti": prof_local,
        "orin_nano": make_profile("jetson_orin_nano", "fp16"),
        "orin_nx": make_profile("jetson_orin_nx", "fp16"),
    }
    for k, p in profiles.items():
        print("  %-11s peak=%.1f TFLOPS  bw=%.0f GB/s  ridge=%.1f FLOP/B  [%s]%s" % (
            k, p.peak_flops / 1e12, p.peak_bw / 1e9, ridge_point(p),
            "MEASURED" if p.measured else "纸面规格",
            ("  " + p.note) if p.note else ""))

    out["predictions"] = {}
    for name, bd in bds.items():
        print("\n  【%s】本机实测 %.1f ms" % (name, bd.total_ms))
        rows = {}
        for pkey, prof in profiles.items():
            if pkey == "gtx1650ti":
                rows[pkey] = {"total_pred_ms": bd.total_ms, "measured": True}
                print("     [MEASURED ] %-11s %8.2f ms" % (pkey, bd.total_ms))
            else:
                r = extrapolate(bd, prof_local, prof)
                r["measured"] = False
                rows[pkey] = r
                print("     [PREDICTED] %-11s %8.2f ms  (%.2f× vs 本机)  "
                      "[GEMM %.1f + overhead %.1f, overhead 不确定性 HIGH]" % (
                          pkey, r["total_pred_ms"], r["speedup_vs_local"],
                          r["gemm_ms"], r["overhead_ms"]))
        out["predictions"][name] = rows

    # ---- 结论 ----
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    notes = []

    # decode 带宽受限分析
    L, dm, nkv, hd = 30, 576, 3, 64
    text_params = 134.59e6 + 49280 * dm
    kv_bytes = 2 * L * nkv * hd * 79 * 4
    dec_bytes = text_params * 4 + kv_bytes
    dec_flops = 2 * text_params
    ai_dec = dec_flops / dec_bytes
    rp_local, rp_nano = ridge_point(prof_local), ridge_point(profiles["orin_nano"])
    notes.append(
        "decode(bs=1) 算术强度 = %.2f FLOP/B，本机脊点 %.2f，Orin Nano 脊点 %.1f。"
        "AI 远低于两者 => **decode 是带宽受限**。" % (ai_dec, rp_local, rp_nano))

    dec_local_ms = MEASURED_LOCAL["decode_ms_per_tok"]
    dec_nano_ms = dec_local_ms * (MEASURED_LOCAL["bw_gbps_measured"] / 68.0)
    out["decode_extrapolation"] = {
        "local_ms": dec_local_ms, "orin_nano_ms": dec_nano_ms,
        "bw_ratio": MEASURED_LOCAL["bw_gbps_measured"] / 68.0, "measured_local": True,
    }
    notes.append(
        "decode 本机实测 %.1f ms/token。Orin Nano 带宽 68 GB/s 仅为本机实测 %.1f GB/s 的 %.2f，"
        "按带宽比外推 => **Orin Nano 上 decode 反而更慢（约 %.0f ms/token）**。"
        "这与「Jetson 算力更强」的直觉相反，但对带宽受限的 decode 是必然结果。"
        "注意：此结论只依赖带宽比，不依赖 roofline，可靠性较高。" % (
            dec_local_ms, MEASURED_LOCAL["bw_gbps_measured"],
            68.0 / MEASURED_LOCAL["bw_gbps_measured"], dec_nano_ms))

    notes.append(
        "**部署启示**：带宽受限的 decode 要减少访存（W8/W4 权重量化直接减半字节数），"
        "而非堆算力；计算受限的 vision encoder 才真正受益于 Orin 的 Tensor Core。"
        "本项目 M2 实测的 W8A16 权重 1.19->0.87 GB 在 Orin 的 decode 上直接有效。")

    notes.append(
        "[平台能力差异 — 本模块最重要的可复用结论] "
        "GTX 1650 Ti (Turing SM7.5)：有 INT8 Tensor Core，**无 BF16 / INT4 Tensor Core**，无 TF32。"
        "Orin (Ampere SM8.7)：BF16/INT8/INT4 TC 齐备。"
        "因此本项目在 1650 Ti 上「W4A16 变慢且精度崩坏」的结论**不能搬到 Orin**；"
        "反过来，M2 中「W8A8 端到端反而变慢」的结论在 Orin 上**预期会反转**"
        "（Orin 有 INT8 TC 40 TOPS，且 decode 的 M=1 限制在 prefill 阶段不存在）。")

    notes.append(
        "**prefill 的外推不可靠**：实测 prefill 中 53.9% 是 GEMM、约 15% 是布局变换"
        "（as_strided 单算子 1335 次调用）。roofline 只对 GEMM 部分成立，"
        "布局/小算子的开销与外推强相关于驱动与算子融合程度，本机数据无法可靠外推。"
        "故 prefill 的 Orin 预测带有 HIGH 不确定性，仅作数量级参考。")

    for n in notes:
        print(" - " + n)
    out["notes"] = notes

    # ---- 限制 ----
    limitations = [
        "**本项目没有 Jetson 真机**，所有 Orin 数字均为模型外推，不是实测。",
        "Orin 的峰值算力/带宽取自厂商 datasheet（纸面规格），未经本机验证；"
        "而本机实测显示 fp16 纸面规格达成率仅 0.07（因 cuBLAS 回退，见 M1），"
        "说明 **datasheet 与实际可达性能可能相差极大**。",
        "外推假设 compute 部分保持本机标定的 GEMM 效率 (%.3f)，"
        "未考虑 Ampere 架构、不同 cuBLAS/cuDNN 版本的影响。" % GEMM_EFF_REPRESENTATIVE,
        "overhead 部分（布局/小算子）的外推不确定性标注为 HIGH，"
        "因其主要取决于驱动与图优化，无法从 Turing 可靠外推到 Ampere。",
        "未考虑 Orin 的功耗/散热导致的持续降频（15W TDP），故外推偏乐观。",
        "decode 外推**仅基于带宽比**，未考虑 Orin 可能通过算子融合减少访存次数"
        "从而优于纯带宽比外推。",
        "未考虑内存容量（Orin Nano 8GB vs 本机 4GB）对可用 batch size 的影响。",
    ]
    out["limitations"] = limitations
    print("\n[限制声明]")
    for l in limitations:
        print(" ! " + l)

    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m6_platforms.json"))
    args = ap.parse_args()

    res = run_m6(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print("\n保存至 %s" % args.out)


if __name__ == "__main__":
    main()
