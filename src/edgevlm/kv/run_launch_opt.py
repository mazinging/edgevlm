"""M3-B：验证「减少 kernel 启动开销」是最有效的 decode 优化.

基于 M3-A 发现：decode 44ms/tok，其中 ~46.7ms 可由 210 个 kernel 的
启动开销（0.222ms/kernel）解释 -> 瓶颈是 launch overhead，不是算力/带宽。

待验证的优化手段：
  T1. torch.compile（inductor 可做算子融合，减少 kernel 数）
  T2. CUDA Graph 捕获 decode 步（消除重复 launch）
  T3. 模型级 batch（M 变大，摊薄启动开销）—— M3-A 已证吞吐 ×4.4

本脚本重点验证 T1/T2，因为它们直接针对 launch overhead。

用法:
    python -m edgevlm.kv.run_launch_opt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR
from edgevlm.hardware import _ramp_clocks
from edgevlm.bench.baseline import CudaTimer, reset_peak, peak_gb
from edgevlm.kv.run_context_scan import load_model, build_multi_image_inputs


def time_generate(model, inputs, device, max_new_tokens=16, warmup=2, iters=3):
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        ts = []
        out = None
        for _ in range(iters):
            with CudaTimer(device) as t:
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            ts.append(t.dt)
    ts.sort()
    med = ts[len(ts) // 2]
    n_out = out.shape[1] - n_in
    return {"latency_s": med, "tok_per_s": n_out / med, "out_tokens": int(n_out),
            "min_s": ts[0], "max_s": ts[-1]}


def profile_kernel_count(model, inputs, device) -> Dict:
    """用 torch profiler 统计 decode 单步的 kernel 数量与耗时分布."""
    info = {}
    try:
        from torch.profiler import profile, ProfilerActivity
        inputs = inputs.to(device)
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                torch.cuda.synchronize()
        evts = prof.key_averages()
        cuda_evts = [e for e in evts if e.device_type is not None and "cuda" in str(e.device_type).lower()]
        total_cuda = sum(e.self_device_time_total for e in evts)
        info["num_events"] = len(evts)
        info["total_cuda_us"] = total_cuda
        # 按 self time 排序的 top kernel
        top = sorted(evts, key=lambda e: e.self_device_time_total, reverse=True)[:12]
        info["top_kernels"] = [
            {"name": e.key[:70], "calls": e.count,
             "cuda_us": round(e.self_device_time_total, 1),
             "pct": round(100 * e.self_device_time_total / max(total_cuda, 1e-9), 2)}
            for e in top
        ]
        info["supported"] = True
    except Exception as e:
        import traceback
        info["supported"] = False
        info["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        info["traceback"] = traceback.format_exc()[-500:]
    return info


def try_compile(model, device) -> Optional[object]:
    """T1: torch.compile."""
    try:
        m = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        return m
    except Exception as e:
        print("    torch.compile 失败: %s" % str(e)[:120])
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m3_launch_opt.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 4.0)

    proc, model = load_model(args.model)
    model = model.to(args.device).eval()
    inputs = build_multi_image_inputs(proc, 1, 512, 512).to(args.device)

    print("=" * 100)
    print("M3-B 减少 kernel 启动开销 | %s | %s" % (MODELS[args.model]["repo"], args.device))
    print("=" * 100)

    results: Dict[str, dict] = {}

    # --- 基线 ---
    print("\n[1] 基线 (eager, fp32)")
    base = time_generate(model, inputs, args.device, args.max_new_tokens)
    results["eager"] = base
    print("    %.2f tok/s  (%.3fs)" % (base["tok_per_s"], base["latency_s"]))

    # --- kernel 剖析 ---
    print("\n[2] kernel 剖析 (profiler)")
    kp = profile_kernel_count(model, inputs, args.device)
    results["kernel_profile"] = kp
    if kp.get("supported"):
        print("    事件数: %s   CUDA 总耗时: %.1f ms" % (
            kp["num_events"], kp["total_cuda_us"] / 1000))
        print("    %-52s %6s %10s %7s" % ("kernel", "calls", "cuda_us", "pct"))
        for k in kp["top_kernels"]:
            print("    %-52s %6d %10.1f %6.2f%%" % (
                k["name"], k["calls"], k["cuda_us"], k["pct"]))
    else:
        print("    失败: %s" % kp.get("error", "")[:100])

    # --- T1: torch.compile ---
    print("\n[3] T1: torch.compile (mode=reduce-overhead)")
    try:
        cmodel = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        # 触发编译
        with torch.inference_mode():
            _ = cmodel.generate(**inputs, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()
        comp = time_generate(cmodel, inputs, args.device, args.max_new_tokens,
                             warmup=3, iters=3)
        results["torch_compile"] = comp
        print("    %.2f tok/s   加速 %.2fx  (编译耗时已排除)" % (
            comp["tok_per_s"], comp["tok_per_s"] / base["tok_per_s"]))
    except Exception as e:
        import traceback
        results["torch_compile"] = {"supported": False,
                                    "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                                    "traceback": traceback.format_exc()[-800:]}
        print("    失败: %s" % str(e)[:150])

    # --- T3: batch ---
    print("\n[4] T3: 批处理（摊薄启动开销）")
    batch_res = []
    for bs in [1, 2, 4, 8]:
        try:
            binputs = {}
            for k, v in inputs.items():
                binputs[k] = v.repeat(bs, *([1] * (v.dim() - 1))) if v.dim() >= 1 else v
            binputs = {k: v.to(args.device) for k, v in binputs.items()}
            r = time_generate(model, binputs, args.device, args.max_new_tokens,
                              warmup=1, iters=2)
            r["batch"] = bs
            r["total_tok_per_s"] = r["tok_per_s"] * bs
            batch_res.append(r)
            print("    bs=%-2d  单流 %6.2f tok/s   总吞吐 %7.2f tok/s   加速 %.2fx" % (
                bs, r["tok_per_s"], r["total_tok_per_s"],
                r["total_tok_per_s"] / base["tok_per_s"]))
            del binputs
            torch.cuda.empty_cache()
        except Exception as e:
            print("    bs=%-2d 失败: %s" % (bs, str(e)[:80]))
    results["batched"] = batch_res

    payload = {"config": {"model": args.model, "device": args.device,
                          "max_new_tokens": args.max_new_tokens},
               "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n保存至 %s" % args.out)


if __name__ == "__main__":
    main()
