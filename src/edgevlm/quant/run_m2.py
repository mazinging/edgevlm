"""M2 实验：量化方案的精度 / 显存 / 时延 Pareto 分析.

关键问题：
  Q1. INT8 的 4x matmul 加速能否传导到端到端推理？
      （M1 显示 decode 占 69.5%，而 decode 是 memory-bound 的逐 token 矩阵向量乘，
       理论上 INT8 主要省带宽而非算力 -> 加速可能远小于 4x）
  Q2. W4A16 在无 INT4 TC 的机器上是否真的变慢？
  Q3. 量化哪个模块最划算？

用法:
    python -m edgevlm.quant.run_m2 --out results/m2_quant.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR
from edgevlm.hardware import _ramp_clocks
from edgevlm.bench.baseline import build_inputs, CudaTimer, reset_peak, peak_gb
from edgevlm.quant.model_quant import (
    apply_quantization, measure_model_weight_bytes, count_params, SCOPES,
)


def load_fp_model(model_key: str, dtype: torch.dtype = torch.float32, attn: str = "eager"):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    repo = MODELS[model_key]["repo"]
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForVision2Seq.from_pretrained(
        repo, torch_dtype=dtype, _attn_implementation=attn)
    return proc, model


def gen_text(proc, model, inputs, device: str, max_new_tokens: int = 24) -> str:
    inputs = inputs.to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    return proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


def bench_latency(model, inputs, device: str, max_new_tokens: int = 24,
                  warmup: int = 2, iters: int = 3) -> Dict:
    """端到端 generate 计时（多次取中位数，抗抖动）."""
    inputs = inputs.to(device)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        out = None
        for _ in range(iters):
            with CudaTimer(device) as t:
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            times.append(t.dt)
    times.sort()
    med = times[len(times) // 2]
    n_out = out.shape[1] - n_in
    return {
        "latency_s": med,
        "latency_min_s": times[0],
        "latency_max_s": times[-1],
        "out_tokens": int(n_out),
        "tok_per_s": n_out / med,
    }


def build_configs() -> List[Dict]:
    """待测配置清单 —— 每组回答一个具体问题."""
    return [
        # 基线
        {"name": "fp32 (baseline)", "scheme": None, "scope": None, "dtype": "float32"},
        # W8A8：验证 INT8 TC 是否传导到端到端
        {"name": "W8A8 / text", "scheme": "w8a8", "scope": "text", "dtype": "float32"},
        {"name": "W8A8 / text+vision", "scheme": "w8a8", "scope": "all_no_lmhead", "dtype": "float32"},
        {"name": "W8A8 / all", "scheme": "w8a8", "scope": "all", "dtype": "float32"},
        # W4A16：验证无 INT4 TC 时的表现
        {"name": "W4A16 / text", "scheme": "w4a16", "scope": "text", "dtype": "float32"},
        # W8A16：weight-only，与 W8A8 对比激活量化的代价
        {"name": "W8A16 / text", "scheme": "w8a16", "scope": "text", "dtype": "float32"},
    ]


def run_one(cfg: Dict, model_key: str, device: str, max_new_tokens: int,
            image_size: int, ref_text: Optional[str], proc) -> Dict:
    rec = {"config": cfg["name"], "scheme": cfg["scheme"], "scope": cfg["scope"]}
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[cfg["dtype"]]
    try:
        _, model = load_fp_model(model_key, dt, "eager")
        model = model.to(device).eval()

        rec["params_m"] = count_params(model) / 1e6
        rec["weight_bytes_fp"] = measure_model_weight_bytes(model)

        if cfg["scheme"]:
            scope = cfg["scope"]
            if scope == "all_no_lmhead":
                # text + vision，但不含 lm_head（词表矩阵对精度敏感）
                info = {"modules": {}, "total_replaced": 0}
                for s in ["text", "vision"]:
                    sub = apply_quantization(model, cfg["scheme"], s)
                    info["total_replaced"] += sub["total_replaced"]
                    info["modules"].update(sub["modules"])
                rec["quant_info"] = info
            else:
                rec["quant_info"] = apply_quantization(model, cfg["scheme"], scope)

            model = model.to(device).eval()
            rec["weight_bytes_quant"] = measure_model_weight_bytes(model)
            rec["weight_compress"] = (rec["weight_bytes_fp"] /
                                      max(rec["weight_bytes_quant"], 1))

        inputs = build_inputs(proc, image_size=image_size)

        reset_peak(device)
        lat = bench_latency(model, inputs, device, max_new_tokens)
        rec.update(lat)
        rec["peak_gb"] = peak_gb(device)

        txt = gen_text(proc, model, inputs, device, max_new_tokens)
        rec["text"] = txt[:150]
        if ref_text is not None:
            rec["text_match"] = (txt.strip() == ref_text.strip())
            rec["text_prefix_match"] = (txt[:30] == ref_text[:30])
        rec["supported"] = True

    except Exception as e:
        import traceback
        rec["supported"] = False
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        rec["traceback"] = traceback.format_exc()[-600:]
    finally:
        try:
            del model
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m2_quant.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 4.0)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODELS[args.model]["repo"])
    proc.image_processor.size = {"longest_edge": args.image_size}
    proc.image_processor.max_image_size = {"longest_edge": args.image_size}

    configs = build_configs()
    print("=" * 108)
    print("M2 量化 Pareto 分析 | %s | %s | %d tokens" % (
        MODELS[args.model]["repo"], args.device, args.max_new_tokens))
    print("=" * 108)

    results = []
    ref_text = None
    for cfg in configs:
        r = run_one(cfg, args.model, args.device, args.max_new_tokens,
                    args.image_size, ref_text, proc)
        if cfg["scheme"] is None and r.get("supported"):
            ref_text = r.get("text", "")
        results.append(r)
        if r.get("supported"):
            print("  %-22s lat=%6.2fs  %6.2f tok/s  peak=%5.2fGB  w=%5.1fMB  match=%s" % (
                cfg["name"], r["latency_s"], r["tok_per_s"], r["peak_gb"],
                r.get("weight_bytes_quant", r.get("weight_bytes_fp", 0)) / 1e6,
                r.get("text_match", "-")))
        else:
            print("  %-22s FAILED: %s" % (cfg["name"], r.get("error", "")[:70]))

    payload = {"config": {"model": args.model, "device": args.device,
                          "max_new_tokens": args.max_new_tokens,
                          "image_size": args.image_size},
               "reference_text": ref_text, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    ok = [r for r in results if r.get("supported")]
    if ok:
        base = next((r for r in ok if r["scheme"] is None), ok[0])
        print("\n" + "=" * 108)
        print("%-22s %9s %9s %9s %9s %9s" % (
            "config", "tok/s", "speedup", "peak_GB", "weights_MB", "compress"))
        print("-" * 108)
        for r in ok:
            print("%-22s %9.2f %8.2fx %9.2f %9.1f %8.2fx" % (
                r["config"], r["tok_per_s"], r["tok_per_s"] / base["tok_per_s"],
                r["peak_gb"],
                r.get("weight_bytes_quant", r.get("weight_bytes_fp", 0)) / 1e6,
                r.get("weight_compress", 1.0)))
    print("\n保存至 %s" % args.out)


if __name__ == "__main__":
    main()
