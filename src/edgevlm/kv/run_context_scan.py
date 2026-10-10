"""M3 实验 A：上下文/多图/分辨率 对序列长度与延迟的影响（KV Cache 优化的前提）.

要回答：
  Q1. 多图 / 高分辨率下，视觉 token 数与序列长度如何增长？
  Q2. decode 阶段延迟是否随上下文长度线性增长（KV 读取是 memory-bound）？
  Q3. KV Cache 实际占用多少显存？GQA(3 heads) 压缩后还剩多少优化空间？

这些是「KV Cache 量化是否值得做」的前置事实。

用法:
    python -m edgevlm.kv.run_context_scan
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR
from edgevlm.hardware import _ramp_clocks
from edgevlm.bench.baseline import CudaTimer, reset_peak, peak_gb


def load_model(model_key: str, dtype=torch.float32, attn: str = "eager"):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    repo = MODELS[model_key]["repo"]
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForVision2Seq.from_pretrained(
        repo, torch_dtype=dtype, _attn_implementation=attn)
    return proc, model


def build_multi_image_inputs(proc, n_images: int, image_size: int,
                             max_image_size: int, prompt: str = "Describe these images."):
    """构造 n_images 张图的输入."""
    from PIL import Image
    proc.image_processor.size = {"longest_edge": max_image_size}
    proc.image_processor.max_image_size = {"longest_edge": max_image_size}
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8))
            for _ in range(n_images)]
    content = []
    for _ in range(n_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    return proc(text=text, images=imgs, return_tensors="pt")


def kv_cache_bytes(model, seq_len: int, batch: int = 1, dtype_bytes: int = 4) -> Dict:
    """解析计算 KV Cache 大小 —— 不实测，直接由架构推导.

    SmolVLM text_config: 30 层, 3 KV heads, head_dim 64
    KV bytes = 2(K,V) * layers * kv_heads * head_dim * seq * batch * dtype_bytes
    """
    cfg = model.config.text_config
    layers = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    total = per_token * seq_len * batch
    return {
        "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
        "bytes_per_token": per_token,
        "total_bytes": total,
        "total_mb": total / 1e6,
    }


def measure_decode_scaling(proc, model, device: str, n_images_list: List[int],
                           image_size: int, max_new_tokens: int = 16) -> List[Dict]:
    """测量不同上下文长度下的 prefill / decode 延迟."""
    results = []
    for n_img in n_images_list:
        rec = {"n_images": n_img, "image_size": image_size}
        try:
            inputs = build_multi_image_inputs(proc, n_img, image_size, image_size)
            inputs = inputs.to(device)
            pv = inputs["pixel_values"]
            n_tiles = pv.shape[1] if pv.dim() == 5 else 1
            rec["n_tiles"] = int(n_tiles)
            rec["seq_len"] = int(inputs["input_ids"].shape[1])

            kv = kv_cache_bytes(model, rec["seq_len"])
            rec["kv_mb"] = kv["total_mb"]
            rec["kv_bytes_per_token"] = kv["bytes_per_token"]

            with torch.inference_mode():
                # 预热
                _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
                if device == "cuda":
                    torch.cuda.synchronize()

                # prefill 计时
                reset_peak(device)
                with CudaTimer(device) as t:
                    out = model(**inputs, use_cache=True)
                rec["prefill_ms"] = t.dt * 1000
                rec["prefill_peak_gb"] = peak_gb(device)

                # 端到端
                torch.cuda.reset_peak_memory_stats()
                with CudaTimer(device) as t:
                    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False)
                rec["e2e_ms"] = t.dt * 1000
                rec["e2e_peak_gb"] = peak_gb(device)
                n_out = gen.shape[1] - rec["seq_len"]
                rec["out_tokens"] = int(n_out)
                rec["decode_ms_per_tok"] = max(
                    rec["e2e_ms"] - rec["prefill_ms"], 0) / max(n_out, 1)
                rec["tok_per_s"] = n_out / (rec["e2e_ms"] / 1000)
            rec["supported"] = True
        except Exception as e:
            rec["supported"] = False
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:150])
            if device == "cuda":
                torch.cuda.empty_cache()
        results.append(rec)
        if rec["supported"]:
            print("  %2d img  tiles=%-3s seq=%-5d  prefill=%7.1fms  decode=%6.1fms/tok  "
                  "e2e=%6.2fs  kv=%6.2fMB  peak=%5.2fGB" % (
                      n_img, rec["n_tiles"], rec["seq_len"], rec["prefill_ms"],
                      rec["decode_ms_per_tok"], rec["e2e_ms"]/1000,
                      rec["kv_mb"], rec["e2e_peak_gb"]))
        else:
            print("  %2d img  FAILED: %s" % (n_img, rec["error"][:80]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m3_context_scan.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 4.0)

    proc, model = load_model(args.model)
    model = model.to(args.device).eval()

    print("=" * 108)
    print("M3-A 上下文/多图 缩放扫描 | %s | %s" % (MODELS[args.model]["repo"], args.device))
    cfg = model.config.text_config
    print("text_config: %d 层, %d KV heads, head_dim=%d (GQA: %d Q / %d KV)" % (
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
        cfg.num_attention_heads, cfg.num_key_value_heads))
    print("=" * 108)

    n_images_list = [1, 2, 4, 8]
    res = measure_decode_scaling(proc, model, args.device, n_images_list,
                                 args.image_size, args.max_new_tokens)

    payload = {"config": {"model": args.model, "device": args.device,
                          "image_size": args.image_size,
                          "max_new_tokens": args.max_new_tokens},
               "text_config": {"layers": cfg.num_hidden_layers,
                               "kv_heads": cfg.num_key_value_heads,
                               "q_heads": cfg.num_attention_heads,
                               "head_dim": cfg.head_dim},
               "results": res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n保存至 %s" % args.out)

    ok = [r for r in res if r.get("supported")]
    if len(ok) >= 2:
        print("\n=== 缩放趋势 ===")
        for r in ok:
            print("  seq=%-5d  prefill=%7.1fms  decode=%6.2fms/tok  kv=%6.2fMB" % (
                r["seq_len"], r["prefill_ms"], r["decode_ms_per_tok"], r["kv_mb"]))
        if len(ok) >= 2:
            dl = ok[-1]["decode_ms_per_tok"] / max(ok[0]["decode_ms_per_tok"], 1e-6)
            sl = ok[-1]["seq_len"] / max(ok[0]["seq_len"], 1)
            print("\n  序列长度 x%.1f  ->  decode 延迟 x%.2f" % (sl, dl))
            print("  (若接近线性 => decode 受 KV 访存主导，KV 优化有价值)")


if __name__ == "__main__":
    main()
