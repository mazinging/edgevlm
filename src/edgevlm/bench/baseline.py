"""M1 基线剖析：端到端延迟分解 + 精度/attn 后端对比 + 算子级 profiling.

用法:
    python -m edgevlm.bench.baseline --model smolvlm-256m --out results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import torch

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR, ARTIFACTS_DIR
from edgevlm.hardware import build_capability_matrix, _ramp_clocks


# --------------------------------------------------------------------------
# 计时工具
# --------------------------------------------------------------------------
class CudaTimer:
    """CUDA 同步计时器（支持 CPU 回退）."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0
        return False


def reset_peak(device: str = "cuda") -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_gb(device: str = "cuda") -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0


# --------------------------------------------------------------------------
# 输入构造
# --------------------------------------------------------------------------
def make_image(size: int = 512, seed: int = 0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def build_inputs(proc, image_size: int = 512, prompt_text: str = "Describe this image.",
                 max_image_size: int = 512, n_images: int = 1, seed: int = 0):
    """构造多模态输入.

    注意 workaround：SmolVLM 上游 preprocessor_config 中 size.longest_edge(2048)
    与 max_image_size.longest_edge(512) 冲突，必须对齐，否则断言失败。
    """
    proc.image_processor.size = {"longest_edge": max_image_size}
    proc.image_processor.max_image_size = {"longest_edge": max_image_size}

    imgs = [make_image(image_size, seed + i) for i in range(n_images)]
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    return proc(text=prompt, images=imgs, return_tensors="pt")


# --------------------------------------------------------------------------
# 模型加载
# --------------------------------------------------------------------------
def load_model(model_key: str, dtype: torch.dtype, attn: str, device: str):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    repo = MODELS[model_key]["repo"]
    proc = AutoProcessor.from_pretrained(repo)
    kwargs = {"torch_dtype": dtype}
    if attn:
        kwargs["_attn_implementation"] = attn
    model = AutoModelForVision2Seq.from_pretrained(repo, **kwargs).to(device).eval()
    return proc, model


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# --------------------------------------------------------------------------
# 阶段分解：视觉塔 / 预填充 / 解码
# --------------------------------------------------------------------------
def profile_stages(proc, model, inputs, device: str, max_new_tokens: int = 24) -> Dict:
    """分离测量 视觉编码 / prefill / decode 三阶段耗时.

    这回答「优化应该投在哪里」—— 如果视觉塔占 60%，
    那么 LLM 侧的 KV cache 优化收益上限就只有 40%。

    API 要点（实测确认）：
      * processor 输出的 pixel_values 是 5D (B, n_tiles, C, H, W)，
        因 do_image_splitting 开启；vision_model 需要 4D，须把 tiles 展平进 batch。
      * connector 返回 Tensor（pixel shuffle 后），不是 BaseModelOutput。
    """
    inputs = inputs.to(device)
    dtype = next(model.parameters()).dtype
    cast = (lambda v: v.to(dtype)) if dtype != torch.float32 else (lambda v: v.float())
    inputs = {k: (cast(v) if v.is_floating_point() else v) for k, v in inputs.items()}

    out: Dict[str, float] = {}

    # 展平 tiles: (B, n_tiles, C, H, W) -> (B*n_tiles, C, H, W)
    pv = inputs["pixel_values"]
    if pv.dim() == 5:
        b, n_tiles, c, h, w = pv.shape
        pv_flat = pv.reshape(b * n_tiles, c, h, w)
    else:
        n_tiles = 1
        pv_flat = pv
    out["n_tiles"] = int(n_tiles)
    out["tile_hw"] = int(pv_flat.shape[-1])

    vm = model.model.vision_model
    conn = model.model.connector
    text_model = model.model.text_model

    with torch.inference_mode():
        # --- 1) 视觉塔 ---
        for _ in range(2):
            vis = vm(pixel_values=pv_flat)
        if device == "cuda":
            torch.cuda.synchronize()
        with CudaTimer(device) as t:
            for _ in range(3):
                vis = vm(pixel_values=pv_flat)
        out["vision_ms"] = t.dt / 3 * 1000
        vis_hidden = vis.last_hidden_state

        # --- 2) connector ---
        with CudaTimer(device) as t:
            for _ in range(3):
                c_out = conn(vis_hidden)
        out["connector_ms"] = t.dt / 3 * 1000
        out["visual_tokens"] = int(c_out.shape[1])

        # --- 3) 语言塔 prefill ---
        n_in = inputs["input_ids"].shape[1]
        out["prefill_in_tokens"] = int(n_in)
        with CudaTimer(device) as t:
            for _ in range(3):
                pf = model(**inputs, use_cache=True)
        out["prefill_ms"] = t.dt / 3 * 1000

        # 纯语言塔 prefill（用等长输入，隔离视觉输入开销）
        try:
            emb = text_model.get_input_embeddings()(inputs["input_ids"])
            emb = cast(emb)
            with CudaTimer(device) as t:
                for _ in range(3):
                    _ = text_model(inputs_embeds=emb, use_cache=True)
            out["lm_prefill_ms"] = t.dt / 3 * 1000
        except Exception as e:
            out["lm_prefill_ms"] = float("nan")
            out["lm_prefill_err"] = str(e)[:80]

        # --- 4) decode：端到端 generate ---
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        with CudaTimer(device) as t:
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out["e2e_ms"] = t.dt * 1000
        n_out = gen.shape[1] - n_in
        out["out_tokens"] = int(n_out)
        out["tok_per_s"] = n_out / t.dt
        # decode 单 token = (总时间 - prefill - vision) / 新 token 数
        overhead = out["vision_ms"] + out["prefill_ms"]
        out["decode_ms_per_tok"] = max(t.dt * 1000 - overhead, 0.0) / max(n_out, 1)

        text = proc.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0]
        out["text"] = text[:120]
    return out


# --------------------------------------------------------------------------
# 精度 × attention 后端 扫描
# --------------------------------------------------------------------------
def sweep_configs(model_key: str, device: str, max_new_tokens: int = 24,
                  image_size: int = 512) -> List[Dict]:
    results = []
    for dname, dtype in DTYPES.items():
        for attn in ["eager", "sdpa"]:
            rec = {"model": model_key, "dtype": dname, "attn": attn, "device": device}
            try:
                proc, model = load_model(model_key, dtype, attn, device)
            except Exception as e:
                # 已知：Idefics3VisionTransformer 不支持 sdpa，
                # 这是有效发现（视觉塔无法用 SDPA 加速），不作为失败静默丢弃。
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])
                rec["supported"] = False
                results.append(rec)
                continue
            try:
                reset_peak(device)
                inputs = build_inputs(proc, image_size=image_size)
                st = profile_stages(proc, model, inputs, device, max_new_tokens)
                rec.update(st)
                rec["peak_gb"] = peak_gb(device)
                rec["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
                rec["supported"] = True
            except Exception as e:
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])
                rec["supported"] = False
            finally:
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            results.append(rec)
    return results


def param_breakdown(model) -> Dict[str, float]:
    """参数量分布 — 决定量化该往哪投."""
    def n(m):
        return sum(p.numel() for p in m.parameters()) / 1e6
    d = {
        "total": n(model),
        "vision_model": n(model.model.vision_model),
        "connector": n(model.model.connector),
        "text_model": n(model.model.text_model),
    }
    try:
        d["lm_head"] = n(model.lm_head)
    except Exception:
        d["lm_head"] = float("nan")
    d["other"] = d["total"] - d["vision_model"] - d["connector"] - d["text_model"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "baseline.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 4.0)

    caps = build_capability_matrix()
    print("=" * 96)
    print("M1 基线剖析 | %s | %s" % (MODELS[args.model]["repo"], args.device))
    g = caps.gpu or {}
    print("GPU: %s  SM %s  %.1fGB" % (g.get("name"), g.get("compute_cap"), g.get("total_mem_gb", 0)))
    print("BF16-TC=%s  INT8-TC=%s  INT4-TC=%s" % (
        caps.capabilities["bf16_tensor_core"], caps.capabilities["int8_tensor_core"],
        caps.capabilities["int4_tensor_core"]))
    print("=" * 96)

    res = sweep_configs(args.model, args.device, args.max_new_tokens, args.image_size)

    hdr = "%-6s %-6s %9s %9s %9s %9s %8s %8s" % (
        "dtype", "attn", "vision_ms", "prefill", "decode/t", "tok/s", "peak_GB", "status")
    print(hdr)
    print("-" * 96)
    for r in res:
        if not r.get("supported"):
            print("%-6s %-6s %s" % (r["dtype"], r["attn"], "UNSUPPORTED: " + r["error"][:62]))
        else:
            print("%-6s %-6s %9.2f %9.2f %9.2f %9.2f %8.2f %8s" % (
                r["dtype"], r["attn"], r.get("vision_ms", float("nan")),
                r.get("prefill_ms", float("nan")), r.get("decode_ms_per_tok", float("nan")),
                r.get("tok_per_s", float("nan")), r.get("peak_gb", float("nan")), "ok"))

    payload = {"capability_matrix": caps.to_dict(), "results": res,
               "config": {"model": args.model, "device": args.device,
                          "max_new_tokens": args.max_new_tokens, "image_size": args.image_size}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n保存至 %s" % args.out)

    # 阶段占比（取最快配置）
    ok = [r for r in res if r.get("supported")]
    if ok:
        best = max(ok, key=lambda r: r.get("tok_per_s", 0))
        print("\n=== 最快配置: %s/%s  (%.2f tok/s) ===" % (
            best["dtype"], best["attn"], best["tok_per_s"]))
        vis, pf = best["vision_ms"], best["prefill_ms"]
        dec = best["decode_ms_per_tok"] * best["out_tokens"]
        tot = vis + pf + dec
        print("  %-10s %8.2f ms  %5.1f%%" % ("vision", vis, 100*vis/tot))
        print("  %-10s %8.2f ms  %5.1f%%" % ("prefill", pf, 100*pf/tot))
        print("  %-10s %8.2f ms  %5.1f%%  (%.2f ms/token x %d)" % (
            "decode", dec, 100*dec/tot, best["decode_ms_per_tok"], best["out_tokens"]))
        print("  %-10s %8.2f ms" % ("total", tot))
        print("  视觉 token 数: %s (n_tiles=%s)" % (
            best.get("visual_tokens"), best.get("n_tiles")))


if __name__ == "__main__":
    main()
