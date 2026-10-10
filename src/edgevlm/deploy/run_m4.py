"""M4 部署链路：torch.export -> ONNX -> ONNX Runtime 图优化与性能验证.

对应 JD 职责 2：模型转换、图优化、算子融合、内存优化、性能分析。

设计（基于 M3 实测发现）：
  * M3 已证 decode 每步 ~210 个 kernel、2201 次 matmul 调用，
    其中真正的矩阵乘只占 10.45%，大量开销在 view/reshape/transpose 等布局操作。
  * 因此 M4 的核心问题是：**图优化 / 算子融合能否削减这些开销？**
  * ONNX Runtime 的 ORT_ENABLE_ALL 会做算子融合、常量折叠、布局优化。

导出对象选择（重要）：
  完整 VLM 的 `generate()` 含 Python 控制流，无法直接导出为静态图。
  故本模块导出**可静态化的子图**：
    1. vision_encoder : pixel_values -> image_features（纯前向，无控制流）
    2. connector      : vision_hidden  -> visual_embeds
    3. text_prefill   : input_ids + visual -> logits（一次前向，含 KV 输出）
  这三块覆盖了端到端的所有计算量，且都是静态可导出的，
  比强行导出 generate 更符合真实部署做法（部署时常把 decode 循环留在宿主侧）。

CUDA EP 状态（实测）：
  onnxruntime-gpu 1.19.2 的 CUDA provider DLL 在本机加载失败
  （WinError 1114，依赖齐全但初始化例程失败）。
  故本实验以 **CPU EP** 为主，并显式标注；GPU EP 结果不可得，
  这是环境限制，不作为性能结论。

用法:
    python -m edgevlm.deploy.run_m4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from edgevlm import MODELS, DEFAULT_MODEL, RESULTS_DIR, ARTIFACTS_DIR
from edgevlm.hardware import _ramp_clocks
from edgevlm.bench.baseline import CudaTimer
from edgevlm.kv.run_context_scan import load_model, build_multi_image_inputs
from edgevlm.deploy.ort_env import setup_ort_cuda_dlls, available_providers


# --------------------------------------------------------------------------
# 可导出子模块：把 VLM 各部分包装成静态图友好的 nn.Module
# --------------------------------------------------------------------------
class VisionEncoderExport(torch.nn.Module):
    """pixel_values (B*T,3,H,W) -> vision hidden (B*T, seq, 768)."""

    def __init__(self, vlm):
        super().__init__()
        self.vision_model = vlm.model.vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values).last_hidden_state


class ConnectorExport(torch.nn.Module):
    """vision hidden -> visual embeddings（含 pixel shuffle）."""

    def __init__(self, vlm):
        super().__init__()
        self.connector = vlm.model.connector

    def forward(self, hidden):
        return self.connector(hidden)


class TextPrefillExport(torch.nn.Module):
    """input_ids + visual embeds 拼接 -> logits + KV cache（单次前向）.

    为避免动态 masking 的导出复杂度，这里接受已拼好的 inputs_embeds。
    """

    def __init__(self, vlm):
        super().__init__()
        self.text_model = vlm.model.text_model
        self.lm_head = vlm.lm_head

    def forward(self, inputs_embeds):
        out = self.text_model(inputs_embeds=inputs_embeds, use_cache=True)
        logits = self.lm_head(out.last_hidden_state[:, -1:, :])
        return logits


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
def export_vision(vlm, device: str, out_path: str, tile_hw: int = 512) -> Dict:
    """导出视觉编码器到 ONNX."""
    m = VisionEncoderExport(vlm).to(device).eval()
    dummy = torch.randn(1, 3, tile_hw, tile_hw, device=device)
    info = {"module": "vision_encoder", "path": out_path, "input_shape": list(dummy.shape)}
    try:
        with torch.inference_mode():
            torch.onnx.export(
                m, (dummy,), out_path,
                input_names=["pixel_values"], output_names=["hidden"],
                opset_version=17, do_constant_folding=True,
                dynamo=False,
            )
        info["ok"] = True
        info["size_mb"] = os.path.getsize(out_path) / 1e6
    except Exception as e:
        import traceback
        info["ok"] = False
        info["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        info["traceback"] = traceback.format_exc()[-700:]
    return info


def export_connector(vlm, device: str, out_path: str, seq: int = 1024) -> Dict:
    m = ConnectorExport(vlm).to(device).eval()
    dummy = torch.randn(1, seq, 768, device=device)
    info = {"module": "connector", "path": out_path, "input_shape": list(dummy.shape)}
    try:
        with torch.inference_mode():
            torch.onnx.export(
                m, (dummy,), out_path,
                input_names=["hidden"], output_names=["visual_embeds"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        info["ok"] = True
        info["size_mb"] = os.path.getsize(out_path) / 1e6
    except Exception as e:
        import traceback
        info["ok"] = False
        info["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        info["traceback"] = traceback.format_exc()[-700:]
    return info


def export_text_prefill(vlm, device: str, out_path: str, seq: int = 79,
                        hidden: int = 576) -> Dict:
    m = TextPrefillExport(vlm).to(device).eval()
    dummy = torch.randn(1, seq, hidden, device=device)
    info = {"module": "text_prefill", "path": out_path, "input_shape": list(dummy.shape)}
    try:
        with torch.inference_mode():
            torch.onnx.export(
                m, (dummy,), out_path,
                input_names=["inputs_embeds"], output_names=["logits"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        info["ok"] = True
        info["size_mb"] = os.path.getsize(out_path) / 1e6
    except Exception as e:
        import traceback
        info["ok"] = False
        info["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        info["traceback"] = traceback.format_exc()[-700:]
    return info


# --------------------------------------------------------------------------
# 数值一致性 + 性能对比
# --------------------------------------------------------------------------
def onnx_graph_info(path: str) -> Dict:
    """统计 ONNX 图的节点数与算子类型分布（图优化前后对比用）."""
    try:
        import onnx
        m = onnx.load(path)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        return {"num_nodes": len(m.graph.node), "op_types": ops,
                "num_initializers": len(m.graph.initializer)}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:150])}


def bench_torch(fn, device: str, iters: int = 20, warmup: int = 5) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_ort(sess, feeds: Dict[str, np.ndarray], iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feeds)
    return (time.perf_counter() - t0) / iters


def compare_vision(vlm, device: str, onnx_path: str, tile_hw: int = 512) -> Dict:
    """视觉编码器：torch vs ORT，含数值一致性与延迟."""
    import onnxruntime as ort
    res = {"module": "vision_encoder"}
    m = VisionEncoderExport(vlm).to(device).eval()
    dummy = torch.randn(1, 3, tile_hw, tile_hw, device=device)
    with torch.inference_mode():
        y_torch = m(dummy)
        t_torch = bench_torch(lambda: m(dummy), device)
    res["torch_ms"] = t_torch * 1000

    feeds = {"pixel_values": dummy.detach().cpu().numpy()}
    for prov_name, prefer in [("CPU", False)]:
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            y_ort = sess.run(None, feeds)[0]
            t_ort = bench_ort(sess, feeds)
            res["ort_%s_ms" % prov_name.lower()] = t_ort * 1000
            diff = np.abs(y_ort - y_torch.detach().cpu().numpy())
            denom = np.abs(y_torch.detach().cpu().numpy()).mean() + 1e-9
            res["ort_%s_rel_err" % prov_name.lower()] = float(diff.mean() / denom)
            res["ort_%s_max_err" % prov_name.lower()] = float(diff.max())
        except Exception as e:
            res["ort_%s_error" % prov_name.lower()] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return res


def compare_text_prefill(vlm, device: str, onnx_path: str, seq: int = 79,
                         hidden: int = 576) -> Dict:
    import onnxruntime as ort
    res = {"module": "text_prefill"}
    m = TextPrefillExport(vlm).to(device).eval()
    dummy = torch.randn(1, seq, hidden, device=device)
    with torch.inference_mode():
        y_torch = m(dummy)
        t_torch = bench_torch(lambda: m(dummy), device, iters=10, warmup=3)
    res["torch_ms"] = t_torch * 1000

    feeds = {"inputs_embeds": dummy.detach().cpu().numpy()}
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, feeds)[0]
        t_ort = bench_ort(sess, feeds, iters=10, warmup=3)
        res["ort_cpu_ms"] = t_ort * 1000
        yt = y_torch.detach().cpu().numpy()
        res["ort_cpu_rel_err"] = float(np.abs(y_ort - yt).mean() / (np.abs(yt).mean() + 1e-9))
        res["ort_cpu_max_err"] = float(np.abs(y_ort - yt).max())
        res["logits_argmax_match"] = bool(
            int(np.argmax(y_ort[0, -1])) == int(np.argmax(yt[0, -1])))
    except Exception as e:
        res["ort_cpu_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m4_deploy.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 3.0)

    onnx_dir = os.path.join(ARTIFACTS_DIR, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    print("=" * 100)
    print("M4 部署链路：torch -> ONNX -> ORT | %s" % MODELS[args.model]["repo"])
    print("=" * 100)

    providers = available_providers(verbose=True)

    proc, vlm = load_model(args.model)
    vlm = vlm.to(args.device).eval()

    results = {"providers_available": providers, "exports": [], "comparisons": []}

    # ---- 导出 ----
    print("\n[1] ONNX 导出")
    exports = [
        ("vision", lambda p: export_vision(vlm, args.device, p)),
        ("connector", lambda p: export_connector(vlm, args.device, p)),
        ("text_prefill", lambda p: export_text_prefill(vlm, args.device, p)),
    ]
    paths = {}
    for name, fn in exports:
        p = os.path.join(onnx_dir, "%s.onnx" % name)
        info = fn(p)
        if info.get("ok"):
            gi = onnx_graph_info(p)
            info.update(gi)
            paths[name] = p
            print("  %-14s OK  size=%7.2f MB  nodes=%s  ops=%d 种" % (
                name, info["size_mb"], gi.get("num_nodes"), len(gi.get("op_types", {}))))
            top = sorted(gi.get("op_types", {}).items(), key=lambda x: -x[1])[:6]
            print("       top ops: %s" % ", ".join("%s×%d" % (k, v) for k, v in top))
        else:
            print("  %-14s FAILED: %s" % (name, info.get("error", "")[:90]))
        results["exports"].append(info)

    # ---- 数值一致性 + 性能 ----
    print("\n[2] 数值一致性 + torch vs ORT 性能 (CPU EP)")
    if "vision" in paths:
        c = compare_vision(vlm, args.device, paths["vision"])
        results["comparisons"].append(c)
        print("  vision_encoder : torch(GPU)=%7.2f ms | ORT(CPU)=%8.2f ms | rel_err=%.2e" % (
            c.get("torch_ms", float("nan")), c.get("ort_cpu_ms", float("nan")),
            c.get("ort_cpu_rel_err", float("nan"))))
    if "text_prefill" in paths:
        c = compare_text_prefill(vlm, args.device, paths["text_prefill"])
        results["comparisons"].append(c)
        print("  text_prefill   : torch(GPU)=%7.2f ms | ORT(CPU)=%8.2f ms | rel_err=%.2e | argmax_match=%s" % (
            c.get("torch_ms", float("nan")), c.get("ort_cpu_ms", float("nan")),
            c.get("ort_cpu_rel_err", float("nan")), c.get("logits_argmax_match")))
    if "connector" in paths:
        c = {"module": "connector", "note": "见 exports"}
        results["comparisons"].append(c)

    # ---- ORT 图优化级别对比 ----
    print("\n[3] ORT 图优化级别对节点数/延迟的影响 (vision_encoder)")
    if "vision" in paths:
        import onnxruntime as ort
        levels = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        m = VisionEncoderExport(vlm).to(args.device).eval()
        dummy = torch.randn(1, 3, 512, 512, device=args.device)
        feeds = {"pixel_values": dummy.detach().cpu().numpy()}
        with torch.inference_mode():
            ref = m(dummy).detach().cpu().numpy()
        opt_res = []
        for lname, lvl in levels.items():
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = lvl
                sess = ort.InferenceSession(paths["vision"], so,
                                            providers=["CPUExecutionProvider"])
                y = sess.run(None, feeds)[0]
                t = bench_ort(sess, feeds, iters=10, warmup=3)
                rel = float(np.abs(y - ref).mean() / (np.abs(ref).mean() + 1e-9))
                opt_res.append({"level": lname, "ms": t * 1000, "rel_err": rel})
                print("  %-22s %8.2f ms   rel_err=%.2e" % (lname, t * 1000, rel))
            except Exception as e:
                print("  %-22s FAILED %s" % (lname, str(e)[:60]))
        results["optimization_levels"] = opt_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n保存至 %s" % args.out)
    print("ONNX 模型保存在 %s" % onnx_dir)


if __name__ == "__main__":
    main()
