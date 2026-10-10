"""ONNX Runtime 环境修复 —— 让 CUDA Execution Provider 真正可用.

问题（实测）：
  `pip install onnxruntime-gpu` 后 `ort.get_available_providers()` 只有
  ['AzureExecutionProvider', 'CPUExecutionProvider']，CUDA EP 静默回退到 CPU。
  指定 providers=['CUDAExecutionProvider'] 不会报错，但会静默使用 CPU —— 
  **这种静默回退会导致基准测试得出完全错误的结论**，必须显式检测。

根因：
  `onnxruntime_providers_cuda.dll` 依赖 CUDA 12 运行时
  （cublasLt64_12.dll / cudnn64_9.dll 等），但这些不在系统 PATH 上。
  torch 2.4.1+cu121 的 `torch/lib` 目录里恰好带有这些 DLL。

修复：
  在 import onnxruntime **之前**，把 torch/lib 加入 DLL 搜索路径
  （Python 3.8+ 用 os.add_dll_directory，Windows 专用）。
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def _torch_lib_dir() -> Optional[str]:
    try:
        import torch
        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        return d if os.path.isdir(d) else None
    except Exception:
        return None


def _cuda_toolkit_bin() -> List[str]:
    """系统 CUDA toolkit 的 bin 目录（可能含 cublas/cudnn）."""
    out = []
    cp = os.environ.get("CUDA_PATH")
    if cp:
        b = os.path.join(cp, "bin")
        if os.path.isdir(b):
            out.append(b)
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        for v in sorted(os.listdir(base), reverse=True):
            b = os.path.join(base, v, "bin")
            if os.path.isdir(b):
                out.append(b)
    return out


_APPLIED = False


def setup_ort_cuda_dlls(verbose: bool = False) -> List[str]:
    """把 torch/lib 等目录加入 DLL 搜索路径，返回已加入的目录."""
    global _APPLIED
    added: List[str] = []
    if sys.platform != "win32":
        return added

    cands = []
    tl = _torch_lib_dir()
    if tl:
        cands.append(tl)
    cands.extend(_cuda_toolkit_bin())

    for d in cands:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
                added.append(d)
            # 同时加入 PATH（部分加载路径只认 PATH）
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
    if verbose and added:
        print("[ort_env] 已加入 DLL 搜索路径:")
        for a in added:
            print("   ", a)
    _APPLIED = True
    return added


def get_ort_session_options():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def available_providers(verbose: bool = True) -> List[str]:
    """获取 ORT 可用 providers（先修 DLL 路径）."""
    setup_ort_cuda_dlls(verbose=False)
    import onnxruntime as ort
    provs = ort.get_available_providers()
    if verbose:
        print("[ort_env] available providers:", provs)
        if "CUDAExecutionProvider" not in provs:
            print("[ort_env] 警告：CUDA EP 不可用，将只能使用 CPU。")
            print("[ort_env] 原因通常是 onnxruntime-gpu 的 CUDA 12 依赖 DLL 未找到。")
    return provs


def make_session(model_path: str, prefer_gpu: bool = True, verbose: bool = True):
    """创建 ORT session，并**显式验证**实际使用的 provider.

    重要：ORT 在 provider 不可用时会静默回退到 CPU，
    因此必须检查 `sess.get_providers()`，否则会把 CPU 结果误当成 GPU 结果。
    """
    setup_ort_cuda_dlls(verbose=verbose)
    import onnxruntime as ort

    provs = ort.get_available_providers()
    use = []
    if prefer_gpu and "CUDAExecutionProvider" in provs:
        use.append("CUDAExecutionProvider")
    use.append("CPUExecutionProvider")

    so = get_ort_session_options()
    sess = ort.InferenceSession(model_path, so, providers=use)
    actual = sess.get_providers()
    if verbose:
        print("[ort_env] 请求 providers: %s" % use)
        print("[ort_env] 实际 providers: %s" % actual)
        if prefer_gpu and "CUDAExecutionProvider" not in actual:
            print("[ort_env] *** 注意：已回退到 CPU，结果不能标注为 GPU ***")
    return sess


if __name__ == "__main__":
    print("torch lib:", _torch_lib_dir())
    print("cuda bins:", _cuda_toolkit_bin())
    available_providers(verbose=True)
