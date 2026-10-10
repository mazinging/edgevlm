"""EdgeVLM-Kit — 端侧 VLM/VLA 推理优化与部署验证工具链.

统一入口：环境变量固化 + 路径管理。
所有子模块必须先 import 本包，以确保 HF 缓存/TMP 落在工作区内。
"""

import os
import sys

__version__ = "0.1.0"

# --- 工作区根目录：本文件位于 <root>/src/edgevlm/__init__.py ---
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))

# 允许直接运行脚本
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# --- 关键：所有可写目录必须落在工作区内（沙箱约束）---
CACHE_DIR = _ensure_dir(os.path.join(ROOT, ".cache"))
HF_HOME = _ensure_dir(os.path.join(CACHE_DIR, "huggingface"))
TMP_DIR = _ensure_dir(os.path.join(ROOT, ".tmp"))
RESULTS_DIR = _ensure_dir(os.path.join(ROOT, "results"))
ARTIFACTS_DIR = _ensure_dir(os.path.join(ROOT, "artifacts"))
FIGURES_DIR = _ensure_dir(os.path.join(RESULTS_DIR, "figures"))

# --- 环境变量固化（必须在 import transformers 之前生效）---
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TMP", TMP_DIR)
os.environ.setdefault("TEMP", TMP_DIR)

# Windows 控制台默认 GBK，中文输出会乱码 —— 强制 UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 模型注册表
MODELS = {
    "smolvlm-256m": {
        "repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "params_m": 256.5,
        "note": "主力模型：SigLIP-base(12L) + Llama(30L, GQA 9Q/3KV)",
    },
    "smolvlm-500m": {
        "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "params_m": 500.0,
        "note": "扩展模型：验证结论是否随规模变化",
    },
}

DEFAULT_MODEL = "smolvlm-256m"
