"""M5 实验：VLA 动作序列生成 —— flow matching 训练 + 去噪步数加速 + 蒸馏.

核心问题（对应 JD 职责 1「动作序列生成加速」）：
  Q1. flow matching 的动作生成质量如何随**去噪步数**变化？
      （步数 -> 延迟的线性关系 vs 质量的非线性损失）
  Q2. **步数蒸馏**能否用 1~2 步逼近多步的质量？
  Q3. action chunk 长度如何影响推理频率与质量？

用法:
    python -m edgevlm.vla.run_m5 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from edgevlm import RESULTS_DIR
from edgevlm.hardware import _ramp_clocks
from edgevlm.vla.env import (
    EnvConfig, ReachEnv, ExpertPolicy, INSTRUCTIONS,
    make_dataset, evaluate_policy,
)
from edgevlm.vla.model import (
    VLAConfig, VLAPolicy, build_chunk_dataset, to_torch_batch,
)


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def train_vla(ds_t: Dict, cfg: VLAConfig, device: str, epochs: int = 40,
              batch_size: int = 128, lr: float = 3e-4, seed: int = 0,
              verbose: bool = True) -> VLAPolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = VLAPolicy(cfg).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = ds_t["images"].shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx] for k, v in ds_t.items()}
            tb = to_torch_batch(batch, device)
            loss, _ = policy.loss(tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("    epoch %3d  loss=%.5f" % (ep, tot / max(cnt, 1)))
    return policy


# --------------------------------------------------------------------------
# 策略包装：把 VLA 变成 env 可调用的策略
# --------------------------------------------------------------------------
class VLAPolicyRunner:
    """用 VLA 的 action chunk 驱动环境（chunk 执行完再重新推理）."""

    def __init__(self, policy: VLAPolicy, device: str, n_steps: int = 10,
                 replan_every: Optional[int] = None):
        self.policy = policy
        self.device = device
        self.n_steps = n_steps
        self.replan_every = replan_every      # None -> 用完整个 chunk 再重规划
        self.buffer: List[np.ndarray] = []
        # 累计统计（跨 episode 累加，不受 reset 影响）
        self.total_infers = 0
        self.all_infer_times: List[float] = []
        self.n_episodes = 0

    def reset(self):
        """每个 episode 开始时清空动作缓冲；累计统计继续保留."""
        self.buffer = []
        self.n_episodes += 1

    def __call__(self, obs: Dict, env: ReachEnv) -> np.ndarray:
        if not self.buffer:
            t0 = time.perf_counter()
            chunk = self.policy.act(
                obs["image"], int(obs["instruction_id"]),
                obs["state"][:self.policy.cfg.state_dim].astype(np.float32),
                device=self.device, n_steps=self.n_steps)
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.all_infer_times.append(time.perf_counter() - t0)
            self.total_infers += 1
            k = self.replan_every or len(chunk)
            self.buffer = [c for c in chunk[:k]]
        return self.buffer.pop(0)


# --------------------------------------------------------------------------
# 步数蒸馏：让学生模型少步即可复现教师多步的输出
# --------------------------------------------------------------------------
def distill_steps(teacher: VLAPolicy, student: VLAPolicy, ds_t: Dict,
                  device: str, student_steps: int = 2, epochs: int = 30,
                  batch_size: int = 128, lr: float = 3e-4, seed: int = 0,
                  verbose: bool = True) -> VLAPolicy:
    """步数蒸馏.

    思路：教师用多步（如 10 步）积分得到高质量动作块 a*；
    学生用少步（如 2 步）也要产出 a*。
    直接对**最终动作块**做回归，而非匹配速度场 ——
    这比 ReFlow / consistency distillation 简单，且对本任务足够。

    注意：学生从教师初始化，保证起点相同，只改变"步数能力"。
    """
    torch.manual_seed(seed)
    student.load_state_dict(teacher.state_dict())
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = ds_t["images"].shape[0]
    teacher.eval()
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx] for k, v in ds_t.items()}
            tb = to_torch_batch(batch, device)

            with torch.no_grad():
                a_teacher = teacher.sample(
                    tb["image"], tb["instruction"], tb["state"],
                    n_steps=teacher.cfg.default_infer_steps)

            a_student = student.sample_with_grad(
                tb["image"], tb["instruction"], tb["state"], n_steps=student_steps)

            loss = F.mse_loss(a_student, a_teacher)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("    distill epoch %3d  loss=%.6f" % (ep, tot / max(cnt, 1)))
    return student


# --------------------------------------------------------------------------
# 动作质量评测（不依赖任务成功率）
# --------------------------------------------------------------------------
def action_quality(policy: VLAPolicy, ds_t: Dict, device: str,
                   n_steps: int, n_eval: int = 512, seed: int = 7) -> Dict:
    """在留出样本上评测动作重建质量."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = ds_t["images"].shape[0]
    idx = rng.choice(n, size=min(n_eval, n), replace=False)
    batch = {k: v[idx] for k, v in ds_t.items()}
    tb = to_torch_batch(batch, device)

    t0 = time.perf_counter()
    a_pred = policy.sample(tb["image"], tb["instruction"], tb["state"], n_steps=n_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    a_true = tb["action_chunk"].view(a_pred.shape)
    err = (a_pred - a_true)
    mse = float(err.pow(2).mean())
    l2 = float(err.norm(dim=-1).mean())            # 每个动作块的 L2 误差
    cos = float(F.cosine_similarity(
        a_pred.reshape(-1), a_true.reshape(-1), dim=0))

    # 平滑度：动作块内相邻步的差分（越小越平滑，越接近专家）
    smooth_pred = float((a_pred[:, 1:] - a_pred[:, :-1]).abs().mean())
    smooth_true = float((a_true[:, 1:] - a_true[:, :-1]).abs().mean())

    return {
        "n_steps": n_steps,
        "mse": mse,
        "l2_per_chunk": l2,
        "cosine": cos,
        "smoothness_pred": smooth_pred,
        "smoothness_true": smooth_true,
        "batch_latency_ms": dt * 1000,
        "per_sample_ms": dt * 1000 / len(idx),
    }


def model_size_mb(policy: VLAPolicy) -> float:
    n = sum(p.numel() for p in policy.parameters())
    return n * 4 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--distill-epochs", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "m5_vla.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        _ramp_clocks(args.device, 3.0)

    print("=" * 100)
    print("M5 VLA 动作序列生成：flow matching + 步数蒸馏")
    print("=" * 100)

    # ---------- 数据 ----------
    print("\n[1] 生成专家数据")
    ds = make_dataset(args.n_episodes, seed=0)
    print("    transitions=%d  expert_success=%.2f" % (
        ds["n_transitions"], ds["expert_success_rate"]))
    chunk_ds = build_chunk_dataset(ds, chunk_size=args.chunk_size,
                                   episode_lengths=ds.get("episode_lengths"))

    n = chunk_ds["images"].shape[0]
    n_val = max(64, int(n * 0.12))
    perm = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = {k: v[train_idx] for k, v in chunk_ds.items()}
    val_ds = {k: v[val_idx] for k, v in chunk_ds.items()}
    print("    train=%d  val=%d  chunk_size=%d" % (
        len(train_idx), len(val_idx), args.chunk_size))

    cfg = VLAConfig(chunk_size=args.chunk_size)

    # ---------- 训练教师 ----------
    print("\n[2] 训练教师模型 (flow matching, 多步推理)")
    policy = train_vla(train_ds, cfg, args.device, epochs=args.epochs)
    policy.eval()
    print("    params=%.3f M  size=%.2f MB" % (
        sum(p.numel() for p in policy.parameters()) / 1e6, model_size_mb(policy)))

    # ---------- 步数扫描 ----------
    print("\n[3] 去噪步数扫描（速度 vs 质量）")
    step_results = []
    for ns in [1, 2, 4, 6, 10, 20]:
        q = action_quality(policy, val_ds, args.device, n_steps=ns)
        step_results.append(q)
        print("    steps=%-3d  mse=%.5f  l2=%.4f  cos=%.4f  %.3f ms/sample" % (
            ns, q["mse"], q["l2_per_chunk"], q["cosine"], q["per_sample_ms"]))

    # ---------- 步数蒸馏 ----------
    print("\n[4] 步数蒸馏：学生 2 步 vs 教师 10 步")
    student = VLAPolicy(cfg).to(args.device)
    student = distill_steps(policy, student, train_ds, args.device,
                            student_steps=2, epochs=args.distill_epochs)
    student.eval()

    distill_results = {"teacher": {}, "student": {}}
    for tag, pol in [("teacher", policy), ("student", student)]:
        for ns in ([10, 20] if tag == "teacher" else [1, 2, 4]):
            q = action_quality(pol, val_ds, args.device, n_steps=ns)
            distill_results[tag]["steps_%d" % ns] = q
            print("    %-8s steps=%-3d mse=%.5f l2=%.4f cos=%.4f %.3f ms/sample" % (
                tag, ns, q["mse"], q["l2_per_chunk"], q["cosine"], q["per_sample_ms"]))

    # ---------- 闭环任务成功率 ----------
    print("\n[5] 闭环评测（任务成功率）")
    env_cfg = EnvConfig()
    succ = {}
    for tag, pol, ns in [("expert", None, None),
                         ("teacher_10step", policy, 10),
                         ("teacher_2step", policy, 2),
                         ("student_2step", student, 2),
                         ("student_1step", student, 1)]:
        runner = ExpertPolicy(env_cfg) if pol is None else \
            VLAPolicyRunner(pol, args.device, n_steps=ns)
        res = evaluate_policy(env_cfg, runner, n_episodes=100)
        # 每条 episode 的平均重规划次数（专家无推理，记为 None）
        if hasattr(runner, "total_infers"):
            n_ep = max(runner.n_episodes, 1)
            res["mean_infers_per_episode"] = runner.total_infers / n_ep
            it = runner.all_infer_times
            res["mean_infer_ms"] = float(np.mean(it) * 1000) if it else None
        else:
            res["mean_infers_per_episode"] = None
            res["mean_infer_ms"] = None
        succ[tag] = res
        print("    %-16s success=%.2f  steps=%.2f  final_dist=%.4f  "
              "infers/ep=%.2f  infer=%.2fms" % (
                  tag, res["success_rate"], res["mean_steps"], res["mean_final_dist"],
                  res.get("mean_infers_per_episode") or 0.0,
                  res.get("mean_infer_ms") or 0.0))

    payload = {
        "config": {"n_episodes": args.n_episodes, "chunk_size": args.chunk_size,
                   "epochs": args.epochs, "distill_epochs": args.distill_epochs,
                   "device": args.device},
        "model_params_m": sum(p.numel() for p in policy.parameters()) / 1e6,
        "step_scan": step_results,
        "distillation": distill_results,
        "closed_loop": succ,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n保存至 %s" % args.out)

    # ---------- 汇总 ----------
    print("\n=== 加速比 vs 质量损失 ===")
    base = next(r for r in step_results if r["n_steps"] == 10)
    for r in step_results:
        print("  steps=%-3d  速度 x%5.2f  MSE x%6.2f  L2 x%5.2f  cos=%.4f" % (
            r["n_steps"], base["per_sample_ms"] / r["per_sample_ms"],
            r["mse"] / base["mse"], r["l2_per_chunk"] / base["l2_per_chunk"],
            r["cosine"]))


if __name__ == "__main__":
    main()
