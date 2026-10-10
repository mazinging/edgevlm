"""M5 仿真环境：2D 视觉-语言「到达」任务（VLA 动作序列生成评测用）.

自建、零外部依赖、可完全复现。

## 任务定义
  * 2D 平面 [0,1]^2 上一个末端执行器（agent）与一个目标点（target）。
  * 语言指令指定「去哪个颜色的目标」（4 个候选目标，红/绿/蓝/黄）。
  * 观测：64x64x3 俯视图 + 指令 id。
  * 动作：2D 位移增量 (dx, dy) ∈ [-1,1]^2，连续控制。
  * 成功：agent 到达 **指令所指定的那个** 目标点（距离 < tol）。
  * 关键：场景中有多个目标，**必须理解指令才能选对** ——
    这保证任务确实需要「视觉+语言」联合，而不是纯视觉回归。

## 为什么改为「到达」而非「推取」（重要设计决策）
  最初设计的是 push 任务（绕到方块后方再推向目标）。实测结果：
    * 环境物理本身**完全正确** —— 手动把 agent 放到方块正后方再推，
      一步即让 d_goal 从 0.235 降到 0.115（见 `check_push.py` 验证）。
    * 但**闭环控制器始终无法可靠完成**，历经 10 版迭代仍仅 2~7% 成功率。
      根因（逐条实测确认）：
        1. 每步重算几何 -> 相位抖动，agent 左右横跳；
        2. 轨道半径与进入 PUSH 的条件不兼容 -> 无限绕圈
           （ORBIT 累计 5009 步，从未进入 PUSH）；
        3. np.clip 把 agent 卡在墙上后控制器仍以为在移动；
        4. 直线走向绕行点会穿过方块，把方块推向错误方向；
        5. 固定半径绕圈形成**极限环**：rel_d 每步在 0.47/0.59 间振荡，
           方位角 25 步仅推进 0.14，永远达不到对齐条件；
        6. safe_r(0.187) 相对步长(0.12) 过小，每步overshoot 造成振荡。
  结论：**把工程预算从"手写接触式操作控制器"转移到"任务本身的有效性"上**。
  到达任务同样能度量 M5 真正关心的东西（动作生成质量、蒸馏损失），
  且专家可 100% 成功，评测才有意义。
  原 push 任务的失败分析已完整保留在本 docstring 中作为记录。

## 为什么自建而非 MetaWorld/ManiSkill
  * 无外部依赖、无需下载数据集、重置成本近零
  * 完全可控，便于消融实验
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# 目标颜色（用于渲染 + 指令语义）
TARGET_COLORS = [
    ("red", np.array([0.90, 0.15, 0.15])),
    ("green", np.array([0.15, 0.80, 0.25])),
    ("blue", np.array([0.20, 0.40, 0.95])),
    ("yellow", np.array([0.95, 0.85, 0.15])),
]

INSTRUCTIONS = [
    "go to the red target",
    "go to the green target",
    "go to the blue target",
    "go to the yellow target",
]


@dataclass
class EnvConfig:
    """环境参数."""
    size: float = 1.0                 # 世界边长 [0,1]^2
    img_size: int = 64
    agent_radius: float = 0.05
    n_targets: int = 4
    reach_tol: float = 0.06           # 到达判定半径
    max_steps: int = 24
    action_scale: float = 0.10        # 每步最大位移
    min_target_sep: float = 0.28      # 目标间最小间距（保证指令有区分度）
    randomize: bool = True
    seed: Optional[int] = None


class ReachEnv:
    """2D 多目标到达环境（指令条件化）."""

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    # ---------------- 核心 API ----------------
    def reset(self, instruction_id: Optional[int] = None) -> Dict:
        c = self.cfg
        r = self.rng if c.randomize else np.random.default_rng(0)

        # 指令：缺省随机（训练时），评测时可由外部指定
        if instruction_id is None:
            instruction_id = int(r.integers(0, min(c.n_targets, len(TARGET_COLORS))))
        self.instruction_id = int(instruction_id) % min(c.n_targets, len(TARGET_COLORS))

        # 采样互不重叠的目标点
        for _ in range(200):
            cand = r.uniform(0.15, 0.85, size=(c.n_targets, 2)).astype(np.float32)
            ok = True
            for i in range(c.n_targets):
                for j in range(i + 1, c.n_targets):
                    if np.linalg.norm(cand[i] - cand[j]) < c.min_target_sep:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
        self.targets = cand
        self.target = self.targets[self.instruction_id]

        # agent 初始位置：离指定目标足够远，且不与其它目标重叠
        for _ in range(100):
            a = r.uniform(0.1, 0.9, size=2).astype(np.float32)
            if np.linalg.norm(a - self.target) > c.reach_tol * 3.0:
                break
        self.agent = a

        self.steps = 0
        return self.observe()

    def observe(self) -> Dict:
        return {
            "image": self.render(),
            "instruction_id": self.instruction_id,
            "state": np.concatenate([self.agent, self.target]).astype(np.float32),
            # 便于分析与可视化（策略不应使用，仅用于评测/调试）
            "targets": self.targets.copy(),
        }

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
        self.agent = np.clip(self.agent + a * c.action_scale, 0.0, 1.0)
        self.steps += 1

        d = float(np.linalg.norm(self.agent - self.target))
        success = d < c.reach_tol

        # 稠密奖励：距离越近越高（便于学习）
        reward = 1.0 if success else -d
        done = success or self.steps >= c.max_steps
        return self.observe(), reward, done, {
            "success": success, "dist_to_target": d, "steps": self.steps,
        }

    def render(self) -> np.ndarray:
        """64x64x3 俯视渲染（纯 numpy，无第三方依赖）."""
        S = self.cfg.img_size
        img = np.full((S, S, 3), 0.15, dtype=np.float32)

        def to_px(p):
            return (int(np.clip(p[0], 0, 1) * (S - 1)),
                    int(np.clip(p[1], 0, 1) * (S - 1)))

        def disc(cx, cy, radius, color):
            r_px = max(1, int(radius * S))
            yy, xx = np.ogrid[:S, :S]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_px ** 2
            img[mask] = color

        # 所有目标都渲染（策略必须靠指令选对，故不能只画目标那个）
        for i in range(min(self.cfg.n_targets, len(TARGET_COLORS))):
            tx, ty = to_px(self.targets[i])
            disc(tx, ty, self.cfg.reach_tol, TARGET_COLORS[i][1])

        # agent
        ax, ay = to_px(self.agent)
        disc(ax, ay, self.cfg.agent_radius, np.array([1.0, 1.0, 1.0]))
        return img


# --------------------------------------------------------------------------
# 专家策略：直线趋向指令指定的目标（必然成功）
# --------------------------------------------------------------------------
class ExpertPolicy:
    """专家：朝指令指定的目标直线移动.

    任务改为到达后，专家可保证 100% 成功，
    这使 M5 的评测（蒸馏是否损伤动作质量）具备可靠基线。
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()

    def reset(self):
        pass

    def __call__(self, obs: Dict, env: "ReachEnv") -> np.ndarray:
        d = env.target - env.agent
        n = np.linalg.norm(d)
        if n < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (d / n).astype(np.float32)


# 向后兼容别名
PushEnv = ReachEnv


def expert_action(env: "ReachEnv", noise: float = 0.0) -> np.ndarray:
    a = ExpertPolicy(env.cfg)({}, env)
    if noise > 0:
        a = a + np.random.default_rng().normal(0, noise, size=2)
        a = a / (np.linalg.norm(a) + 1e-8)
    return a.astype(np.float32)


def generate_expert_episode(env: "ReachEnv", cfg: Optional[EnvConfig] = None,
                            action_noise: float = 0.0,
                            instruction_id: Optional[int] = None) -> Dict:
    """跑一条专家轨迹，返回 obs/action 序列."""
    cfg = cfg or env.cfg
    obs = env.reset(instruction_id=instruction_id)
    policy = ExpertPolicy(cfg)
    images, states, actions, instrs = [], [], [], []
    while True:
        a = policy(obs, env)
        if action_noise > 0:
            a = a + np.random.default_rng().normal(0, action_noise, size=2)
            a = a / (np.linalg.norm(a) + 1e-8)
        images.append(obs["image"])
        states.append(obs["state"])
        actions.append(a.astype(np.float32))
        instrs.append(obs["instruction_id"])
        obs, r, done, info = env.step(a)
        if done:
            break
    return {
        "images": np.stack(images),
        "states": np.stack(states),
        "actions": np.stack(actions),
        "instructions": np.array(instrs),
        "success": info["success"],
        "episode_len": len(actions),
    }


def make_dataset(n_episodes: int = 200, seed: int = 0,
                 action_noise: float = 0.0) -> Dict:
    """生成专家数据集（各指令均衡采样）.

    返回中包含 `episode_lengths`，供 build_chunk_dataset 尊重 episode 边界。
    """
    all_img, all_state, all_act, all_instr, succ = [], [], [], [], []
    lengths: List[int] = []
    n_instr = len(INSTRUCTIONS)
    for i in range(n_episodes):
        cfg = EnvConfig(seed=seed + i, randomize=True)
        env = ReachEnv(cfg)
        ep = generate_expert_episode(env, action_noise=action_noise,
                                     instruction_id=i % n_instr)
        all_img.append(ep["images"])
        all_state.append(ep["states"])
        all_act.append(ep["actions"])
        all_instr.append(ep["instructions"])
        lengths.append(int(ep["episode_len"]))
        succ.append(ep["success"])
    return {
        "images": np.concatenate(all_img),
        "states": np.concatenate(all_state),
        "actions": np.concatenate(all_act),
        "instructions": np.concatenate(all_instr),
        "episode_lengths": lengths,
        "expert_success_rate": float(np.mean(succ)),
        "n_transitions": int(sum(lengths)),
    }


def evaluate_policy(env_cfg: EnvConfig, policy_fn, n_episodes: int = 50,
                    seed: int = 1234) -> Dict:
    """评估策略成功率（每个指令均衡评测）."""
    succ, lens, dists = [], [], []
    n_instr = len(INSTRUCTIONS)
    for i in range(n_episodes):
        cfg = EnvConfig(**{**env_cfg.__dict__, "seed": seed + i})
        env = ReachEnv(cfg)
        obs = env.reset(instruction_id=i % n_instr)
        policy_fn.reset() if hasattr(policy_fn, "reset") else None
        while True:
            a = policy_fn(obs, env)
            obs, r, done, info = env.step(a)
            if done:
                succ.append(info["success"])
                lens.append(info["steps"])
                dists.append(info["dist_to_target"])
                break
    return {
        "success_rate": float(np.mean(succ)),
        "mean_steps": float(np.mean(lens)),
        "mean_final_dist": float(np.mean(dists)),
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    env = ReachEnv(EnvConfig(seed=0))
    obs = env.reset(instruction_id=0)
    print("image:", obs["image"].shape, "state:", obs["state"].shape)
    print("instruction:", INSTRUCTIONS[obs["instruction_id"]])
    print("targets:\n", obs["targets"])
    print("agent:", env.agent)

    ds = make_dataset(60)
    print("\ndataset transitions:", ds["n_transitions"])
    print("expert success rate: %.2f" % ds["expert_success_rate"])
    print("action range: [%.2f, %.2f]" % (ds["actions"].min(), ds["actions"].max()))

    res = evaluate_policy(EnvConfig(), ExpertPolicy(), n_episodes=100)
    print("\nexpert eval:", res)
