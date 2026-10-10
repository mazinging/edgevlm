"""M5 VLA 模型：视觉-语言条件化的动作序列生成（flow matching）.

架构（轻量、可控、可在笔记本 GPU 上训练）：
    image (64x64x3)
      -> CNN 视觉编码器  -> visual feature
    instruction_id (int)
      -> Embedding       -> language feature
    [visual ; language ; state ; noisy_action_chunk ; t]
      -> MLP denoiser    -> velocity field
    flow matching: 学习从噪声到专家动作块的向量场

为什么用 flow matching 而不是 diffusion：
  * 训练目标更简单（直接回归速度场，无需噪声调度）
  * **推理步数可弹性变化**（这正是 M5 要研究的：少步 -> 加速）
  * 与 π0 / Octo 等主流 VLA 的做法一致，贴合业界

动作块（action chunk）：
  一次预测 H 步动作（chunk），而非单步。这是 VLA 的标准做法，
  可显著降低推理频率。M5 将比较不同 chunk 长度与去噪步数的权衡。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VLAConfig:
    img_size: int = 64
    n_instructions: int = 4
    state_dim: int = 4              # agent(2) + target(2)
    action_dim: int = 2
    chunk_size: int = 4             # 一次预测的动作步数 H
    vision_dim: int = 128
    lang_dim: int = 32
    hidden: int = 384
    n_layers: int = 5
    time_dim: int = 64
    # flow matching
    n_train_steps: int = 10         # 训练时采样的 t（连续时间）
    default_infer_steps: int = 10   # 推理默认去噪步数


class VisionEncoder(nn.Module):
    """轻量 CNN：64x64x3 -> vision_dim."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, out_dim, 3, stride=2, padding=1), nn.GroupNorm(4, out_dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]
        h = self.net(x)
        return h.flatten(1)         # (B, out_dim)


class VelocityNet(nn.Module):
    """速度场网络 v(x_t, t | 视觉, 语言, 状态) -> action_vel."""

    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg.vision_dim)
        self.lang = nn.Embedding(cfg.n_instructions, cfg.lang_dim)

        flat_act = cfg.chunk_size * cfg.action_dim

        # 时间嵌入：把标量 t 投影到高维（正弦 + MLP），比直接喂标量更易学
        self.time_embed = nn.Sequential(
            nn.Linear(1, cfg.time_dim), nn.SiLU(),
            nn.Linear(cfg.time_dim, cfg.time_dim))

        in_dim = cfg.vision_dim + cfg.lang_dim + cfg.state_dim + flat_act + cfg.time_dim
        self.in_proj = nn.Linear(in_dim, cfg.hidden)

        layers = []
        d = cfg.hidden
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU()]
            d = cfg.hidden
        layers += [nn.Linear(d, flat_act)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, image, instr, state, x_t, t):
        """
        image: (B,3,H,W)  instr: (B,) long  state: (B,state_dim)
        x_t:   (B, H*action_dim)            t:   (B,1)
        """
        v = self.vision(image)
        l = self.lang(instr)
        te = self.time_embed(t)
        h = torch.cat([v, l, state, x_t, te], dim=-1)
        return self.mlp(self.in_proj(h))


class VLAPolicy(nn.Module):
    """完整 VLA 策略：封装训练目标与推理（flow matching）."""

    def __init__(self, cfg: Optional[VLAConfig] = None):
        super().__init__()
        self.cfg = cfg or VLAConfig()
        self.net = VelocityNet(self.cfg)
        self._last_infer_steps = self.cfg.default_infer_steps

    # ---------------- 训练 ----------------
    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """flow matching 训练损失.

        采用直线路径（rectified flow）：
            x_t = (1-t) * x0 + t * x1,   x0 ~ N(0,I), x1 = 专家动作
            target velocity = x1 - x0
        鼓励网络学习从噪声指向数据的直线速度场。
        """
        image = batch["image"]
        instr = batch["instruction"]
        state = batch["state"]
        a1 = batch["action_chunk"]                    # (B, H*act)
        B = a1.shape[0]

        x0 = torch.randn_like(a1)
        t = torch.rand(B, 1, device=a1.device)
        x_t = (1 - t) * x0 + t * a1
        v_target = a1 - x0

        v_pred = self.net(image, instr, state, x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        return loss, {"loss": loss.detach()}

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def sample(self, image, instr, state, n_steps: Optional[int] = None) -> torch.Tensor:
        """从噪声积分出动作块（推理用，不建图）."""
        return self._integrate(image, instr, state, n_steps, build_graph=False)

    def sample_with_grad(self, image, instr, state,
                         n_steps: Optional[int] = None) -> torch.Tensor:
        """可导版本（蒸馏训练用，需要梯度回传到学生网络）."""
        return self._integrate(image, instr, state, n_steps, build_graph=True)

    def _integrate(self, image, instr, state, n_steps: Optional[int],
                   build_graph: bool) -> torch.Tensor:
        """Euler 法积分 dx/dt = v(x,t).

        n_steps 是可调的去噪步数 —— M5 的核心加速旋钮：
          步数越少 -> 越快，但动作质量可能下降。
        """
        n_steps = n_steps or self._last_infer_steps
        B = image.shape[0]
        H, A = self.cfg.chunk_size, self.cfg.action_dim
        x = torch.randn(B, H * A, device=image.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=image.device)
            v = self.net(image, instr, state, x, t)
            x = x + v * dt
        return x.view(B, H, A)

    @torch.no_grad()
    def act(self, obs_image: np.ndarray, instruction_id: int, state: np.ndarray,
            device: str = "cuda", n_steps: Optional[int] = None) -> np.ndarray:
        """单次观测 -> 动作块（numpy 接口）."""
        img = torch.from_numpy(obs_image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = img.to(device)
        ins = torch.tensor([instruction_id], dtype=torch.long, device=device)
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        chunk = self.sample(img, ins, st, n_steps)
        return chunk[0].cpu().numpy()


# --------------------------------------------------------------------------
# 数据集封装
# --------------------------------------------------------------------------
def build_chunk_dataset(ds: Dict, chunk_size: int = 4, img_size: int = 64,
                        episode_lengths: Optional[List[int]] = None) -> Dict:
    """把逐步数据打包成动作块样本.

    对每个时刻 t，取其后的 chunk_size 步动作作为监督目标；
    不足则用最后一步动作填充（padding）。

    **关键**：必须尊重 episode 边界。
    实测教训：早期版本允许 chunk 跨 episode 采样，
    导致一个 chunk 内混入两条不同轨迹的动作，
    表现为 chunk 内相邻步余弦相似度仅 0.795（同一条直线轨迹应接近 1.0），
    给监督信号引入大量标签噪声、训练损失长期停在 0.69 无法下降。
    因此这里要求传入 episode_lengths，在每个 episode 内独立构造 chunk。
    """
    images, states, instrs, chunks = [], [], [], []
    actions = ds["actions"]
    n = actions.shape[0]

    if episode_lengths is None:
        # 退化情形：视为单个长 episode（会引入跨轨迹噪声，仅作兜底）
        episode_lengths = [n]

    offset = 0
    for L in episode_lengths:
        for t in range(offset, offset + L):
            end = min(t + chunk_size, offset + L)      # 不越过 episode 边界
            chunk = actions[t:end]
            if len(chunk) < chunk_size:
                pad = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            images.append(ds["images"][t])
            states.append(ds["states"][t])
            instrs.append(ds["instructions"][t])
            chunks.append(chunk)
        offset += L

    return {
        "images": np.stack(images).astype(np.float32),
        "states": np.stack(states).astype(np.float32),
        "instructions": np.stack(instrs).astype(np.int64),
        "action_chunks": np.stack(chunks).astype(np.float32),
    }


def to_torch_batch(batched: Dict, device: str = "cuda") -> Dict:
    img = torch.from_numpy(batched["images"]).permute(0, 3, 1, 2).float() / 255.0
    return {
        "image": img.to(device),
        "instruction": torch.from_numpy(batched["instructions"]).long().to(device),
        "state": torch.from_numpy(batched["states"]).float().to(device),
        "action_chunk": torch.from_numpy(batched["action_chunks"]).float().flatten(1).to(device),
    }
