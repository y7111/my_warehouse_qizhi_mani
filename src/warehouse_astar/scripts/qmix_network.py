#!/usr/bin/env python3
"""
qmix_network.py  ─  QMIX 神经网络定义

包含：
  AgentQNet  : 单智能体 Q 网络（参数共享）
  QMixNet    : 混合网络，将各智能体 Q 值混合为全局 Q_tot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNet(nn.Module):
    """
    单智能体局部 Q 网络。
    输入: 局部观测 obs (obs_dim,)
    输出: 每个动作的 Q 值 (n_actions,)
    两个智能体共享同一份参数。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixNet(nn.Module):
    """
    QMIX 单调混合网络。
    用超网络（Hypernetwork）从全局状态生成混合层权重，
    并对权重取绝对值以保证单调性（IGM 条件）。

    forward 参数:
      agent_qs : [batch, n_agents]   ── 各智能体选中动作的 Q 值
      state    : [batch, state_dim]  ── 全局状态（两辆车观测拼接）
    返回:
      q_tot    : [batch, 1]
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_hidden: int = 32,
        hyper_hidden: int = 64,
    ):
        super().__init__()
        self.n_agents     = n_agents
        self.mixing_hidden = mixing_hidden

        # ── 第一层：超网络生成权重和偏置 ───────────────────────────────
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)

        # ── 第二层：超网络生成权重；偏置用小网络保留非线性 ─────────────
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        bs = agent_qs.size(0)

        # ── 第一混合层 ────────────────────────────────────────────────
        # 权重取绝对值保证非负（单调性）
        w1 = torch.abs(self.hyper_w1(state))            # [bs, n_agents * mix_h]
        w1 = w1.view(bs, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(state).view(bs, 1, self.mixing_hidden)

        # [bs,1,n_agents] × [bs,n_agents,mix_h] → [bs,1,mix_h]
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # ── 第二混合层 ────────────────────────────────────────────────
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2              # [bs, 1, 1]
        return q_tot.view(bs, 1)
