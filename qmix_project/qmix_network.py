#!/usr/bin/env python3
"""
qmix_network.py — QMIX 神经网络定义（v5）

包含：
  AttentionAgentQNet : 双路 Attention 版单智能体 Q 网络（主方案，4车共享参数）
                       货架注意力：self → Query，10货架 → Key/Value
                       车间注意力：self → Query，3辆其他车 → Key/Value
  MLPAgentQNet       : 普通 MLP 版（对照基线，保持 hidden=64 原始结构）
  QMixNet            : 混合网络，结构不变，参数更新为 n_agents=4, state_dim=74
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAgentQNet(nn.Module):
    """
    双路 Attention 版单智能体 Q 网络。

    obs 拆分（严格对应 warehouse_env._get_obs 的顺序）：
      obs[:, 0          : self_end]   = self_feat  (pos+carrying+dest)   19维
      obs[:, self_end   : other_end]  = other_feats (其他3车 pos+carrying) 18维
      obs[:, other_end  : ]           = shelf_feats (10货架 goods+pos)    50维

    货架注意力（Shelf Attention）：self_feat → Query，shelf_feats → Key/Value
      → shelf_ctx [batch, d_model]，shelf_w [batch, 1, n_shelves]

    车间注意力（Agent Attention）：self_feat → Query，other_feats → Key/Value
      → agent_ctx [batch, d_model]，agent_w [batch, 1, n_other]

    货架头输入：[self_feat(19) || shelf_ctx(32) || agent_ctx(32)] = 83维 → Q(0..9)
    投递头输入：self_feat(19) ONLY → Q(10..12)（隔离 attention 干扰）
    """

    def __init__(
        self,
        obs_dim:   int = 87,
        n_actions: int = 13,
        n_agents:  int = 4,
        n_shelves: int = 10,
        d_model:   int = 32,
        n_heads:   int = 2,
        hidden:    int = 128,
    ):
        super().__init__()

        # obs 切分边界
        self.self_feat_dim  = 2 + 4 + n_actions          # 19
        # obs_dim 断言：将死参数转为主动校验，传入值必须与内部切分一致
        expected_obs_dim = self.self_feat_dim + 6 * (n_agents - 1) + 5 * n_shelves
        assert obs_dim == expected_obs_dim, (
            f"obs_dim 不匹配：传入 {obs_dim}，期望 {expected_obs_dim}"
        )
        self.other_end      = self.self_feat_dim + 6 * (n_agents - 1)  # 37
        self.n_other        = n_agents - 1               # 3
        self.shelf_feat_dim = 5                          # goods(3)+pos(2)
        self.agent_feat_dim = 6                          # pos(2)+carrying(4)
        self.n_shelves      = n_shelves
        self.d_model        = d_model

        # 货架注意力（独立投影）
        self.shelf_q = nn.Linear(self.self_feat_dim, d_model)
        self.shelf_k = nn.Linear(self.shelf_feat_dim, d_model)
        self.shelf_v = nn.Linear(self.shelf_feat_dim, d_model)
        self.shelf_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # 车间注意力（独立投影）
        self.agent_q = nn.Linear(self.self_feat_dim, d_model)
        self.agent_k = nn.Linear(self.agent_feat_dim, d_model)
        self.agent_v = nn.Linear(self.agent_feat_dim, d_model)
        self.agent_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)  # n_heads=1，序列长度仅3

        # 货架头：[self_feat(19) || shelf_ctx(d_model) || agent_ctx(d_model)] → Q(0..n_shelves-1)
        shelf_in = self.self_feat_dim + d_model + d_model  # 83
        self.shelf_head = nn.Sequential(
            nn.Linear(shelf_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_shelves),
        )

        # 投递头：仅 self_feat(19) → Q(n_shelves..n_actions-1)，隔离 attention 上下文干扰
        n_delivery = n_actions - n_shelves  # 3
        self.delivery_head = nn.Sequential(
            nn.Linear(self.self_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_delivery),
        )

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        """
        Args:
            obs:         [batch, obs_dim]
            return_attn: True 时额外返回 (shelf_w, agent_w)
                         shelf_w [batch, 1, n_shelves]，agent_w [batch, 1, n_other]
        Returns:
            q_vals [batch, n_actions]，或 (q_vals, shelf_w, agent_w)
        """
        bs = obs.size(0)

        self_feat   = obs[:, :self.self_feat_dim]               # [bs, 19]
        other_feats = obs[:, self.self_feat_dim:self.other_end] \
                         .view(bs, self.n_other, self.agent_feat_dim)  # [bs, 3, 6]
        shelf_feats = obs[:, self.other_end:] \
                         .view(bs, self.n_shelves, self.shelf_feat_dim)  # [bs, 10, 5]

        # 货架注意力
        s_q = self.shelf_q(self_feat).unsqueeze(1)  # [bs, 1, d_model]
        s_k = self.shelf_k(shelf_feats)             # [bs, 10, d_model]
        s_v = self.shelf_v(shelf_feats)             # [bs, 10, d_model]
        shelf_ctx, shelf_w = self.shelf_attn(s_q, s_k, s_v)
        shelf_ctx = shelf_ctx.squeeze(1)            # [bs, d_model]

        # 车间注意力
        a_q = self.agent_q(self_feat).unsqueeze(1)  # [bs, 1, d_model]
        a_k = self.agent_k(other_feats)             # [bs, 3, d_model]
        a_v = self.agent_v(other_feats)             # [bs, 3, d_model]
        agent_ctx, agent_w = self.agent_attn(a_q, a_k, a_v)
        agent_ctx = agent_ctx.squeeze(1)            # [bs, d_model]

        x_attn     = torch.cat([self_feat, shelf_ctx, agent_ctx], dim=-1)  # [bs, 83]
        q_shelf    = self.shelf_head(x_attn)       # [bs, n_shelves]
        q_delivery = self.delivery_head(self_feat) # [bs, 3]
        q_vals     = torch.cat([q_shelf, q_delivery], dim=-1)  # [bs, n_actions]

        if return_attn:
            return q_vals, shelf_w, agent_w
        return q_vals


class MLPAgentQNet(nn.Module):
    """
    对照基线：普通两层 MLP，hidden=64。
    保持原始结构（不扩维），以确保与 AttentionAgentQNet 的对比反映架构差异而非参数量差异。
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

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(obs)


class QMixNet(nn.Module):
    """
    QMIX 单调混合网络（结构不变，参数更新为 n_agents=4, state_dim=74）。

    forward 参数:
      agent_qs : [batch, n_agents]   各智能体选中动作的 Q 值
      state    : [batch, state_dim]  全局状态
    返回:
      q_tot    : [batch, 1]
    """

    def __init__(
        self,
        n_agents:      int = 4,
        state_dim:     int = 74,
        mixing_hidden: int = 32,
        hyper_hidden:  int = 64,
    ):
        super().__init__()
        self.n_agents      = n_agents
        self.mixing_hidden = mixing_hidden

        # 第一层：超网络生成权重和偏置
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)

        # 第二层：超网络生成权重；偏置用小网络保留非线性
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

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bs = agent_qs.size(0)

        # 第一混合层（权重取绝对值保证单调性）
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(state).view(bs, 1, self.mixing_hidden)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # 第二混合层
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(bs, 1)
