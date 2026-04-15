#!/usr/bin/env python3
"""
qmix_train.py  ─  QMIX 多智能体训练脚本

运行方式：
    rosrun warehouse_astar qmix_train.py

训练结果保存至 ~/qmix_checkpoints/
  qmix_epXXX.pt   ── 模型检查点
  rewards.npy      ── 每局总奖励曲线
  losses.npy       ── 每局平均 loss 曲线
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import rospy

# 将脚本目录加入 sys.path，确保 import warehouse_env / qmix_network 可用
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from warehouse_env import WarehouseEnv
from qmix_network import AgentQNet, QMixNet


# ═══════════════════════════════════════════════════════════════════════
#  超参数
# ═══════════════════════════════════════════════════════════════════════

OBS_DIM       = 9      # 单个智能体观测维度
STATE_DIM     = 18     # 全局状态 = 两智能体观测拼接
N_AGENTS      = 2
N_ACTIONS     = 5
HIDDEN        = 64     # Q 网络隐层宽度
MIX_HIDDEN    = 32     # Mixing 网络隐层宽度
HYPER_HIDDEN  = 64     # 超网络隐层宽度

LR            = 1e-4
GAMMA         = 0.99
BATCH_SIZE    = 32
BUFFER_SIZE   = 5000   # 经验池容量（步数）
TARGET_UPDATE = 20     # 每隔多少 episode 硬更新目标网络
GRAD_CLIP     = 10.0

EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY_EP  = 500    # 在前 N 个 episode 内线性衰减 epsilon

MAX_EPISODES  = 1000
SAVE_INTERVAL = 10     # 每隔多少 episode 保存检查点

CKPT_DIR = os.path.expanduser('~/qmix_checkpoints')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════
#  经验回放池
# ═══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """存储 (obs_list, actions, reward, next_obs_list, done) 转移样本。"""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs_list, actions, reward: float,
             next_obs_list, done: bool):
        self.buf.append((obs_list, actions, reward, next_obs_list, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs_list, actions, rewards, next_obs_list, dones = zip(*batch)
        return obs_list, actions, rewards, next_obs_list, dones

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════════════════
#  QMIX 智能体（包含训练逻辑）
# ═══════════════════════════════════════════════════════════════════════

class QMIXAgent:
    def __init__(self):
        # ── 在线网络 ──────────────────────────────────────────────────
        self.q_net  = AgentQNet(OBS_DIM, N_ACTIONS, HIDDEN).to(DEVICE)
        self.mix_net = QMixNet(N_AGENTS, STATE_DIM, MIX_HIDDEN, HYPER_HIDDEN).to(DEVICE)

        # ── 目标网络（定期硬拷贝）────────────────────────────────────
        self.q_target   = AgentQNet(OBS_DIM, N_ACTIONS, HIDDEN).to(DEVICE)
        self.mix_target = QMixNet(N_AGENTS, STATE_DIM, MIX_HIDDEN, HYPER_HIDDEN).to(DEVICE)
        self._sync_target()

        # ── 优化器（两个网络联合优化）────────────────────────────────
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mix_net.parameters()),
            lr=LR,
        )

        self.buffer  = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START

    # ── 动作选择 ──────────────────────────────────────────────────────

    def select_actions(self, obs_list):
        """ε-greedy 策略，返回 [action_alpha, action_beta]。"""
        actions = []
        for obs in obs_list:
            if random.random() < self.epsilon:
                actions.append(random.randint(0, N_ACTIONS - 1))
            else:
                with torch.no_grad():
                    o = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                    q = self.q_net(o)
                    actions.append(int(q.argmax(dim=1).item()))
        return actions

    # ── ε 衰减 ────────────────────────────────────────────────────────

    def update_epsilon(self, episode: int):
        ratio = min(1.0, episode / EPS_DECAY_EP)
        self.epsilon = EPS_START + ratio * (EPS_END - EPS_START)

    # ── 目标网络硬更新 ────────────────────────────────────────────────

    def _sync_target(self):
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.mix_target.load_state_dict(self.mix_net.state_dict())

    # ── 单步训练 ──────────────────────────────────────────────────────

    def train_step(self):
        """从经验池采样一个 batch 并更新网络，返回 loss（池不足时返回 None）。"""
        if len(self.buffer) < BATCH_SIZE:
            return None

        obs_b, acts_b, rews_b, nobs_b, done_b = self.buffer.sample(BATCH_SIZE)

        # ── 转为张量 ──────────────────────────────────────────────────
        # obs_t : [bs, n_agents, obs_dim]
        obs_t  = torch.FloatTensor(
            np.array([[o for o in ol] for ol in obs_b])
        ).to(DEVICE)
        nobs_t = torch.FloatTensor(
            np.array([[o for o in ol] for ol in nobs_b])
        ).to(DEVICE)
        acts_t = torch.LongTensor(np.array(acts_b)).to(DEVICE)   # [bs, n_agents]
        rews_t = torch.FloatTensor(np.array(rews_b)).unsqueeze(1).to(DEVICE)  # [bs,1]
        done_t = torch.FloatTensor(np.array(done_b)).unsqueeze(1).to(DEVICE)  # [bs,1]

        # 全局状态 = 两智能体观测拼接
        state_t  = obs_t.view(BATCH_SIZE, -1)   # [bs, state_dim]
        nstate_t = nobs_t.view(BATCH_SIZE, -1)

        # ── 当前 Q 值（取选中动作的 Q）────────────────────────────────
        q_vals = []
        for i in range(N_AGENTS):
            q_i = self.q_net(obs_t[:, i, :])                       # [bs, n_actions]
            q_i = q_i.gather(1, acts_t[:, i].unsqueeze(1))         # [bs, 1]
            q_vals.append(q_i)
        agent_qs = torch.cat(q_vals, dim=1)                         # [bs, n_agents]

        # ── 目标 Q 值（Double-DQN 风格：在线网络选动作，目标网络估值）
        with torch.no_grad():
            next_q_vals = []
            for i in range(N_AGENTS):
                # 在线网络选最优动作
                best_a = self.q_net(nobs_t[:, i, :]).argmax(dim=1, keepdim=True)
                # 目标网络估该动作的 Q 值
                nq_i = self.q_target(nobs_t[:, i, :]).gather(1, best_a)
                next_q_vals.append(nq_i)
            next_agent_qs = torch.cat(next_q_vals, dim=1)          # [bs, n_agents]

            q_tot_next = self.mix_target(next_agent_qs, nstate_t)  # [bs, 1]
            target = rews_t + GAMMA * (1.0 - done_t) * q_tot_next  # [bs, 1]

        # ── 混合当前 Q，计算 loss ──────────────────────────────────────
        q_tot = self.mix_net(agent_qs, state_t)                     # [bs, 1]
        loss  = nn.MSELoss()(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mix_net.parameters()),
            GRAD_CLIP,
        )
        self.optimizer.step()

        return loss.item()

    # ── 模型保存 / 加载 ───────────────────────────────────────────────

    def save(self, directory: str, episode: int):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'qmix_ep{episode:04d}.pt')
        torch.save({
            'episode':   episode,
            'epsilon':   self.epsilon,
            'q_net':     self.q_net.state_dict(),
            'mix_net':   self.mix_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        rospy.loginfo('[QMIX] 模型保存 → %s', path)

    def load(self, ckpt_path: str) -> int:
        ck = torch.load(ckpt_path, map_location=DEVICE)
        self.q_net.load_state_dict(ck['q_net'])
        self.mix_net.load_state_dict(ck['mix_net'])
        self._sync_target()
        self.optimizer.load_state_dict(ck['optimizer'])
        self.epsilon = ck['epsilon']
        rospy.loginfo('[QMIX] 模型加载 ← %s (ep=%d)', ckpt_path, ck['episode'])
        return ck['episode']


# ═══════════════════════════════════════════════════════════════════════
#  训练主循环
# ═══════════════════════════════════════════════════════════════════════

def train(resume_ckpt: str = None):
    env   = WarehouseEnv()
    agent = QMIXAgent()

    start_ep = 1
    if resume_ckpt:
        start_ep = agent.load(resume_ckpt) + 1

    # 恢复训练时加载已有历史，保证曲线连续
    rewards_path = os.path.join(CKPT_DIR, 'rewards.npy')
    losses_path  = os.path.join(CKPT_DIR, 'losses.npy')
    if resume_ckpt and os.path.exists(rewards_path):
        rewards_hist = list(np.load(rewards_path))
        losses_hist  = list(np.load(losses_path)) if os.path.exists(losses_path) else []
        rospy.loginfo('[QMIX] 已加载历史曲线 %d 条', len(rewards_hist))
    else:
        rewards_hist = []
        losses_hist  = []

    rospy.loginfo('[QMIX] 训练开始 | 设备: %s | episode: %d → %d',
                  DEVICE, start_ep, MAX_EPISODES)

    for ep in range(start_ep, MAX_EPISODES + 1):
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            actions = agent.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)

            # 共享团队奖励，裁剪后存入经验池（防止大奖励/惩罚造成Q值震荡）
            clipped_reward = max(-20.0, min(20.0, rewards[0]))
            agent.buffer.push(obs, actions, clipped_reward, next_obs, float(done))

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += rewards[0]
            obs = next_obs

        # ── episode 结束后的更新 ──────────────────────────────────────
        agent.update_epsilon(ep)

        if ep % TARGET_UPDATE == 0:
            agent._sync_target()
            rospy.loginfo('[QMIX] 目标网络已同步 (ep=%d)', ep)

        # ── 记录 ─────────────────────────────────────────────────────
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_hist.append(ep_reward)
        losses_hist.append(mean_loss)

        flipped_flag = ' [FLIP]' if info.get('flipped') else ''
        rospy.loginfo(
            '[QMIX] ep %4d | reward %7.1f | loss %.4f | eps %.3f | shelf %s | step %d%s',
            ep, ep_reward, mean_loss, agent.epsilon,
            ''.join('Y' if s else 'N' for s in info['shelf_picked']),
            info['step'],
            flipped_flag,
        )

        # ── 定期保存 ─────────────────────────────────────────────────
        if ep % SAVE_INTERVAL == 0:
            agent.save(CKPT_DIR, ep)
            np.save(os.path.join(CKPT_DIR, 'rewards.npy'),
                    np.array(rewards_hist))
            np.save(os.path.join(CKPT_DIR, 'losses.npy'),
                    np.array(losses_hist))

    rospy.loginfo('[QMIX] 训练完成，共 %d 局', MAX_EPISODES)
    env.close()


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点继续训练，填写 .pt 文件路径')
    # rosrun 会传额外参数，ignore_unknown 防止报错
    args, _ = parser.parse_known_args()

    train(resume_ckpt=args.resume)
