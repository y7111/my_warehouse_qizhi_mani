"""
train.py — QMIX 训练主循环（v5）

用法：
  python train.py --model attention   # 训练 AttentionAgentQNet → checkpoints_v5_attn/
  python train.py --model mlp         # 训练 MLPAgentQNet      → checkpoints_v5_mlp/
  python train.py --model attention --resume checkpoints_v5_attn/qmix_ep1000.pt

对照实验：两套超参数完全相同，仅网络结构不同（MLP hidden=64 以保证对比公平性）。
"""

import os
import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from warehouse_env import WarehouseEnv
from qmix_network import AttentionAgentQNet, MLPAgentQNet, QMixNet

# ── 超参数 ────────────────────────────────────────────────────────────────────
N_AGENTS      = 4
N_SHELVES     = 10
MAX_STEPS     = 300        # 已在旧版验证，新地图暂沿用，早期若平均投递<6次再考虑调整

N_EPISODES    = 6000

BUFFER_SIZE   = 20000
BATCH_SIZE    = 32
MIN_BUFFER    = 500

LR            = 2e-4       # 从 5e-4 降低，减少 Q 值震荡
GAMMA         = 0.99
TARGET_UPDATE = 100        # 从 50 增大，让 target 网络更稳定
UPDATE_INTERVAL = 4        # 每4步做一次梯度更新，防灾难性遗忘（旧版已验证）

EPS_START     = 1.0
EPS_END       = 0.10       # 保留底线探索，防止贪婪策略锁死后崩溃
EPS_DECAY     = 1500       # 原6000太慢：ep1100时ε仍0.835，4000轮后才开始学

SAVE_INTERVAL = 200        # 更密存档，便于事后取最佳 checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 经验回放 ──────────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    'Transition',
    ['obs', 'state', 'actions', 'reward', 'next_obs', 'next_state', 'done']
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def get_epsilon(episode: int) -> float:
    ratio = min(episode / EPS_DECAY, 1.0)
    return EPS_START + ratio * (EPS_END - EPS_START)


def select_action_for_agent(obs: np.ndarray, agent_net, epsilon: float, n_actions: int) -> int:
    n_shelves     = n_actions - 3
    carrying_type = int(np.argmax(obs[2:6]))  # 0=无货 1=A 2=B 3=C

    # 探索：纯随机（含错误投递区），初始奖励从低处起，曲线上升趋势清晰
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    # 贪婪：有货选最优投递区，空手选最优货架
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        q     = agent_net(obs_t)
    if carrying_type != 0:
        return int(q[:, n_shelves:].argmax().item()) + n_shelves
    else:
        return int(q[:, :n_shelves].argmax().item())


def compute_loss(
    batch,
    agent_net, agent_net_target,
    mix_net,   mix_net_target,
    n_agents:  int,
    n_actions: int,
    optimizer,
) -> float:
    # obs_b: [bs, n_agents, obs_dim]  state_b: [bs, state_dim]
    assert all(t.obs[i] is not None for t in batch for i in range(n_agents)), \
        "buffer 中存在 None obs，transition 存储逻辑有误"
    obs_b        = torch.FloatTensor(
        np.array([[t.obs[i]      for i in range(n_agents)] for t in batch])
    ).to(DEVICE)
    next_obs_b   = torch.FloatTensor(
        np.array([[t.next_obs[i] for i in range(n_agents)] for t in batch])
    ).to(DEVICE)
    state_b      = torch.FloatTensor(np.array([t.state      for t in batch])).to(DEVICE)
    next_state_b = torch.FloatTensor(np.array([t.next_state for t in batch])).to(DEVICE)
    actions_b    = torch.LongTensor( np.array([t.actions    for t in batch])).to(DEVICE)
    reward_b     = torch.FloatTensor(np.array([t.reward     for t in batch])).unsqueeze(1).to(DEVICE)
    done_b       = torch.FloatTensor(np.array([t.done       for t in batch])).unsqueeze(1).to(DEVICE)

    bs = len(batch)

    # 当前 Q 值
    obs_flat  = obs_b.view(bs * n_agents, -1)
    q_all     = agent_net(obs_flat).view(bs, n_agents, -1)          # [bs, n_agents, n_actions]
    agent_qs  = q_all.gather(2, actions_b.unsqueeze(2)).squeeze(2)  # [bs, n_agents]
    q_tot     = mix_net(agent_qs, state_b)                          # [bs, 1]

    # Target Q 值（与动作选择一致：carrying时只取投递区max，空手时只取货架max）
    with torch.no_grad():
        next_obs_flat = next_obs_b.view(bs * n_agents, -1)
        q_next_all    = agent_net_target(next_obs_flat).view(bs, n_agents, -1)
        n_shelves     = n_actions - 3
        carrying_next = (next_obs_b[:, :, 2:6].argmax(dim=2) != 0)          # [bs, n_agents]
        shelf_max     = q_next_all[:, :, :n_shelves].max(dim=2)[0]           # [bs, n_agents]
        delivery_max  = q_next_all[:, :, n_shelves:].max(dim=2)[0]           # [bs, n_agents]
        best_next_qs  = torch.where(carrying_next, delivery_max, shelf_max)  # [bs, n_agents]
        q_tot_next    = mix_net_target(best_next_qs, next_state_b)           # [bs, 1]
        target        = reward_b + GAMMA * (1 - done_b) * q_tot_next

    loss = nn.MSELoss()(q_tot, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent_net.parameters()) + list(mix_net.parameters()), 10.0
    )
    optimizer.step()
    return loss.item()


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train(model_type: str = 'attention', resume_from: str = None):
    env = WarehouseEnv(n_agents=N_AGENTS, n_shelves=N_SHELVES, max_steps=MAX_STEPS)

    if model_type == 'attention':
        agent_net        = AttentionAgentQNet(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            n_agents=N_AGENTS, n_shelves=N_SHELVES,
        ).to(DEVICE)
        agent_net_target = AttentionAgentQNet(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            n_agents=N_AGENTS, n_shelves=N_SHELVES,
        ).to(DEVICE)
        save_dir = 'checkpoints_v5_attn'
    else:
        agent_net        = MLPAgentQNet(env.obs_dim, env.n_actions, hidden=64).to(DEVICE)
        agent_net_target = MLPAgentQNet(env.obs_dim, env.n_actions, hidden=64).to(DEVICE)
        save_dir = 'checkpoints_v5_mlp'

    mix_net        = QMixNet(N_AGENTS, env.state_dim).to(DEVICE)
    mix_net_target = QMixNet(N_AGENTS, env.state_dim).to(DEVICE)

    agent_net_target.load_state_dict(agent_net.state_dict())
    mix_net_target.load_state_dict(mix_net.state_dict())

    optimizer = optim.Adam(
        list(agent_net.parameters()) + list(mix_net.parameters()), lr=LR
    )
    buffer = ReplayBuffer(BUFFER_SIZE)

    start_episode = 1
    if resume_from:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        agent_net.load_state_dict(ckpt['agent_net'])
        mix_net.load_state_dict(ckpt['mix_net'])
        agent_net_target.load_state_dict(ckpt['agent_net'])
        mix_net_target.load_state_dict(ckpt['mix_net'])
        start_episode = ckpt['episode'] + 1
        print(f'从 {resume_from} 续训，起始 episode={start_episode}')

    os.makedirs(save_dir, exist_ok=True)

    ep_rewards    = []
    ep_deliveries = []

    for episode in range(start_episode, N_EPISODES + 1):
        obs, state   = env.reset()
        needs_action = list(env.needs_action)
        ep_reward    = 0.0
        ep_steps     = 0
        epsilon      = get_epsilon(episode)

        # 每个 agent 上次决策时刻的快照（全体 agent 同一时刻的 obs/state/actions）
        snap_obs     = [None] * N_AGENTS
        snap_state   = [None] * N_AGENTS
        snap_actions = [None] * N_AGENTS

        # 团队累积奖励（经典 QMIX 协作信号）
        cum_reward = 0.0
        last_cum   = [0.0] * N_AGENTS

        for _ in range(MAX_STEPS):

            # ── 阶段1：为刚完成任务的 agent 选新动作 ────────────────────────
            actions = [-1] * N_AGENTS
            for i in range(N_AGENTS):
                if needs_action[i]:
                    actions[i] = select_action_for_agent(
                        obs[i], agent_net, epsilon, env.n_actions
                    )

            # ── 阶段2：存 transition（用快照 obs + 累积奖励） ────────────────
            # next_obs = 当前 obs（agent i 刚到达目标，准备接受新决策的时刻）
            # 所有 agent 的 obs 来自同一时刻，时间一致
            for i in range(N_AGENTS):
                if needs_action[i] and snap_obs[i] is not None:
                    buffer.push(
                        snap_obs[i],
                        snap_state[i],
                        snap_actions[i],
                        cum_reward - last_cum[i],   # 团队累积奖励
                        [o.copy() for o in obs],
                        state.copy(),
                        float(False)
                    )

            # ── 阶段3：更新快照（记录本次决策时刻的联合状态） ───────────────
            # joint_act：本次决策的动作向量
            #   - 刚完成的 agent：取新选的动作
            #   - 仍在路途的 agent：取其当前目标（env.agent_action[j]）
            joint_act = [
                actions[j] if actions[j] >= 0
                else (env.agent_action[j] if env.agent_action[j] >= 0 else 0)
                for j in range(N_AGENTS)
            ]
            for i in range(N_AGENTS):
                if needs_action[i]:
                    snap_obs[i]     = [o.copy() for o in obs]
                    snap_state[i]   = state.copy()
                    snap_actions[i] = list(joint_act)
                    last_cum[i]     = cum_reward

            # ── 阶段4：环境步进 ──────────────────────────────────────────────
            next_obs, next_state, reward, done, info, needs_action = env.step(actions)
            cum_reward += reward
            ep_reward  += reward

            obs, state = next_obs, next_state
            ep_steps  += 1

            if len(buffer) >= MIN_BUFFER and ep_steps % UPDATE_INTERVAL == 0:
                batch = buffer.sample(BATCH_SIZE)
                compute_loss(batch, agent_net, agent_net_target,
                             mix_net, mix_net_target, N_AGENTS, env.n_actions, optimizer)

            if done:
                break

        ep_rewards.append(ep_reward)
        ep_deliveries.append(info['deliveries'])

        if episode % TARGET_UPDATE == 0:
            agent_net_target.load_state_dict(agent_net.state_dict())
            mix_net_target.load_state_dict(mix_net.state_dict())

        if episode % 100 == 0:
            avg_r = np.mean(ep_rewards[-100:])
            avg_d = np.mean(ep_deliveries[-100:])
            # 采样一个 obs 检查 Q 值范围，验证网络是否在学习
            if len(buffer) >= MIN_BUFFER:
                sample_obs = torch.FloatTensor(buffer.buf[-1].obs[0]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    q_sample = agent_net(sample_obs)
                q_min, q_max = q_sample.min().item(), q_sample.max().item()
            else:
                q_min, q_max = 0.0, 0.0
            print(f'[{model_type}] Episode {episode:5d} | ε={epsilon:.3f} | '
                  f'平均奖励={avg_r:7.2f} | 平均送达={avg_d:.2f} | '
                  f'Q范围=[{q_min:.2f}, {q_max:.2f}]')

        if episode % SAVE_INTERVAL == 0:
            torch.save({
                'agent_net': agent_net.state_dict(),
                'mix_net':   mix_net.state_dict(),
                'episode':   episode,
                'model':     model_type,
            }, f'{save_dir}/qmix_ep{episode}.pt')
            print(f'  → 已保存: {save_dir}/qmix_ep{episode}.pt')

    # 最终模型和训练曲线
    torch.save({
        'agent_net': agent_net.state_dict(),
        'mix_net':   mix_net.state_dict(),
        'episode':   N_EPISODES,
        'model':     model_type,
    }, f'{save_dir}/qmix_final.pt')
    np.save(f'{save_dir}/ep_rewards.npy',    np.array(ep_rewards))
    np.save(f'{save_dir}/ep_deliveries.npy', np.array(ep_deliveries))

    print(f'\n训练完成 ({model_type})，模型和曲线已保存至 {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='attention', choices=['attention', 'mlp'],
                        help='网络类型：attention（主方案）或 mlp（对照基线）')
    parser.add_argument('--resume', default=None,
                        help='从 checkpoint 续训，如 checkpoints_v5_attn/qmix_ep1000.pt')
    args = parser.parse_args()
    train(model_type=args.model, resume_from=args.resume)
