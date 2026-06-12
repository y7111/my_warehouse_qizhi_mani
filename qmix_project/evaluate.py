"""
evaluate.py — 贪婪策略评估
用法: python evaluate.py checkpoints_v3/qmix_ep1000.pt [--episodes 100]
"""

import argparse, random
import numpy as np
import torch

from warehouse_env import WarehouseEnv
from qmix_network import AgentQNet, QMixNet

N_AGENTS  = 2
N_SHELVES = 4
MAX_STEPS = 300
HIDDEN    = 64
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    env  = WarehouseEnv(n_agents=N_AGENTS, n_shelves=N_SHELVES, max_steps=MAX_STEPS)
    agent_net = AgentQNet(env.obs_dim, env.n_actions, HIDDEN).to(DEVICE)
    mix_net   = QMixNet(N_AGENTS, env.state_dim, hyper_hidden=HIDDEN).to(DEVICE)
    agent_net.load_state_dict(ckpt['agent_net'])
    mix_net.load_state_dict(ckpt['mix_net'])
    agent_net.eval()
    mix_net.eval()
    return agent_net, mix_net, env, ckpt.get('episode', '?')


def greedy_action(obs, agent_net):
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        return int(agent_net(obs_t).argmax().item())


def run_episode(env, agent_net):
    obs, state = env.reset()
    needs_action = list(env.needs_action)
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        actions = [-1] * N_AGENTS
        for i in range(N_AGENTS):
            if needs_action[i]:
                actions[i] = greedy_action(obs[i], agent_net)

        obs, state, reward, done, info, needs_action = env.step(actions)
        total_reward += reward
        if done:
            break

    return total_reward, info['deliveries']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='checkpoint 文件路径')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    agent_net, mix_net, env, ep = load_model(args.checkpoint)
    print(f'已加载: {args.checkpoint}  (训练到 ep{ep})')
    print(f'贪婪评估 {args.episodes} 个 episode...\n')

    rewards, deliveries = [], []
    for i in range(1, args.episodes + 1):
        r, d = run_episode(env, agent_net)
        rewards.append(r)
        deliveries.append(d)
        if i % 20 == 0:
            print(f'  [{i:3d}/{args.episodes}] '
                  f'avg_reward={np.mean(rewards[-20:]):7.2f}  '
                  f'avg_deliver={np.mean(deliveries[-20:]):.2f}')

    print(f'\n===== 最终评估结果 =====')
    print(f'平均奖励:   {np.mean(rewards):.2f}  (±{np.std(rewards):.2f})')
    print(f'平均送达数: {np.mean(deliveries):.2f}  (±{np.std(deliveries):.2f})')
    print(f'最大送达:   {max(deliveries)}')
    print(f'最小送达:   {min(deliveries)}')
    print(f'送达≥10 比例: {sum(1 for d in deliveries if d >= 10) / len(deliveries) * 100:.1f}%')


if __name__ == '__main__':
    main()
