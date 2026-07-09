"""
run_compare.py — 任务分配策略对比实验主程序

对 random / nearest / roundrobin / qmix 四种策略，在【完全相同的环境随机序列】
（固定种子）下各跑 N 局，统计四项指标并输出 CSV + 控制台表格：

  吞吐 deliveries   每局送达件数（越高越好）—— 核心指标
  行驶 moves        每局累计移动格数（越低越好）—— 空驶/能耗
  冲突 conflicts    每局让步+卡死重规划次数（越低越好）—— 协同顺畅度
  等待 avg_wait     货物“上架→被取走”平均步数（越低越好）—— 响应速度

用法:
  cd qmix_project
  python3 experiments/run_compare.py --episodes 100 \
          --ckpt checkpoints_v5_attn/qmix_ep800.pt
"""

import os
import sys
import csv
import argparse
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warehouse_env import WarehouseEnv
from qmix_network import AttentionAgentQNet
from experiments.baselines import build_policy

N_AGENTS  = 4
N_SHELVES = 10
MAX_STEPS = 300
DEVICE    = torch.device('cpu')


def make_env(restock):
    return WarehouseEnv(n_agents=N_AGENTS, n_shelves=N_SHELVES,
                        max_steps=MAX_STEPS, restock_delay=restock)


def load_qmix(ckpt_path):
    env = WarehouseEnv(n_agents=N_AGENTS, n_shelves=N_SHELVES, max_steps=MAX_STEPS)
    net = AttentionAgentQNet(env.obs_dim, env.n_actions,
                             n_agents=N_AGENTS, n_shelves=N_SHELVES).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    net.load_state_dict(ckpt['agent_net'])
    net.eval()
    return net


def run_episode(env, policy, seed):
    # 固定 env 随机序列：所有策略同一个 seed → 面对同样的初始库存与刷新序列
    random.seed(seed)
    np.random.seed(seed)
    obs, state = env.reset()
    needs = list(env.needs_action)

    for _ in range(MAX_STEPS):
        actions = [-1] * env.n_agents
        for i in range(env.n_agents):
            if needs[i]:
                actions[i] = policy(env, i)
        obs, state, reward, done, info, needs = env.step(actions)
        if done:
            break

    waits = info['wait_times']
    return {
        'deliveries': info['deliveries'],
        'moves':      info['moves'],
        'conflicts':  info['conflicts'],
        'avg_wait':   float(np.mean(waits)) if waits else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--ckpt', default='checkpoints_v5_attn/qmix_ep800.pt')
    ap.add_argument('--out', default='experiments/results.csv')
    ap.add_argument('--restock', type=int, nargs='+', default=[40],
                    help='货架补货空置步数，可给多个做缺货扫描，如 --restock 20 40 80 120')
    args = ap.parse_args()

    agent_net = load_qmix(args.ckpt)
    kinds = ['random', 'roundrobin', 'nearest', 'qmix']
    seeds = list(range(1000, 1000 + args.episodes))   # 跨策略复用同一组种子
    metrics = ['deliveries', 'moves', 'conflicts', 'avg_wait']
    label = {'deliveries': '吞吐↑', 'moves': '行驶↓', 'conflicts': '冲突↓', 'avg_wait': '等待↓'}

    raw_rows = []
    for restock in args.restock:
        env = make_env(restock)
        summary = {}
        for kind in kinds:
            policy = build_policy(kind, agent_net=agent_net, device=DEVICE)
            acc = {m: [] for m in metrics}
            for s in seeds:
                r = run_episode(env, policy, s)
                for m in metrics:
                    acc[m].append(r[m])
                raw_rows.append({'restock': restock, 'policy': kind, 'seed': s, **r})
            summary[kind] = {m: (float(np.mean(acc[m])), float(np.std(acc[m]))) for m in metrics}

        print('\n══════ 补货空置={}步  对比结果（{}局, mean±std）══════'.format(restock, args.episodes))
        header = '{:12s}'.format('策略') + ''.join('{:>16s}'.format(label[m]) for m in metrics)
        print(header)
        print('-' * len(header))
        for kind in kinds:
            row = '{:12s}'.format(kind)
            for m in metrics:
                mu, sd = summary[kind][m]
                row += '{:>16s}'.format('{:.1f}±{:.1f}'.format(mu, sd))
            print(row)
        qd, nd = summary['qmix']['deliveries'][0], summary['nearest']['deliveries'][0]
        if nd > 0:
            print('  → QMIX 吞吐相对“就近贪心”: {:+.1f}%'.format((qd - nd) / nd * 100))

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['restock', 'policy', 'seed'] + metrics)
        w.writeheader()
        w.writerows(raw_rows)
    print('\n明细已写入 {}'.format(args.out))


if __name__ == '__main__':
    main()
