#!/usr/bin/env python3
"""
qmix_eval.py  ─  QMIX 推理评估脚本（不训练，只跑模型）

用法：
    # 自动加载最新检查点
    rosrun warehouse_astar qmix_eval.py

    # 指定检查点
    rosrun warehouse_astar qmix_eval.py --ckpt ~/qmix_checkpoints/qmix_ep0200.pt

    # 运行指定局数后退出
    rosrun warehouse_astar qmix_eval.py --episodes 10
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import rospy

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from warehouse_env import WarehouseEnv
from qmix_network import AgentQNet

# ── 与训练脚本保持一致的超参 ──────────────────────────────────────────
OBS_DIM   = 9
N_ACTIONS = 5
HIDDEN    = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = os.path.expanduser('~/qmix_checkpoints')


def find_latest_ckpt(directory: str) -> str | None:
    """返回目录下编号最大的 .pt 文件路径，找不到返回 None。"""
    pts = sorted(glob.glob(os.path.join(directory, 'qmix_ep*.pt')))
    return pts[-1] if pts else None


def load_q_net(ckpt_path: str) -> AgentQNet:
    ck = torch.load(ckpt_path, map_location=DEVICE)
    net = AgentQNet(OBS_DIM, N_ACTIONS, HIDDEN).to(DEVICE)
    net.load_state_dict(ck['q_net'])
    net.eval()
    ep = ck.get('episode', '?')
    eps = ck.get('epsilon', '?')
    rospy.loginfo('[Eval] 加载检查点 %s  (ep=%s, ε_train=%.3f)',
                  os.path.basename(ckpt_path), ep, eps if isinstance(eps, float) else 0)
    return net


@torch.no_grad()
def select_actions(net: AgentQNet, obs_list):
    """纯贪心策略（ε=0）。"""
    actions = []
    for obs in obs_list:
        o = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        q = net(o)
        actions.append(int(q.argmax(dim=1).item()))
    return actions


ACTION_NAMES = {0: '→A取货点', 1: '→B取货点', 2: '↑取北侧', 3: '↓取南侧', 4: '⟲返回'}


def run_eval(ckpt_path: str, n_episodes: int):
    env = WarehouseEnv()
    net = load_q_net(ckpt_path)

    total_rewards = []
    total_picks   = []

    for ep in range(1, n_episodes + 1):
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        rospy.loginfo('[Eval] ══ Episode %d 开始 ══', ep)

        while not done and not rospy.is_shutdown():
            actions = select_actions(net, obs)

            act_str = '  '.join(
                f'{"α" if i==0 else "β"}:{ACTION_NAMES[a]}'
                for i, a in enumerate(actions)
            )
            rospy.loginfo('[Eval] step %2d  动作: %s', step + 1, act_str)

            obs, rewards, done, info = env.step(actions)
            ep_reward += rewards[0]
            step += 1

        picked = sum(info['shelf_picked'])
        total_rewards.append(ep_reward)
        total_picks.append(picked)

        rospy.loginfo(
            '[Eval] Episode %d 结束 | 奖励 %.1f | 取货 %d/4 | 步数 %d | 货架 %s',
            ep, ep_reward, picked, info['step'],
            ''.join('✓' if s else '✗' for s in info['shelf_picked']),
        )

    # ── 汇总统计 ──────────────────────────────────────────────────────
    rospy.loginfo('=' * 55)
    rospy.loginfo('[Eval] 评估完成  共 %d 局', n_episodes)
    rospy.loginfo('[Eval] 平均奖励  %.1f ± %.1f',
                  np.mean(total_rewards), np.std(total_rewards))
    rospy.loginfo('[Eval] 平均取货  %.2f / 4  (完成率 %.0f%%)',
                  np.mean(total_picks),
                  100.0 * sum(p == 4 for p in total_picks) / max(1, n_episodes))
    rospy.loginfo('=' * 55)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='.pt 检查点路径（默认自动加载最新）')
    parser.add_argument('--episodes', type=int, default=5,
                        help='评估局数（默认 5）')
    args, _ = parser.parse_known_args()

    rospy.init_node('qmix_eval', anonymous=True)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = find_latest_ckpt(CKPT_DIR)
        if ckpt is None:
            rospy.logerr('[Eval] 未找到检查点，请先训练或用 --ckpt 指定路径')
            sys.exit(1)
        rospy.loginfo('[Eval] 自动选择最新检查点: %s', os.path.basename(ckpt))

    run_eval(os.path.expanduser(ckpt), args.episodes)
