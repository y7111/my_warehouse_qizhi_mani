#!/usr/bin/env python3
"""
plot_curves.py  ─  QMIX 训练曲线可视化

用法：
    python3 ~/catkin_ws/src/warehouse_astar/scripts/plot_curves.py

    # 指定检查点目录
    python3 plot_curves.py --dir ~/qmix_checkpoints

    # 只显示最近 N 局
    python3 plot_curves.py --last 200

输出：
    - 屏幕显示交互图窗
    - 同目录保存 training_curves.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # Gazebo/RViz 环境下用 TkAgg 后端
import matplotlib.pyplot as plt

CKPT_DIR = os.path.expanduser('~/qmix_checkpoints')


def moving_avg(data, window=10):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot(ckpt_dir: str, last_n: int, start_ep: int = 1, end_ep: int = 0):
    rewards_path = os.path.join(ckpt_dir, 'rewards.npy')
    losses_path  = os.path.join(ckpt_dir, 'losses.npy')

    if not os.path.exists(rewards_path):
        print(f'[错误] 找不到 {rewards_path}，请先运行 qmix_train.py')
        return

    rewards = np.load(rewards_path)
    losses  = np.load(losses_path) if os.path.exists(losses_path) else None

    total_eps = len(rewards)
    if last_n and last_n < total_eps:
        rewards  = rewards[-last_n:]
        losses   = losses[-last_n:] if losses is not None else None
        start_ep = start_ep + total_eps - last_n

    # --end 截断：只保留 start_ep 到 end_ep 的数据
    if end_ep > 0 and end_ep >= start_ep:
        keep = end_ep - start_ep + 1
        rewards = rewards[:keep]
        if losses is not None:
            losses = losses[:keep]

    eps = np.arange(start_ep, start_ep + len(rewards))

    # ── 画布布局 ──────────────────────────────────────────────────────
    n_plots = 2 if losses is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        f'QMIX Training Curves  '
        f'(ep {start_ep} - {start_ep + len(rewards) - 1},  total {total_eps} eps)',
        fontsize=14)

    # ── Reward curve ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, rewards, color='steelblue', alpha=0.35, linewidth=0.8, label='Episode reward')
    if len(rewards) >= 10:
        ma = moving_avg(rewards, 10)
        ax.plot(eps[9:], ma, color='steelblue', linewidth=2.0, label='Moving avg (10 eps)')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_ylabel('Total Reward')
    ax.set_xlabel('Episode')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Loss curve ────────────────────────────────────────────────────
    if losses is not None:
        ax = axes[1]
        ax.plot(eps, losses, color='tomato', alpha=0.35, linewidth=0.8, label='Avg loss per ep')
        if len(losses) >= 10:
            ml = moving_avg(losses, 10)
            ax.plot(eps[9:], ml, color='tomato', linewidth=2.0, label='Moving avg (10 eps)')
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Episode')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────
    out_path = os.path.join(ckpt_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[图表] 已保存到 {out_path}')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',   type=str, default=CKPT_DIR,
                        help='检查点目录（默认 ~/qmix_checkpoints）')
    parser.add_argument('--last',  type=int, default=0,
                        help='只显示最近 N 局（0=全部）')
    parser.add_argument('--start', type=int, default=1,
                        help='横坐标起始 episode')
    parser.add_argument('--end',   type=int, default=0,
                        help='横坐标截止 episode（0=不截断）')
    args = parser.parse_args()

    plot(os.path.expanduser(args.dir), args.last, args.start, args.end)
