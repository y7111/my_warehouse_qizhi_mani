"""
plot_compare.py — 读取 results.csv，出对比图（答辩/报告用）

  fig1  metrics_vs_scarcity.png : 四项指标随“缺货程度（补货空置步数）”的变化曲线
  fig2  throughput_box.png      : 训练工况(补货=40)下各策略吞吐分布箱线图（看稳定性）

用法: cd qmix_project && python3 experiments/plot_compare.py
"""

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, 'results.csv')

POLICIES = ['random', 'roundrobin', 'nearest', 'qmix']
PLABEL   = {'random': '随机分配', 'roundrobin': '轮转分配',
            'nearest': '就近贪心', 'qmix': 'QMIX(本作品)'}
PCOLOR   = {'random': '#9e9e9e', 'roundrobin': '#ffb300',
            'nearest': '#42a5f5', 'qmix': '#e53935'}
METRICS  = [('deliveries', '吞吐（送达件数/局）↑'),
            ('conflicts',  '冲突次数/局 ↓'),
            ('moves',      '行驶格数/局 ↓'),
            ('avg_wait',   '平均等待步数 ↓')]


def load():
    rows = defaultdict(list)   # (restock, policy, metric) -> [values]
    with open(CSV) as f:
        for r in csv.DictReader(f):
            for m, _ in METRICS:
                rows[(int(r['restock']), r['policy'], m)].append(float(r[m]))
    return rows


def main():
    rows = load()
    restocks = sorted({k[0] for k in rows})

    # ── fig1: 指标 vs 缺货程度 ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (m, title) in zip(axes.flat, METRICS):
        for p in POLICIES:
            mu  = np.array([np.mean(rows[(rs, p, m)]) for rs in restocks])
            sem = np.array([np.std(rows[(rs, p, m)]) / np.sqrt(len(rows[(rs, p, m)]))
                            for rs in restocks])   # 标准误：均值差异是否显著的正确刻度
            ax.plot(restocks, mu, '-o', color=PCOLOR[p], label=PLABEL[p], lw=2)
            ax.fill_between(restocks, mu - sem, mu + sem, color=PCOLOR[p], alpha=0.15)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('补货空置步数（越大越缺货 →）')
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=10, loc='best')
    fig.suptitle('四种任务分配策略对比（每点 100 局均值，阴影=±1标准误）', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(HERE, 'metrics_vs_scarcity.png')
    fig.savefig(out1, dpi=130)
    print('已保存', out1)

    # ── fig2: 吞吐分布箱线图（训练工况 restock=40，没有则取中间档）──
    target = 40 if 40 in restocks else restocks[len(restocks) // 2]
    data = [rows[(target, p, 'deliveries')] for p in POLICIES]
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, patch_artist=True, labels=[PLABEL[p] for p in POLICIES],
                    medianprops=dict(color='black'))
    for patch, p in zip(bp['boxes'], POLICIES):
        patch.set_facecolor(PCOLOR[p]); patch.set_alpha(0.7)
    ax.set_title('补货空置={}步 下各策略吞吐分布（箱体越短越稳定）'.format(target), fontsize=13)
    ax.set_ylabel('送达件数/局')
    ax.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    out2 = os.path.join(HERE, 'throughput_box.png')
    fig2.savefig(out2, dpi=130)
    print('已保存', out2)


if __name__ == '__main__':
    main()
