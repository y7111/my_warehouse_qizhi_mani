"""
plot_compare_en.py — English-labeled version of plot_compare.py (avoids CJK font issues).
Reads results.csv and overwrites the two figures with all-English text.

  metrics_vs_scarcity.png : four metrics vs scarcity (restock idle steps)
  throughput_box.png      : throughput distribution at training condition (restock=40)

Usage: cd qmix_project && python3 experiments/plot_compare_en.py
"""

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, 'results.csv')

POLICIES = ['random', 'roundrobin', 'nearest', 'qmix']
PLABEL   = {'random': 'Random', 'roundrobin': 'Round-Robin',
            'nearest': 'Nearest-Greedy', 'qmix': 'QMIX (ours)'}
PCOLOR   = {'random': '#9e9e9e', 'roundrobin': '#ffb300',
            'nearest': '#42a5f5', 'qmix': '#e53935'}
METRICS  = [('deliveries', 'Throughput (deliveries / episode)  (higher is better)'),
            ('conflicts',  'Conflicts / episode  (lower is better)'),
            ('moves',      'Travel (grid steps) / episode  (lower is better)'),
            ('avg_wait',   'Average wait (steps)  (lower is better)')]


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

    # -- fig1: metrics vs scarcity --
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (m, title) in zip(axes.flat, METRICS):
        for p in POLICIES:
            mu  = np.array([np.mean(rows[(rs, p, m)]) for rs in restocks])
            sem = np.array([np.std(rows[(rs, p, m)]) / np.sqrt(len(rows[(rs, p, m)]))
                            for rs in restocks])
            ax.plot(restocks, mu, '-o', color=PCOLOR[p], label=PLABEL[p], lw=2)
            ax.fill_between(restocks, mu - sem, mu + sem, color=PCOLOR[p], alpha=0.15)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Restock idle steps  (larger = scarcer ->)')
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=10, loc='best')
    fig.suptitle('Comparison of Four Task-Allocation Policies '
                 '(100 episodes / point, shaded = +/-1 SEM)', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(HERE, 'metrics_vs_scarcity.png')
    fig.savefig(out1, dpi=130)
    print('saved', out1)

    # -- fig2: throughput boxplot (training condition restock=40) --
    target = 40 if 40 in restocks else restocks[len(restocks) // 2]
    data = [rows[(target, p, 'deliveries')] for p in POLICIES]
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, patch_artist=True, labels=[PLABEL[p] for p in POLICIES],
                    medianprops=dict(color='black'))
    for patch, p in zip(bp['boxes'], POLICIES):
        patch.set_facecolor(PCOLOR[p]); patch.set_alpha(0.7)
    ax.set_title('Throughput Distribution at Restock={} steps '
                 '(shorter box = more stable)'.format(target), fontsize=13)
    ax.set_ylabel('Deliveries / episode')
    ax.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    out2 = os.path.join(HERE, 'throughput_box.png')
    fig2.savefig(out2, dpi=130)
    print('saved', out2)


if __name__ == '__main__':
    main()
