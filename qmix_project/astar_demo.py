"""
astar_demo.py — A* + 固定任务队列，完成一轮4货物搬运

任务分配：
  AGV0: 货架2(6,2) → 货架0(2,2)  完成后停到(9,5)
  AGV1: 货架3(6,9) → 货架1(2,9)  完成后停到(9,7)

直接操作 env 内部状态，绕过 action 空间限制，避免"完成后停在 goal 口堵路"问题。
"""

import sys, os, time, random
from typing import Optional, Tuple
sys.path.insert(0, os.path.dirname(__file__))
from warehouse_env import WarehouseEnv, GOODS_NONE, GOODS_A, GOODS_B

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ── 停车位（row8 中段/右段，不在任何关键通道上）─────────────────────────────
# row9 中 col2(goal_A)、col7 均是唯一横向通道，不能停；
# row8 中 (8,3)(8,7) 是柱子，将 row8 分成三段，段内不堵路。
PARK = [(8, 5), (8, 8)]


# ── 智能体状态机 ─────────────────────────────────────────────────────────────

class AgentFSM:
    """每辆车的任务状态机，维护待执行货架队列。"""

    def __init__(self, agent_id: int, shelf_queue: list):
        self.id          = agent_id
        self.queue       = list(shelf_queue)   # 货架索引列表
        self.state       = 'fetch'             # fetch | deliver | park | done
        self.deliveries  = 0
        self.log         = []

    def next_dest(self, env) -> Optional[Tuple]:
        """返回下一个目标坐标；None 表示已完成，不再移动。"""
        carrying = env.agent_carrying[self.id]

        if self.state == 'done':
            return None

        if self.state == 'fetch':
            if not self.queue:
                self.state = 'park'
                return PARK[self.id]
            shelf_idx = self.queue[0]            # 先看，不弹出
            return env.shelf_positions[shelf_idx]

        if self.state == 'deliver':
            if carrying == GOODS_A:
                return env.target_a
            if carrying == GOODS_B:
                return env.target_b
            # 空载到了 deliver 状态 → 继续 fetch
            self.state = 'fetch'
            return self.next_dest(env)

        if self.state == 'park':
            return PARK[self.id]

        return None

    def on_arrival(self, env, pos: tuple) -> bool:
        """
        处理到达事件，更新携带状态。
        返回 True 表示需要重新规划下一段路径。
        """
        carrying = env.agent_carrying[self.id]

        # ── 到达货架 ─────────────────────────────────────────────────────────
        if pos in env.shelf_positions and self.state == 'fetch':
            shelf_idx = env.shelf_positions.index(pos)
            if self.queue and self.queue[0] == shelf_idx:
                self.queue.pop(0)
            if carrying == GOODS_NONE:
                env.agent_carrying[self.id] = env.shelf_goods[shelf_idx]
                env.shelf_goods[shelf_idx]  = random.choice([GOODS_A, GOODS_B])
                self.state = 'deliver'
                g = env.agent_carrying[self.id]
                self.log.append(f'取货{"A" if g==GOODS_A else "B"} @ {pos}')
            return True

        # ── 到达目标点 ───────────────────────────────────────────────────────
        if pos == env.target_a:
            if carrying == GOODS_A:
                self.deliveries += 1
                self.log.append(f'送达A @ {pos}')
            env.agent_carrying[self.id] = GOODS_NONE
            self.state = 'fetch' if self.queue else 'park'
            return True

        if pos == env.target_b:
            if carrying == GOODS_B:
                self.deliveries += 1
                self.log.append(f'送达B @ {pos}')
            env.agent_carrying[self.id] = GOODS_NONE
            self.state = 'fetch' if self.queue else 'park'
            return True

        # ── 到达停车位 ───────────────────────────────────────────────────────
        if pos == PARK[self.id] and self.state == 'park':
            self.state = 'done'
            self.log.append(f'停车 @ {pos}')
            return False

        return False


# ── 路径分配 ─────────────────────────────────────────────────────────────────

def plan_path(env, agent_id: int, dest: tuple):
    """用 A* 为 agent_id 规划到 dest 的路径，其他车的当前位置作为动态障碍。"""
    start   = tuple(env.agent_pos[agent_id])
    blocked = {tuple(env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
    path    = env._astar(start, dest, blocked)
    if not path:                               # 动态障碍堵死时退回无动态障碍版本
        path = env._astar(start, dest, set())
    env.agent_paths[agent_id] = path
    env.agent_dest[agent_id]  = dest


# ── 可视化 ───────────────────────────────────────────────────────────────────

CLR = dict(
    wall='#2C2C2C', empty='#F5F5DC',
    shelf_A='#4FC3F7', shelf_B='#EF9A9A', shelf_done='#B0BEC5',
    goal_a='#81C784', goal_b='#FFB74D',
    agv0='#7B1FA2', agv1='#C62828',
    path0='#CE93D8', path1='#EF9A9A',
)


def draw(ax, env, fsms, step, total_del, target_del):
    ax.cla()
    H, W = env.H, env.W
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis()
    ax.set_aspect('equal'); ax.axis('off')

    # 底色 + 网格
    ax.add_patch(plt.Rectangle((0,0), W, H, color=CLR['empty'], zorder=0))
    for r in range(H+1): ax.axhline(r, color='#CCC', lw=0.4, zorder=1)
    for c in range(W+1): ax.axvline(c, color='#CCC', lw=0.4, zorder=1)

    # 墙
    for (r,c) in env.static_walls:
        ax.add_patch(plt.Rectangle((c,r), 1, 1, color=CLR['wall'], zorder=2))

    # 货架
    visited = set()
    for fsm in fsms:
        if fsm.state in ('deliver','park','done'):
            idx = [env.shelf_positions.index(env.shelf_positions[q])
                   for q in range(env.n_shelves)
                   if env.shelf_positions[q] not in [env.shelf_positions[x] for x in fsm.queue]]
    all_visited = {env.shelf_positions[k] for fsm in fsms
                   for k in range(env.n_shelves)
                   if k not in fsm.queue and fsm.id==0 or k not in fsm.queue and fsm.id==1}

    done_shelves = set()
    for fsm in fsms:
        original = [2,0] if fsm.id==0 else [3,1]
        remaining = set(fsm.queue)
        for s in original:
            if s not in remaining:
                done_shelves.add(s)

    for k, pos in enumerate(env.shelf_positions):
        r, c = pos
        g = env.shelf_goods[k]
        color = CLR['shelf_done'] if k in done_shelves else \
                (CLR['shelf_A'] if g==GOODS_A else CLR['shelf_B'])
        ax.add_patch(plt.Rectangle((c,r),1,1, color=color, zorder=2))
        ax.text(c+.5, r+.5, ['?','A','B'][g],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', zorder=5)

    # 目标点
    for pos, label, color in [
        (env.target_a, 'A\n终点', CLR['goal_a']),
        (env.target_b, 'B\n终点', CLR['goal_b']),
    ]:
        r, c = pos
        ax.add_patch(plt.Rectangle((c,r),1,1, color=color, zorder=2))
        ax.text(c+.5, r+.5, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#333', zorder=5)

    # 停车位标记
    for i, (pr,pc) in enumerate(PARK):
        ax.add_patch(plt.Rectangle((pc+.1,pr+.1),.8,.8,
                     fill=False, edgecolor=['#7B1FA2','#C62828'][i],
                     linestyle='--', lw=1, zorder=3))

    # 路径
    path_colors = [CLR['path0'], CLR['path1']]
    for i, path in enumerate(env.agent_paths):
        for (pr,pc) in path:
            ax.add_patch(plt.Rectangle((pc+.15,pr+.15),.7,.7,
                         color=path_colors[i], alpha=.3, zorder=3))

    # AGV
    agv_colors = [CLR['agv0'], CLR['agv1']]
    carry_label = ['','A','B']
    for i, pos in enumerate(env.agent_pos):
        r, c = pos
        ax.add_patch(plt.Circle((c+.5,r+.5), .38, color=agv_colors[i], zorder=6))
        carry = env.agent_carrying[i]
        lbl = f'{i}\n{carry_label[carry]}' if carry else str(i)
        ax.text(c+.5, r+.5, lbl, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=7)

    # 状态标题
    state_str = [f'AGV{fsm.id}[{fsm.state}]={["空","A","B"][env.agent_carrying[fsm.id]]}'
                 for fsm in fsms]
    ax.set_title(
        f'Step {step}  |  送达 {total_del}/{target_del}  |  '
        + '  '.join(state_str),
        fontsize=9, pad=3
    )


# ── 主仿真循环 ────────────────────────────────────────────────────────────────

def run(render_mode='matplotlib', delay=0.15, seed=None):
    if seed is not None:
        random.seed(seed)

    env = WarehouseEnv(n_agents=2, n_shelves=4, max_steps=800)
    env.reset()

    # 固定任务队列：AGV0 负责左侧货架，AGV1 负责右侧货架
    fsms = [
        AgentFSM(0, [2, 0]),   # shelf2(6,2) → shelf0(2,2)
        AgentFSM(1, [3, 1]),   # shelf3(6,9) → shelf1(2,9)
    ]

    TARGET = 4   # 一轮 = 4次送达
    MAX_STEPS = 600

    print(f'\n{"="*55}')
    print(f'  A* + 固定任务分配  |  目标：{TARGET} 次送达')
    print(f'  货架初始: { ["A" if g==GOODS_A else "B" for g in env.shelf_goods] }')
    print(f'  AGV0: 货架2→0  |  AGV1: 货架3→1')
    print(f'{"="*55}\n')

    # 初始化路径
    for fsm in fsms:
        dest = fsm.next_dest(env)
        if dest:
            plan_path(env, fsm.id, dest)

    fig, ax = None, None
    if render_mode == 'matplotlib' and HAS_MPL:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))

    step = 0
    total_del = 0

    while total_del < TARGET and step < MAX_STEPS:
        step += 1
        env.step_count = step

        # 1. 移动
        env._move_agents()

        # 2. 到达检测
        for fsm in fsms:
            if fsm.state == 'done':
                env.agent_dest[fsm.id]  = None
                env.agent_paths[fsm.id] = []
                continue

            pos  = tuple(env.agent_pos[fsm.id])
            dest = env.agent_dest[fsm.id]

            if dest and pos == dest:
                need_replan = fsm.on_arrival(env, pos)
                if need_replan:
                    new_dest = fsm.next_dest(env)
                    if new_dest:
                        plan_path(env, fsm.id, new_dest)
                        print(f'  [Step {step:3d}] AGV{fsm.id} → {new_dest}  '
                              f'状态={fsm.state}  '
                              f'携={["空","A","B"][env.agent_carrying[fsm.id]]}')
                    else:
                        env.agent_dest[fsm.id]  = None
                        env.agent_paths[fsm.id] = []
                else:
                    env.agent_dest[fsm.id]  = None
                    env.agent_paths[fsm.id] = []

        total_del = sum(fsm.deliveries for fsm in fsms)

        # 3. 渲染
        if render_mode == 'terminal' and step % 1 == 0:
            env.render()
            time.sleep(delay)
        elif render_mode == 'matplotlib' and fig is not None:
            draw(ax, env, fsms, step, total_del, TARGET)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    # ── 结果 ─────────────────────────────────────────────────────────────────
    print(f'\n{"="*55}')
    if total_del >= TARGET:
        print(f'  完成！共用 {step} 步，{TARGET} 件货物全部送达')
    else:
        print(f'  超时！{step} 步后仅完成 {total_del}/{TARGET} 次送达')

    for fsm in fsms:
        print(f'  AGV{fsm.id}: {fsm.deliveries} 次送达  |  ' + ' → '.join(fsm.log))
    print(f'{"="*55}\n')

    if render_mode == 'matplotlib' and fig is not None:
        draw(ax, env, fsms, step, total_del, TARGET)
        fig.canvas.draw()
        plt.ioff()
        plt.show()

    return step, total_del


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['matplotlib','terminal','none'],
                   default='matplotlib')
    p.add_argument('--delay', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    if args.mode == 'matplotlib' and not HAS_MPL:
        print('[警告] matplotlib 不可用，切换 terminal 模式')
        args.mode = 'terminal'

    run(render_mode=args.mode, delay=args.delay, seed=args.seed)
