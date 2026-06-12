"""
warehouse_env.py — 多AGV仓库 2D 网格仿真（v5）

地图：20×16 十字形+多走廊（H=16, W=20）
  北臂 R0-R4：中间竖廊 col7-12，瓶颈 R3 仅 col8/col11 可通
  横臂 R5-R9：全通，R7 有走廊隔墙（col2-4，col15-17）
  南臂 R10-R15：对称北臂，瓶颈 R11 仅 col8/col11 可通

关键坐标（来自 PLAN.md，已确认）：
  货架（10个）: (1,8)(1,11)(6,1)(6,4)(8,1)(8,4)(6,15)(6,18)(8,15)(8,18)
  投递区: A=(13,7)  B=(13,9)  C=(13,11)
  机器人起点: (15,8)(15,9)(15,11)(15,12)

观测向量（87维）：
  self_feat(19) = pos(2) + carrying_onehot(4) + dest_onehot(13)
  other_feats(18) = 3车 × (pos(2) + carrying_onehot(4))
  shelf_feats(50) = 10货架 × (goods_onehot(3) + pos(2))

全局状态（74维）：
  4车 × (pos(2) + carrying_onehot(4)) + 10货架 × (goods_onehot(3) + pos(2))
"""

import heapq
import random
import numpy as np
from typing import List, Tuple, Optional, Set


GOODS_NONE = 0
GOODS_A    = 1
GOODS_B    = 2
GOODS_C    = 3

RESTOCK_DELAY = 40   # 货架被取货后空置步数，逼出智能体对库存状态的感知

DEFAULT_SHELF_POSITIONS = [
    (1, 8), (1, 11),
    (6, 1), (6, 4), (8, 1), (8, 4),
    (6, 15), (6, 18), (8, 15), (8, 18),
]

DEFAULT_DELIVERY_ZONES = {
    GOODS_A: (13, 7),
    GOODS_B: (13, 9),
    GOODS_C: (13, 11),
}

DEFAULT_AGENT_STARTS = [(15, 8), (15, 9), (15, 11), (15, 12)]


def _build_default_walls() -> Set[Tuple[int, int]]:
    walls = set()
    for r in range(16):
        for c in range(20):
            # 北臂 R0-R4：col7-12 为走廊，其余为墙
            if 0 <= r <= 4 and not (7 <= c <= 12):
                walls.add((r, c))
            # 北臂瓶颈 R3：走廊内 col7/9/10/12 封堵，只留 col8 和 col11
            if r == 3 and c in (7, 9, 10, 12):
                walls.add((r, c))
            # 南臂 R10-R15：对称北臂
            if 10 <= r <= 15 and not (7 <= c <= 12):
                walls.add((r, c))
            # 南臂瓶颈 R11
            if r == 11 and c in (7, 9, 10, 12):
                walls.add((r, c))
            # 横臂走廊隔墙 R7
            if r == 7 and c in (2, 3, 4, 15, 16, 17):
                walls.add((r, c))
    return walls


DEFAULT_STATIC_WALLS = _build_default_walls()


class WarehouseEnv:
    """
    多AGV仓库 2D 网格仿真，gym 风格接口。

    接口：
        obs_list, state = env.reset()
        obs_list, state, reward, done, info, needs_action = env.step(actions)

    needs_action[i] == True 时，agent i 完成当前任务，QMIX 需要为其分配新目标。
    """

    def __init__(
        self,
        n_agents:    int                           = 4,
        n_shelves:   int                           = 10,
        max_steps:   int                           = 300,
        shelf_positions: Optional[List[Tuple]]     = None,
        delivery_zones:  Optional[dict]            = None,
        agent_starts:    Optional[List[Tuple]]     = None,
        static_walls:    Optional[Set[Tuple]]      = None,
    ):
        self.H, self.W   = 16, 20
        self.n_agents    = n_agents
        self.n_shelves   = n_shelves
        self.max_steps   = max_steps

        self.shelf_positions = (shelf_positions or DEFAULT_SHELF_POSITIONS)[:n_shelves]
        self.delivery_zones  = delivery_zones or DEFAULT_DELIVERY_ZONES
        self.agent_starts    = (agent_starts or DEFAULT_AGENT_STARTS)[:n_agents]
        self.static_walls    = static_walls if static_walls is not None else DEFAULT_STATIC_WALLS

        # action: 0..n_shelves-1 → 货架；n_shelves/+1/+2 → 投递区 A/B/C
        self.n_actions = n_shelves + 3
        self.ACTION_A  = n_shelves
        self.ACTION_B  = n_shelves + 1
        self.ACTION_C  = n_shelves + 2

        # obs_dim = self(2+4+n_actions) + other×(n_agents-1)×6 + shelf×n_shelves×5
        self.obs_dim   = (2 + 4 + self.n_actions) + 6 * (n_agents - 1) + 5 * n_shelves
        # state_dim = 4×6 + 10×5 = 74
        self.state_dim = 6 * n_agents + 5 * n_shelves

        # 运行时状态（reset 后初始化）
        self.agent_pos      = None
        self.agent_carrying = None
        self.agent_dest     = None
        self.agent_paths    = None
        self.agent_action   = None  # 当前被分配的动作编号（-1 表示无任务）
        self.needs_action   = None
        self.shelf_goods         = None
        self.shelf_restock_timer = None   # >0 表示空置剩余步数
        self._stuck_count        = None
        self.step_count     = 0
        self.total_deliveries = 0

    # ═══════════════════════════════════════════════════════════════════════════
    #  Gym 接口
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self):
        self.step_count       = 0
        self.total_deliveries = 0

        self.shelf_goods         = [random.randint(1, 3) for _ in range(self.n_shelves)]
        self.shelf_restock_timer = [0] * self.n_shelves
        self.agent_pos      = list(self.agent_starts)
        self.agent_carrying = [GOODS_NONE] * self.n_agents
        self.agent_dest     = [None]       * self.n_agents
        self.agent_paths    = [[]          for _ in range(self.n_agents)]
        self.agent_action   = [-1]         * self.n_agents
        self.needs_action   = [True]       * self.n_agents
        self._stuck_count   = [0]          * self.n_agents

        return self._get_obs(), self._get_state()

    def step(self, actions: List[int]):
        for i in range(self.n_agents):
            if self.needs_action[i]:
                self._set_destination(i, actions[i])

        self._move_agents()

        # 货架补货计时器
        for k in range(self.n_shelves):
            if self.shelf_restock_timer[k] > 0:
                self.shelf_restock_timer[k] -= 1
                if self.shelf_restock_timer[k] == 0:
                    self.shelf_goods[k] = random.randint(1, 3)

        reward = -0.05
        self.needs_action = [False] * self.n_agents

        for i in range(self.n_agents):
            if self.agent_dest[i] is None:
                continue
            if tuple(self.agent_pos[i]) == tuple(self.agent_dest[i]):
                r, triggered = self._handle_arrival(i)
                reward += r
                if triggered:
                    self.needs_action[i] = True

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            'deliveries':  self.total_deliveries,
            'step':        self.step_count,
            'carrying':    list(self.agent_carrying),
            'shelf_goods': list(self.shelf_goods),
        }

        return self._get_obs(), self._get_state(), reward, done, info, list(self.needs_action)

    # ═══════════════════════════════════════════════════════════════════════════
    #  内部逻辑
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_destination(self, agent_id: int, action: int):
        assert action >= 0
        if action < self.n_shelves:
            dest = self.shelf_positions[action]
        elif action == self.ACTION_A:
            dest = self.delivery_zones[GOODS_A]
        elif action == self.ACTION_B:
            dest = self.delivery_zones[GOODS_B]
        else:
            dest = self.delivery_zones[GOODS_C]

        # 目标未变则只更新 action 编号，不重新规划 A*（避免每步无效重算）
        if dest == self.agent_dest[agent_id]:
            self.agent_action[agent_id] = action
            return

        self.agent_dest[agent_id]   = dest
        self.agent_action[agent_id] = action

        blocked = {
            tuple(self.agent_pos[j])
            for j in range(self.n_agents) if j != agent_id
        }
        path = self._astar(tuple(self.agent_pos[agent_id]), tuple(dest), blocked)
        if not path and tuple(self.agent_pos[agent_id]) != tuple(dest):
            path = self._astar(tuple(self.agent_pos[agent_id]), tuple(dest), set())
        self.agent_paths[agent_id] = path

    def _move_agents(self):
        # Step 1: 各车意图下一格
        intended = []
        for i in range(self.n_agents):
            if self.agent_paths[i]:
                intended.append(self.agent_paths[i][0])
            else:
                intended.append(tuple(self.agent_pos[i]))

        # Step 2: swap 死锁检测，高编号车让步
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (intended[i] == tuple(self.agent_pos[j]) and
                        intended[j] == tuple(self.agent_pos[i])):
                    intended[j] = tuple(self.agent_pos[j])

        # Step 3: 按编号顺序分配格子（低编号优先）
        reserved = {tuple(self.agent_pos[i]) for i in range(self.n_agents)}
        next_pos = list(self.agent_pos)
        moved    = [False] * self.n_agents

        for i in range(self.n_agents):
            if not self.agent_paths[i]:
                continue
            candidate = intended[i]
            if candidate != tuple(self.agent_pos[i]) and candidate not in reserved:
                next_pos[i] = candidate
                reserved.discard(tuple(self.agent_pos[i]))
                reserved.add(candidate)
                self.agent_paths[i].pop(0)
                moved[i] = True

        # Step 4: 重规划逻辑
        # - 有目标但路径为空（A* 失败或初始规划失败）：立即重规划，不等 5 步
        # - 有路径但连续 5 步未移动：触发重规划（新地图路径长，阈值比旧版 2 步更大）
        for i in range(self.n_agents):
            dest = self.agent_dest[i]
            if dest is None:
                self._stuck_count[i] = 0
                continue

            at_dest = tuple(next_pos[i]) == tuple(dest)

            if not at_dest and not self.agent_paths[i]:
                # A* 返回空路径但尚未到达目标：立即重规划，避免静默卡死
                blocked = {tuple(next_pos[j]) for j in range(self.n_agents) if j != i}
                new_path = self._astar(tuple(next_pos[i]), tuple(dest), blocked)
                if not new_path:
                    new_path = self._astar(tuple(next_pos[i]), tuple(dest), set())
                self.agent_paths[i] = new_path
                self._stuck_count[i] = 0
            elif not moved[i] and self.agent_paths[i]:
                self._stuck_count[i] += 1
                if self._stuck_count[i] >= 5:
                    blocked = {tuple(next_pos[j]) for j in range(self.n_agents) if j != i}
                    new_path = self._astar(tuple(next_pos[i]), tuple(dest), blocked)
                    if not new_path:
                        new_path = self._astar(tuple(next_pos[i]), tuple(dest), set())
                    self.agent_paths[i] = new_path
                    self._stuck_count[i] = 0
            else:
                self._stuck_count[i] = 0

        self.agent_pos = [tuple(p) for p in next_pos]

    def _handle_arrival(self, agent_id: int):
        pos    = tuple(self.agent_pos[agent_id])
        reward = 0.0

        # 到达货架：空手则取货，有货则直接要求新决策
        if pos in self.shelf_positions:
            shelf_idx = self.shelf_positions.index(pos)
            if self.agent_carrying[agent_id] == GOODS_NONE:
                if self.shelf_goods[shelf_idx] != GOODS_NONE:
                    # 货架有货：取走，启动补货倒计时
                    self.agent_carrying[agent_id]    = self.shelf_goods[shelf_idx]
                    self.shelf_goods[shelf_idx]      = GOODS_NONE
                    self.shelf_restock_timer[shelf_idx] = RESTOCK_DELAY
            self.agent_action[agent_id] = -1
            return reward, True

        # 到达投递区
        for goods_type, zone_pos in self.delivery_zones.items():
            if pos == zone_pos:
                carried = self.agent_carrying[agent_id]
                if carried == GOODS_NONE:
                    pass
                elif carried == goods_type:
                    reward += 10.0
                    self.total_deliveries += 1
                else:
                    reward -= 10.0
                self.agent_carrying[agent_id] = GOODS_NONE
                self.agent_action[agent_id]   = -1
                return reward, True

        return reward, False

    # ═══════════════════════════════════════════════════════════════════════════
    #  A* 路径规划
    # ═══════════════════════════════════════════════════════════════════════════

    def _astar(self, start: Tuple, goal: Tuple, blocked: set) -> List[Tuple]:
        if start == goal:
            return []

        all_blocked = self.static_walls | blocked

        def h(p):
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        open_heap = [(h(start), 0, start)]
        came_from = {}
        g_cost    = {start: 0}

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)

            if cur == goal:
                path, node = [], cur
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            if g > g_cost.get(cur, float('inf')):
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    continue
                if nb in all_blocked and nb != goal:
                    continue
                ng = g_cost[cur] + 1
                if ng < g_cost.get(nb, float('inf')):
                    g_cost[nb]    = ng
                    came_from[nb] = cur
                    heapq.heappush(open_heap, (ng + h(nb), ng, nb))

        return []

    # ═══════════════════════════════════════════════════════════════════════════
    #  观测 & 状态
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> List[np.ndarray]:
        obs_list = []
        for i in range(self.n_agents):
            obs = []

            # self_feat: pos(2) + carrying_onehot(4) + dest_onehot(13) = 19
            obs += [self.agent_pos[i][0] / self.H, self.agent_pos[i][1] / self.W]
            obs += self._carrying_onehot(self.agent_carrying[i])
            dest_oh = [0.0] * self.n_actions
            if self.agent_action[i] >= 0:
                dest_oh[self.agent_action[i]] = 1.0
            obs += dest_oh

            # other_feats: (n_agents-1) × (pos(2) + carrying_onehot(4)) = 18
            for j in range(self.n_agents):
                if j == i:
                    continue
                obs += [self.agent_pos[j][0] / self.H, self.agent_pos[j][1] / self.W]
                obs += self._carrying_onehot(self.agent_carrying[j])

            # shelf_feats: n_shelves × (goods_onehot(3) + pos(2)) = 50
            for k in range(self.n_shelves):
                obs += self._goods_onehot(self.shelf_goods[k])
                obs += [self.shelf_positions[k][0] / self.H,
                        self.shelf_positions[k][1] / self.W]

            obs_list.append(np.array(obs, dtype=np.float32))
        return obs_list

    def _get_state(self) -> np.ndarray:
        state = []
        for i in range(self.n_agents):
            state += [self.agent_pos[i][0] / self.H, self.agent_pos[i][1] / self.W]
            state += self._carrying_onehot(self.agent_carrying[i])
        for k in range(self.n_shelves):
            state += self._goods_onehot(self.shelf_goods[k])
            state += [self.shelf_positions[k][0] / self.H,
                      self.shelf_positions[k][1] / self.W]
        return np.array(state, dtype=np.float32)

    @staticmethod
    def _carrying_onehot(goods: int) -> List[float]:
        v = [0.0, 0.0, 0.0, 0.0]
        v[goods] = 1.0
        return v

    @staticmethod
    def _goods_onehot(goods: int) -> List[float]:
        return [
            1.0 if goods == GOODS_A else 0.0,
            1.0 if goods == GOODS_B else 0.0,
            1.0 if goods == GOODS_C else 0.0,
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    #  可视化
    # ═══════════════════════════════════════════════════════════════════════════

    def render(self):
        if self.agent_pos is None:
            print("环境未初始化，请先调用 reset()")
            return
        grid = [['.' for _ in range(self.W)] for _ in range(self.H)]

        for r, c in self.static_walls:
            if 0 <= r < self.H and 0 <= c < self.W:
                grid[r][c] = '■'

        goods_char = {GOODS_A: 'A', GOODS_B: 'B', GOODS_C: 'C'}
        for k, pos in enumerate(self.shelf_positions):
            grid[pos[0]][pos[1]] = goods_char.get(self.shelf_goods[k], '?')

        for goods_type, pos in self.delivery_zones.items():
            grid[pos[0]][pos[1]] = goods_char[goods_type].lower()

        for i, pos in enumerate(self.agent_pos):
            c     = self.agent_carrying[i]
            label = str(i) if c == GOODS_NONE else goods_char.get(c, '?') + str(i)
            grid[pos[0]][pos[1]] = label

        print(f'\n─── Step {self.step_count} | 送达: {self.total_deliveries} ───')
        for row in grid:
            print(' '.join(f'{cell:2s}' for cell in row))

        carry_char = {GOODS_NONE: '无', GOODS_A: 'A', GOODS_B: 'B', GOODS_C: 'C'}
        for i in range(self.n_agents):
            na = self.needs_action[i] if self.needs_action else '?'
            print(f'  AGV{i}: {self.agent_pos[i]}  携带={carry_char[self.agent_carrying[i]]}'
                  f'  需要决策={na}')
        print(f'  货架: {[goods_char.get(g, "?") for g in self.shelf_goods]}')
