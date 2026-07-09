"""
baselines.py — 任务分配对比实验的策略集合

【公平性设计】所有策略共享同一套“系统级”能力，只在最后一步——
“在可选货架里挑哪一个”——上区分，从而严格隔离“分配智能”这一个变量：

  系统级公共能力（所有策略相同，模拟部署端 cargo_manager + _select_action 的掩码）：
    1. 载货 → 一律送往对应投递区（不比投递）
    2. 空架掩码：货架没货 → 不可选
    3. 预约掩码：别的车已经在前往的货架 → 不可选（杜绝两车抢同一架）

  策略差异（仅“在可选货架里怎么排序选第一个”）：
    random     可选货架里随机挑
    nearest    可选货架里挑 A* 距离最近的（最强非学习基线）
    roundrobin 可选货架里按全局轮转挑
    qmix       可选货架里挑 QMIX 网络 Q 值最高的（本作品）

接口统一为 policy(env, agent_id) -> action_int
"""

import random
import numpy as np
import torch

from warehouse_env import GOODS_NONE


# ─────────────────────────────────────────────────────────────────────────────
#  系统级公共能力
# ─────────────────────────────────────────────────────────────────────────────
def _delivery_action(env, agent_id):
    """载货 → 对应投递区动作（货物 1/2/3 → A/B/C）。"""
    return env.n_shelves + (env.agent_carrying[agent_id] - 1)


def _reserved_shelves(env, agent_id):
    """别的车当前正前往的货架（预约），本车不可再选。"""
    res = set()
    for j in range(env.n_agents):
        if j == agent_id:
            continue
        a = env.agent_action[j]
        if 0 <= a < env.n_shelves:
            res.add(a)
    return res


def _available_shelves(env, agent_id):
    """可选货架：有货 且 未被别车预约。
    依次放宽：有货且没被预约 → 有货 → 全部（兜底，避免无动作可返）。"""
    reserved = _reserved_shelves(env, agent_id)
    avail = [k for k in range(env.n_shelves)
             if env.shelf_goods[k] != GOODS_NONE and k not in reserved]
    if avail:
        return avail
    avail = [k for k in range(env.n_shelves) if env.shelf_goods[k] != GOODS_NONE]
    if avail:
        return avail
    return list(range(env.n_shelves))   # 全空，随便去一个等补货


def _astar_dist(env, start, goal):
    if tuple(start) == tuple(goal):
        return 0
    path = env._astar(tuple(start), tuple(goal), set())
    return len(path) if path else 9999


# ─────────────────────────────────────────────────────────────────────────────
#  策略 1：随机分配（性能下限）
# ─────────────────────────────────────────────────────────────────────────────
class RandomPolicy:
    name = 'random'

    def __init__(self):
        self.rng = random.Random(12345)   # 独立 RNG，不污染 env 全局随机流

    def __call__(self, env, agent_id):
        if env.agent_carrying[agent_id] != GOODS_NONE:
            return _delivery_action(env, agent_id)
        return self.rng.choice(_available_shelves(env, agent_id))


# ─────────────────────────────────────────────────────────────────────────────
#  策略 2：就近贪心（最强非学习基线）
# ─────────────────────────────────────────────────────────────────────────────
class NearestPolicy:
    name = 'nearest'

    def __call__(self, env, agent_id):
        if env.agent_carrying[agent_id] != GOODS_NONE:
            return _delivery_action(env, agent_id)
        pos = env.agent_pos[agent_id]
        return min(_available_shelves(env, agent_id),
                   key=lambda k: _astar_dist(env, pos, env.shelf_positions[k]))


# ─────────────────────────────────────────────────────────────────────────────
#  策略 3：轮转分配
# ─────────────────────────────────────────────────────────────────────────────
class RoundRobinPolicy:
    name = 'roundrobin'

    def __init__(self):
        self._next = 0

    def __call__(self, env, agent_id):
        if env.agent_carrying[agent_id] != GOODS_NONE:
            return _delivery_action(env, agent_id)
        avail = _available_shelves(env, agent_id)
        k = avail[self._next % len(avail)]
        self._next += 1
        return k


# ─────────────────────────────────────────────────────────────────────────────
#  策略 4：QMIX（本作品）—— 注意力网络，在可选货架里取 Q 最大
# ─────────────────────────────────────────────────────────────────────────────
class QMixPolicy:
    name = 'qmix'

    def __init__(self, agent_net, device):
        self.net = agent_net
        self.device = device

    def __call__(self, env, agent_id):
        if env.agent_carrying[agent_id] != GOODS_NONE:
            return _delivery_action(env, agent_id)
        obs = env._get_obs()[agent_id]
        with torch.no_grad():
            q = self.net(torch.FloatTensor(obs).unsqueeze(0).to(self.device))[0].numpy()
        avail = set(_available_shelves(env, agent_id))
        q_shelf = q[:env.n_shelves].copy()
        for k in range(env.n_shelves):
            if k not in avail:
                q_shelf[k] = -1e9                  # 与部署端一致的掩码
        return int(q_shelf.argmax())


def build_policy(kind, agent_net=None, device=None):
    if kind == 'random':
        return RandomPolicy()
    if kind == 'nearest':
        return NearestPolicy()
    if kind == 'roundrobin':
        return RoundRobinPolicy()
    if kind == 'qmix':
        assert agent_net is not None
        return QMixPolicy(agent_net, device)
    raise ValueError(kind)
