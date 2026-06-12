#!/usr/bin/env python3
"""traffic_manager.py — 中央交通管制（格子预约），多车零碰撞、不来回蹭。

原理（像火车信号系统）：
  地图切 1m 格子；每辆车沿 A* 路线，提前锁定前方若干格（滚动窗口）。
  一个格子同一时刻只发给一辆车 → 物理上不可能两车进同一格 → 不可能撞。
  锁不到下一格的车就在原地停住等（绝不绕、不试探）→ 不来回蹭。
  move_base 只负责把车开到"锁到的最远那格"，那格保证是空的。

话题：
  订阅  /robot_i/odom                  → 每辆车当前格
  订阅  /robot_i/goal_cell  "r c"       → 每辆车的目标格（QMIX 选完目标后发；"-1 -1"=无目标）
  发布  /robot_i/drive_to   "x y"       → 让该车开到这个世界坐标（某格中心）
  发布  /traffic/debug      String      → 占用表 ASCII（调试用）
"""

import threading
import heapq
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

import grid_map
from grid_map import GRID_H, GRID_W, grid_to_xy, xy_to_grid

MAP_YAML  = '/home/jasper/mani_ws/src/warehouse_qmix/maps/warehouse_cross.yaml'
N_AGENTS  = 4
LOOKAHEAD = 4        # 提前锁定的格子数（滚动窗口）；越大越顺、占用越多
STEP_HZ   = 5.0      # 调度频率
DEADLOCK_T = 6.0     # 某车停滞超过这么久 → 记为疑似死锁（日志告警）
ADVANCE_DIST = 0.7   # 车离当前目标格中心 < 此距离 → 视为已到该格，提前推进下一格
                     # （比 move_base 的到点容差大，胡萝卜始终在车前方，不会停在格边界）

# ── 单行方向约束（防窄口迎面死锁）────────────────────────────────────────
# key=(r,c)，value=允许"进入该格"的移动方向集合：'D'下 'U'上 'L'左 'R'右。
# 底部窄口 row11 两条道：左道(7,8)只许向下(去投递)，右道(11,12)只许向上(离场)。
ONEWAY = {}
for _r in (10, 11, 12):
    ONEWAY[(_r, 7)]  = {'D'}
    ONEWAY[(_r, 8)]  = {'D'}
    ONEWAY[(_r, 11)] = {'U'}
    ONEWAY[(_r, 12)] = {'U'}


def _move_dir(a, b):
    """从格 a 进入相邻格 b 的方向字符。"""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 1:  return 'D'
    if dr == -1: return 'U'
    if dc == 1:  return 'R'
    if dc == -1: return 'L'
    return '?'


def _dir_ok(a, b):
    """单行约束：进入 b 的方向是否被允许。"""
    allow = ONEWAY.get(b)
    if not allow:
        return True
    return _move_dir(a, b) in allow


def astar(free, start, goal, blocked):
    """4 连通 A*。free=可行格；blocked=要避开的格集合（其他车占的格，goal 除外）。
    返回从 start 到 goal 的格子列表（含两端），无解返回 None。"""
    if start == goal:
        return [start]

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    openq = [(h(start), 0, start)]
    came = {start: None}
    gcost = {start: 0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        r, c = cur
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            nb = (nr, nc)
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if not free[nr][nc]:
                continue
            if nb in blocked and nb != goal:
                continue
            if not _dir_ok(cur, nb):
                continue
            ng = g + 1
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng
                came[nb] = cur
                heapq.heappush(openq, (ng + h(nb), ng, nb))
    return None


class TrafficManager:

    def __init__(self):
        self.lock = threading.Lock()
        self.free = grid_map.load_free_grid(MAP_YAML)

        self.pos  = {}    # rid -> (r,c) 当前格（odom 四舍五入）
        self.posf = {}    # rid -> (x,y) 连续世界坐标
        self.goal = {}    # rid -> (r,c) 目标格，或 None
        self.reserved = {}        # (r,c) -> rid，全局占用表
        self.held = {i: set() for i in range(N_AGENTS)}   # rid -> 它锁着的格集合
        self.last_target = {i: None for i in range(N_AGENTS)}
        self.blocked_since = {i: None for i in range(N_AGENTS)}

        self.drive_pub = {}
        for i in range(N_AGENTS):
            self.drive_pub[i] = rospy.Publisher(
                '/robot_{}/drive_to'.format(i), String, queue_size=1, latch=True)
            rospy.Subscriber('/robot_{}/odom'.format(i), Odometry,
                             lambda m, ii=i: self._odom_cb(m, ii))
            rospy.Subscriber('/robot_{}/goal_cell'.format(i), String,
                             lambda m, ii=i: self._goal_cb(m, ii))
        self.debug_pub = rospy.Publisher('/traffic/debug', String, queue_size=1)

        n_free = sum(sum(1 for v in row if v) for row in self.free)
        rospy.loginfo('[traffic] 地图加载完成，可行格 %d/%d，单行约束 %d 格',
                      n_free, GRID_H * GRID_W, len(ONEWAY))

    # ── 回调 ────────────────────────────────────────────────────────────
    def _odom_cb(self, msg, rid):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        with self.lock:
            self.pos[rid]  = xy_to_grid(x, y)
            self.posf[rid] = (x, y)

    def _goal_cb(self, msg, rid):
        try:
            parts = msg.data.split()
            r, c = int(parts[0]), int(parts[1])
        except Exception:
            return
        with self.lock:
            self.goal[rid] = None if (r < 0 or c < 0) else (r, c)

    # ── 占用表操作（均在持锁下调用）──────────────────────────────────────
    def _release(self, rid, keep):
        """释放 rid 锁着但不在 keep 集合里的格。"""
        for cell in list(self.held[rid]):
            if cell not in keep:
                if self.reserved.get(cell) == rid:
                    del self.reserved[cell]
                self.held[rid].discard(cell)

    def _claim(self, rid, cell):
        """把 cell 锁给 rid；成功返回 True。已被别人锁则 False。"""
        owner = self.reserved.get(cell)
        if owner is not None and owner != rid:
            return False
        self.reserved[cell] = rid
        self.held[rid].add(cell)
        return True

    # ── 主调度循环 ──────────────────────────────────────────────────────
    def step(self):
        with self.lock:
            pos  = dict(self.pos)
            posf = dict(self.posf)
            goal = dict(self.goal)

            # 1) 物理占用优先：每辆车先牢牢锁住自己当前所在的格
            #    （万一被别人锁了，物理占用者抢回来，对方下一轮自然停下）
            for rid in range(N_AGENTS):
                cur = pos.get(rid)
                if cur is None:
                    continue
                owner = self.reserved.get(cur)
                if owner is not None and owner != rid:
                    self.held[owner].discard(cur)
                self.reserved[cur] = rid
                self.held[rid].add(cur)

            # 2) 按车号优先级（小号先）扩展各自的前向滚动窗口
            now = rospy.Time.now()
            for rid in range(N_AGENTS):
                odom_cell = pos.get(rid)
                if odom_cell is None:
                    continue
                g = goal.get(rid)

                # 有效当前格：若车已开到"当前目标格"附近(<ADVANCE_DIST)，就把规划起点
                # 提前到目标格 → 胡萝卜持续前移，不会卡在格边界（核心修复）。
                cur = odom_cell
                tgt = self.last_target.get(rid)
                if tgt is not None and rid in posf:
                    tx, ty = grid_to_xy(*tgt)
                    if (posf[rid][0] - tx) ** 2 + (posf[rid][1] - ty) ** 2 < ADVANCE_DIST ** 2:
                        cur = tgt

                # 无目标 / 已到目标 → 只保留当前格，原地待命
                if g is None or g == cur:
                    self._release(rid, {cur, odom_cell})
                    self._set_target(rid, cur)
                    self.blocked_since[rid] = None
                    continue

                # 其他车当前占的格 = A* 要避开的障碍（目标格除外）
                blocked = set()
                for oid in range(N_AGENTS):
                    if oid == rid:
                        continue
                    if oid in pos:
                        blocked.add(pos[oid])
                    blocked |= self.held[oid]

                path = astar(self.free, cur, g, blocked)

                if not path or len(path) < 2:
                    # 暂时无路（被占）→ 守住当前格停下等
                    self._release(rid, {cur, odom_cell})
                    self._set_target(rid, cur)
                    if self.blocked_since[rid] is None:
                        self.blocked_since[rid] = now
                    elif (now - self.blocked_since[rid]).to_sec() > DEADLOCK_T:
                        rospy.logwarn_throttle(
                            3.0, '[traffic] robot_%d 停滞 >%.0fs（疑似拥堵/死锁）@%s 目标%s',
                            rid, DEADLOCK_T, cur, g)
                    continue

                # 沿路径锁前方最多 LOOKAHEAD 格（遇到锁不到的就停）
                keep = {cur, odom_cell}           # 物理格也保住，避免抢占重叠期被别人拿走
                booked = []                       # 已锁到的前向格（按路径顺序）
                for cell in path[1:1 + LOOKAHEAD]:
                    if self._claim(rid, cell):
                        keep.add(cell)
                        booked.append(cell)
                    else:
                        break
                self._release(rid, keep)

                # 行驶目标 = 当前直线段末端（第一个转弯前那格）。
                # 只给直线目标，move_base 直着开，不会抄近道切进没预约的格。
                target = cur
                if booked:
                    target = booked[0]
                    d0 = (booked[0][0] - cur[0], booked[0][1] - cur[1])
                    prev = booked[0]
                    for cell in booked[1:]:
                        d = (cell[0] - prev[0], cell[1] - prev[1])
                        if d != d0:
                            break
                        target = cell
                        prev = cell
                self._set_target(rid, target)

                if target == cur:
                    if self.blocked_since[rid] is None:
                        self.blocked_since[rid] = now
                    elif (now - self.blocked_since[rid]).to_sec() > DEADLOCK_T:
                        rospy.logwarn_throttle(
                            3.0, '[traffic] robot_%d 停滞 >%.0fs（前方被占）@%s 目标%s',
                            rid, DEADLOCK_T, cur, g)
                else:
                    self.blocked_since[rid] = None

        self._publish_debug(pos, goal)

    def _set_target(self, rid, cell):
        if self.last_target[rid] == cell:
            return
        self.last_target[rid] = cell
        x, y = grid_to_xy(*cell)
        self.drive_pub[rid].publish(String(data='{:.3f} {:.3f}'.format(x, y)))

    def _publish_debug(self, pos, goal):
        marks = {}
        for cell, rid in self.reserved.items():
            marks[cell] = str(rid)
        for rid, cell in pos.items():
            marks[cell] = chr(ord('A') + rid)   # 当前格用 A/B/C/D 大写
        txt = grid_map.render(self.free, marks)
        self.debug_pub.publish(String(data=txt))


def main():
    rospy.init_node('traffic_manager')
    tm = TrafficManager()
    rate = rospy.Rate(STEP_HZ)
    rospy.loginfo('[traffic] 调度启动 @%.0fHz, 前瞻 %d 格', STEP_HZ, LOOKAHEAD)
    while not rospy.is_shutdown():
        tm.step()
        rate.sleep()


if __name__ == '__main__':
    main()
