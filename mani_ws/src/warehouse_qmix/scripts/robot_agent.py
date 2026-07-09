#!/usr/bin/env python3
"""
robot_agent.py — 单 AGV QMIX 推理 + 航点导航节点

ROS 参数：
  ~robot_id (int): 车辆编号 0-3

发布：
  /robot_{id}/waterplus/navi_waypoint  目标航点名
  /robot_{id}/carrying                 携带状态 Int32
  /warehouse/pickup                    "id shelf_idx"
  /warehouse/delivery                  "id goods_type"

订阅：
  /robot_{id}/odom                     自身位置
  /robot_{j}/odom  (j≠id)             其他车位置
  /robot_{j}/carrying (j≠id)          其他车携带状态
  /warehouse/shelf_goods               货架状态
  /robot_{id}/waterplus/navi_result    导航结果
"""

import sys
import math
import threading
import rospy
import numpy as np
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalID

sys.path.append('/home/jasper/qmix_project')
import torch
from qmix_network import AttentionAgentQNet

# ── 常量 ──────────────────────────────────────────────────────────────────
N_AGENTS  = 4
N_SHELVES = 10
N_ACTIONS = 13
GRID_H    = 16
GRID_W    = 20

GOODS_NONE = 0
GOODS_A    = 1
GOODS_B    = 2
GOODS_C    = 3
GOODS_NAME = {0: '空', 1: 'A', 2: 'B', 3: 'C'}

# 区域软偏好：0/1 偏好左侧+顶部，2/3 偏好右侧+顶部
# 顶部货架 (0,1) 两侧都覆盖，避免 starvation
ZONE_SHELVES = {
    0: [0, 1, 2, 3, 4, 5],
    1: [0, 1, 2, 3, 4, 5],
    2: [0, 1, 6, 7, 8, 9],
    3: [0, 1, 6, 7, 8, 9],
}
ZONE_BONUS = 3.0   # 主场货架 Q 值加成（相对 Q 值量级约 10-30）

# 固定货架格坐标（对应训练 DEFAULT_SHELF_POSITIONS，用于 obs 构造）
SHELF_GRID_POS = [
    (1, 8), (1, 11),
    (6, 1), (6, 4), (8, 1), (8, 4),
    (6, 15), (6, 18), (8, 15), (8, 18),
]

CHECKPOINT = '/home/jasper/qmix_project/checkpoints_v5_attn/qmix_ep800.pt'

STATE_SELECTING      = 0
STATE_GOING_SHELF    = 1
STATE_GOING_DELIVERY = 2
# 进/出投递区改用 _go_route 多段路线（IN_ROUTE/OUT_ROUTE），不再单设闸口状态

STUCK_TIMEOUT   = 12.0   # 综合计时器（位置+转向都不变）超时 → 兜底脱困
HEADON_CHECK_T  =  3.0   # 位置计时器（仅平移）超时 → 检测让行
STUCK_DIST      =  0.08  # 位移门限（米）
STUCK_YAW       =  0.08  # 转向门限（弧度），约 4.6°
ESCAPE_VEL      = -0.25  # 后退速度（m/s）
ESCAPE_DURATION =  1.5   # 后退时长（秒）
REAR_SAFE_DIST  =  0.55  # 后方至少这么远才允许后退，否则改旋转
APPROACH_DIST   =  2.2   # 前方车在这个格子数内 → 才避让（≈1.1m，原3.5太远，开阔区瞎触发）
PAUSE_MAX_WAIT  = 15.0   # 优先车原地停等的最长时间，超时按原航点继续（兜底）
YIELD_COOLDOWN  =  4.0   # 对同一辆车两次主动让行的最小间隔（秒），防 0.5s 反复触发抖动
OTHER_MOVE_WINDOW = 2.0  # 对方近这么多秒内有平移才算"在动"；停着不动的车不空等它，直接绕

# 投递区训练网格坐标（A/B/C）
DELIVERY_GRID_POS = [(13, 7), (13, 9), (13, 11)]

# 各 action 对应航点（WP1-13）的真实 Gazebo 坐标 —— move_base 实际停靠点
# 注意：取自 ~/waypoints.xml，是货架旁 1m 的停车点，不是货架中心
WP_POS = [
    (-1.5, 5.5), ( 1.5, 5.5),                            # 货架 0,1（南侧停靠）
    (-7.5, 1.5), (-4.5, 1.5), (-7.5, -0.5), (-4.5, -0.5),  # 货架 2-5（东侧停靠）
    ( 4.5, 1.5), ( 7.5, 1.5), ( 4.5, -0.5), ( 7.5, -0.5),  # 货架 6-9（西侧停靠）
    (-2.5, -5.5), (-0.5, -5.5), ( 1.5, -5.5),           # 投递区 A,B,C
]
ARRIVE_TOL = 0.6   # 最终货架/投递点：离目标 < 0.6m 才算真到达（防假到达拿错/投错）
GATE_ARRIVE_TOL = 1.2  # 中间过站点（窄口引导桩 WP14-17）：只是穿缝引导，贴墙差几十厘米也算过

# ── 底部单行进出（治根：投递区窄口"一缝进、一缝出"，杜绝迎面顶牛）──────────
# 每条缝上下两端各放一个站点，"顶→底/底→顶"那一步把穿越锁死在该缝里，
# move_base 无法抄到另一条缝（这是单站点版的修复：单点管不住走哪条缝）。
IN_ROUTE  = [('16', (-1.5, -2.5)), ('14', (-1.5, -4.5))]  # 左缝 顶→底，下行进投递区
OUT_ROUTE = [('15', ( 2.5, -4.5)), ('17', ( 2.5, -2.5))]  # 右缝 底→顶，上行离开底部
BOTTOM_EXIT_ROW = 11             # 车 row >= 此值视为"在底部"，去货架须先经出站路线

# ── 区域容量限制 ───────────────────────────────────────────────────────────
# 十字形地图分为 4 个翼区，每个区最多同时允许 ZONE_MAX 台车
ZONE_MAX = 2

# 区域边界（行范围，列范围），用训练网格坐标
ZONE_BOUNDS = {
    'top':    ((0,  4), (6,  13)),   # 顶翼：货架 0,1
    'left':   ((4, 10), (0,   7)),   # 左翼：货架 2,3,4,5
    'right':  ((4, 10), (13, 19)),   # 右翼：货架 6,7,8,9
    'bottom': ((10, 15), (6,  13)),  # 底翼：投递区+起点
}

# 每个货架属于哪个区
SHELF_ZONE = {
    0: 'top',  1: 'top',
    2: 'left', 3: 'left', 4: 'left', 5: 'left',
    6: 'right', 7: 'right', 8: 'right', 9: 'right',
}

# ── 投递区出口"单车闸口"参数（grid 坐标）────────────────────────────────────
GATE_ROWS       = (11, 12)   # 出口单车道窄口（被 wall_s_btl 卡出来）
GATE_APPROACH   = 10         # 窄口北侧入口行
BOTTOM_COLS     = (6, 13)    # 底部列范围
BOTTOM_MIN_ROW  = 11         # row≥11 视为"在底部区域"（窄口+投递+起点）
DEPART_MAX_WAIT = 20.0       # 待发最长等待，超时强制出发（防新卡死兜底）
N_DELIVER       = 3          # 投递点数 A/B/C
ENTER_MAX_WAIT  = 25.0       # 进场投递排队最长等待，超时强制进场（兜底）


def _grid_zone(row, col):
    """返回网格位置 (row,col) 所在区域名，不在任何翼区则返回 'center'"""
    for name, ((r0, r1), (c0, c1)) in ZONE_BOUNDS.items():
        if r0 <= row <= r1 and c0 <= col <= c1:
            return name
    return 'center'


def _in_gate(r, c):
    """是否在出口单车道窄口内"""
    return GATE_ROWS[0] <= r <= GATE_ROWS[1] and BOTTOM_COLS[0] <= c <= BOTTOM_COLS[1]


def _in_bottom(r, c):
    """是否在底部区域（窄口+投递区+起点）"""
    return r >= BOTTOM_MIN_ROW and BOTTOM_COLS[0] <= c <= BOTTOM_COLS[1]


def gazebo_to_grid(x, y):
    """Gazebo (x,y) → 训练网格 (row,col)"""
    row = int(round(7.5 - y))
    col = int(round(x + 9.5))
    return max(0, min(GRID_H - 1, row)), max(0, min(GRID_W - 1, col))


def carrying_onehot(c):
    oh = [0.0, 0.0, 0.0, 0.0]
    if 0 <= c < 4:
        oh[c] = 1.0
    return oh


def goods_onehot(g):
    oh = [0.0, 0.0, 0.0]
    if 1 <= g <= 3:
        oh[g - 1] = 1.0
    return oh


class RobotAgent:

    def __init__(self, robot_id):
        self.rid  = robot_id
        self.lock = threading.Lock()

        # 自身状态
        self.pos            = (GRID_H - 1, 8 + robot_id)  # 初始估计
        self.carrying       = GOODS_NONE
        self.current_action = -1   # -1 = 无目标
        self.current_shelf  = -1   # 当前目标货架（GOING_SHELF 时有效）
        self.state          = STATE_SELECTING

        # 其他车辆状态
        self.other_pos      = {}   # j → (row, col)
        self.other_carrying = {}   # j → int
        self.other_targets  = {}   # j → shelf_idx（-1 表示去投递区或未知）
        self.other_raw      = {}   # j → (x, y) 上次原始坐标（判断对方是否在动）
        self.other_move_t   = {}   # j → 对方上次明显平移的时刻

        # 货架状态（由 cargo_manager 广播）
        self.shelf_goods = [GOODS_NONE] * N_SHELVES
        self.reserved    = [-1] * N_SHELVES   # 货架预约表（cargo_manager 唯一发号）
        self.deliver_reserved = [-1] * N_DELIVER   # 投递点预约表（cargo_manager 唯一发号）

        # 加载 QMIX 网络（参数共享，所有车加载同一 checkpoint）
        self.net = AttentionAgentQNet(obs_dim=87, n_actions=N_ACTIONS,
                                      n_agents=N_AGENTS, n_shelves=N_SHELVES)
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['agent_net'])
        self.net.eval()
        rospy.loginfo('[robot_%d] QMIX 加载完成', self.rid)

        # 卡住检测（原始 Gazebo 坐标 + 朝向）
        self._raw_pos       = None
        self._last_yaw      = None
        self._last_move_t   = None   # 综合计时器：位置或转向有变化就重置
        self._last_pos_t    = None   # 位置计时器：只有真实平移才重置
        self._navigating    = False
        self._arrived       = False  # 本段导航是否已就近判定到达（防止重复触发）
        self._nav_goal_pos  = None   # 当前这一段导航的目标点 (x,y)（含进/出站点）
        self._nav_goal_wp   = None   # 当前这一段导航的航点名
        self._legs          = []     # 剩余航段队列 [(wp_str,(x,y)), ...]，最后一段是最终目标
        self._final_kind    = None   # 'shelf' / 'delivery'：最后一段到达后干啥
        self._first_depart  = True   # 开局首次出车：自由全局规划散开，之后才走右缝单行
        self._last_yield_t  = {}     # {j: rospy.Time} 上次对 j 号车主动让行的时刻（冷却用）

        # 激光雷达数据（用于脱困方向判断）
        self._scan_ranges   = None
        self._scan_angle_min = 0.0
        self._scan_angle_inc = 0.0

        ns = 'robot_{}'.format(robot_id)

        # 发布者
        self.wp_pub       = rospy.Publisher(
            '/{}/waterplus/navi_waypoint'.format(ns), String, queue_size=1)
        self.carrying_pub = rospy.Publisher(
            '/{}/carrying'.format(ns), Int32, queue_size=1, latch=True)
        self.cmd_pub      = rospy.Publisher(
            '/{}/cmd_vel'.format(ns), Twist, queue_size=10)
        self.cancel_pub   = rospy.Publisher(
            '/{}/move_base/cancel'.format(ns), GoalID, queue_size=1)
        self.pickup_pub   = rospy.Publisher(
            '/warehouse/pickup',   String, queue_size=1)
        self.delivery_pub = rospy.Publisher(
            '/warehouse/delivery', String, queue_size=1)
        self.claim_pub    = rospy.Publisher(
            '/warehouse/claim',    String, queue_size=1)
        self.deliver_claim_pub = rospy.Publisher(
            '/warehouse/deliver_claim', String, queue_size=1)

        # 订阅
        rospy.Subscriber('/{}/odom'.format(ns), Odometry, self._odom_cb)
        rospy.Subscriber('/warehouse/shelf_goods', String, self._shelf_goods_cb)
        rospy.Subscriber('/warehouse/shelf_reserved', String, self._reserved_cb)
        rospy.Subscriber('/warehouse/deliver_reserved', String, self._deliver_reserved_cb)
        rospy.Subscriber('/{}/waterplus/navi_result'.format(ns),
                         String, self._navi_result_cb)
        rospy.Subscriber('/{}/scan_fixed'.format(ns), LaserScan, self._scan_cb)

        for j in range(N_AGENTS):
            if j == robot_id:
                continue
            ns_j = 'robot_{}'.format(j)
            rospy.Subscriber('/{}/odom'.format(ns_j), Odometry,
                             lambda msg, jj=j: self._other_odom_cb(msg, jj))
            rospy.Subscriber('/{}/carrying'.format(ns_j), Int32,
                             lambda msg, jj=j: self._other_carrying_cb(msg, jj))
            rospy.Subscriber('/{}/waterplus/navi_waypoint'.format(ns_j), String,
                             lambda msg, jj=j: self._other_waypoint_cb(msg, jj))

        threading.Thread(target=self._watchdog, daemon=True).start()
        threading.Thread(target=self._arrival_monitor, daemon=True).start()

    # ── 回调 ──────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        row, col = gazebo_to_grid(x, y)
        with self.lock:
            self.pos = (row, col)

        # 从四元数提取 yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        now = rospy.Time.now()
        new_raw = (x, y)

        # 检测位置平移
        translated = False
        if self._raw_pos is None:
            translated = True
        else:
            dist = math.hypot(new_raw[0] - self._raw_pos[0],
                              new_raw[1] - self._raw_pos[1])
            if dist >= STUCK_DIST:
                translated = True

        # 检测转向
        rotated = False
        if self._last_yaw is not None:
            dyaw = abs((yaw - self._last_yaw + math.pi) % (2 * math.pi) - math.pi)
            if dyaw >= STUCK_YAW:
                rotated = True

        if translated:
            self._last_pos_t  = now   # 位置计时器：仅平移重置
            self._last_move_t = now   # 综合计时器也重置
            self._last_yaw    = yaw
        elif rotated:
            self._last_move_t = now   # 综合计时器：转向也重置
            self._last_yaw    = yaw

        self._raw_pos = new_raw

    def _other_odom_cb(self, msg, j):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        row, col = gazebo_to_grid(x, y)
        now = rospy.Time.now()
        with self.lock:
            self.other_pos[j] = (row, col)
            prev = self.other_raw.get(j)
            if prev is None or math.hypot(x - prev[0], y - prev[1]) >= STUCK_DIST:
                self.other_move_t[j] = now   # 对方有明显平移 → 记下时刻
            self.other_raw[j] = (x, y)

    def _other_moving(self, j):
        """对方近 OTHER_MOVE_WINDOW 秒内是否还在平移。
        用来区分『真·迎面/横穿』(要让)与『停在路边不走的车』(别空等，直接绕)。"""
        t = self.other_move_t.get(j)
        if t is None:
            return False
        return (rospy.Time.now() - t).to_sec() < OTHER_MOVE_WINDOW

    # ── 区域容量查询 ──────────────────────────────────────────────────────

    def _zone_other_count(self, zone_name):
        """统计其他车中：物理上在该区 或 目标货架在该区 的车辆数（不含自身）"""
        with self.lock:
            op = dict(self.other_pos)
            ot = dict(self.other_targets)
        counted = set()
        count = 0
        for j, (jr, jc) in op.items():
            if _grid_zone(jr, jc) == zone_name:
                counted.add(j)
                count += 1
        for j, target in ot.items():
            if j in counted:
                continue
            if SHELF_ZONE.get(target) == zone_name:
                count += 1
        return count

    # ── 路径冲突检测辅助 ──────────────────────────────────────────────────

    def _dest_grid_pos(self):
        """当前导航目标的训练网格坐标"""
        with self.lock:
            action = self.current_action
        if 0 <= action < N_SHELVES:
            return SHELF_GRID_POS[action]
        if N_SHELVES <= action < N_SHELVES + 3:
            return DELIVERY_GRID_POS[action - N_SHELVES]
        return None

    def _robot_in_my_path(self, require_higher_id):
        """
        判断是否有其他车挡在我前进路径上。
        require_higher_id=True  → 只匹配 ID > self.rid 的车（低让高场景）
        require_higher_id=False → 只匹配 ID < self.rid 的车（高等低场景）
        返回 (found, robot_id)
        """
        dest = self._dest_grid_pos()
        if dest is None:
            return False, -1
        with self.lock:
            my_pos = self.pos
            op     = dict(self.other_pos)

        dr = dest[0] - my_pos[0]
        dc = dest[1] - my_pos[1]
        dist_to_dest = math.hypot(dr, dc)
        if dist_to_dest < 1.5:
            return False, -1   # 已接近目标，不触发让行

        ur = dr / dist_to_dest   # 朝目标方向的单位向量
        uc = dc / dist_to_dest

        for j, (jr, jc) in op.items():
            if require_higher_id and j <= self.rid:
                continue
            if not require_higher_id and j >= self.rid:
                continue
            vr = jr - my_pos[0]
            vc = jc - my_pos[1]
            dist = math.hypot(vr, vc)
            if dist < 0.5 or dist > 3.0:   # 原6.0太远，开阔区把朝同向散开的车都误判成挡路
                continue
            dot = (vr * ur + vc * uc) / dist   # cos(角度)
            if dot > 0.8:   # 在前方 ±37° 范围内（原0.6/±53°锥太宽）
                return True, j
        return False, -1

    def _wait_for_path_clear(self, avoid_j, timeout=15.0):
        """阻塞直到 avoid_j 号车不再挡在我的路径上"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(2)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            found, j = self._robot_in_my_path(require_higher_id=True)
            if not found or j != avoid_j:
                break
            rate.sleep()
        rospy.sleep(0.3)

    # ── 卡住检测 ──────────────────────────────────────────────────────────

    def _watchdog(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            rate.sleep()
            if not self._navigating or self._last_move_t is None:
                continue
            now = rospy.Time.now()

            # 注：窄口对撞已由"底部单行进出"从结构上杜绝，原"出口让路退车"逻辑已移除。

            # ── 主动避让（走廊死锁兜底，DWA 注入避障为主）────────────────────
            # 关键：只有自己『真的卡住、连 HEADON_CHECK_T 秒都没前进』才介入。
            # 正常行驶中绝不打断 —— 否则一辆停在路边的车会把正常路过的车反复拽进
            # 原地停等，看着就是"站着不动、rviz 没路径"。
            #   对方在动(真·迎面)且是高ID → 我优先 → 停一下等它过；
            #   对方停着不走 / 我是高ID让行     → 重发目标让 DWA 绕过去，绝不空等。
            pos_elapsed = (now - self._last_pos_t).to_sec() if self._last_pos_t else 0.0
            acted = False
            if pos_elapsed >= HEADON_CHECK_T:
                for see_higher in (True, False):
                    found, j = self._robot_in_my_path(require_higher_id=see_higher)
                    if not found:
                        continue
                    with self.lock:
                        my_pos  = self.pos
                        other_p = self.other_pos.get(j)
                    if other_p is None:
                        continue
                    dist = math.hypot(other_p[0] - my_pos[0], other_p[1] - my_pos[1])
                    if dist >= APPROACH_DIST:
                        continue
                    # 冷却：刚对这辆车让过行就先别再让，防反复取消目标/抖动
                    last_t = self._last_yield_t.get(j)
                    if last_t is not None and (now - last_t).to_sec() < YIELD_COOLDOWN:
                        continue
                    self._last_yield_t[j] = now
                    if see_higher and self._other_moving(j):
                        # 前方是高ID且在动 → 我优先 → 停一下等它过，路线不变
                        rospy.loginfo('[robot_%d] 前方 robot_%d(高ID)在动 %.1f 格，原地停等',
                                      self.rid, j, dist)
                        self._pause_and_hold(j)
                    else:
                        # 对方停着不走 或 我是高ID让行 → 重规划绕过，绝不空等
                        rospy.logwarn('[robot_%d] 前方 robot_%d 挡路(停着/我让行)，重规划绕行',
                                      self.rid, j)
                        self._yield_reroute(j)
                    acted = True
                    break
            if acted:
                continue

            # ── 综合计时器超时 → 真正卡死（墙角/货架），兜底脱困 ─────────
            any_elapsed = (now - self._last_move_t).to_sec()
            if any_elapsed >= STUCK_TIMEOUT:
                rospy.logwarn('[robot_%d] 综合卡住 %.0fs，执行脱困', self.rid, any_elapsed)
                self._escape()

    def _go_route(self, gate_legs, final_wp, final_pos, final_state, final_kind):
        """开一条多段路线：先依次过 gate_legs 各站点，最后到 (final_wp,final_pos)。
        全程状态保持 final_state（供 obs/避让用），到最后一段才执行 final_kind 动作。"""
        with self.lock:
            self.state = final_state
        self._legs = list(gate_legs) + [(final_wp, final_pos)]
        self._final_kind = final_kind
        self._advance_leg()

    def _cur_tol(self):
        """当前这段的到达容差：最后一段（货架/投递）用严容差 ARRIVE_TOL；
        还有后续站点的中间段（窄口引导桩）用宽容差 GATE_ARRIVE_TOL，
        否则贴墙的 WP15/WP17 永远到不了 0.6m → move_base 报 done 却被判假到达 → 无限重试卡死。"""
        return ARRIVE_TOL if len(self._legs) <= 1 else GATE_ARRIVE_TOL

    def _advance_leg(self):
        """驶向队列中的下一段。"""
        if not self._legs:
            return
        wp_str, pos = self._legs[0]
        self._nav_goal_wp  = wp_str
        self._nav_goal_pos = pos
        self._arrived      = False
        self.wp_pub.publish(String(data=wp_str))
        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True

    def _on_leg_arrive(self):
        """到达当前段：还有后续站点就继续，否则执行最终动作（取货/投递）。"""
        if self._legs:
            self._legs.pop(0)
        if self._legs:
            rospy.loginfo('[robot_%d] 过站点 → 下一段 WP%s', self.rid, self._legs[0][0])
            self._advance_leg()
            return
        rospy.loginfo('[robot_%d] 就近到达最终目标(<%.1fm)', self.rid, ARRIVE_TOL)
        if self._final_kind == 'shelf':
            self._on_arrive_shelf()
        else:
            self._on_arrive_delivery()

    def _arrival_monitor(self):
        """就近到达判定：车开到离本段目标 < ARRIVE_TOL 即判定到达，
        不必死等 move_base 把最后几十厘米的姿态修到完美 → 根治货架前前后蠕动 / 卡住倒车。"""
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            rate.sleep()
            with self.lock:
                st = self.state
            if st not in (STATE_GOING_SHELF, STATE_GOING_DELIVERY):
                continue
            if self._arrived or not self._navigating:
                continue
            pos = self._nav_goal_pos
            if pos is None or self._raw_pos is None:
                continue
            if math.hypot(self._raw_pos[0] - pos[0], self._raw_pos[1] - pos[1]) < self._cur_tol():
                self._arrived    = True
                self._navigating = False
                self.cancel_pub.publish(GoalID())   # 停掉 move_base，别再原地蠕动
                self._on_leg_arrive()

    def _escape(self, head_on=False, avoid_j=-1):
        self._navigating = False
        # 取消 move_base 当前目标，让 DWA 停止发布速度
        self.cancel_pub.publish(GoalID())
        rospy.sleep(0.1)
        rear_dist = self._sector_min(180, half_deg=40)

        if rear_dist < REAR_SAFE_DIST:
            rospy.logwarn('[robot_%d] 后方 %.2fm 有障碍，改为旋转脱困', self.rid, rear_dist)
            self._do_rotate()
        else:
            twist = Twist()
            twist.linear.x = ESCAPE_VEL
            end_t = rospy.Time.now() + rospy.Duration(ESCAPE_DURATION)
            rate  = rospy.Rate(20)
            while rospy.Time.now() < end_t and not rospy.is_shutdown():
                self.cmd_pub.publish(twist)
                rate.sleep()
            self.cmd_pub.publish(Twist())

        if head_on and avoid_j >= 0:
            rospy.loginfo('[robot_%d] 已退让，等待 robot_%d 通过路径...', self.rid, avoid_j)
            self._wait_for_path_clear(avoid_j, timeout=15.0)
        else:
            rospy.sleep(0.5)

        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True
        self._retry_nav()

    def _pause_and_hold(self, avoid_j):
        """优先车：保持原路线，原地刹停、压住 DWA，等高ID车开过去后按原航点继续。
        只『停』不『绕』——避免两车同时机动。"""
        self._navigating = False
        self.cancel_pub.publish(GoalID())   # 停掉 move_base/DWA，免得它自己乱绕
        rospy.sleep(0.1)
        stop = Twist()
        deadline = rospy.Time.now() + rospy.Duration(PAUSE_MAX_WAIT)
        rate = rospy.Rate(10)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            self.cmd_pub.publish(stop)      # 持续发 0 速，确保真的钉在原地
            found, j = self._robot_in_my_path(require_higher_id=True)
            if not found or j != avoid_j:   # 对方已不在我正前方 → 可以继续
                break
            if not self._other_moving(avoid_j):  # 对方停下不走了 → 别空等，改自己绕行
                rospy.loginfo('[robot_%d] robot_%d 停住不走，改重规划绕行', self.rid, avoid_j)
                self._yield_reroute(avoid_j)
                return
            rate.sleep()
        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True
        self._retry_nav()                   # 按原航点继续，路线不变

    def _yield_reroute(self, avoid_j):
        """让行车：目标不变，重新规划一条路绕过停着的对方；不阻塞等待，自己一直在动。
        开阔区不再后退（那会造成'微微倒车'抖动），直接重发目标让 move_base 绕行；
        只有真正被堵到前方贴脸时才原地转一下错开。"""
        self._navigating = False
        self.cancel_pub.publish(GoalID())
        rospy.sleep(0.1)
        # 前方贴脸（move_base 一时绕不开）才转一下错开；否则直接重发目标自然绕行
        if self._sector_min(0, half_deg=30) < REAR_SAFE_DIST:
            self._do_rotate()
        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True
        self._retry_nav()   # 重发目标，让 move_base 绕开停住的对方

    def _do_rotate(self):
        """原地旋转；偶数ID左转、奇数ID右转，避免两车在窄道同向死转"""
        twist = Twist()
        twist.angular.z = 0.5 if self.rid % 2 == 0 else -0.5
        end_t = rospy.Time.now() + rospy.Duration(2.0)
        rate  = rospy.Rate(20)
        while rospy.Time.now() < end_t and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.cmd_pub.publish(Twist())

    def _yield_throat(self):
        """载货下行车退出窄口（向北后退），等底部离场车走完再重发投递航点。"""
        self._navigating = False
        self.cancel_pub.publish(GoalID())
        rospy.sleep(0.1)
        # 后退 = 向北退出窄口；后方为开阔横厅，通常安全
        if self._sector_min(180, half_deg=40) >= REAR_SAFE_DIST:
            twist = Twist()
            twist.linear.x = ESCAPE_VEL
            end_t = rospy.Time.now() + rospy.Duration(ESCAPE_DURATION)
            rate  = rospy.Rate(20)
            while rospy.Time.now() < end_t and not rospy.is_shutdown():
                self.cmd_pub.publish(twist)
                rate.sleep()
            self.cmd_pub.publish(Twist())

        # 等底部不再有空载待发车（或最多 10s）
        deadline = rospy.Time.now() + rospy.Duration(10.0)
        rate = rospy.Rate(2)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            with self.lock:
                op = dict(self.other_pos)
                oc = dict(self.other_carrying)
            if not any(oc.get(jj, GOODS_NONE) == GOODS_NONE and _in_bottom(*op[jj])
                       for jj in op):
                break
            rate.sleep()

        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True
        self._retry_nav()   # 重发投递航点，重新下去

    def _other_carrying_cb(self, msg, j):
        with self.lock:
            self.other_carrying[j] = msg.data

    def _other_waypoint_cb(self, msg, j):
        """跟踪其他车当前目标货架，用于避免重复抢占"""
        try:
            wp_num = int(msg.data)
            target = wp_num - 1 if 1 <= wp_num <= N_SHELVES else -1
        except ValueError:
            target = -1
        with self.lock:
            self.other_targets[j] = target

    def _scan_cb(self, msg):
        self._scan_ranges    = list(msg.ranges)
        self._scan_angle_min = msg.angle_min
        self._scan_angle_inc = msg.angle_increment

    def _sector_min(self, angle_deg, half_deg=30):
        """返回机器人坐标系中某扇区内的最近激光距离（0°=正前方）"""
        if not self._scan_ranges:
            return float('inf')
        center = math.radians(angle_deg)
        half   = math.radians(half_deg)
        mn = float('inf')
        for i, r in enumerate(self._scan_ranges):
            if not (math.isfinite(r) and r > 0.05):
                continue
            a = self._scan_angle_min + i * self._scan_angle_inc
            diff = (a - center + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half:
                mn = min(mn, r)
        return mn

    def _shelf_goods_cb(self, msg):
        try:
            goods = [int(x) for x in msg.data.split(',')]
            if len(goods) == N_SHELVES:
                with self.lock:
                    self.shelf_goods = goods
        except Exception:
            pass

    def _reserved_cb(self, msg):
        try:
            rsv = [int(x) for x in msg.data.split(',')]
            if len(rsv) == N_SHELVES:
                with self.lock:
                    self.reserved = rsv
        except Exception:
            pass

    def _deliver_reserved_cb(self, msg):
        try:
            rsv = [int(x) for x in msg.data.split(',')]
            if len(rsv) == N_DELIVER:
                with self.lock:
                    self.deliver_reserved = rsv
        except Exception:
            pass

    def _navi_result_cb(self, msg):
        if self._arrived:
            return   # 已就近判定到达并处理，忽略 move_base 的迟到结果
        self._navigating = False
        with self.lock:
            cur_state = self.state
        if cur_state not in (STATE_GOING_SHELF, STATE_GOING_DELIVERY):
            return

        if msg.data == 'done':
            # 到达核对：核实车真的开到本段目标点附近，挡住假"到达"
            pos = self._nav_goal_pos
            raw = self._raw_pos
            if pos is not None and raw is not None:
                dist = math.hypot(raw[0] - pos[0], raw[1] - pos[1])
                tol = self._cur_tol()
                if dist >= tol:
                    rospy.logwarn('[robot_%d] 收到 done 但离本段目标 %.2fm（>%.1f），判为假到达，重试',
                                  self.rid, dist, tol)
                    self._retry_nav()   # 重发航点，不切状态
                    return
            self._arrived = True
            self._on_leg_arrive()
        else:
            rospy.logwarn('[robot_%d] 导航失败，2s 后重试', self.rid)
            threading.Timer(2.0, self._retry_nav).start()

    # ── 状态机事件 ────────────────────────────────────────────────────────

    def _on_arrive_shelf(self):
        with self.lock:
            shelf_idx   = self.current_shelf
            goods       = self.shelf_goods[shelf_idx] if shelf_idx >= 0 else GOODS_NONE

        if goods == GOODS_NONE:
            # 货架被其他车取走，重新选择
            rospy.logwarn('[robot_%d] 货架%d 为空，重新选择', self.rid, shelf_idx)
            with self.lock:
                self.state          = STATE_SELECTING
                self.current_action = -1
            threading.Timer(1.0, self._do_select).start()
            return

        with self.lock:
            self.carrying = goods
            self.state    = STATE_GOING_DELIVERY
            self.current_action = N_SHELVES + (goods - 1)  # 10/11/12

        rospy.loginfo('[robot_%d] 取货架%d，货物 %s', self.rid, shelf_idx, GOODS_NAME[goods])
        self.carrying_pub.publish(Int32(data=goods))
        self.pickup_pub.publish(String(data='{} {}'.format(self.rid, shelf_idx)))

        # 取完货立即离开货架往投递区走，别堵着货架空等投递点。
        # 投递点预约只发一次做排序记录；真冲突由 DWA 注入 + 左缝单行兜住，
        # 排队自然发生在投递点附近，而不是冻在货架前。
        point = goods - 1                       # A/B/C → 0/1/2
        self.deliver_claim_pub.publish(String(data='{} {}'.format(self.rid, point)))
        action = N_SHELVES + (goods - 1)
        rospy.loginfo('[robot_%d] 经左缝进投递区 %s', self.rid, GOODS_NAME[goods])
        self._go_route(IN_ROUTE, str(action + 1), WP_POS[action],
                       STATE_GOING_DELIVERY, 'delivery')

    def _on_arrive_delivery(self):
        with self.lock:
            goods = self.carrying
            self.carrying       = GOODS_NONE
            self.state          = STATE_SELECTING
            self.current_action = -1

        rospy.loginfo('[robot_%d] 投递 %s ✓', self.rid, GOODS_NAME[goods])
        self.carrying_pub.publish(Int32(data=GOODS_NONE))
        self.delivery_pub.publish(String(data='{} {}'.format(self.rid, goods)))
        # 选下一个货架；离开底部由 _do_select 经"出站点"单行上行（右缝），不会对撞
        threading.Timer(0.3, self._do_select).start()

    def _depart_when_ready(self, t0=None):
        """出口单车闸口（上行方向）：窄口空 且 没有更小车号的待发车 才出发；
        按车号先来后到，最低车号永远能走 → 破活锁。超时则强制出发兜底。"""
        now = rospy.Time.now()
        if t0 is None:
            t0 = now
        with self.lock:
            op = dict(self.other_pos)
            oc = dict(self.other_carrying)

        gate_busy = any(_in_gate(r, c) for (r, c) in op.values())
        lower_departer = any(
            j < self.rid and oc.get(j, GOODS_NONE) == GOODS_NONE and _in_bottom(*op[j])
            for j in op)

        if (gate_busy or lower_departer) and (now - t0).to_sec() < DEPART_MAX_WAIT:
            rospy.loginfo('[robot_%d] 出口闸口占用/有低号车待发，1s 后再出发', self.rid)
            threading.Timer(1.0, lambda: self._depart_when_ready(t0)).start()
            return
        if (now - t0).to_sec() >= DEPART_MAX_WAIT:
            rospy.logwarn('[robot_%d] 出口等待超时，强制出发', self.rid)
        self._do_select()

    def _retry_nav(self):
        """重发当前这一段的航点（导航失败/让行后重试）"""
        with self.lock:
            state = self.state
        if state == STATE_SELECTING or self._nav_goal_wp is None:
            self._do_select()
            return

        rospy.loginfo('[robot_%d] 重试 WP%s', self.rid, self._nav_goal_wp)
        self.wp_pub.publish(String(data=self._nav_goal_wp))
        self._arrived     = False
        self._last_move_t = rospy.Time.now()
        self._last_pos_t  = rospy.Time.now()
        self._navigating  = True

    # ── obs 构造 + QMIX 推理 ──────────────────────────────────────────────

    def _build_obs(self):
        with self.lock:
            my_row, my_col = self.pos
            my_carrying    = self.carrying
            my_action      = self.current_action
            shelf_goods    = list(self.shelf_goods)
            other_pos      = dict(self.other_pos)
            other_carrying = dict(self.other_carrying)

        obs = []

        # self_feat (19): pos(2) + carrying_onehot(4) + dest_onehot(13)
        obs += [my_row / GRID_H, my_col / GRID_W]
        obs += carrying_onehot(my_carrying)
        dest_oh = [0.0] * N_ACTIONS
        if 0 <= my_action < N_ACTIONS:
            dest_oh[my_action] = 1.0
        obs += dest_oh

        # other_feats (18): (N_AGENTS-1) × [pos(2) + carrying_onehot(4)]
        for j in range(N_AGENTS):
            if j == self.rid:
                continue
            row, col = other_pos.get(j, (GRID_H - 1, 8 + j))
            c        = other_carrying.get(j, GOODS_NONE)
            obs += [row / GRID_H, col / GRID_W]
            obs += carrying_onehot(c)

        # shelf_feats (50): N_SHELVES × [goods_onehot(3) + pos(2)]
        for k in range(N_SHELVES):
            obs += goods_onehot(shelf_goods[k])
            shelf_row, shelf_col = SHELF_GRID_POS[k]
            obs += [shelf_row / GRID_H, shelf_col / GRID_W]

        return np.array(obs, dtype=np.float32)

    def _select_action(self):
        """调用 QMIX 选择下一个动作，返回 action 编号，-1 表示等待补货"""
        with self.lock:
            my_carrying    = self.carrying
            shelf_goods    = list(self.shelf_goods)
            reserved       = list(self.reserved)

        obs_t = torch.from_numpy(self._build_obs()).unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs_t)[0].numpy()  # [13]

        if my_carrying != GOODS_NONE:
            # 携带货物 → 强制去投递区
            return int(q[N_SHELVES:].argmax()) + N_SHELVES

        # 未携带 → 选有货的货架
        non_empty = [k for k in range(N_SHELVES) if shelf_goods[k] != GOODS_NONE]
        if not non_empty:
            return -1  # 全空，等待

        q_shelves = q[:N_SHELVES].copy()
        for k in range(N_SHELVES):
            if shelf_goods[k] == GOODS_NONE:
                q_shelves[k] = -1e9

        # 排除其他车已预约的货架（cargo_manager 唯一发号，杜绝两车抢同一架）
        for k in range(N_SHELVES):
            if reserved[k] != -1 and reserved[k] != self.rid:
                q_shelves[k] = -1e9

        # 主场区域软偏好：给本车负责区域的货架加 Q 值加成
        for k in ZONE_SHELVES.get(self.rid, []):
            if q_shelves[k] > -1e8:
                q_shelves[k] += ZONE_BONUS

        # 区域容量限制：目标区已有 ZONE_MAX 台其他车 → 大幅降权（软限制，不死锁）
        for k in range(N_SHELVES):
            if q_shelves[k] <= -1e8:
                continue
            zone = SHELF_ZONE.get(k)
            if zone and self._zone_other_count(zone) >= ZONE_MAX:
                q_shelves[k] -= 800
                rospy.logdebug('[robot_%d] %s 区已满，shelf%d 降权', self.rid, zone, k)

        return int(q_shelves.argmax())

    def _do_select(self, attempt=0):
        with self.lock:
            if self.state != STATE_SELECTING:
                return

        action = self._select_action()

        if action == -1:
            rospy.loginfo('[robot_%d] 所有货架空置，2s 后重试', self.rid)
            threading.Timer(2.0, self._do_select).start()
            return

        # 取货架前先向 cargo_manager 预约，确认占到了再出发（防两车抢同一架）
        if action < N_SHELVES:
            self.claim_pub.publish(String(data='{} {}'.format(self.rid, action)))
            rospy.sleep(0.3)   # 等 cargo_manager 发号回声
            with self.lock:
                granted = (self.reserved[action] == self.rid)
            if not granted:
                if attempt < 3:
                    rospy.loginfo('[robot_%d] 货架%d 被抢，改选（第%d次）',
                                  self.rid, action, attempt + 1)
                    self._do_select(attempt + 1)
                    return
                rospy.logwarn('[robot_%d] 连续抢占失败，1s 后重试', self.rid)
                threading.Timer(1.0, self._do_select).start()
                return

        with self.lock:
            self.current_action = action
            if action < N_SHELVES:
                self.current_shelf = action
            my_row = self.pos[0]

        if action < N_SHELVES and my_row >= BOTTOM_EXIT_ROW:
            if self._first_depart:
                # 开局首次出车：自由全局规划，允许走任意缝先散开，避免四车挤右缝
                self._first_depart = False
                rospy.loginfo('[robot_%d] 开局首次出车，自由规划 → 货架%d', self.rid, action)
                self._go_route([], str(action + 1), WP_POS[action],
                               STATE_GOING_SHELF, 'shelf')
            else:
                # 稳态：经"出站路线"单行离开（右缝底→顶，焊死上行），再驶向货架
                rospy.loginfo('[robot_%d] 在底部，经右缝出 → 货架%d', self.rid, action)
                self._go_route(OUT_ROUTE, str(action + 1), WP_POS[action],
                               STATE_GOING_SHELF, 'shelf')
        elif action < N_SHELVES:
            rospy.loginfo('[robot_%d] action=%d → 货架WP%s', self.rid, action, action + 1)
            self._go_route([], str(action + 1), WP_POS[action],
                           STATE_GOING_SHELF, 'shelf')
        else:
            # 极少走到：载货状态下 _do_select 选到投递 → 经左缝进投递区
            rospy.loginfo('[robot_%d] action=%d → 经左缝投递', self.rid, action)
            self._go_route(IN_ROUTE, str(action + 1), WP_POS[action],
                           STATE_GOING_DELIVERY, 'delivery')

    # ── 启动 ──────────────────────────────────────────────────────────────

    def start(self):
        # 发布初始携带状态
        self.carrying_pub.publish(Int32(data=GOODS_NONE))

        rospy.loginfo('[robot_%d] 等待 cargo_manager 初始化...', self.rid)
        deadline = rospy.Time.now() + rospy.Duration(45.0)
        while not rospy.is_shutdown():
            with self.lock:
                any_goods = any(g != GOODS_NONE for g in self.shelf_goods)
            if any_goods:
                break
            if rospy.Time.now() > deadline:
                rospy.logwarn('[robot_%d] 超时等待 shelf_goods，强制启动', self.rid)
                break
            rospy.sleep(1.0)

        # 错开各车启动时间，避免同时发起导航
        rospy.sleep(self.rid * 1.5)
        rospy.loginfo('[robot_%d] 开始任务', self.rid)
        self._do_select()


def main():
    rospy.init_node('robot_agent', anonymous=False)
    robot_id = rospy.get_param('~robot_id', 0)
    rospy.loginfo('启动 robot_agent，robot_id=%d', robot_id)

    agent = RobotAgent(robot_id)
    agent.start()
    rospy.spin()


if __name__ == '__main__':
    main()
