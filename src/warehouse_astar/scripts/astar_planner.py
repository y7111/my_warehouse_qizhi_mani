#!/usr/bin/env python3
"""
astar_planner.py
基于占用栅格地图的 A* 路径规划服务节点。

服务: /astar_planner/plan_path  (warehouse_astar/AStarPlan)
  请求: start (PoseStamped), goal (PoseStamped)
  响应: success (bool), path (nav_msgs/Path)

发布: /astar_path (nav_msgs/Path) — 供 RViz 可视化
"""
import heapq
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from warehouse_astar.srv import AStarPlan, AStarPlanResponse

OBSTACLE_THRESH = 65   # 占用概率 > 65 视为障碍
ROBOT_INFLATE_M = 0.22 # 膨胀半径，与 move_base robot_radius 一致


class AStarPlannerNode:

    def __init__(self):
        rospy.init_node('astar_planner')

        self.map_info  = None
        self.grid_orig = None   # 原始地图数组 (H×W, int8)
        self.grid_inf  = None   # 膨胀后地图数组

        rospy.Subscriber('/map', OccupancyGrid, self._map_cb, queue_size=1)
        self.path_pub = rospy.Publisher(
            '/astar_path', Path, queue_size=1, latch=True)
        rospy.Service('/astar_planner/plan_path', AStarPlan, self._plan_cb)

        rospy.loginfo('[A*] 等待地图...')
        rospy.wait_for_message('/map', OccupancyGrid)
        rospy.loginfo('[A*] 地图就绪，A* 规划服务已启动')

    # ── 地图回调 ──────────────────────────────────────────────────────

    def _map_cb(self, msg):
        self.map_info = msg.info
        arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self.grid_orig = arr
        t0 = rospy.Time.now()
        self.grid_inf = self._inflate(arr, msg.info.resolution)
        dt = (rospy.Time.now() - t0).to_sec()
        rospy.loginfo_once(
            '[A*] 地图 %dx%d  分辨率=%.3fm  膨胀耗时=%.3fs',
            msg.info.width, msg.info.height, msg.info.resolution, dt)

    def _inflate(self, grid, res):
        """对已知障碍物做方形膨胀，使机器人保持安全距离"""
        r = max(1, int(math.ceil(ROBOT_INFLATE_M / res)))
        result = grid.copy()

        # 只膨胀已知障碍（>OBSTACLE_THRESH），不膨胀未知区域(-1)
        occ_ys, occ_xs = np.where(grid > OBSTACLE_THRESH)
        if len(occ_ys) == 0:
            return result

        # 用 bounding-box 提速：只在障碍附近区域操作
        y_min = max(0,             int(occ_ys.min()) - r - 5)
        y_max = min(grid.shape[0], int(occ_ys.max()) + r + 5)
        x_min = max(0,             int(occ_xs.min()) - r - 5)
        x_max = min(grid.shape[1], int(occ_xs.max()) + r + 5)

        roi_obs = (grid[y_min:y_max, x_min:x_max] > OBSTACLE_THRESH).astype(np.uint8)

        try:
            from scipy.ndimage import maximum_filter
            inflated = maximum_filter(roi_obs, size=2 * r + 1) > 0
        except ImportError:
            # 无 scipy 时手动膨胀（较慢但正确）
            inflated = np.zeros_like(roi_obs, dtype=bool)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    shifted = np.roll(np.roll(roi_obs, dy, axis=0), dx, axis=1)
                    inflated |= shifted > 0

        result[y_min:y_max, x_min:x_max][inflated] = 100
        return result

    # ── 坐标转换 ──────────────────────────────────────────────────────

    def _w2g(self, wx, wy):
        """世界坐标 → 栅格坐标（列, 行）"""
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        res = self.map_info.resolution
        return int((wx - ox) / res), int((wy - oy) / res)

    def _g2w(self, gx, gy):
        """栅格坐标 → 世界坐标（格子中心）"""
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        res = self.map_info.resolution
        return ox + (gx + 0.5) * res, oy + (gy + 0.5) * res

    def _valid(self, gx, gy):
        """判断栅格是否可通行"""
        if gx < 0 or gy < 0:
            return False
        if gx >= self.map_info.width or gy >= self.map_info.height:
            return False
        v = int(self.grid_inf[gy, gx])
        return 0 <= v <= OBSTACLE_THRESH   # 已知且未被膨胀

    # ── A* 核心 ───────────────────────────────────────────────────────

    def _astar(self, sx, sy, gx, gy):
        """
        A* 搜索，8方向移动。
        返回栅格坐标路径列表 [(x,y), ...]，失败返回 None。
        """
        DIRS = [
            ( 1,  0, 1.000), (-1,  0, 1.000),
            ( 0,  1, 1.000), ( 0, -1, 1.000),
            ( 1,  1, 1.414), ( 1, -1, 1.414),
            (-1,  1, 1.414), (-1, -1, 1.414),
        ]

        def h(x, y):
            return math.hypot(x - gx, y - gy)

        # heap 元素: (f, g, x, y)
        open_heap = [(h(sx, sy), 0.0, sx, sy)]
        g_cost  = {(sx, sy): 0.0}
        parent  = {}

        while open_heap:
            f, g, cx, cy = heapq.heappop(open_heap)

            if (cx, cy) == (gx, gy):
                # 回溯路径
                path, node = [], (cx, cy)
                while node in parent:
                    path.append(node)
                    node = parent[node]
                path.append((sx, sy))
                path.reverse()
                return path

            if g > g_cost.get((cx, cy), 1e18):
                continue

            for dx, dy, cost in DIRS:
                nx, ny = cx + dx, cy + dy
                if not self._valid(nx, ny):
                    continue
                ng = g + cost
                if ng < g_cost.get((nx, ny), 1e18):
                    g_cost[(nx, ny)] = ng
                    parent[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_heap, (ng + h(nx, ny), ng, nx, ny))

        return None  # 无可达路径

    # ── 路径简化 (Theta* 风格 LoS 裁剪) ─────────────────────────────

    def _simplify(self, path):
        """用视线检测去掉多余中间节点"""
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._los(path[i], path[j]):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _los(self, p1, p2):
        """Bresenham 直线，检查路径上是否全部可通行"""
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if not self._valid(x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0  += sx
            if e2 <  dx:
                err += dx
                y0  += sy

    # ── 工具：就近找可通行格 ─────────────────────────────────────────

    def _nearest_valid(self, gx, gy, max_r=20):
        if self._valid(gx, gy):
            return (gx, gy)
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        cand = (gx + dx, gy + dy)
                        if self._valid(*cand):
                            return cand
        return None

    # ── 服务回调 ──────────────────────────────────────────────────────

    def _plan_cb(self, req):
        resp = AStarPlanResponse()

        if self.grid_inf is None:
            rospy.logwarn('[A*] 地图尚未就绪')
            resp.success = False
            return resp

        sx = req.start.pose.position.x
        sy = req.start.pose.position.y
        ex = req.goal.pose.position.x
        ey = req.goal.pose.position.y

        sg = self._w2g(sx, sy)
        eg = self._w2g(ex, ey)

        # 起点/终点若在障碍内则就近偏移
        sg = self._nearest_valid(*sg)
        eg = self._nearest_valid(*eg)

        if sg is None or eg is None:
            rospy.logwarn('[A*] 起点或终点附近无可通行格')
            resp.success = False
            return resp

        rospy.loginfo('[A*] 规划: (%.2f, %.2f) → (%.2f, %.2f)', sx, sy, ex, ey)
        t0 = rospy.Time.now()
        grid_path = self._astar(sg[0], sg[1], eg[0], eg[1])
        dt = (rospy.Time.now() - t0).to_sec()

        if grid_path is None:
            rospy.logwarn('[A*] 搜索失败，无可达路径')
            resp.success = False
            return resp

        simplified = self._simplify(grid_path)
        rospy.loginfo('[A*] 找到路径: %d 格 → 简化 %d 节点  耗时 %.3fs',
                      len(grid_path), len(simplified), dt)

        # 构建 nav_msgs/Path
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp    = rospy.Time.now()

        for i, (pgx, pgy) in enumerate(simplified):
            wx, wy = self._g2w(pgx, pgy)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # 最后一个点使用目标朝向
        if path_msg.poses:
            path_msg.poses[-1].pose.orientation = req.goal.pose.orientation

        self.path_pub.publish(path_msg)

        resp.success = True
        resp.path    = path_msg
        return resp

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    AStarPlannerNode().run()
