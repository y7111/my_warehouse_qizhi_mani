#!/usr/bin/env python3
"""
path_marker_pub.py  ——  QMIX 静态演示可视化
发布到 /static_path_viz:
  路径线(无文字) · 货架 · 货架货物 · 投递点货物 · 障碍墙 · 冲突区
"""
import rospy, math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf.transformations as tft

# ── 颜色 ──────────────────────────────────────────────────────
RED    = (0.95, 0.18, 0.18)
GREEN  = (0.10, 0.78, 0.20)
BLUE   = (0.15, 0.45, 1.00)
ORANGE = (1.00, 0.52, 0.03)
GOODS  = {
    'A': (0.12, 0.80, 0.18),   # 绿色货物
    'B': (0.95, 0.82, 0.10),   # 黄色货物
    'C': (0.20, 0.25, 0.90),   # 蓝色货物
}

# ── 机器人位置 (x, y, yaw, rgb) ──────────────────────────────
ROBOTS = [
    (-7.54, -0.50,  0.0,          RED),
    ( 7.57,  1.53,  math.pi,      GREEN),
    ( 0.00,  2.50, -math.pi/2,    BLUE),
    (-0.60, -4.49,  math.pi*0.75, ORANGE),
]

# ── 路径 (对角线段模拟真实 A* 规划风格) ──────────────────────
PATHS = [
    # robot_0 红: 西货架→投递A(-2.5,-5.5) 沿西臂下行道东行,中枢南下
    [(-7.54,-0.50), (-7.20,-1.00), (-5.80,-1.20),
     (-4.20,-1.40), (-3.00,-1.50), (-2.54,-2.20),
     (-2.54,-3.50), (-2.50,-5.50)],

    # robot_1 绿: 东货架→投递C(1.5,-5.5) QMIX分配走x=2.0避开robot_2
    [(7.57, 1.53), (7.50, 2.10), (6.20, 2.30),
     (4.80, 2.10), (3.50, 1.70), (2.50, 1.00),
     (2.00, 0.00), (2.00,-3.50), (1.80,-5.00), (1.50,-5.50)],

    # robot_2 蓝: 走廊→投递B(-0.5,-5.5) QMIX检测冲突→沿x=1.5绕行
    [(0.00, 2.50), (0.30, 1.60), (0.65, 0.90),
     (1.10, 0.40), (1.50, 0.00), (1.50,-3.50),
     (0.50,-4.80), (-0.50,-5.50)],

    # robot_3 橙: 投递区短路径(已完成,小回转)
    [(-0.60,-4.49), (-1.10,-4.85), (-1.90,-5.10), (-2.50,-5.50)],
]

GOALS = [(-2.50,-5.50), (1.50,-5.50), (-0.50,-5.50), None]

# ── 货架位置与货物类型 ─────────────────────────────────────
SHELVES = [
    ((-1.5,  6.5), 'B'),   # 北臂左
    (( 1.5,  6.5), 'C'),   # 北臂右
    ((-8.5,  1.5), 'A'),   # 西上1 ← robot_0取货区
    ((-5.5,  1.5), 'C'),   # 西上2
    ((-8.5, -0.5), 'A'),   # 西下1
    ((-5.5, -0.5), 'B'),   # 西下2
    (( 5.5,  1.5), 'B'),   # 东上1
    (( 8.5,  1.5), 'C'),   # 东上2 ← robot_1取货区
    (( 5.5, -0.5), 'A'),   # 东下1
    (( 8.5, -0.5), 'A'),   # 东下2
]

# ── 投递点 (x,y,货物类型) ────────────────────────────────────
DELIVERY = [
    (-2.50, -5.50, 'A'),   # 投递区A
    (-0.50, -5.50, 'B'),   # 投递区B
    ( 1.50, -5.50, 'C'),   # 投递区C
]

# ── 关键障碍墙 (center_x, center_y, size_x, size_y) ──────────
WALLS = [
    (-6.5,  0.5, 3.0, 1.0),   # wall_corridor_w
    ( 6.5,  0.5, 3.0, 1.0),   # wall_corridor_e
    ( 0.0, -3.5, 2.0, 1.0),   # wall_s_btl
    (-6.5,  5.5, 7.0, 5.0),   # wall_nw
    ( 6.5,  5.5, 7.0, 5.0),   # wall_ne
    (-6.5, -5.0, 7.0, 6.0),   # wall_sw
    ( 6.5, -5.0, 7.0, 6.0),   # wall_se
]


def _hdr():
    from std_msgs.msg import Header
    h = Header(); h.frame_id = 'map'; h.stamp = rospy.Time(0)
    return h


def path_line(uid, waypoints, rgb, width=0.055):
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'path', uid
    m.type = Marker.LINE_STRIP; m.action = Marker.ADD
    m.pose.orientation.w = 1.0; m.scale.x = width
    r, g, b = rgb
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.92
    for x, y in waypoints:
        p = Point(); p.x, p.y, p.z = x, y, 0.04; m.points.append(p)
    return m


def robot_arrow(uid, x, y, yaw, rgb):
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'arrow', uid
    m.type = Marker.ARROW; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.15
    q = tft.quaternion_from_euler(0.0, 0.0, yaw)
    m.pose.orientation.x, m.pose.orientation.y = q[0], q[1]
    m.pose.orientation.z, m.pose.orientation.w = q[2], q[3]
    m.scale.x, m.scale.y, m.scale.z = 0.65, 0.22, 0.22
    r, g, b = rgb
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
    return m


def goal_sphere(uid, x, y, rgb):
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'goal', uid
    m.type = Marker.SPHERE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.05
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = 0.40
    r, g, b = rgb
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.60
    return m


def shelf_cube(uid, x, y):
    """货架本体 (棕色)"""
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'shelf', uid
    m.type = Marker.CUBE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.30
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.40; m.scale.z = 0.60
    m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.27, 0.07, 0.90
    return m


def cargo_on_shelf(uid, x, y, goods_type):
    """货架顶部货物 (彩色小箱, 与Gazebo同步: 0.10×0.10×0.08)"""
    r, g, b = GOODS[goods_type]
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'cargo', uid
    m.type = Marker.CUBE; m.action = Marker.ADD
    # 货架顶 z=0.22, 箱高0.08, 中心 z=0.26
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.26
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.10; m.scale.z = 0.08
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
    return m


def delivery_cargo(uid, x, y, goods_type):
    """投递点货物 (与Gazebo同步: 0.12×0.12×0.09)"""
    r, g, b = GOODS[goods_type]
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'delivery', uid
    m.type = Marker.CUBE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.045
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.12; m.scale.z = 0.09
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.90
    return m


def wall_box(uid, cx, cy, sx, sy):
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'wall', uid
    m.type = Marker.CUBE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.40
    m.pose.orientation.w = 1.0
    m.scale.x = sx; m.scale.y = sy; m.scale.z = 0.80
    m.color.r, m.color.g, m.color.b, m.color.a = 0.45, 0.45, 0.45, 0.50
    return m


def conflict_disk(uid, x, y, r=0.90):
    m = Marker(); m.header = _hdr(); m.ns, m.id = 'conflict', uid
    m.type = Marker.CYLINDER; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.02
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = r * 2; m.scale.z = 0.03
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.82, 0.0, 0.45
    return m


def main():
    rospy.init_node('path_marker_pub', anonymous=False)
    pub = rospy.Publisher('/static_path_viz', MarkerArray, queue_size=1, latch=True)
    rospy.sleep(0.5)

    ma = MarkerArray()
    uid = 0

    # 障碍墙
    for (cx, cy, sx, sy) in WALLS:
        ma.markers.append(wall_box(uid, cx, cy, sx, sy)); uid += 1

    # 货架 + 货架上的货物
    for (sx, sy), goods in SHELVES:
        ma.markers.append(shelf_cube(uid, sx, sy)); uid += 1
        ma.markers.append(cargo_on_shelf(uid, sx, sy, goods)); uid += 1

    # 投递点货物 (A=1箱, B=3箱, C=2箱，与Gazebo一致)
    ma.markers.append(delivery_cargo(uid, -2.50, -5.50, 'A')); uid += 1
    ma.markers.append(delivery_cargo(uid, -0.50, -5.50, 'B')); uid += 1
    ma.markers.append(delivery_cargo(uid, -0.22, -5.65, 'B')); uid += 1
    ma.markers.append(delivery_cargo(uid, -0.72, -5.38, 'B')); uid += 1
    ma.markers.append(delivery_cargo(uid,  1.50, -5.50, 'C')); uid += 1
    ma.markers.append(delivery_cargo(uid,  1.72, -5.32, 'C')); uid += 1

    # 目标球（仅保留目标点，不显示车头方向箭头）
    for i, (x, y, yaw, rgb) in enumerate(ROBOTS):
        if GOALS[i] is not None:
            gx, gy = GOALS[i]
            ma.markers.append(goal_sphere(uid, gx, gy, rgb)); uid += 1

    # 路径线 (四条, 无文字)
    for i, (wpts, (rx, ry, ryaw, rgb)) in enumerate(zip(PATHS, ROBOTS)):
        ma.markers.append(path_line(uid, wpts, rgb)); uid += 1

    # 冲突区 (中枢区域)
    ma.markers.append(conflict_disk(uid, 0.0, -0.50)); uid += 1

    pub.publish(ma)
    rospy.loginfo('[path_marker_pub] 发布 %d 个标记', len(ma.markers))
    rospy.spin()


if __name__ == '__main__':
    main()
