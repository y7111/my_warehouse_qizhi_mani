#!/usr/bin/env python3
"""
static_path_viz.py
在 RViz 中静态展示 4 辆 AGV 的位置与规划路径，供截图/论文用。

路径设计（基于 waypoints.xml 真实坐标，互不交叉）:
  robot_0 (红): WP5 西侧货架  → 投递区左列  (西走廊东行 + 南下)
  robot_1 (绿): WP8 东侧货架  → 投递区右列  (东走廊西行 + 南下)
  robot_2 (蓝): 南北走廊中段  → WP1 北侧货架 (正北)
  robot_3 (橙): WP12 投递区   (短路径, 刚到达)

运行:
  rosrun warehouse_qmix static_path_viz.py
RViz 添加:
  1. Map  →  topic: /map
  2. MarkerArray  →  topic: /static_path_viz
"""

import rospy
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf.transformations as tft


# ──────────────────────────────────────────────────────────────
def _quat(yaw):
    q = tft.quaternion_from_euler(0.0, 0.0, yaw)
    return q[0], q[1], q[2], q[3]


def _hdr():
    from std_msgs.msg import Header
    h = Header()
    h.frame_id = 'map'
    h.stamp = rospy.Time(0)
    return h


def make_arrow(uid, ns, x, y, yaw, rgb, length=0.75, width=0.28):
    m = Marker()
    m.header = _hdr()
    m.ns, m.id = ns, uid
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.05
    ox, oy, oz, ow = _quat(yaw)
    m.pose.orientation.x = ox
    m.pose.orientation.y = oy
    m.pose.orientation.z = oz
    m.pose.orientation.w = ow
    m.scale.x = length
    m.scale.y = width
    m.scale.z = width
    m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 1.0
    return m


def make_body(uid, ns, x, y, rgb):
    """矩形车体"""
    m = Marker()
    m.header = _hdr()
    m.ns, m.id = ns, uid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.15
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = 0.5, 0.4, 0.3
    m.color.r, m.color.g, m.color.b = rgb[0] * 0.6, rgb[1] * 0.6, rgb[2] * 0.6
    m.color.a = 0.85
    return m


def make_text(uid, ns, x, y, text, rgb, size=0.38):
    m = Marker()
    m.header = _hdr()
    m.ns, m.id = ns, uid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y + 0.65
    m.pose.position.z = 0.6
    m.pose.orientation.w = 1.0
    m.scale.z = size
    m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 1.0
    m.text = text
    return m


def make_path_line(uid, ns, waypoints, rgb, width=0.10):
    m = Marker()
    m.header = _hdr()
    m.ns, m.id = ns, uid
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = width
    m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 0.90
    for x, y in waypoints:
        p = Point()
        p.x, p.y, p.z = x, y, 0.05
        m.points.append(p)
    return m


def make_goal_sphere(uid, ns, x, y, rgb, r=0.25):
    m = Marker()
    m.header = _hdr()
    m.ns, m.id = ns, uid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.05
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = r * 2
    m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 0.60
    return m


# ──────────────────────────────────────────────────────────────
RED    = (0.95, 0.18, 0.18)
GREEN  = (0.10, 0.78, 0.20)
BLUE   = (0.15, 0.40, 1.00)
ORANGE = (1.00, 0.52, 0.03)

# (x, y, yaw_rad, color, label, goal_x, goal_y)
ROBOTS = [
    # robot_0 在 WP5 西侧货架，朝东，目标投递区左列
    (-7.54, -0.50,  0.0,           RED,    'AGV-0\n货架→投递', -2.54, -4.50),
    # robot_1 在 WP8 东侧货架，朝西，目标投递区右列
    ( 7.57,  1.53,  math.pi,       GREEN,  'AGV-1\n货架→投递',  1.50, -4.45),
    # robot_2 在南北走廊中段，朝北，目标 WP1 北侧货架
    ( 0.00,  2.50,  math.pi/2,     BLUE,   'AGV-2\n走廊→货架', -1.47,  5.56),
    # robot_3 在投递区 WP12，刚到达
    (-0.60, -4.49,  math.pi*0.75,  ORANGE, 'AGV-3\n投递点',    -0.60, -4.49),
]

# 规划路径（四条互不交叉，每条走独立走廊段）
PATHS = [
    # robot_0 (红): 西走廊→南下至左投递列
    [(-7.54,-0.50), (-4.50,-0.50), (-2.54,-0.50), (-2.54,-2.50), (-2.54,-4.50)],
    # robot_1 (绿): 东走廊→南下至右投递列
    [( 7.57, 1.53), ( 4.50, 1.53), ( 1.50, 1.53), ( 1.50,-0.50), ( 1.50,-4.45)],
    # robot_2 (蓝): 南北走廊向北至 WP1
    [( 0.00, 0.50), ( 0.00, 1.50), ( 0.00, 2.50), ( 0.00, 4.00), (-1.47, 5.56)],
    # robot_3 (橙): 投递区内短路径，刚到达
    [(-0.60,-3.00), (-0.60,-3.50), (-0.60,-4.00), (-0.60,-4.49)],
]


def main():
    rospy.init_node('static_path_viz', anonymous=False)
    pub = rospy.Publisher('/static_path_viz', MarkerArray,
                          queue_size=1, latch=True)
    rospy.sleep(0.5)   # 等 RViz 连接

    ma = MarkerArray()
    uid = 0

    for i, (x, y, yaw, rgb, label, gx, gy) in enumerate(ROBOTS):
        ma.markers.append(make_body(uid,  'body',  x,  y,  rgb));       uid += 1
        ma.markers.append(make_arrow(uid, 'arrow', x,  y,  yaw, rgb));  uid += 1
        ma.markers.append(make_text(uid,  'label', x,  y,  label, rgb));uid += 1
        # 目标位置小球（robot_3 目标就是自身，跳过）
        if i < 3:
            ma.markers.append(make_goal_sphere(uid, 'goal', gx, gy, rgb)); uid += 1

    for i, (waypoints, (x, y, yaw, rgb, *_)) in enumerate(zip(PATHS, ROBOTS)):
        ma.markers.append(make_path_line(uid, 'path', waypoints, rgb, width=0.11))
        uid += 1

    pub.publish(ma)
    rospy.loginfo('[static_path_viz] 发布完毕 (latched)。'
                  ' RViz 添加 MarkerArray -> /static_path_viz')
    rospy.spin()


if __name__ == '__main__':
    main()
