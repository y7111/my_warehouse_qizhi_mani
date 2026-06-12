#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟邻车雷达 neighbor_laser_pub
================================
问题：四车的真实激光装在 z=0.30，比车身顶（0.28）还高 → 各车激光从别人头顶扫过，
      DWA 代价地图里"看不见"别的车 → 不会避让 → 撞车。
方案：本节点订阅所有车的 /odom（精确连续位姿），为每辆车实时合成一圈"虚拟激光"，
      把其它车的车身画成激光返回，发到 /robot_i/neighbor_scan。
      该话题作为局部代价地图的第二个 LaserScan 障碍源 → DWA 像绕墙一样连续避开别的车。
      用 LaserScan（而非点云）是因为代价地图对激光天生逐帧清除，邻车移动后不留拖影。
"""
import math
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

N_ROBOTS   = 4
ROBOT_R    = 0.32     # 把每辆车当成半径 0.32m 的圆（车 0.57×0.45 的外接半径≈0.36，略收一点）
N_BEAMS    = 180      # 2° 分辨率
RANGE_MIN  = 0.10
RANGE_MAX  = 8.0
PUB_RATE   = 15.0     # Hz
ANGLE_MIN  = -math.pi
ANGLE_INC  = 2.0 * math.pi / N_BEAMS


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NeighborLaserPub(object):
    def __init__(self):
        self.n = rospy.get_param('~n_robots', N_ROBOTS)
        self.pose = {}   # i -> (x, y, yaw)
        self.pubs = {}
        for i in range(self.n):
            rospy.Subscriber('/robot_{}/odom'.format(i), Odometry,
                             lambda msg, ii=i: self._odom_cb(msg, ii))
            self.pubs[i] = rospy.Publisher('/robot_{}/neighbor_scan'.format(i),
                                           LaserScan, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / PUB_RATE), self._tick)
        rospy.loginfo('[neighbor_laser_pub] 启动，%d 车，虚拟邻车雷达就绪', self.n)

    def _odom_cb(self, msg, i):
        p = msg.pose.pose
        self.pose[i] = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _tick(self, _evt):
        now = rospy.Time.now()
        for i in range(self.n):
            if i not in self.pose:
                continue
            xi, yi, yawi = self.pose[i]
            ranges = [RANGE_MAX] * N_BEAMS
            c, s = math.cos(-yawi), math.sin(-yawi)   # 世界→车体 旋转
            for j in range(self.n):
                if j == i or j not in self.pose:
                    continue
                xj, yj, _ = self.pose[j]
                dx, dy = xj - xi, yj - yi
                bx = dx * c - dy * s     # 车体系：x 前 y 左
                by = dx * s + dy * c
                d = math.hypot(bx, by)
                if d > RANGE_MAX + ROBOT_R:
                    continue
                bearing = math.atan2(by, bx)
                surf = max(d - ROBOT_R, RANGE_MIN)      # 车身表面距离
                half = math.asin(min(1.0, ROBOT_R / max(d, ROBOT_R)))  # 圆的角半宽
                k0 = int(math.floor((bearing - half - ANGLE_MIN) / ANGLE_INC))
                k1 = int(math.ceil((bearing + half - ANGLE_MIN) / ANGLE_INC))
                for k in range(k0, k1 + 1):
                    kk = k % N_BEAMS
                    if surf < ranges[kk]:
                        ranges[kk] = surf
            scan = LaserScan()
            scan.header.stamp = now
            scan.header.frame_id = 'robot_{}/base_footprint'.format(i)
            scan.angle_min = ANGLE_MIN
            scan.angle_max = ANGLE_MIN + ANGLE_INC * (N_BEAMS - 1)
            scan.angle_increment = ANGLE_INC
            scan.range_min = RANGE_MIN
            scan.range_max = RANGE_MAX
            scan.ranges = ranges
            self.pubs[i].publish(scan)


if __name__ == '__main__':
    rospy.init_node('neighbor_laser_pub')
    NeighborLaserPub()
    rospy.spin()
