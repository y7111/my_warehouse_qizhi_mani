#!/usr/bin/env python3
"""
laser_frame_fix.py — 修正激光扫描的 frame_id

问题：libgazebo_ros_laser 发布的 scan frame_id='laser'（无命名空间），
      但 TF 树里只有 robot_N/laser，costmap 无法完成坐标变换。

解决：订阅 scan，将 frame_id 改为 robot_N/laser，重发到 scan_fixed。
      costmap 订阅 scan_fixed，TF 查找正常。
"""

import rospy
from sensor_msgs.msg import LaserScan


def main():
    rospy.init_node('laser_frame_fix', anonymous=True)
    ns = rospy.get_namespace().rstrip('/')  # e.g. "/robot_0"
    fixed_frame = ns.lstrip('/') + '/laser'  # e.g. "robot_0/laser"

    pub = rospy.Publisher('scan_fixed', LaserScan, queue_size=10)

    def cb(msg):
        msg.header.frame_id = fixed_frame
        pub.publish(msg)

    rospy.Subscriber('scan', LaserScan, cb)
    rospy.loginfo('[laser_frame_fix] %s/scan → frame_id=%s → scan_fixed', ns, fixed_frame)
    rospy.spin()


if __name__ == '__main__':
    main()
