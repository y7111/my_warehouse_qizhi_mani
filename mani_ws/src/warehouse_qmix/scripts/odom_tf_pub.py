#!/usr/bin/env python3
"""
odom_tf_pub.py — 把 odom 话题重新发布为 TF 变换

用途：libgazebo_ros_planar_move 在 world 嵌入模式下 broadcastTF 失效，
      此节点订阅 odom 并广播 frame_id → child_frame_id 的 TF 变换。

每个机器人命名空间下运行一个实例：
  在 robot_0 namespace 下订阅 /robot_0/odom
  广播 robot_0/odom → robot_0/base_footprint
"""

import rospy
import tf
from nav_msgs.msg import Odometry


def odom_cb(msg, br):
    # +1ms offset: planar_move plugin also broadcasts TF at msg.header.stamp
    # causing TF_REPEATED_DATA. A 1ms shift makes timestamps distinct.
    stamp = rospy.Time(msg.header.stamp.secs,
                       msg.header.stamp.nsecs + 1000000)
    br.sendTransform(
        (msg.pose.pose.position.x,
         msg.pose.pose.position.y,
         msg.pose.pose.position.z),
        (msg.pose.pose.orientation.x,
         msg.pose.pose.orientation.y,
         msg.pose.pose.orientation.z,
         msg.pose.pose.orientation.w),
        stamp,
        msg.child_frame_id,   # robot_X/base_footprint
        msg.header.frame_id   # robot_X/odom
    )


def main():
    rospy.init_node('odom_tf_pub', anonymous=True)
    br = tf.TransformBroadcaster()
    rospy.Subscriber('odom', Odometry, lambda msg: odom_cb(msg, br))
    rospy.loginfo('[odom_tf_pub] 已启动，订阅 %s/odom',
                  rospy.get_namespace().rstrip('/'))
    rospy.spin()


if __name__ == '__main__':
    main()
