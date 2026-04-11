#!/usr/bin/env python3
"""
init_robot_poses.py
启动后自动向两个机器人的 AMCL 发布精确初始位姿，消除粒子云散布导致的位置偏移。

原理：向 /robot_xxx/initialpose 发布 PoseWithCovarianceStamped，
      协方差矩阵设为极小值，强制 AMCL 粒子集中在正确位置。
"""
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import math

# 与 two_robots_nav.launch 中的 spawn 坐标严格一致
ROBOTS = {
    'robot_alpha': {'x': -1.5, 'y': 0.0, 'yaw': 0.0},
    'robot_beta':  {'x':  1.5, 'y': 0.0, 'yaw': math.pi},
}


def yaw_to_quat_z_w(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def make_initial_pose(x, y, yaw):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rospy.Time.now()

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0

    qz, qw = yaw_to_quat_z_w(yaw)
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw

    # 协方差矩阵（6x6展开为36元素），xy和yaw设为极小值
    # 索引 0=xx, 7=yy, 35=yaw-yaw
    cov = [0.0] * 36
    cov[0]  = 1e-6   # x 方差
    cov[7]  = 1e-6   # y 方差
    cov[35] = 1e-6   # yaw 方差
    msg.pose.covariance = cov

    return msg


def main():
    rospy.init_node('init_robot_poses', anonymous=False)

    pubs = {}
    for robot_name in ROBOTS:
        topic = f'/{robot_name}/initialpose'
        pubs[robot_name] = rospy.Publisher(
            topic, PoseWithCovarianceStamped, queue_size=1, latch=True)

    # 等待 AMCL 节点就绪（订阅者上线）
    rospy.loginfo("[init_poses] 等待 AMCL 节点就绪...")
    rospy.sleep(3.0)

    for robot_name, cfg in ROBOTS.items():
        msg = make_initial_pose(cfg['x'], cfg['y'], cfg['yaw'])
        pubs[robot_name].publish(msg)
        rospy.loginfo("[init_poses] %s 初始位姿已发布: (%.1f, %.1f, yaw=%.3f)",
                      robot_name, cfg['x'], cfg['y'], cfg['yaw'])

    # 保持短暂存活确保消息被接收（latch=True 会重发给后来的订阅者）
    rospy.sleep(2.0)
    rospy.loginfo("[init_poses] 初始化完成，节点退出")


if __name__ == '__main__':
    main()
