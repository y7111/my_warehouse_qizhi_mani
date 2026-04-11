#!/usr/bin/env python3
"""
simple_tour.py - 双车仓储演示
分两阶段：先走到走廊中间位置（验证基本运动），再去货架取货点。
"""
import math
import threading
import rospy
import actionlib
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus


# ── 初始位姿（与 launch spawn 一致）────────────────────────────────
INIT_POSES = {
    'robot_alpha': (-1.0, 0.0, 0.0),
    'robot_beta':  ( 1.0, 0.0, math.pi),
}

# ── 导航点序列（先走到走廊开阔位置，再去货架旁）──────────────────
# 走廊中点：x=±0.5, y=0  ← 空旷无障碍，验证基本运动
# 货架取货：y=±1.2，远离货架0.4m，远离墙壁0.3m
WAYPOINTS = {
    'robot_alpha': [
        (-0.5,  0.0,  0.0,      '走廊中点'),
        (-0.7,  1.2, -math.pi/2, 'rack1北侧取货'),
        (-0.5,  0.0,  0.0,      '返回走廊'),
        (-0.7, -1.2,  math.pi/2, 'rack3南侧取货'),
    ],
    'robot_beta': [
        ( 0.5,  0.0,  math.pi,   '走廊中点'),
        ( 0.7,  1.2, -math.pi/2, 'rack2北侧取货'),
        ( 0.5,  0.0,  math.pi,   '返回走廊'),
        ( 0.7, -1.2,  math.pi/2, 'rack4南侧取货'),
    ],
}


def publish_initial_pose(robot_name, x, y, yaw):
    topic = '/' + robot_name + '/initialpose'
    pub = rospy.Publisher(topic, PoseWithCovarianceStamped, queue_size=1, latch=True)
    rospy.sleep(0.5)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp    = rospy.Time.now()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    cov = [0.0] * 36
    cov[0]  = 1e-4
    cov[7]  = 1e-4
    cov[35] = 1e-4
    msg.pose.covariance = cov
    pub.publish(msg)
    rospy.loginfo('[%s] AMCL 初始位姿已设置 (%.1f, %.1f, yaw=%.2f)',
                  robot_name, x, y, yaw)


def make_goal(x, y, yaw):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp    = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
    goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
    return goal


def go_to(client, x, y, yaw, label, timeout=90.0):
    rospy.loginfo('  → 前往 %s (%.2f, %.2f)', label, x, y)
    client.send_goal(make_goal(x, y, yaw))
    finished = client.wait_for_result(rospy.Duration(timeout))
    state = client.get_state()
    if finished and state == GoalStatus.SUCCEEDED:
        rospy.loginfo('  ✓ 到达 %s', label)
        return True
    client.cancel_goal()
    rospy.logwarn('  ✗ %s 未到达（state=%d），继续下一个目标', label, state)
    return False


def run_robot(robot_name):
    x0, y0, yaw0 = INIT_POSES[robot_name]

    # 1. 强制设置 AMCL 初始位姿
    publish_initial_pose(robot_name, x0, y0, yaw0)

    # 2. 连接 move_base
    server = '/' + robot_name + '/move_base'
    rospy.loginfo('[%s] 等待 move_base...', robot_name)
    client = actionlib.SimpleActionClient(server, MoveBaseAction)
    client.wait_for_server()

    # 3. 等 AMCL 收敛
    rospy.loginfo('[%s] AMCL 收敛等待 3s...', robot_name)
    rospy.sleep(3.0)

    # 4. 依次前往各航点
    for wp in WAYPOINTS[robot_name]:
        x, y, yaw, label = wp
        go_to(client, x, y, yaw, label)
        rospy.sleep(1.0)

    rospy.loginfo('[%s] 全部航点完成', robot_name)


if __name__ == '__main__':
    rospy.init_node('simple_tour')

    rospy.loginfo('等待系统启动 3s...')
    rospy.sleep(3.0)

    t_alpha = threading.Thread(target=run_robot, args=('robot_alpha',))
    t_beta  = threading.Thread(target=run_robot, args=('robot_beta',))

    t_alpha.start()
    t_beta.start()
    t_alpha.join()
    t_beta.join()

    rospy.loginfo('=== 全部完成 ===')
