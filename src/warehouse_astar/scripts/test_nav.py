#!/usr/bin/env python3
"""
test_nav.py
单车逐一导航测试：先让 robot_beta 依次前往4个货架取货点（robot_alpha静止），
再让 robot_alpha 依次前往4个货架取货点（robot_beta静止）。
"""
import math
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus


# 4个货架取货停靠点 (x, y, yaw)
SHELF_PICKUP = [
    (-0.7,  0.0,  math.pi / 2),   # 货架1：左侧，朝北
    ( 0.7,  0.0,  math.pi / 2),   # 货架2：右侧，朝北
    (-0.7,  0.0, -math.pi / 2),   # 货架3：左侧，朝南
    ( 0.7,  0.0, -math.pi / 2),   # 货架4：右侧，朝南
]

# 离墙安全起始位置
HOME = {
    'robot_alpha': (-1.5, 0.0, 0.0),
    'robot_beta':  ( 1.5, 0.0, math.pi),
}


def yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def navigate(client, x, y, yaw, label):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp    = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation = yaw_to_quat(yaw)

    rospy.loginfo('[test] %s → (%.2f, %.2f)', label, x, y)
    client.send_goal(goal)
    client.wait_for_result()

    ok = client.get_state() == GoalStatus.SUCCEEDED
    rospy.loginfo('[test] %s → %s', label, '✓ 成功' if ok else '✗ 失败')
    return ok


def test_robot(robot_name):
    rospy.loginfo('========== 开始测试 %s ==========', robot_name)

    ns = '/' + robot_name + '/move_base'
    client = actionlib.SimpleActionClient(ns, MoveBaseAction)
    rospy.loginfo('[test] 等待 %s move_base...', robot_name)
    client.wait_for_server()

    # 步骤0：先离开墙壁
    hx, hy, hyaw = HOME[robot_name]
    navigate(client, hx, hy, hyaw, f'{robot_name} 安全起始位置')
    rospy.sleep(1.0)

    # 步骤1~4：依次前往4个货架
    for i, (x, y, yaw) in enumerate(SHELF_PICKUP, 1):
        navigate(client, x, y, yaw, f'{robot_name} 货架{i}')
        rospy.sleep(2.0)

    rospy.loginfo('========== %s 测试完成 ==========', robot_name)


if __name__ == '__main__':
    rospy.init_node('test_nav', anonymous=False)

    rospy.loginfo('[test] === 阶段一：robot_beta 导航，robot_alpha 静止 ===')
    test_robot('robot_beta')
    rospy.sleep(3.0)

    rospy.loginfo('[test] === 阶段二：robot_alpha 导航，robot_beta 静止 ===')
    test_robot('robot_alpha')

    rospy.loginfo('[test] 全部测试完成！')
