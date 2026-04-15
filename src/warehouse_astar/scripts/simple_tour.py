#!/usr/bin/env python3
"""
simple_tour.py - 双车仓储演示
策略：停在两货架正中间 → 收臂转向 → 伸臂取货 → 收臂转向 → 伸臂取货
"""
import math
import threading
import rospy
import tf
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest
from std_srvs.srv import Empty

_tf_listener = None

# ── 初始位姿 ────────────────────────────────────────────────────────
INIT_POSES = {
    'robot_alpha': (-1.0, 0.0, 0.0),
    'robot_beta':  ( 1.0, 0.0, math.pi),
}

# ── 控制参数 ────────────────────────────────────────────────────────
TURN_SPEED  = 0.4    # 转速 rad/s
MOVE_SPEED  = 0.25   # 行进速度 m/s
TURN_TOL    = 0.08   # 转向容忍 rad
ARRIVE_TOL  = 0.30   # 到达容忍 m
CONTROL_HZ  = 10

# ── 手臂姿态 ────────────────────────────────────────────────────────
ARM_FOLD    = [0.0, -2.0, 0.0, 0.0]   # 收起：竖直向上，不碍事
ARM_EXTEND  = [0.0,  0.0, 0.0, 0.0]   # 伸出：水平向前，准备取货


def get_yaw(q):
    return 2.0 * math.atan2(q.z, q.w)


def angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def get_pose(robot_name, timeout=3.0):
    global _tf_listener
    target = robot_name + '/base_footprint'
    try:
        _tf_listener.waitForTransform('map', target, rospy.Time(0),
                                      rospy.Duration(timeout))
        (trans, rot) = _tf_listener.lookupTransform('map', target, rospy.Time(0))
        return trans[0], trans[1], 2.0 * math.atan2(rot[2], rot[3])
    except Exception:
        return None


def set_arm(robot_name, positions):
    """设置关节角度（gravity=0 + 无传动，无需暂停物理）"""
    try:
        srv = rospy.ServiceProxy('/gazebo/set_model_configuration',
                                 SetModelConfiguration)
        req = SetModelConfigurationRequest()
        req.model_name      = robot_name
        req.urdf_param_name = 'robot_description'
        req.joint_names     = ['joint1', 'joint2', 'joint3', 'joint4']
        req.joint_positions = positions
        srv(req)
    except Exception as e:
        rospy.logwarn('[%s] 手臂指令失败: %s', robot_name, e)


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
    cov[0] = cov[7] = cov[35] = 1e-4
    msg.pose.covariance = cov
    pub.publish(msg)
    rospy.loginfo('[%s] AMCL 初始位姿已设置', robot_name)


def go_to(robot_name, goal_x, goal_y, label, timeout=60.0):
    """先转向再直行，全程不倒车"""
    cmd_pub = rospy.Publisher('/' + robot_name + '/cmd_vel',
                              Twist, queue_size=1)
    rospy.sleep(0.1)
    rate    = rospy.Rate(CONTROL_HZ)
    deadline = rospy.Time.now() + rospy.Duration(timeout)

    rospy.loginfo('[%s] → %s (%.2f, %.2f)', robot_name, label, goal_x, goal_y)
    while not rospy.is_shutdown():
        if rospy.Time.now() > deadline:
            cmd_pub.publish(Twist())
            rospy.logwarn('[%s] %s 超时，跳过', robot_name, label)
            return False

        pose = get_pose(robot_name)
        if pose is None:
            rate.sleep(); continue

        cx, cy, cyaw = pose
        dx, dy = goal_x - cx, goal_y - cy
        dist = math.hypot(dx, dy)
        if dist < ARRIVE_TOL:
            cmd_pub.publish(Twist())
            rospy.loginfo('[%s] ✓ 到达 %s', robot_name, label)
            return True

        desired_yaw = math.atan2(dy, dx)
        yaw_err = angle_diff(desired_yaw, cyaw)

        cmd = Twist()
        if abs(yaw_err) > TURN_TOL:
            cmd.angular.z = TURN_SPEED * (1.0 if yaw_err > 0 else -1.0)
        else:
            cmd.linear.x  = max(0.1, min(MOVE_SPEED, MOVE_SPEED * dist / 0.6))
            cmd.angular.z = 0.6 * yaw_err
        cmd_pub.publish(cmd)
        rate.sleep()

    cmd_pub.publish(Twist())
    return False


def turn_to(robot_name, target_yaw, label=''):
    """原地精确转向到指定 yaw（收臂后调用）"""
    cmd_pub = rospy.Publisher('/' + robot_name + '/cmd_vel',
                              Twist, queue_size=1)
    rospy.sleep(0.1)
    rate     = rospy.Rate(CONTROL_HZ)
    deadline = rospy.Time.now() + rospy.Duration(15.0)

    rospy.loginfo('[%s] 转向 %.1f°%s', robot_name, math.degrees(target_yaw),
                  ' ' + label if label else '')
    while not rospy.is_shutdown():
        if rospy.Time.now() > deadline:
            break
        pose = get_pose(robot_name)
        if pose is None:
            rate.sleep(); continue
        yaw_err = angle_diff(target_yaw, pose[2])
        if abs(yaw_err) < 0.05:
            break
        cmd = Twist()
        cmd.angular.z = TURN_SPEED * (1.0 if yaw_err > 0 else -1.0)
        cmd_pub.publish(cmd)
        rate.sleep()
    cmd_pub.publish(Twist())


def pick_from(robot_name, face_yaw, shelf_label):
    """收臂 → 转向 → 伸臂 → 模拟取货停顿 → 收臂"""
    rospy.loginfo('[%s] 准备取货：%s', robot_name, shelf_label)
    set_arm(robot_name, ARM_FOLD)
    rospy.sleep(0.8)
    turn_to(robot_name, face_yaw, shelf_label)
    set_arm(robot_name, ARM_EXTEND)
    rospy.loginfo('[%s] 取货中：%s', robot_name, shelf_label)
    rospy.sleep(2.0)                  # 模拟取货动作
    set_arm(robot_name, ARM_FOLD)
    rospy.sleep(0.5)


def run_robot(robot_name):
    x0, y0, yaw0 = INIT_POSES[robot_name]

    # 1. 收臂
    rospy.wait_for_service('/gazebo/set_model_configuration', timeout=10.0)
    set_arm(robot_name, ARM_FOLD)
    rospy.loginfo('[%s] 手臂已收起', robot_name)

    # 2. 设置 AMCL 初始位姿
    publish_initial_pose(robot_name, x0, y0, yaw0)

    # 3. 等待 TF 就绪
    rospy.loginfo('[%s] 等待定位...', robot_name)
    for i in range(20):
        if get_pose(robot_name, timeout=2.0) is not None:
            rospy.loginfo('[%s] 定位就绪', robot_name)
            break
        rospy.logwarn('[%s] 等待中 (%d/20)', robot_name, i + 1)

    rospy.sleep(1.0)

    if robot_name == 'robot_alpha':
        # alpha：走到 rack1/rack3 正中间 (-0.7, 0)
        go_to(robot_name, -0.7, 0.0, '货架1&3中间')
        pick_from(robot_name,  math.pi / 2, 'rack1 北侧')   # 朝北取 rack1
        pick_from(robot_name, -math.pi / 2, 'rack3 南侧')   # 朝南取 rack3
        set_arm(robot_name, ARM_FOLD)
        go_to(robot_name, -1.0, 0.0, '返回起点')

    else:  # robot_beta
        # beta：走到 rack2/rack4 正中间 (0.7, 0)
        go_to(robot_name, 0.7, 0.0, '货架2&4中间')
        pick_from(robot_name,  math.pi / 2, 'rack2 北侧')   # 朝北取 rack2
        pick_from(robot_name, -math.pi / 2, 'rack4 南侧')   # 朝南取 rack4
        set_arm(robot_name, ARM_FOLD)
        go_to(robot_name, 1.0, 0.0, '返回起点')

    rospy.loginfo('[%s] 任务完成', robot_name)


if __name__ == '__main__':
    rospy.init_node('simple_tour')
    _tf_listener = tf.TransformListener()
    rospy.loginfo('等待系统启动...')
    rospy.sleep(5.0)
    # 确保物理引擎处于运行状态
    try:
        rospy.ServiceProxy('/gazebo/unpause_physics', Empty)()
    except Exception:
        pass

    t_alpha = threading.Thread(target=run_robot, args=('robot_alpha',))
    t_beta  = threading.Thread(target=run_robot, args=('robot_beta',))
    t_alpha.start()
    t_beta.start()
    t_alpha.join()
    t_beta.join()
    rospy.loginfo('=== 全部完成 ===')
