#!/usr/bin/env python3
"""
demo_setup.py
等 Gazebo 就绪后，把4辆AGV传送到展示位置。
路径标记发布由 path_marker_pub.py 独立负责。
"""
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
import tf.transformations as tft

SHOWCASE = [
    (-7.54, -0.50,  0.0),            # robot_0 西货架, 朝东
    ( 7.57,  1.53,  math.pi),        # robot_1 东货架, 朝西
    ( 0.00,  2.50, -math.pi / 2),    # robot_2 走廊, 朝南
    (-0.60, -4.49,  math.pi * 0.75), # robot_3 投递区, 朝西北
]


def _quat(yaw):
    return tft.quaternion_from_euler(0.0, 0.0, yaw)


def teleport_all(set_state):
    for i, (x, y, yaw) in enumerate(SHOWCASE):
        q = _quat(yaw)
        ms = ModelState()
        ms.model_name = f'robot_{i}'
        ms.pose.position.x = x
        ms.pose.position.y = y
        ms.pose.position.z = 0.05
        ms.pose.orientation.x = q[0]
        ms.pose.orientation.y = q[1]
        ms.pose.orientation.z = q[2]
        ms.pose.orientation.w = q[3]
        ms.twist.linear.x = ms.twist.linear.y = ms.twist.angular.z = 0.0
        ms.reference_frame = 'world'
        try:
            set_state(ms)
            rospy.loginfo('[demo] robot_%d → (%.2f, %.2f, yaw=%.2f)', i, x, y, yaw)
        except Exception as e:
            rospy.logwarn('[demo] robot_%d 传送失败: %s', i, e)


def main():
    rospy.init_node('demo_setup', anonymous=False)

    rospy.loginfo('[demo] 等待 Gazebo 服务 ...')
    try:
        rospy.wait_for_service('/gazebo/set_model_state', timeout=60)
        rospy.wait_for_service('/gazebo/pause_physics',   timeout=60)
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=60)
    except rospy.ROSException as e:
        rospy.logerr('[demo] Gazebo 服务超时: %s', e)
        return

    set_state  = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    pause_phys = rospy.ServiceProxy('/gazebo/pause_physics',   Empty)
    unpause    = rospy.ServiceProxy('/gazebo/unpause_physics',  Empty)

    try:
        pause_phys()
        rospy.loginfo('[demo] 物理引擎已暂停')
    except Exception as e:
        rospy.logwarn('[demo] pause_physics 失败: %s', e)

    rospy.sleep(0.3)
    teleport_all(set_state)
    rospy.sleep(0.4)
    teleport_all(set_state)   # 发两次确保生效
    rospy.sleep(0.3)

    try:
        unpause()
        rospy.loginfo('[demo] 物理引擎已恢复')
    except Exception as e:
        rospy.logwarn('[demo] unpause_physics 失败: %s', e)

    rospy.loginfo('[demo] 传送完成')


if __name__ == '__main__':
    main()
