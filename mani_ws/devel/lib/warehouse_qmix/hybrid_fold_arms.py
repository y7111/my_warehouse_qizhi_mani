#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hybrid_fold_arms.py — 世界加载后,把嵌入式 4 车的机械臂摆成折叠姿势(全新文件)。
嵌入 world 的模型无法用 spawn 的 -J 设初始关节,这里用 /gazebo/set_model_configuration 设置。"""
import rospy
from gazebo_msgs.srv import SetModelConfiguration

FOLD = (['joint2', 'joint3', 'joint4'], [-1.57, 1.35, 0.24])  # 折叠姿势

if __name__ == '__main__':
    rospy.init_node('hybrid_fold_arms')
    rospy.wait_for_service('/gazebo/set_model_configuration')
    svc = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    rospy.sleep(2.0)
    for i in range(4):
        name = 'robot_{}'.format(i)
        try:
            r = svc(name, '', FOLD[0], FOLD[1])
            rospy.loginfo('[fold] %s: %s', name, r.status_message)
        except Exception as e:
            rospy.logwarn('[fold] %s 失败: %s', name, e)
        rospy.sleep(0.3)
    rospy.loginfo('[fold] 4 车机械臂折叠完成')
