#!/usr/bin/env python3
"""等所有机器人 spawn 完成后，恢复 Gazebo 物理仿真。"""

import rospy
from std_srvs.srv import Empty

rospy.init_node('unpause_helper', anonymous=True)
delay = rospy.get_param('~delay', 10.0)
rospy.loginfo('[unpause] %.0f 秒后恢复物理仿真...', delay)
rospy.sleep(delay)
rospy.wait_for_service('/gazebo/unpause_physics', timeout=30.0)
rospy.ServiceProxy('/gazebo/unpause_physics', Empty)()
rospy.loginfo('[unpause] 物理仿真已恢复！')
