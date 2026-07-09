#!/usr/bin/env python3
"""依次导航到航点 1-13，到达后打印日志，全部完成后退出。"""

import rospy
from std_msgs.msg import String

WAYPOINTS = [str(i) for i in range(1, 14)]   # "1" ~ "13"

current_idx = 0
waypoint_pub = None


def send_next():
    global current_idx
    if current_idx >= len(WAYPOINTS):
        rospy.loginfo('[test_all_wp] 全部 %d 个航点巡航完毕！', len(WAYPOINTS))
        rospy.signal_shutdown('完成')
        return
    wp = WAYPOINTS[current_idx]
    rospy.loginfo('[test_all_wp] (%d/%d) 前往航点 %s ...',
                  current_idx + 1, len(WAYPOINTS), wp)
    msg = String()
    msg.data = wp
    waypoint_pub.publish(msg)


def navi_result_cb(msg):
    global current_idx
    if msg.data != 'done':
        return
    rospy.loginfo('[test_all_wp] 到达航点 %s ✓', WAYPOINTS[current_idx])
    current_idx += 1
    send_next()


def main():
    global waypoint_pub
    rospy.init_node('test_all_wp', anonymous=False)
    waypoint_pub = rospy.Publisher('/waterplus/navi_waypoint',
                                   String, queue_size=10)
    rospy.Subscriber('/waterplus/navi_result', String, navi_result_cb)
    rospy.sleep(1.0)
    send_next()
    rospy.spin()


if __name__ == '__main__':
    main()
