#!/usr/bin/env python3
"""
test_four_wp.py — 四车航点巡航测试（含卡住自动脱困）

卡住判定：连续 15s 位移 < 0.08m → 判为卡住
脱困动作：后退 1.5s，然后重发当前目标航点
"""

import math
import threading
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

ROBOT_ROUTES = {
    0: ['1', '2', '12'],
    1: ['3', '4', '11'],
    2: ['7', '8', '13'],
    3: ['5', '6', '12'],
}

ROBOT_START_DELAY = {0: 0, 1: 5, 2: 10, 3: 15}

STUCK_TIMEOUT   = 15.0   # 超过这么多秒没动就判为卡住
STUCK_DIST      = 0.08   # 位移门限（米）
ESCAPE_VEL      = -0.25  # 后退速度（m/s）
ESCAPE_DURATION = 1.5    # 后退时长（秒）

N_ROBOTS = 4


class RobotNavigator:
    def __init__(self, robot_id, route):
        self.robot_id = robot_id
        self.route    = route
        self.idx      = 0
        self.lock     = threading.Lock()

        ns = '/robot_{}'.format(robot_id)

        self.pub_wp = rospy.Publisher(
            '{}/waterplus/navi_waypoint'.format(ns),
            String, queue_size=10)

        self.pub_cmd = rospy.Publisher(
            '{}/cmd_vel'.format(ns),
            Twist, queue_size=10)

        rospy.Subscriber(
            '{}/waterplus/navi_result'.format(ns),
            String, self._result_cb)

        # 卡住检测状态
        self._pos          = None
        self._last_move_t  = None
        self._navigating   = False

        rospy.Subscriber(
            '{}/odom'.format(ns),
            Odometry, self._odom_cb)

        threading.Thread(target=self._watchdog, daemon=True).start()

    # ── 里程计回调 ──────────────────────────────────────────────────
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        new_pos = (p.x, p.y)
        if self._pos is not None:
            dist = math.hypot(new_pos[0] - self._pos[0],
                              new_pos[1] - self._pos[1])
            if dist >= STUCK_DIST:
                self._last_move_t = rospy.Time.now()
        else:
            self._last_move_t = rospy.Time.now()
        self._pos = new_pos

    # ── 卡住看门狗 ─────────────────────────────────────────────────
    def _watchdog(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            rate.sleep()
            if not self._navigating or self._last_move_t is None:
                continue
            elapsed = (rospy.Time.now() - self._last_move_t).to_sec()
            if elapsed >= STUCK_TIMEOUT:
                rospy.logwarn('[robot_%d] 卡住 %.0fs，执行脱困',
                              self.robot_id, elapsed)
                self._escape()

    def _escape(self):
        self._navigating = False          # 暂停看门狗计时

        twist = Twist()
        twist.linear.x = ESCAPE_VEL
        end_t = rospy.Time.now() + rospy.Duration(ESCAPE_DURATION)
        rate  = rospy.Rate(20)
        while rospy.Time.now() < end_t and not rospy.is_shutdown():
            self.pub_cmd.publish(twist)
            rate.sleep()
        self.pub_cmd.publish(Twist())     # 停车
        rospy.sleep(0.3)

        self._last_move_t = rospy.Time.now()
        self._navigating  = True
        self._send_current()              # 重发当前目标

    # ── 导航结果回调 ────────────────────────────────────────────────
    def _result_cb(self, msg):
        if msg.data == 'done':
            with self.lock:
                wp = self.route[self.idx]
                rospy.loginfo('[robot_%d] 到达 WP%s ✓', self.robot_id, wp)
                self.idx = (self.idx + 1) % len(self.route)
            self._send_current()
        else:
            rospy.logwarn('[robot_%d] WP%s 导航失败，重试',
                          self.robot_id, self.route[self.idx])
            rospy.sleep(1.0)
            self._send_current()

    # ── 发送当前航点 ────────────────────────────────────────────────
    def _send_current(self):
        wp = self.route[self.idx]
        msg = String()
        msg.data = wp
        self.pub_wp.publish(msg)
        self._navigating  = True
        self._last_move_t = rospy.Time.now()
        rospy.loginfo('[robot_%d] 前往 WP%s (%d/%d)',
                      self.robot_id, wp, self.idx + 1, len(self.route))

    # ── 启动入口 ────────────────────────────────────────────────────
    def start(self):
        delay = ROBOT_START_DELAY.get(self.robot_id, 0)
        rospy.loginfo('[robot_%d] 延迟 %ds 后启动', self.robot_id, delay)
        rospy.sleep(delay + 0.5)
        self._send_current()


def main():
    rospy.init_node('test_four_wp', anonymous=False)

    navigators = [RobotNavigator(rid, route)
                  for rid, route in ROBOT_ROUTES.items()]

    rospy.loginfo('[test_four_wp] 等待导航栈就绪...')
    rospy.sleep(3.0)

    threads = []
    for nav in navigators:
        t = threading.Thread(target=nav.start, daemon=True)
        threads.append(t)
        t.start()

    rospy.loginfo('[test_four_wp] 四车巡航已启动（含卡住自动脱困）')
    rospy.spin()


if __name__ == '__main__':
    main()
