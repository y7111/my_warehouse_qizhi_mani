#!/usr/bin/env python3
"""
robot_agent.py
单个机器人的任务执行状态机。
每个机器人运行一个独立实例，通过 ~robot_name 参数区分。

状态机:
  IDLE → GOING_TO_SHELF → PICKING → GOING_TO_GOAL → DELIVERING → IDLE

导航流程:
  1. 调用 /astar_planner/plan_path 获取 A* 规划路径并发布到 /astar_path 可视化
  2. 按简化航点序列通过 move_base 跟踪路径
"""
import math
import rospy
import actionlib
import tf
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from warehouse_astar.srv import RequestTask, AStarPlan


def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class RobotAgent:
    IDLE           = 'IDLE'
    GOING_TO_SHELF = 'GOING_TO_SHELF'
    PICKING        = 'PICKING'
    GOING_TO_GOAL  = 'GOING_TO_GOAL'
    DELIVERING     = 'DELIVERING'

    PICK_TIME    = 3.0
    DELIVER_TIME = 2.0

    # 启动时先移到离墙安全的初始位置，避免贴墙导航异常
    HOME_POSITIONS = {
        'robot_alpha': (-1.5, 0.0, 0.0),      # 离西墙，朝东
        'robot_beta':  ( 1.5, 0.0, math.pi),  # 离东墙，朝西
    }

    def __init__(self):
        rospy.init_node('robot_agent', anonymous=True)

        self.robot_name   = rospy.get_param('~robot_name', 'robot_alpha')
        self.state        = self.IDLE
        self.current_task = None

        ns = '/' + self.robot_name

        # TF 监听器（用于获取机器人当前位姿）
        self.tf_listener = tf.TransformListener()

        # move_base action client
        self.move_base = actionlib.SimpleActionClient(
            ns + '/move_base', MoveBaseAction)
        rospy.loginfo('[%s] 等待 move_base...', self.robot_name)
        self.move_base.wait_for_server()
        rospy.loginfo('[%s] move_base 已连接', self.robot_name)

        # A* 规划服务
        rospy.loginfo('[%s] 等待 A* 规划服务...', self.robot_name)
        rospy.wait_for_service('/astar_planner/plan_path')
        self.astar_srv = rospy.ServiceProxy('/astar_planner/plan_path', AStarPlan)
        rospy.loginfo('[%s] A* 服务已连接', self.robot_name)

        # 发布配送完成事件
        self.delivery_pub = rospy.Publisher(
            '/warehouse/delivery_complete', String, queue_size=5)

        # 发布当前状态
        self.state_pub = rospy.Publisher(
            '/warehouse/' + self.robot_name + '/state', String, queue_size=5)

        # 仓储任务服务
        rospy.loginfo('[%s] 等待 /warehouse/request_task 服务...', self.robot_name)
        rospy.wait_for_service('/warehouse/request_task')
        self.request_task_srv = rospy.ServiceProxy(
            '/warehouse/request_task', RequestTask)
        rospy.loginfo('[%s] 机器人代理启动完成', self.robot_name)

    # ── 获取当前位姿 ────────────────────────────────────────────────

    def get_current_pose(self):
        """通过 TF 查询机器人在 map 坐标系下的当前位姿"""
        base_frame = self.robot_name + '/base_footprint'
        try:
            self.tf_listener.waitForTransform(
                'map', base_frame, rospy.Time(0), rospy.Duration(1.0))
            trans, rot = self.tf_listener.lookupTransform(
                'map', base_frame, rospy.Time(0))
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp    = rospy.Time.now()
            ps.pose.position.x = trans[0]
            ps.pose.position.y = trans[1]
            ps.pose.orientation.x = rot[0]
            ps.pose.orientation.y = rot[1]
            ps.pose.orientation.z = rot[2]
            ps.pose.orientation.w = rot[3]
            return ps
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            rospy.logwarn('[%s] TF 查询失败: %s', self.robot_name, e)
            return None

    # ── 导航核心 ────────────────────────────────────────────────────

    def _send_move_base_goal(self, pose_stamped):
        """向 move_base 发送单个目标，等待结果，返回是否成功"""
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp    = rospy.Time.now()
        self.move_base.send_goal(goal)
        self.move_base.wait_for_result()
        return self.move_base.get_state() == GoalStatus.SUCCEEDED

    def navigate_to(self, x, y, yaw, label='目标'):
        """
        调用 A* 服务获取规划路径（发布到 /astar_path 供 RViz 可视化），
        然后直接将终点目标发给 move_base 执行导航。
        """
        # A* 路径可视化（不影响导航执行）
        current_pose = self.get_current_pose()
        if current_pose is not None:
            goal_ps = PoseStamped()
            goal_ps.header.frame_id = 'map'
            goal_ps.header.stamp    = rospy.Time.now()
            goal_ps.pose.position.x = x
            goal_ps.pose.position.y = y
            goal_ps.pose.orientation = yaw_to_quaternion(yaw)
            try:
                resp = self.astar_srv(current_pose, goal_ps)
                if resp.success:
                    rospy.loginfo('[%s] A* 规划 %d 个航点 → %s',
                                  self.robot_name, len(resp.path.poses), label)
                else:
                    rospy.logwarn('[%s] A* 规划失败，将由 move_base 自行规划', self.robot_name)
            except rospy.ServiceException as e:
                rospy.logwarn('[%s] A* 服务调用失败: %s', self.robot_name, e)

        # 直接发送终点目标给 move_base（让 move_base 完整执行导航）
        rospy.loginfo('[%s] 前往 %s (%.2f, %.2f)', self.robot_name, label, x, y)
        ps = PoseStamped()
        ps.header.frame_id    = 'map'
        ps.header.stamp       = rospy.Time.now()
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.orientation   = yaw_to_quaternion(yaw)
        ok = self._send_move_base_goal(ps)
        if ok:
            rospy.loginfo('[%s] 到达 %s', self.robot_name, label)
        else:
            rospy.logwarn('[%s] 导航 %s 失败', self.robot_name, label)
        return ok

    # ── 状态机 ──────────────────────────────────────────────────────

    def do_idle(self):
        try:
            resp = self.request_task_srv(self.robot_name)
        except rospy.ServiceException as e:
            rospy.logwarn('[%s] 请求任务失败: %s', self.robot_name, e)
            rospy.sleep(2.0)
            return

        if resp.success:
            self.current_task = resp
            rospy.loginfo('[%s] 接到任务: 货架%d (商品%s) → 目标%s',
                          self.robot_name, resp.shelf_id,
                          resp.item_type, resp.item_type)
            self.state = self.GOING_TO_SHELF
        else:
            rospy.loginfo('[%s] 暂无任务，2秒后重试', self.robot_name)
            rospy.sleep(2.0)

    def do_go_to_shelf(self):
        t = self.current_task
        ok = self.navigate_to(t.pickup_x, t.pickup_y, t.pickup_yaw,
                              f'货架{t.shelf_id}')
        if ok:
            self.state = self.PICKING

    def do_pick(self):
        item = self.current_task.item_type
        rospy.loginfo('[%s] 开始取货（商品%s），模拟 %.0f 秒...',
                      self.robot_name, item, self.PICK_TIME)
        rospy.sleep(self.PICK_TIME)
        rospy.loginfo('[%s] 取货完成（商品%s）', self.robot_name, item)
        self.state = self.GOING_TO_GOAL

    def do_go_to_goal(self):
        t = self.current_task
        ok = self.navigate_to(t.goal_x, t.goal_y, t.goal_yaw,
                              f'目标{t.item_type}')
        if ok:
            self.state = self.DELIVERING

    def do_deliver(self):
        t = self.current_task
        rospy.loginfo('[%s] 开始卸货（商品%s），模拟 %.0f 秒...',
                      self.robot_name, t.item_type, self.DELIVER_TIME)
        rospy.sleep(self.DELIVER_TIME)

        msg = String()
        msg.data = f'{self.robot_name} {t.shelf_id}'
        self.delivery_pub.publish(msg)

        rospy.loginfo('[%s] 配送完成！商品%s 已送达目标%s',
                      self.robot_name, t.item_type, t.item_type)
        self.current_task = None
        self.state = self.IDLE

    # ── 主循环 ──────────────────────────────────────────────────────

    def run(self):
        # 启动时先离开墙壁，移到中间开阔区域
        if self.robot_name in self.HOME_POSITIONS:
            hx, hy, hyaw = self.HOME_POSITIONS[self.robot_name]
            rospy.loginfo('[%s] 初始化：先移到安全起始位置 (%.1f, %.1f)', self.robot_name, hx, hy)
            self.navigate_to(hx, hy, hyaw, '安全起始位置')

        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.state_pub.publish(self.state)

            if   self.state == self.IDLE:
                self.do_idle()
            elif self.state == self.GOING_TO_SHELF:
                self.do_go_to_shelf()
            elif self.state == self.PICKING:
                self.do_pick()
            elif self.state == self.GOING_TO_GOAL:
                self.do_go_to_goal()
            elif self.state == self.DELIVERING:
                self.do_deliver()

            rate.sleep()


if __name__ == '__main__':
    agent = RobotAgent()
    agent.run()
