#!/usr/bin/env python3
"""
robot_agent.py  —  简化单点导航版

每个任务只发两个 move_base 目标：
  1. 取货点（货架旁，让 A* 自动规划路径）
  2. home（放货点，A* 自动规划返回）
"""
import math
import rospy
import actionlib
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Quaternion
from gazebo_msgs.srv import DeleteModel

# 机械臂控制
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arm_ctrl import ArmController

PI = math.pi


def make_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# ──────────────────────────────────────────────────────────────────────
# 任务路线表
# 每段航点 (x, y, yaw)，机器人依次到达每个点
# ──────────────────────────────────────────────────────────────────────
#   yaw 方向约定:  0=朝东  PI=朝西  PI/2=朝北  -PI/2=朝南

TASK_LIST = {
    # pickup_pose：紧贴货架安全边界（货架面 - inflation 0.35 - robot_radius 0.22 = 0.57m 净距）
    # rack_1 南面 y=2.3 → 安全停靠 y=1.73，取 y=1.7
    # rack_2 南面 y=2.8 → 安全停靠 y=2.23，取 y=2.2
    # rack_3 北面 y=-2.8 → 安全停靠 y=-2.23，取 y=-2.2
    # rack_4 北面 y=-2.3 → 安全停靠 y=-1.73，取 y=-1.7
    'robot_alpha': [
        {
            'shelf_id':    1,
            'item':        'A(绿)',
            'cargo_model': 'cargo_A1',
            'pickup_pose': (-3.0,  1.7,  PI/2),   # 紧贴 rack_1 南侧，正面朝北
            'home_pose':   (-5.0,  0.0,  0.0),    # goal_A，朝东放货
        },
        {
            'shelf_id':    3,
            'item':        'A(绿)',
            'cargo_model': 'cargo_A3',
            'pickup_pose': (-1.5, -2.2, -PI/2),   # 紧贴 rack_3 北侧，正面朝南
            'home_pose':   (-5.0,  0.0,  0.0),    # goal_A
        },
    ],
    'robot_beta': [
        {
            'shelf_id':    2,
            'item':        'B(蓝)',
            'cargo_model': 'cargo_B2',
            'pickup_pose': ( 1.5,  2.2,  PI/2),   # 紧贴 rack_2 南侧，正面朝北
            'home_pose':   ( 5.0,  0.0,  PI),     # goal_B，朝西放货
        },
        {
            'shelf_id':    4,
            'item':        'B(蓝)',
            'cargo_model': 'cargo_B4',
            'pickup_pose': ( 3.0, -1.7, -PI/2),   # 紧贴 rack_4 北侧，正面朝南
            'home_pose':   ( 5.0,  0.0,  PI),     # goal_B
        },
    ],
}

# 步骤常量
STEP_READY    = 0
STEP_GO_SHELF = 1
STEP_PICKING  = 2
STEP_GO_GOAL  = 3
STEP_PLACING  = 4
STEP_DONE     = 5
STEP_NAMES    = {0:'READY', 1:'GO_SHELF', 2:'PICKING',
                 3:'GO_GOAL', 4:'PLACING', 5:'DONE'}


class RobotAgent:
    def __init__(self):
        rospy.init_node('robot_agent', anonymous=True)
        self.robot_name = rospy.get_param('~robot_name', 'robot_alpha')
        self.tasks      = TASK_LIST.get(self.robot_name, [])
        self.task_idx   = 0
        self.step       = STEP_READY
        self.delay_cnt  = 0

        # move_base client
        ns = '/' + self.robot_name
        self.mb = actionlib.SimpleActionClient(ns + '/move_base', MoveBaseAction)
        rospy.loginfo('[%s] 等待 move_base...', self.robot_name)
        self.mb.wait_for_server()
        rospy.loginfo('[%s] 就绪，共 %d 个任务', self.robot_name, len(self.tasks))

        # Gazebo 删除模型服务
        rospy.loginfo('[%s] 等待 Gazebo delete_model 服务...', self.robot_name)
        rospy.wait_for_service('/gazebo/delete_model')
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # 机械臂控制器
        self.arm = ArmController(self.robot_name)
        self.arm.fold()   # 启动时收臂

        # 发布者
        self.pickup_pub   = rospy.Publisher('/warehouse/pickup',            String, queue_size=5)
        self.delivery_pub = rospy.Publisher('/warehouse/delivery_complete', String, queue_size=5)
        self.state_pub    = rospy.Publisher(f'/warehouse/{self.robot_name}/state', String, queue_size=5)

    # ── 单点导航 ──────────────────────────────────────────────────────
    def nav_to(self, x: float, y: float, yaw: float, label: str = '') -> bool:
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp    = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = make_quat(yaw)

        rospy.loginfo('[%s] → %s  (%.2f, %.2f)', self.robot_name, label, x, y)
        self.mb.send_goal(goal)
        self.mb.wait_for_result()
        ok = (self.mb.get_state() == GoalStatus.SUCCEEDED)
        rospy.loginfo('[%s] %s %s', self.robot_name, '✓' if ok else '✗', label)
        return ok

    # ── 主循环 ────────────────────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.state_pub.publish(STEP_NAMES.get(self.step, '?'))

            # READY ────────────────────────────────────────────────────
            if self.step == STEP_READY:
                if self.task_idx < len(self.tasks):
                    t = self.tasks[self.task_idx]
                    rospy.loginfo('[%s] ── 开始任务 %d/%d: rack_%d %s ──',
                                  self.robot_name, self.task_idx + 1,
                                  len(self.tasks), t['shelf_id'], t['item'])
                    self.step = STEP_GO_SHELF
                else:
                    self.step = STEP_DONE

            # GO_SHELF ─────────────────────────────────────────────────
            elif self.step == STEP_GO_SHELF:
                t = self.tasks[self.task_idx]
                x, y, yaw = t['pickup_pose']
                if self.nav_to(x, y, yaw, f'rack_{t["shelf_id"]}'):
                    self.delay_cnt = 0
                    self.step = STEP_PICKING
                else:
                    rospy.logwarn('[%s] 导航失败，5秒后重试...', self.robot_name)
                    rospy.sleep(5.0)

            # PICKING ──────────────────────────────────────────────────
            elif self.step == STEP_PICKING:
                if self.delay_cnt == 0:
                    t = self.tasks[self.task_idx]
                    rospy.loginfo('[%s] 开始取货 %s', self.robot_name, t['item'])
                    # ① 伸臂 + 张开手爪 → 夹紧
                    self.arm.pick_sequence()
                    # ② 夹紧瞬间删除 Gazebo 模型（货物被夹走）
                    try:
                        self.delete_model(t['cargo_model'])
                        rospy.loginfo('[%s] Gazebo 已移除 %s', self.robot_name, t['cargo_model'])
                    except Exception as e:
                        rospy.logwarn('[%s] 删除模型失败: %s', self.robot_name, e)
                    # ③ 收臂，进入带货行驶姿态
                    self.arm.pick_end()
                    self.pickup_pub.publish(f'{self.robot_name} {t["shelf_id"]}')
                    rospy.loginfo('[%s] ✓ 取货完成 rack_%d %s',
                                  self.robot_name, t['shelf_id'], t['item'])
                    self.delay_cnt = 1
                if self.delay_cnt == 1:
                    self.delay_cnt = 0
                    self.step = STEP_GO_GOAL

            # GO_GOAL ──────────────────────────────────────────────────
            elif self.step == STEP_GO_GOAL:
                t = self.tasks[self.task_idx]
                x, y, yaw = t['home_pose']
                if self.nav_to(x, y, yaw, f'home_{t["item"][0]}'):
                    self.delay_cnt = 0
                    self.step = STEP_PLACING
                else:
                    rospy.logwarn('[%s] 导航失败，5秒后重试...', self.robot_name)
                    rospy.sleep(5.0)

            # PLACING ──────────────────────────────────────────────────
            elif self.step == STEP_PLACING:
                if self.delay_cnt == 0:
                    t = self.tasks[self.task_idx]
                    rospy.loginfo('[%s] 开始放货 %s', self.robot_name, t['item'])
                    # 伸臂 → 张开手爪放货 → 收臂
                    self.arm.place_sequence()
                    self.delivery_pub.publish(f'{self.robot_name} {t["shelf_id"]}')
                    rospy.loginfo('[%s] ✓ 放货完成 → 下一任务', self.robot_name)
                    self.delay_cnt = 1
                if self.delay_cnt == 1:
                    self.task_idx  += 1
                    self.delay_cnt  = 0
                    self.step       = STEP_READY

            # DONE ─────────────────────────────────────────────────────
            elif self.step == STEP_DONE:
                rospy.loginfo_throttle(10, '[%s] 全部任务完成！', self.robot_name)

            rate.sleep()


if __name__ == '__main__':
    agent = RobotAgent()
    agent.run()
