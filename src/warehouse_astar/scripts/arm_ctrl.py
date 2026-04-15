#!/usr/bin/env python3
"""
arm_ctrl.py  ─  机械臂控制模块

关节角度参考 wpb_mani_tutorials：
  收臂（行驶）: j2=-1.57  j3=1.35  j4=0.24   gripper=0
  取货伸臂:     j2= 1.57  j3=-0.67 j4=-0.9   gripper=open
  放货伸臂:     j2= 1.57  j3=-0.94 j4=-0.59  gripper=open→close

话题（以 robot_alpha 为例）：
  /robot_alpha/joint1_position_controller/command
  /robot_alpha/joint2_position_controller/command
  /robot_alpha/joint3_position_controller/command
  /robot_alpha/joint4_position_controller/command
  /robot_alpha/gripper_position_controller/command
"""

import rospy
from std_msgs.msg import Float64

GRIPPER_OPEN  =  0.019   # 手爪张开（URDF upper limit）
GRIPPER_CLOSE = -0.010   # 手爪夹紧（URDF lower limit）


class ArmController:
    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        ns = '/' + robot_name
        self._j1  = rospy.Publisher(f'{ns}/joint1_position_controller/command', Float64, queue_size=1)
        self._j2  = rospy.Publisher(f'{ns}/joint2_position_controller/command', Float64, queue_size=1)
        self._j3  = rospy.Publisher(f'{ns}/joint3_position_controller/command', Float64, queue_size=1)
        self._j4  = rospy.Publisher(f'{ns}/joint4_position_controller/command', Float64, queue_size=1)
        self._grp = rospy.Publisher(f'{ns}/gripper_position_controller/command', Float64, queue_size=1)
        rospy.sleep(0.5)

    def _send(self, j1, j2, j3, j4, gripper, wait=2.0):
        self._j1.publish(Float64(data=j1))
        self._j2.publish(Float64(data=j2))
        self._j3.publish(Float64(data=j3))
        self._j4.publish(Float64(data=j4))
        self._grp.publish(Float64(data=gripper))
        if wait > 0:
            rospy.sleep(wait)

    # ── 基本姿态 ──────────────────────────────────────────────────────

    def fold(self):
        """收臂向后折叠，行驶安全姿态（来自 mobile_manipulation.cpp）。"""
        rospy.loginfo('[%s arm] fold', self.robot_name)
        self._send(0, -1.57, 1.35, 0.24, GRIPPER_CLOSE, wait=2.0)

    def reach_pick(self):
        """
        取货伸臂：向正前方水平伸出到货架高度（来自 joint_control.cpp case1）。
        手爪张开，准备夹取。
        """
        rospy.loginfo('[%s arm] reach_pick', self.robot_name)
        self._send(0, 1.57, -0.67, -0.9, GRIPPER_OPEN, wait=2.0)

    def close_gripper(self):
        """夹紧手爪（模拟夹住货物）。"""
        rospy.loginfo('[%s arm] grasp', self.robot_name)
        self._grp.publish(Float64(data=GRIPPER_CLOSE))
        rospy.sleep(1.0)

    def reach_place(self):
        """
        放货伸臂：手持货物后水平伸出（来自 mobile_manipulation.cpp STEP_STRETCH_OUT）。
        """
        rospy.loginfo('[%s arm] reach_place', self.robot_name)
        self._send(0, 1.57, -0.94, -0.59, GRIPPER_CLOSE, wait=2.0)

    def open_gripper(self):
        """张开手爪（放下货物）。"""
        rospy.loginfo('[%s arm] release', self.robot_name)
        self._grp.publish(Float64(data=GRIPPER_OPEN))
        rospy.sleep(1.0)

    # ── 完整动作序列 ──────────────────────────────────────────────────

    def pick_sequence(self):
        """
        完整取货序列（robot_agent 在此之后删除 Gazebo 模型）：
          ① reach_pick：伸臂张开手爪（到货架高度）
          ② close_gripper：夹紧
          返回时保持伸展夹紧状态，调用方删模型后再调 pick_end()
        """
        self.reach_pick()
        self.close_gripper()

    def pick_end(self):
        """取货后收臂（带货行驶）。"""
        self.fold()

    def place_sequence(self):
        """
        完整放货序列：
          ① reach_place：伸臂（持货姿态）
          ② open_gripper：张开放货
          ③ fold：收臂
        """
        self.reach_place()
        self.open_gripper()
        self.fold()