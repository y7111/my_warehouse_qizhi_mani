#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_demo.py — 单车完整抓取演示(仓库 world)

流程：生成货架台+货箱 → 导航到货架点 → 贴近 → 伸手 → 合爪吸附 → 抬臂
      → 导航到投递区 → 下探 → 松爪 → 货箱平滑落到地面
机械臂用 /wpb_mani/joint_ctrl 直接控制；货箱用 set_model_state 吸附在 end_effector
(纯视觉无碰撞)。货架台是实体静态 box，货箱摆在台面上不再悬空。
"""

import math
import threading
import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from geometry_msgs.msg import Pose, Point, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# ── 可调参数 ──────────────────────────────────────────────────────────
EE_FWD     = 0.0443                 # end_effector 相对手指连杆中点前向偏移
GRAB_WP    = '1'                    # 取货航点(货架0 站立点 WP1，机器人停 (-1.5,5.5) 朝北)
DELIVER_WP = '11'                   # 投递航点(投递区A WP11)

SHELF_XY   = (-1.5, 6.5)            # 货架0 真实坐标(世界里已有 shelf_0 实体)
SHELF_TOP  = 0.22                   # 已有货架顶面高(shelf_0 尺寸0.4³, pose z=0.11 → 顶=0.22)
BOX_SIZE   = 0.045
BOX_Z      = SHELF_TOP + BOX_SIZE / 2   # 货箱摆在已有货架顶面上
REACH_DIST = 0.57                   # 机器人中心到货箱、机械臂正好够到的距离
GRIPPER_LAT = 0.025                 # 夹爪中线相对车体中心的横向偏置(朝北时夹爪偏西)→对中补偿

# 关节姿势 [joint1,joint2,joint3,joint4]
POSE_REACH = [0.0, 1.10, -0.55, -0.62]  # 伸手抓取态(ee≈台面高度)
POSE_CARRY = [0.0, 0.20, -0.20, -0.40]  # 抬起搬运态
POSE_DROP  = [0.0, 1.20, -0.62, -0.66]  # 投递区下探到最低态
GRIP_OPEN  = 0.07
GRIP_CLOSE = 0.05
# ─────────────────────────────────────────────────────────────────────

_ee = {}
_robot = {}
_attached = {'on': False}


def _rot_x(q, d):
    x, y, z, w = q
    return (d * (1 - 2 * (y * y + z * z)), d * 2 * (x * y + w * z), d * 2 * (x * z - w * y))


def _link_cb(m):
    idx = {n: i for i, n in enumerate(m.name)}
    if 'wpb_mani::gripper_link' in idx and 'wpb_mani::gripper_link_sub' in idx:
        a = m.pose[idx['wpb_mani::gripper_link']]
        b = m.pose[idx['wpb_mani::gripper_link_sub']].position
        mid = ((a.position.x + b.x) / 2, (a.position.y + b.y) / 2, (a.position.z + b.z) / 2)
        q = (a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w)
        off = _rot_x(q, EE_FWD)
        _ee['p'] = (mid[0] + off[0], mid[1] + off[1], mid[2] + off[2])


def _model_cb(m):
    if 'wpb_mani' in m.name:
        p = m.pose[m.name.index('wpb_mani')].position
        _robot['p'] = (p.x, p.y)


class GraspDemo:
    def __init__(self):
        self.jpub = rospy.Publisher('/wpb_mani/joint_ctrl', JointState, queue_size=1)
        self.spub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.vpub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.wp_pub = rospy.Publisher('/waterplus/navi_waypoint', String, queue_size=1)
        rospy.Subscriber('/gazebo/link_states', LinkStates, _link_cb)
        rospy.Subscriber('/gazebo/model_states', ModelStates, _model_cb)
        self._navi_done = threading.Event()
        rospy.Subscriber('/waterplus/navi_result', String, self._navi_cb)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        threading.Thread(target=self._attach_loop, daemon=True).start()

    # ── 机械臂/夹爪 ──
    def arm(self, pose, vel=0.4):
        m = JointState(); m.name = ['joint1', 'joint2', 'joint3', 'joint4']
        m.position = list(pose); m.velocity = [vel] * 4
        self.jpub.publish(m)

    def gripper(self, val):
        m = JointState(); m.name = ['gripper']; m.position = [val]; m.velocity = [1.0]
        self.jpub.publish(m)

    # ── 导航 ──
    def _navi_cb(self, msg):
        if msg.data == 'done':
            self._navi_done.set()

    def goto(self, wp, timeout=60.0):
        rospy.loginfo('[grasp] 导航 → 航点 %s', wp)
        self._navi_done.clear(); rospy.sleep(0.3)
        self.wp_pub.publish(String(data=wp))
        ok = self._navi_done.wait(timeout)
        rospy.loginfo('[grasp] 航点 %s %s', wp, '到达' if ok else '超时')
        return ok

    def drive(self, vx, dur):
        """以 vx(m/s) 前/后行 dur 秒。"""
        t = Twist(); t.linear.x = vx
        end = rospy.Time.now() + rospy.Duration(dur); r = rospy.Rate(20)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            self.vpub.publish(t); r.sleep()
        self.vpub.publish(Twist())

    def approach_box(self, timeout=8.0):
        """向前贴近货箱，直到机器人中心距货箱 ≈ REACH_DIST。"""
        rospy.loginfo('[grasp] 贴近货架...')
        t = Twist(); r = rospy.Rate(20)
        end = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            if 'p' in _robot:
                rx, ry = _robot['p']
                d = math.hypot(SHELF_XY[0] - rx, SHELF_XY[1] - ry)
                if d <= REACH_DIST:
                    break
                t.linear.x = min(0.12, 0.5 * (d - REACH_DIST))
                self.vpub.publish(t)
            r.sleep()
        self.vpub.publish(Twist())
        rospy.loginfo('[grasp] 贴近完成')

    def strafe_align(self, box_x, timeout=6.0):
        """横向(麦轮 linear.y)对中：用【夹爪实测世界坐标】闭环，把夹爪 x 对到货箱 x。
        朝北(yaw≈90°)时 base +y → 世界 -x。误差=夹爪x-货箱x，>0 偏右(东)→向西(linear.y正)。"""
        rospy.loginfo('[grasp] 横向对中(夹爪实测闭环)...')
        t = Twist(); r = rospy.Rate(20)
        end = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            if 'p' in _ee:
                err = _ee['p'][0] - box_x
                if abs(err) < 0.008:
                    break
                t.linear.y = max(-0.06, min(0.06, 1.2 * err))
                self.vpub.publish(t)
            r.sleep()
        self.vpub.publish(Twist())
        rospy.loginfo('[grasp] 对中完成 (残差 %.3f m)',
                      (_ee['p'][0] - box_x) if 'p' in _ee else -1)

    # ── 货箱/货架台 ──
    def _box_sdf(self):
        s = BOX_SIZE
        return ('<?xml version="1.0"?><sdf version="1.6"><model name="goods">'
                '<static>false</static><link name="l"><gravity>0</gravity>'
                f'<visual name="v"><geometry><box><size>{s} {s} {s}</size></box></geometry>'
                '<material><ambient>0.1 0.8 0.1 1</ambient><diffuse>0.1 0.8 0.1 1</diffuse>'
                '</material></visual></link></model></sdf>')

    def spawn_scene(self):
        # 世界里已有 shelf_0 实体货架，只把货箱摆到它顶面；并清掉上一版误建的 shelf0
        for name in ('goods', 'shelf0'):
            try:
                self.delete(name); rospy.sleep(0.2)
            except Exception:
                pass
        self.spawn('goods', self._box_sdf(), '',
                   Pose(position=Point(SHELF_XY[0], SHELF_XY[1], BOX_Z)), 'world')
        rospy.loginfo('[grasp] 货箱已摆到已有货架 shelf_0 顶面')

    def _attach_loop(self):
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            if _attached['on'] and 'p' in _ee:
                self._set_box(_ee['p'])
            r.sleep()

    def _set_box(self, xyz):
        ms = ModelState(); ms.model_name = 'goods'; ms.reference_frame = 'world'
        ms.pose.position.x, ms.pose.position.y, ms.pose.position.z = xyz
        ms.pose.orientation.w = 1.0
        self.spub.publish(ms)

    def smooth_drop(self):
        """松爪后让货箱从当前(ee)高度平滑直直落到地面。"""
        if 'p' not in _ee:
            return
        x, y, z = _ee['p']
        ground = BOX_SIZE / 2
        steps = 25; r = rospy.Rate(40)
        for i in range(steps + 1):
            zz = z + (ground - z) * (i / steps)
            self._set_box((x, y, zz))
            r.sleep()
        rospy.loginfo('[grasp] 货箱已平稳落地')

    # ── 主流程 ──
    def run(self):
        rospy.loginfo('[grasp] === 单车抓取演示开始 ===')
        self.gripper(GRIP_OPEN); self.arm(POSE_CARRY); rospy.sleep(2.0)
        self.spawn_scene()

        self.goto(GRAB_WP); rospy.sleep(1.0)
        self.approach_box()
        self.strafe_align(SHELF_XY[0])   # 横向对中(夹爪x对货箱x),消除偏右

        rospy.loginfo('[grasp] 伸手准备抓取')
        self.gripper(GRIP_OPEN); self.arm(POSE_REACH); rospy.sleep(3.0)

        rospy.loginfo('[grasp] 合爪吸附货箱')
        _attached['on'] = True; self.gripper(GRIP_CLOSE); rospy.sleep(1.5)

        rospy.loginfo('[grasp] 抬臂'); self.arm(POSE_CARRY); rospy.sleep(2.5)
        self.drive(-0.12, 1.5)            # 后退离开货架台再走

        self.goto(DELIVER_WP); rospy.sleep(1.0)
        self.drive(-0.12, 4.0)            # 倒车退离绿色投递点，别整车压在上面；机械臂前伸把货箱放到绿点上

        rospy.loginfo('[grasp] 投递区下探放下')
        self.arm(POSE_DROP); rospy.sleep(3.0)
        self.gripper(GRIP_OPEN); rospy.sleep(0.8)
        _attached['on'] = False
        self.smooth_drop()                # 平滑直落，不再闪现
        rospy.sleep(0.3); self.arm(POSE_CARRY)
        rospy.loginfo('[grasp] === 完成：取-搬-投 一轮 ===')


if __name__ == '__main__':
    rospy.init_node('grasp_demo')
    rospy.sleep(1.0)
    GraspDemo().run()
    rospy.spin()
