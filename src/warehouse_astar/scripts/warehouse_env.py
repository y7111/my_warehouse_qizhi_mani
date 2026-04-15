#!/usr/bin/env python3
"""
warehouse_env.py  ─  QMIX 仓储多智能体 Gym 环境
两辆 wpb_mani 机器人，4 个货架，协同取货任务。

地图布局:
  rack_1(-0.7,+0.6)   rack_2(+0.7,+0.6)
  取货点A(-0.7, 0)     取货点B(+0.7, 0)
  rack_3(-0.7,-0.6)   rack_4(+0.7,-0.6)

  alpha 起点(-1.0, 0)  beta 起点(+1.0, 0)

观测 (每个智能体, 9维):
  [自身x, 自身y, 自身yaw,
   rack1已取, rack2已取, rack3已取, rack4已取,
   另一辆x, 另一辆y]

动作 (每个智能体, 离散5):
  0 → 前往取货点A (-0.7, 0.0)
  1 → 前往取货点B ( 0.7, 0.0)
  2 → 朝北取货 (face +y, π/2)
  3 → 朝南取货 (face -y, -π/2)
  4 → 返回起点

奖励 (全局共享):
  每取一个货架 +10
  全部取完     +50
  每步         -0.1
  两车距离<0.45 -20
"""

import math
import threading
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import (SetModelState,
                              SetModelConfiguration,
                              SetModelConfigurationRequest)

# ── 机器人参数 ────────────────────────────────────────────────────────────
INIT_POSES = {
    'robot_alpha': (-1.0,  0.0, 0.0),
    'robot_beta':  ( 1.0,  0.0, math.pi),
}
AGENT_NAMES = ['robot_alpha', 'robot_beta']

# 取货点（两排货架正中间的走廊）
PICKUP_A = (-0.7, 0.0)
PICKUP_B = ( 0.7, 0.0)

# 手臂姿态
ARM_FOLD   = [0.0, -2.0, 0.0, 0.0]
ARM_EXTEND = [0.0,  0.0, 0.0, 0.0]

# 控制参数
TURN_SPEED = 0.4
MOVE_SPEED = 0.25
TURN_TOL   = 0.08
ARRIVE_TOL = 0.30
CONTROL_HZ = 10
PICK_ZONE  = 0.25   # 距取货点多近才算有效取货（避免从起点直接取货）

# 动作编码
ACT_GOTO_A     = 0
ACT_GOTO_B     = 1
ACT_PICK_NORTH = 2
ACT_PICK_SOUTH = 3
ACT_RETURN     = 4
N_ACTIONS      = 5

OBS_DIM = 9


class WarehouseEnv:
    """ROS/Gazebo 双机器人仓储环境，兼容 Gym 风格接口（无需安装 gym 包）。"""

    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('warehouse_env', anonymous=True)

        self._tf = tf.TransformListener()
        rospy.sleep(1.0)

        self.n_agents      = len(AGENT_NAMES)
        self.n_actions     = N_ACTIONS
        self.obs_dim       = OBS_DIM

        # cmd_vel 发布器
        self._cmd = {
            name: rospy.Publisher(f'/{name}/cmd_vel', Twist, queue_size=1)
            for name in AGENT_NAMES
        }

        self._shelf_picked = [False] * 4
        self._step_count   = 0
        self.max_steps     = 50

    # ═══════════════════════════════════════════════════════════════════
    #  Gym 核心接口
    # ═══════════════════════════════════════════════════════════════════

    def reset(self):
        """重置环境，返回初始观测列表 [obs_alpha, obs_beta]。"""
        self._shelf_picked = [False] * 4
        self._step_count   = 0


        for name, (x, y, yaw) in INIT_POSES.items():
            self._reset_model(name, x, y, yaw)
        rospy.sleep(1.0)   # 等 Gazebo 物理稳定，避免传送后翻车

        for name, (x, y, yaw) in INIT_POSES.items():
            self._publish_initial_pose(name, x, y, yaw)

        for name in AGENT_NAMES:
            self._set_arm(name, ARM_FOLD)

        rospy.sleep(1.0)
        rospy.loginfo('[Env] reset 完成')
        return self._get_obs()

    def step(self, actions):
        """
        执行一步。
        actions : list[int], 长度 = n_agents
        返回    : obs_list, reward_list, done, info
        """
        self._step_count += 1
        prev_picked = self._shelf_picked[:]

        # 两辆车并行执行动作
        threads = [
            threading.Thread(target=self._execute_action, args=(name, act))
            for name, act in zip(AGENT_NAMES, actions)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        # 翻车检测：原地扶起并扣分，不结束 episode（避免碰撞震动误触发截断）
        flipped = [n for n in AGENT_NAMES if self._is_flipped(n)]
        if flipped:
            rospy.logwarn('[Env] 翻车 %s，原地扶起继续', flipped)
            for n in flipped:
                x, y, yaw = INIT_POSES[n]
                self._reset_model(n, x, y, yaw)
            rospy.sleep(0.5)

        obs    = self._get_obs()
        reward = self._compute_reward(prev_picked)
        if flipped:
            reward -= 30.0   # 翻车扣分但不结束
        done   = all(self._shelf_picked) or self._step_count >= self.max_steps
        info   = {'shelf_picked': self._shelf_picked[:],
                  'step': self._step_count,
                  'flipped': flipped if flipped else []}

        return obs, [reward] * self.n_agents, done, info

    def render(self, mode='human'):
        picked = sum(self._shelf_picked)
        status = ' '.join(
            f'rack{i+1}:{"✓" if s else "✗"}'
            for i, s in enumerate(self._shelf_picked)
        )
        print(f'[Step {self._step_count:3d}] 已取 {picked}/4 | {status}')

    def close(self):
        for pub in self._cmd.values():
            pub.publish(Twist())

    # ═══════════════════════════════════════════════════════════════════
    #  动作执行
    # ═══════════════════════════════════════════════════════════════════

    def _execute_action(self, name, action):
        x0, y0, _ = INIT_POSES[name]
        if   action == ACT_GOTO_A:     self._go_to(name, *PICKUP_A, '取货点A')
        elif action == ACT_GOTO_B:     self._go_to(name, *PICKUP_B, '取货点B')
        elif action == ACT_PICK_NORTH: self._pick(name,  math.pi / 2)
        elif action == ACT_PICK_SOUTH: self._pick(name, -math.pi / 2)
        elif action == ACT_RETURN:     self._go_to(name, x0, y0, '起点')

    def _pick(self, name, face_yaw):
        """收臂 → 转向 → 伸臂 → 登记货架 → 收臂"""
        pose = self._get_pose(name)
        if pose is None:
            return

        x, y, _ = pose
        near_A = math.hypot(x - PICKUP_A[0], y - PICKUP_A[1]) < PICK_ZONE
        near_B = math.hypot(x - PICKUP_B[0], y - PICKUP_B[1]) < PICK_ZONE

        if not (near_A or near_B):
            rospy.logwarn('[%s] 不在取货区 (%.2f, %.2f)，动作无效', name, x, y)
            return

        self._set_arm(name, ARM_FOLD)
        rospy.sleep(0.5)
        self._turn_to(name, face_yaw)
        self._set_arm(name, ARM_EXTEND)
        rospy.sleep(1.5)

        # 登记：哪个区域 × 朝哪个方向
        if near_A:
            idx = 0 if face_yaw > 0 else 2   # rack1 or rack3
        else:
            idx = 1 if face_yaw > 0 else 3   # rack2 or rack4

        self._shelf_picked[idx] = True
        rospy.loginfo('[%s] 取到 rack%d，状态: %s', name, idx + 1, self._shelf_picked)

        self._set_arm(name, ARM_FOLD)
        rospy.sleep(0.3)

    # ═══════════════════════════════════════════════════════════════════
    #  奖励计算
    # ═══════════════════════════════════════════════════════════════════

    def _compute_reward(self, prev_picked):
        reward    = -0.1
        new_picks = sum(self._shelf_picked) - sum(prev_picked)
        reward   += new_picks * 10.0
        if all(self._shelf_picked):
            reward += 50.0

        poses = [self._get_pose(n) for n in AGENT_NAMES]

        # 两车互相碰撞惩罚
        if all(p is not None for p in poses):
            dist = math.hypot(poses[0][0] - poses[1][0],
                              poses[0][1] - poses[1][1])
            if dist < 0.45:
                reward -= 10.0
                rospy.logwarn('[Env] 两车碰撞！距离 %.2f m', dist)

        # 撞墙检测：贴墙就传送回起点，防止卡死
        WALL_X =  2.4
        WALL_Y =  1.5
        MARGIN =  0.35
        for i, name in enumerate(AGENT_NAMES):
            p = poses[i]
            if p is None:
                continue
            x, y, _ = p
            if abs(x) > WALL_X - MARGIN or abs(y) > WALL_Y - MARGIN:
                reward -= 10.0
                rospy.logwarn('[Env] %s 撞墙 (%.2f, %.2f)，传送回起点', name, x, y)
                ix, iy, iyaw = INIT_POSES[name]
                self._reset_model(name, ix, iy, iyaw)

        return reward

    # ═══════════════════════════════════════════════════════════════════
    #  观测
    # ═══════════════════════════════════════════════════════════════════

    def _get_obs(self):
        poses = {}
        for name in AGENT_NAMES:
            p = self._get_pose(name)
            poses[name] = p if p is not None else (0.0, 0.0, 0.0)

        shelf = [float(s) for s in self._shelf_picked]
        obs   = []
        for i, name in enumerate(AGENT_NAMES):
            other     = AGENT_NAMES[1 - i]
            x, y, yaw = poses[name]
            ox, oy, _ = poses[other]
            obs.append(np.array([x, y, yaw] + shelf + [ox, oy], dtype=np.float32))
        return obs

    # ═══════════════════════════════════════════════════════════════════
    #  底层控制
    # ═══════════════════════════════════════════════════════════════════

    def _get_pose(self, name, timeout=3.0):
        target = name + '/base_footprint'
        try:
            self._tf.waitForTransform('map', target, rospy.Time(0),
                                      rospy.Duration(timeout))
            (t, r) = self._tf.lookupTransform('map', target, rospy.Time(0))
            return t[0], t[1], 2.0 * math.atan2(r[2], r[3])
        except Exception:
            return None

    def _is_flipped(self, name) -> bool:
        """检测机器人是否翻倒：z 高度异常或 roll/pitch 过大。"""
        target = name + '/base_footprint'
        try:
            self._tf.waitForTransform('map', target, rospy.Time(0),
                                      rospy.Duration(1.0))
            (t, r) = self._tf.lookupTransform('map', target, rospy.Time(0))
            # z > 0.30m 说明机器人明显离地（翻倒后底盘抬高）
            if t[2] > 0.30:
                return True
            # roll/pitch 超过 45° 才认为翻车
            sinp = 2.0 * (r[3] * r[0] - r[1] * r[2])
            sinp = max(-1.0, min(1.0, sinp))
            pitch = math.asin(sinp)
            sinr  = 2.0 * (r[3] * r[1] + r[0] * r[2])
            cosr  = 1.0 - 2.0 * (r[1] * r[1] + r[2] * r[2])
            roll  = math.atan2(sinr, cosr)
            if abs(roll) > 0.785 or abs(pitch) > 0.785:   # 45°
                return True
            return False
        except Exception:
            return False

    def _go_to(self, name, gx, gy, label, timeout=12.0):
        rate     = rospy.Rate(CONTROL_HZ)
        deadline = rospy.Time.now() + rospy.Duration(timeout)

        # 卡住检测：记录上次位置，连续2秒没动就放弃
        stuck_pos   = None
        stuck_timer = rospy.Time.now() + rospy.Duration(2.0)

        rospy.loginfo('[%s] -> %s (%.2f, %.2f)', name, label, gx, gy)

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._cmd[name].publish(Twist())
                rospy.logwarn('[%s] %s timeout', name, label)
                return False

            pose = self._get_pose(name)
            if pose is None:
                rate.sleep(); continue

            cx, cy, cyaw = pose

            # 卡住检测
            if stuck_pos is None:
                stuck_pos = (cx, cy)
            elif rospy.Time.now() > stuck_timer:
                moved = math.hypot(cx - stuck_pos[0], cy - stuck_pos[1])
                if moved < 0.05:   # 2秒内移动不足5cm → 卡住
                    self._cmd[name].publish(Twist())
                    rospy.logwarn('[%s] stuck near (%.2f, %.2f), abort', name, cx, cy)
                    return False
                stuck_pos  = (cx, cy)
                stuck_timer = rospy.Time.now() + rospy.Duration(2.0)

            dist = math.hypot(gx - cx, gy - cy)
            if dist < ARRIVE_TOL:
                self._cmd[name].publish(Twist())
                rospy.loginfo('[%s] arrived %s', name, label)
                return True

            desired_yaw = math.atan2(gy - cy, gx - cx)
            yaw_err     = self._angle_diff(desired_yaw, cyaw)
            cmd = Twist()
            if abs(yaw_err) > TURN_TOL:
                cmd.angular.z = TURN_SPEED * (1.0 if yaw_err > 0 else -1.0)
            else:
                cmd.linear.x  = max(0.1, min(MOVE_SPEED, MOVE_SPEED * dist / 0.6))
                cmd.angular.z = 0.6 * yaw_err
            self._cmd[name].publish(cmd)
            rate.sleep()

        self._cmd[name].publish(Twist())
        return False

    def _turn_to(self, name, target_yaw):
        rate     = rospy.Rate(CONTROL_HZ)
        deadline = rospy.Time.now() + rospy.Duration(15.0)
        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline: break
            pose = self._get_pose(name)
            if pose is None:
                rate.sleep(); continue
            yaw_err = self._angle_diff(target_yaw, pose[2])
            if abs(yaw_err) < 0.05: break
            cmd = Twist()
            cmd.angular.z = TURN_SPEED * (1.0 if yaw_err > 0 else -1.0)
            self._cmd[name].publish(cmd)
            rate.sleep()
        self._cmd[name].publish(Twist())

    def _set_arm(self, name, positions):
        try:
            srv = rospy.ServiceProxy('/gazebo/set_model_configuration',
                                     SetModelConfiguration)
            req = SetModelConfigurationRequest()
            req.model_name      = name
            req.urdf_param_name = 'robot_description'
            req.joint_names     = ['joint1', 'joint2', 'joint3', 'joint4']
            req.joint_positions = positions
            srv(req)
        except Exception as e:
            rospy.logwarn('[%s] 手臂指令失败: %s', name, e)

    def _reset_model(self, name, x, y, yaw):
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state = ModelState()
            state.model_name             = name
            state.pose.position.x        = x
            state.pose.position.y        = y
            state.pose.position.z        = 0.0        # 确保落在地面上
            state.pose.orientation.x     = 0.0        # 清零 roll/pitch，防止翻车
            state.pose.orientation.y     = 0.0
            state.pose.orientation.z     = math.sin(yaw / 2)
            state.pose.orientation.w     = math.cos(yaw / 2)
            # 清零所有残余速度，防止带动量传送
            state.twist.linear.x         = 0.0
            state.twist.linear.y         = 0.0
            state.twist.linear.z         = 0.0
            state.twist.angular.x        = 0.0
            state.twist.angular.y        = 0.0
            state.twist.angular.z        = 0.0
            state.reference_frame        = 'world'
            set_state(state)
        except Exception as e:
            rospy.logwarn('[%s] 位置重置失败: %s', name, e)

    def _publish_initial_pose(self, name, x, y, yaw):
        pub = rospy.Publisher(f'/{name}/initialpose',
                              PoseWithCovarianceStamped, queue_size=1, latch=True)
        rospy.sleep(0.3)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id         = 'map'
        msg.header.stamp            = rospy.Time.now()
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = cov[35] = 1e-4
        msg.pose.covariance = cov
        pub.publish(msg)

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d
