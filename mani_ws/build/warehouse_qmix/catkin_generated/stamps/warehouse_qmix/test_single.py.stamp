#!/usr/bin/env python3
"""
单车单货架单次取货-投递测试节点（简化版）

流程（不依赖视觉检测，避免 grab_box 节点驱动机器人漂移）：
  1. 发布 AMCL 初始位姿 + spawn 货箱
  2. 导航到货架0接近点
  3. 展臂 → 关爪 → 删除 Gazebo 货箱（模拟拾取）→ 收臂
  4. 导航到投递区A
  5. 展臂 → 开爪 → 收臂
  6. 完成

参照：wpb_mani_mobile_manipulation.cpp 关节序列
"""
import threading
import math
import rospy
import actionlib

from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel, DeleteModelRequest

# ── 坐标常量（每格 1.0m，世界范围 20m×16m）──────────────────────────
ROBOT_START_X   =  8.0
ROBOT_START_Y   = -15.0
ROBOT_START_YAW = math.pi / 2      # 面朝北(+y)

# 货架0：grid(1,8) → 物理中心(8.5, -1.5)，顶面 z=0.6
SHELF0_X, SHELF0_Y = 8.5, -1.5
CARGO_Z = 0.675                    # 货箱中心高度

# 接近点：货架正南 1.0m，面朝北（move_base 可达的最近安全距离）
APPROACH_X = SHELF0_X
APPROACH_Y = SHELF0_Y - 1.0       # = -2.5

# 投递区A：grid(13,7) → 物理中心(7.5, -13.5)
DELIVERY_X, DELIVERY_Y = 7.5, -13.5
# 投递接近点：正南 1.0m
DELIVERY_APPROACH_X = DELIVERY_X
DELIVERY_APPROACH_Y = DELIVERY_Y - 1.0   # = -14.5

# ── 发布器 ────────────────────────────────────────────────────────────
mani_ctrl_pub = None

mani_msg    = JointState()
gripper_msg = JointState()


def _make_quat(yaw):
    return Quaternion(x=0.0, y=0.0,
                      z=math.sin(yaw / 2),
                      w=math.cos(yaw / 2))


# ── Gazebo 货箱管理 ────────────────────────────────────────────────────

def _cargo_sdf():
    s = 0.15
    m = 0.05 / 6 * s * s
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="cargo_test">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>{m:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{m:.6f}</iyy><iyz>0</iyz>
          <izz>{m:.6f}</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
        <material>
          <ambient>0.2 0.9 0.2 1</ambient>
          <diffuse>0.2 0.9 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def spawn_cargo():
    rospy.loginfo('[test] 等待 Gazebo spawn 服务...')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    req = SpawnModelRequest()
    req.model_name      = 'cargo_test'
    req.model_xml       = _cargo_sdf()
    req.robot_namespace = ''
    req.reference_frame = 'world'
    req.initial_pose    = Pose(
        position=Point(x=SHELF0_X, y=SHELF0_Y, z=CARGO_Z),
        orientation=Quaternion(w=1.0)
    )
    resp = spawn(req)
    if resp.success:
        rospy.loginfo('[test] 货箱已生成在货架0 (%.1f, %.1f)', SHELF0_X, SHELF0_Y)
    else:
        rospy.logwarn('[test] 货箱生成失败: %s', resp.status_message)


def delete_cargo():
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=3.0)
        delete = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp = delete(DeleteModelRequest(model_name='cargo_test'))
        if resp.success:
            rospy.loginfo('[test] 货箱已从 Gazebo 删除（拾取完成）')
    except Exception as e:
        rospy.logwarn('[test] 删除货箱异常: %s', e)


# ── 机械臂控制 ────────────────────────────────────────────────────────

def _pub_arm(j1, j2, j3, j4, wait=2.5):
    mani_msg.position[0] = j1
    mani_msg.position[1] = j2
    mani_msg.position[2] = j3
    mani_msg.position[3] = j4
    mani_ctrl_pub.publish(mani_msg)
    rospy.sleep(wait)


def _pub_gripper(opening, wait=1.0):
    gripper_msg.position[0] = opening
    mani_ctrl_pub.publish(gripper_msg)
    rospy.sleep(wait)


def arm_reach_shelf():
    """伸臂朝货架（面朝北，货架在前方1m处）。"""
    rospy.loginfo('[test] 展臂→货架方向')
    _pub_gripper(0.07)         # 先张开手爪
    _pub_arm(0.0, 1.0, -0.6, -0.3, wait=3.0)   # 伸向前方偏下


def arm_carry():
    """收臂（携带姿态）。"""
    rospy.loginfo('[test] 收臂（携带）')
    _pub_arm(0.0, 0.0, 0.0, 0.0, wait=2.0)


def arm_stretch_delivery():
    """展臂投递（参照 mobile_manipulation.cpp STRETCH_OUT）。"""
    rospy.loginfo('[test] 展臂→投递方向')
    _pub_arm(0.0, 1.57, -0.94, -0.59, wait=3.0)


def arm_fold_back():
    """投递后收臂（参照 mobile_manipulation.cpp DROP_BOX）。"""
    rospy.loginfo('[test] 收臂（复位）')
    _pub_arm(0.0, -1.57, 1.35, 0.24, wait=2.5)


# ── 导航 ──────────────────────────────────────────────────────────────

def navigate_to(x, y, yaw, label=''):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id    = 'map'
    goal.target_pose.header.stamp       = rospy.Time.now()
    goal.target_pose.pose.position      = Point(x=x, y=y, z=0.0)
    goal.target_pose.pose.orientation   = _make_quat(yaw)

    rospy.loginfo('[test] 导航→%s (%.1f, %.1f)', label, x, y)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server(timeout=rospy.Duration(10.0))
    client.send_goal(goal)
    client.wait_for_result()

    state = client.get_state()
    ok = (state == actionlib.GoalStatus.SUCCEEDED)
    if ok:
        rospy.loginfo('[test] 到达：%s', label)
    else:
        rospy.logwarn('[test] 导航状态=%d，继续流程', state)
    return ok


# ── AMCL 初始位姿 ─────────────────────────────────────────────────────

def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped,
                          queue_size=1, latch=True)
    rospy.sleep(1.5)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id         = 'map'
    msg.header.stamp            = rospy.Time.now()
    msg.pose.pose.position      = Point(x=ROBOT_START_X, y=ROBOT_START_Y, z=0.0)
    msg.pose.pose.orientation   = _make_quat(ROBOT_START_YAW)
    msg.pose.covariance[0]  = 0.25
    msg.pose.covariance[7]  = 0.25
    msg.pose.covariance[35] = 0.07
    pub.publish(msg)
    rospy.loginfo('[test] AMCL 初始位姿已设置 (%.1f, %.1f)', ROBOT_START_X, ROBOT_START_Y)
    rospy.sleep(1.0)


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    global mani_ctrl_pub

    rospy.init_node('test_single', anonymous=False)

    mani_ctrl_pub = rospy.Publisher('/wpb_mani/joint_ctrl', JointState, queue_size=5)

    # 初始化 JointState 消息
    mani_msg.name     = ['joint1', 'joint2', 'joint3', 'joint4']
    mani_msg.position = [0.0, 0.0, 0.0, 0.0]
    mani_msg.velocity = [10.0, 10.0, 10.0, 10.0]

    gripper_msg.name     = ['gripper']
    gripper_msg.position = [0.0]
    gripper_msg.velocity = [10.0]

    # 后台 spin（保持回调正常运行）
    spin_t = threading.Thread(target=rospy.spin, daemon=True)
    spin_t.start()

    rospy.sleep(2.0)

    # ──────────────────────────────────────────────────────────────────
    rospy.loginfo('=' * 60)
    rospy.loginfo('[test] ===  单车测试开始  ===')

    # Step 0: 初始化
    set_initial_pose()
    spawn_cargo()
    rospy.sleep(1.0)

    # Step 1: 前往货架0接近点
    rospy.loginfo('[test] --- STEP 1: 前往货架0 ---')
    navigate_to(APPROACH_X, APPROACH_Y, math.pi / 2, '货架0接近点')

    # Step 2: 抓取（直接臂控）
    rospy.loginfo('[test] --- STEP 2: 模拟抓取 ---')
    arm_reach_shelf()
    rospy.sleep(0.5)
    _pub_gripper(0.0, wait=1.0)    # 关闭手爪（夹住）
    rospy.sleep(0.3)
    delete_cargo()                  # 删除 Gazebo 模型（拾取完成）
    rospy.sleep(0.3)
    arm_carry()

    # Step 3: 前往投递区A
    rospy.loginfo('[test] --- STEP 3: 前往投递区A ---')
    navigate_to(DELIVERY_APPROACH_X, DELIVERY_APPROACH_Y, math.pi / 2, '投递区A')

    # Step 4: 放置
    rospy.loginfo('[test] --- STEP 4: 模拟投递 ---')
    arm_stretch_delivery()
    _pub_gripper(0.07, wait=1.5)   # 张开手爪（放下）
    arm_fold_back()

    rospy.loginfo('=' * 60)
    rospy.loginfo('[test] ===  测试完成！取货-投递流程正常  ===')
    rospy.loginfo('=' * 60)


if __name__ == '__main__':
    main()
