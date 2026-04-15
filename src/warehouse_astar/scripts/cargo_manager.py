#!/usr/bin/env python3
"""
cargo_manager.py  ─  仓库货物生命周期管理节点

功能：
  1. 启动时在 4 个货架上生成彩色货箱（Gazebo SDF 模型）
  2. 订阅 /cargo/pickup 话题 → 删除对应货架上的货箱
  3. 订阅 /warehouse/delivery_complete 话题 → 配送完成后自动补货
  4. 发布 /cargo/status → 实时货架货物状态

货架坐标（对应 my_warehouse.world）：
  rack_1: (-0.7,  0.6, 0.618)
  rack_2: ( 0.7,  0.6, 0.618)
  rack_3: (-0.7, -0.6, 0.618)
  rack_4: ( 0.7, -0.6, 0.618)

货箱放置高度：货架顶面 0.618 m + 半箱高 0.075 m = 0.693 m

用法：
    rosrun warehouse_astar cargo_manager.py
"""

import rospy
import threading
import random
from std_msgs.msg import String
from gazebo_msgs.srv import (SpawnModel, SpawnModelRequest,
                              DeleteModel, DeleteModelRequest)
from geometry_msgs.msg import Pose, Point, Quaternion

# ── 货架配置 ──────────────────────────────────────────────────────────
RACK_POSITIONS = {
    1: (-0.7,  0.6, 0.693),
    2: ( 0.7,  0.6, 0.693),
    3: (-0.7, -0.6, 0.693),
    4: ( 0.7, -0.6, 0.693),
}

# 商品颜色（RGBA）
ITEM_COLORS = {
    'A': (0.2, 0.8, 0.2, 1.0),   # 绿色 → 送往 goal_A
    'B': (0.2, 0.2, 0.9, 1.0),   # 蓝色 → 送往 goal_B
}

BOX_SIZE = 0.15   # 货箱边长（米）


def _make_box_sdf(r, g, b, a=1.0, size=BOX_SIZE):
    """生成一个彩色正方体的 SDF XML 字符串。"""
    half = size / 2.0
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="cargo_box">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>{0.5/6*size*size:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{0.5/6*size*size:.6f}</iyy><iyz>0</iyz>
          <izz>{0.5/6*size*size:.6f}</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class CargoManager:
    """货物生命周期管理节点。"""

    def __init__(self):
        rospy.init_node('cargo_manager')
        self.lock = threading.Lock()

        # shelf_state[shelf_id] = 'A' | 'B' | None
        self.shelf_state = {i: None for i in range(1, 5)}

        # 等待 Gazebo 服务就绪
        rospy.loginfo('[Cargo] 等待 Gazebo 服务...')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self._spawn  = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self._delete = rospy.ServiceProxy('/gazebo/delete_model',    DeleteModel)
        rospy.loginfo('[Cargo] Gazebo 服务已连接')

        # 发布货架状态
        self.status_pub = rospy.Publisher('/cargo/status', String, queue_size=5)
        rospy.Timer(rospy.Duration(2.0), self._publish_status)

        # 订阅取货事件
        rospy.Subscriber('/cargo/pickup',
                         String, self._on_pickup)

        # 订阅配送完成事件，自动补货
        rospy.Subscriber('/warehouse/delivery_complete',
                         String, self._on_delivery_complete)

        # 初始化：随机给每个货架放一箱货物
        rospy.sleep(1.0)
        self._init_shelves()
        rospy.loginfo('[Cargo] 货物管理节点启动完成')

    # ── 初始化 ────────────────────────────────────────────────────────

    # 与 warehouse_manager 保持一致：左架固定A，右架固定B
    SHELF_ITEM = {1: 'A', 2: 'B', 3: 'A', 4: 'B'}

    def _init_shelves(self):
        """启动时按分区固定生成货物（左=A绿色，右=B蓝色）。"""
        for shelf_id in range(1, 5):
            item = self.SHELF_ITEM[shelf_id]
            self._spawn_cargo(shelf_id, item)
        self._log_state()

    # ── Gazebo 模型操作 ───────────────────────────────────────────────

    def _model_name(self, shelf_id: int) -> str:
        return f'cargo_shelf_{shelf_id}'

    def _spawn_cargo(self, shelf_id: int, item_type: str) -> bool:
        """在指定货架上生成货物模型，返回是否成功。"""
        color = ITEM_COLORS.get(item_type, (0.8, 0.8, 0.2, 1.0))
        sdf   = _make_box_sdf(*color)
        name  = self._model_name(shelf_id)

        x, y, z = RACK_POSITIONS[shelf_id]
        pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        )

        req = SpawnModelRequest()
        req.model_name       = name
        req.model_xml        = sdf
        req.robot_namespace  = ''
        req.initial_pose     = pose
        req.reference_frame  = 'world'

        try:
            resp = self._spawn(req)
            if resp.success:
                with self.lock:
                    self.shelf_state[shelf_id] = item_type
                rospy.loginfo('[Cargo] 货架%d 生成商品%s ✓', shelf_id, item_type)
                return True
            else:
                rospy.logwarn('[Cargo] 货架%d 生成失败: %s', shelf_id, resp.status_message)
                return False
        except rospy.ServiceException as e:
            rospy.logwarn('[Cargo] 货架%d 生成异常: %s', shelf_id, e)
            return False

    def _delete_cargo(self, shelf_id: int) -> bool:
        """删除指定货架上的货物模型。"""
        name = self._model_name(shelf_id)
        req  = DeleteModelRequest(model_name=name)
        try:
            resp = self._delete(req)
            if resp.success:
                with self.lock:
                    self.shelf_state[shelf_id] = None
                rospy.loginfo('[Cargo] 货架%d 货物已取走 ✓', shelf_id)
                return True
            else:
                # 模型不存在也不算错误
                rospy.logdebug('[Cargo] 货架%d 删除: %s', shelf_id, resp.status_message)
                with self.lock:
                    self.shelf_state[shelf_id] = None
                return False
        except rospy.ServiceException as e:
            rospy.logwarn('[Cargo] 货架%d 删除异常: %s', shelf_id, e)
            return False

    # ── 话题回调 ──────────────────────────────────────────────────────

    def _on_pickup(self, msg: String):
        """
        /cargo/pickup 话题格式：  "shelf_id"  例如 "1"
        机器人取货完成后发布，触发删除货物模型。
        """
        try:
            shelf_id = int(msg.data.strip())
        except ValueError:
            rospy.logwarn('[Cargo] pickup 消息格式错误: %s', msg.data)
            return
        rospy.loginfo('[Cargo] 收到取货信号 → 货架%d', shelf_id)
        self._delete_cargo(shelf_id)

    def _on_delivery_complete(self, msg: String):
        """
        /warehouse/delivery_complete 话题格式：  "robot_name shelf_id"
        配送完成后自动在对应货架补货（随机 A/B）。
        """
        parts = msg.data.split()
        if len(parts) < 2:
            return
        try:
            shelf_id = int(parts[1])
        except ValueError:
            return

        new_item = self.SHELF_ITEM.get(shelf_id, 'A')   # 固定分区
        rospy.loginfo('[Cargo] 配送完成，货架%d 补充商品%s', shelf_id, new_item)
        # 稍作延迟再补货，视觉上更自然
        rospy.sleep(1.0)
        self._spawn_cargo(shelf_id, new_item)

    # ── 状态发布 ──────────────────────────────────────────────────────

    def _log_state(self):
        with self.lock:
            parts = [f'S{k}:{v or "空"}' for k, v in sorted(self.shelf_state.items())]
        rospy.loginfo('[Cargo] 当前货架: %s', '  '.join(parts))

    def _publish_status(self, _event):
        with self.lock:
            parts = [f'S{k}:{v or "空"}' for k, v in sorted(self.shelf_state.items())]
        self.status_pub.publish('  '.join(parts))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    mgr = CargoManager()
    mgr.run()
