#!/usr/bin/env python3
"""
warehouse_manager.py
中央调度节点：管理4个货架上的商品（A/B随机刷新），分配任务给机器人。

货架坐标（my_warehouse.world）：
  rack_1: (-0.7,  0.5)  取货点在货架南侧走廊
  rack_2: ( 0.7,  0.5)
  rack_3: (-0.7, -0.5)
  rack_4: ( 0.7, -0.5)

目标区域：
  goal_A（绿色）: (-2.0,  0.0)  机器人朝东 yaw=0
  goal_B（蓝色）: ( 2.0,  0.0)  机器人朝西 yaw=pi
"""
import rospy
import random
import threading
from std_msgs.msg import String
from warehouse_astar.srv import RequestTask, RequestTaskResponse

# 每个货架的取货停靠位（机器人在中间走廊 y=0 处面向货架）
# 货架已向外移至 y=±0.6，走廊宽 0.8m，机器人可直接从东西方向进入
# (x, y, yaw)  yaw: pi/2=朝北, -pi/2=朝南
SHELF_PICKUP = {
    1: (-0.7,  0.0,  1.5708),  # rack_1 南侧走廊，朝北取货
    2: ( 0.7,  0.0,  1.5708),  # rack_2 南侧走廊，朝北取货
    3: (-0.7,  0.0, -1.5708),  # rack_3 北侧走廊，朝南取货
    4: ( 0.7,  0.0, -1.5708),  # rack_4 北侧走廊，朝南取货
}

GOAL_POSITIONS = {
    'A': (-1.8, 0.0, 0.0),       # 绿色地标，朝东（离西墙0.6m，安全）
    'B': ( 1.8, 0.0, 3.14159),   # 蓝色地标，朝西（离东墙0.6m，安全）
}


class WarehouseManager:
    def __init__(self):
        rospy.init_node('warehouse_manager')
        self.lock = threading.Lock()

        # shelf_items: {shelf_id: 'A' | 'B' | 'reserved'}
        self.shelf_items = {}
        # task_queue: list of (shelf_id, item_type)
        self.task_queue = []

        # 初始化：4个货架随机分配 A/B
        for shelf_id in range(1, 5):
            item = random.choice(['A', 'B'])
            self.shelf_items[shelf_id] = item
            self.task_queue.append((shelf_id, item))

        rospy.loginfo("[Manager] 初始货架: %s", self.shelf_items)
        rospy.loginfo("[Manager] 初始任务队列: %s", self.task_queue)

        # 任务请求服务
        rospy.Service('/warehouse/request_task', RequestTask, self.handle_request_task)

        # 机器人配送完成回调
        rospy.Subscriber('/warehouse/delivery_complete', String, self.handle_delivery)

        # 状态发布（每秒）
        self.status_pub = rospy.Publisher('/warehouse/shelf_status', String, queue_size=5)
        rospy.Timer(rospy.Duration(1.0), self.publish_status)

        rospy.loginfo("[Manager] 仓储调度节点启动完成")

    def handle_request_task(self, req):
        resp = RequestTaskResponse()
        with self.lock:
            if not self.task_queue:
                resp.success = False
                rospy.loginfo("[Manager] 无可用任务，%s 等待中", req.robot_name)
                return resp

            shelf_id, item_type = self.task_queue.pop(0)
            self.shelf_items[shelf_id] = 'reserved'

        pickup = SHELF_PICKUP[shelf_id]
        goal   = GOAL_POSITIONS[item_type]

        resp.success    = True
        resp.shelf_id   = shelf_id
        resp.item_type  = item_type
        resp.pickup_x   = pickup[0]
        resp.pickup_y   = pickup[1]
        resp.pickup_yaw = pickup[2]
        resp.goal_x     = goal[0]
        resp.goal_y     = goal[1]
        resp.goal_yaw   = goal[2]

        rospy.loginfo("[Manager] 分配任务给 %s: 货架%d (%s) -> 目标%s",
                      req.robot_name, shelf_id, item_type, item_type)
        return resp

    def handle_delivery(self, msg):
        """
        消息格式: "robot_name shelf_id"
        机器人送达后，对应货架立刻刷新新商品并入队。
        """
        parts = msg.data.split()
        if len(parts) != 2:
            rospy.logwarn("[Manager] delivery_complete 消息格式错误: %s", msg.data)
            return

        robot_name = parts[0]
        shelf_id   = int(parts[1])
        new_item   = random.choice(['A', 'B'])

        with self.lock:
            self.shelf_items[shelf_id] = new_item
            self.task_queue.append((shelf_id, new_item))

        rospy.loginfo("[Manager] %s 完成配送，货架%d 刷新为商品%s，任务入队",
                      robot_name, shelf_id, new_item)

    def publish_status(self, event):
        with self.lock:
            parts = [f"S{k}:{v}" for k, v in sorted(self.shelf_items.items())]
            queue_len = len(self.task_queue)
        status = "  ".join(parts) + f"  Queue:{queue_len}"
        self.status_pub.publish(status)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    mgr = WarehouseManager()
    mgr.run()
