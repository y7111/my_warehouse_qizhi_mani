#!/usr/bin/env python3
"""
warehouse_manager.py  —  简化版仓储状态管理

职责：
  - 维护 4 个货架的商品颜色（固定，不刷新）
  - 接收机器人"已取货/已送达"消息，更新状态
  - 每秒发布状态供调试

货架颜色规则（距离哪侧投放点近就生成对应颜色）：
  rack_1 (-0.7, +0.5) → 绿色(A) → 送往 goal_A
  rack_2 (+0.7, +0.5) → 蓝色(B) → 送往 goal_B
  rack_3 (-0.7, -0.5) → 绿色(A) → 送往 goal_A
  rack_4 (+0.7, -0.5) → 蓝色(B) → 送往 goal_B

分工：
  robot_alpha → rack_1, rack_3 (左侧两个绿色货架)
  robot_beta  → rack_2, rack_4 (右侧两个蓝色货架)
"""
import rospy
from std_msgs.msg import String

SHELF_COLOR = {1: 'GREEN(A)', 2: 'BLUE(B)', 3: 'GREEN(A)', 4: 'BLUE(B)'}
SHELF_STATUS = {1: 'waiting', 2: 'waiting', 3: 'waiting', 4: 'waiting'}


def on_pickup(msg):
    """格式: "robot_name shelf_id"  机器人已取货"""
    parts = msg.data.split()
    if len(parts) == 2:
        shelf_id = int(parts[1])
        SHELF_STATUS[shelf_id] = 'picked'
        rospy.loginfo('[Manager] 货架%d 已被 %s 取货', shelf_id, parts[0])


def on_delivered(msg):
    """格式: "robot_name shelf_id"  机器人已送达"""
    parts = msg.data.split()
    if len(parts) == 2:
        shelf_id = int(parts[1])
        SHELF_STATUS[shelf_id] = 'delivered'
        rospy.loginfo('[Manager] 货架%d 商品已送达（%s）', shelf_id, parts[0])


def main():
    rospy.init_node('warehouse_manager')

    rospy.Subscriber('/warehouse/pickup',   String, on_pickup)
    rospy.Subscriber('/warehouse/delivery_complete', String, on_delivered)
    pub = rospy.Publisher('/warehouse/shelf_status', String, queue_size=5)

    rospy.loginfo('[Manager] 启动 ─────────────────────────')
    rospy.loginfo('[Manager] rack1&3(绿A) → robot_alpha → goal_A')
    rospy.loginfo('[Manager] rack2&4(蓝B) → robot_beta  → goal_B')

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        parts = []
        for sid in range(1, 5):
            parts.append(f'rack{sid}[{SHELF_COLOR[sid]}]:{SHELF_STATUS[sid]}')
        pub.publish('  |  '.join(parts))
        rate.sleep()


if __name__ == '__main__':
    main()
