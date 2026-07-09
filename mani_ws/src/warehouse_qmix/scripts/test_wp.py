#!/usr/bin/env python3
"""
test_wp.py — 单车航点导航完整取货-投递循环

流程：
  1. 初始化：随机给 10 个货架分配货物(A/B/C)，spawn 对应颜色货箱
  2. 轮询选下一个有货的货架，发布航点名导航
  3. 到达货架：删除货箱，记录货物类型，启动 20s 补货计时
  4. 导航到对应投递区（A→WP11, B→WP12, C→WP13）
  5. 到达投递区：完成投递，继续下一轮
  6. 所有货架暂时空置时：每 2s 重试一次，等待补货
"""

import math
import random
import threading
import rospy
from std_msgs.msg import String
from gazebo_msgs.srv import (SpawnModel, SpawnModelRequest,
                              DeleteModel, DeleteModelRequest)
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped

# ── 货物类型 ──────────────────────────────────────────────────────────
GOODS_NONE = 0
GOODS_A    = 1
GOODS_B    = 2
GOODS_C    = 3

GOODS_NAME = {GOODS_A: 'A', GOODS_B: 'B', GOODS_C: 'C'}

# ── 地图参数（与 warehouse_cross.world 一致）──────────────────────────
N_SHELVES = 10

SHELF_POSITIONS = [
    (-1.5,  6.5),   # 货架 0 → WP "1"
    ( 1.5,  6.5),   # 货架 1 → WP "2"
    (-8.5,  1.5),   # 货架 2 → WP "3"
    (-5.5,  1.5),   # 货架 3 → WP "4"
    (-8.5, -0.5),   # 货架 4 → WP "5"
    (-5.5, -0.5),   # 货架 5 → WP "6"
    ( 5.5,  1.5),   # 货架 6 → WP "7"
    ( 8.5,  1.5),   # 货架 7 → WP "8"
    ( 5.5, -0.5),   # 货架 8 → WP "9"
    ( 8.5, -0.5),   # 货架 9 → WP "10"
]
CARGO_Z = 0.245     # 货箱在货架上的中心高度

DELIVERY_WP = {GOODS_A: '11', GOODS_B: '12', GOODS_C: '13'}

# ── 货物颜色（A=绿, B=黄, C=蓝）─────────────────────────────────────
GOODS_COLOR = {
    GOODS_A: (0.2, 0.9, 0.2),
    GOODS_B: (0.9, 0.9, 0.2),
    GOODS_C: (0.2, 0.5, 0.9),
}

# ── 补货延迟（对应训练 RESTOCK_DELAY=40 步）──────────────────────────
RESTOCK_DELAY = 20.0   # 秒

# ── 状态机 ────────────────────────────────────────────────────────────
STATE_SELECTING     = 0
STATE_GOING_SHELF   = 1
STATE_GOING_DELIVERY = 2

state         = STATE_SELECTING
current_shelf = -1          # 当前目标货架索引
carrying      = GOODS_NONE  # 当前携带的货物类型
shelf_goods   = [GOODS_NONE] * N_SHELVES

lock = threading.Lock()
waypoint_pub = None


# ── Gazebo 货箱管理 ───────────────────────────────────────────────────

def _cargo_sdf(goods_type):
    s = 0.15
    m = 0.05 / 6 * s * s
    r, g, b = GOODS_COLOR[goods_type]
    return (
        '<?xml version="1.0"?>'
        '<sdf version="1.6"><model name="cargo"><static>false</static>'
        '<link name="link">'
        '<inertial><mass>0.05</mass>'
        f'<inertia><ixx>{m:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>'
        f'<iyy>{m:.6f}</iyy><iyz>0</iyz><izz>{m:.6f}</izz></inertia></inertial>'
        '<collision name="col"><geometry>'
        f'<box><size>{s} {s} {s}</size></box></geometry></collision>'
        '<visual name="vis"><geometry>'
        f'<box><size>{s} {s} {s}</size></box></geometry>'
        '<material>'
        f'<ambient>{r} {g} {b} 1</ambient>'
        f'<diffuse>{r} {g} {b} 1</diffuse>'
        '</material></visual>'
        '</link></model></sdf>'
    )


def spawn_cargo(shelf_idx, goods_type):
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=3.0)
        srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        req = SpawnModelRequest()
        req.model_name      = 'cargo_shelf_{}'.format(shelf_idx)
        req.model_xml       = _cargo_sdf(goods_type)
        req.reference_frame = 'world'
        x, y = SHELF_POSITIONS[shelf_idx]
        req.initial_pose = Pose(position=Point(x=x, y=y, z=CARGO_Z),
                                orientation=Quaternion(w=1.0))
        resp = srv(req)
        if not resp.success:
            rospy.logwarn('[test_wp] spawn cargo_shelf_%d 失败: %s',
                          shelf_idx, resp.status_message)
    except Exception as e:
        rospy.logwarn('[test_wp] spawn 异常: %s', e)


def delete_cargo(shelf_idx):
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=2.0)
        srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        srv(DeleteModelRequest(model_name='cargo_shelf_{}'.format(shelf_idx)))
        rospy.loginfo('[test_wp] cargo_shelf_%d 已删除', shelf_idx)
    except Exception as e:
        rospy.logwarn('[test_wp] delete 异常: %s', e)


# ── 补货（子线程，20s 后执行）────────────────────────────────────────

def _do_restock(shelf_idx):
    new_goods = random.randint(1, 3)
    with lock:
        shelf_goods[shelf_idx] = new_goods
    spawn_cargo(shelf_idx, new_goods)
    rospy.loginfo('[test_wp] 货架%d 补货完成 → 货物%s',
                  shelf_idx, GOODS_NAME[new_goods])


def schedule_restock(shelf_idx):
    t = threading.Timer(RESTOCK_DELAY, _do_restock, args=[shelf_idx])
    t.daemon = True
    t.start()


# ── 航点导航 ──────────────────────────────────────────────────────────

def go_waypoint(wp_name, label=''):
    wp = String()
    wp.data = wp_name
    waypoint_pub.publish(wp)
    rospy.loginfo('[test_wp] 发布航点 %s  %s', wp_name, label)


def select_and_go():
    """选下一个有货的货架并出发；全空则 2s 后重试。"""
    global state, current_shelf

    with lock:
        # 从上次货架的下一个开始轮询
        for offset in range(1, N_SHELVES + 1):
            idx = (current_shelf + offset) % N_SHELVES
            if shelf_goods[idx] != GOODS_NONE:
                current_shelf = idx
                state = STATE_GOING_SHELF
                break
        else:
            idx = -1

    if idx == -1:
        rospy.loginfo('[test_wp] 所有货架空置，2s 后重试...')
        threading.Timer(2.0, select_and_go).start()
        return

    go_waypoint(str(current_shelf + 1),
                '→ 货架{}'.format(current_shelf))


# ── 导航结果回调 ──────────────────────────────────────────────────────

def navi_result_cb(msg):
    global state, carrying

    if msg.data != 'done':
        return

    with lock:
        cur_state = state

    if cur_state == STATE_GOING_SHELF:
        with lock:
            goods = shelf_goods[current_shelf]
            shelf_goods[current_shelf] = GOODS_NONE
            carrying = goods

        rospy.loginfo('[test_wp] 到达货架%d，取货%s',
                      current_shelf, GOODS_NAME[goods])
        delete_cargo(current_shelf)
        schedule_restock(current_shelf)

        with lock:
            state = STATE_GOING_DELIVERY
        go_waypoint(DELIVERY_WP[goods],
                    '→ 投递区{}'.format(GOODS_NAME[goods]))

    elif cur_state == STATE_GOING_DELIVERY:
        with lock:
            goods = carrying
            carrying = GOODS_NONE
            state = STATE_SELECTING

        rospy.loginfo('[test_wp] 投递完成，货物%s ✓', GOODS_NAME[goods])
        select_and_go()


# ── 主函数 ────────────────────────────────────────────────────────────

def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped,
                          queue_size=1, latch=True)
    rospy.sleep(1.0)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id       = 'map'
    msg.header.stamp          = rospy.Time.now()
    msg.pose.pose.position    = Point(x=-2.0, y=-7.0, z=0.0)
    msg.pose.pose.orientation = Quaternion(
        x=0.0, y=0.0,
        z=math.sin(math.pi / 4),
        w=math.cos(math.pi / 4))   # yaw = π/2 面朝北
    msg.pose.covariance[0]  = 0.25
    msg.pose.covariance[7]  = 0.25
    msg.pose.covariance[35] = 0.07
    pub.publish(msg)
    rospy.loginfo('[test_wp] AMCL 初始位姿已发布 (-2.0, -7.0)')
    rospy.sleep(1.0)


def main():
    global waypoint_pub

    rospy.init_node('test_wp', anonymous=False)

    waypoint_pub = rospy.Publisher('/waterplus/navi_waypoint',
                                   String, queue_size=10)
    rospy.Subscriber('/waterplus/navi_result', String, navi_result_cb)

    rospy.sleep(2.0)
    set_initial_pose()

    # 初始化：给 10 个货架随机分配货物并 spawn 货箱
    rospy.loginfo('[test_wp] 初始化货架...')
    for i in range(N_SHELVES):
        goods = random.randint(1, 3)
        with lock:
            shelf_goods[i] = goods
        spawn_cargo(i, goods)
        rospy.sleep(0.3)   # 避免 Gazebo 服务过载

    rospy.loginfo('[test_wp] 初始化完成，开始任务循环')
    select_and_go()

    rospy.spin()


if __name__ == '__main__':
    main()
