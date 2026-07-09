#!/usr/bin/env python3
"""
cargo_manager.py — 全局货物管理节点

订阅 /warehouse/pickup   "robot_id shelf_idx"  → 删除货箱，20s 后补货
订阅 /warehouse/delivery "robot_id goods_type" → 记录日志
广播 /warehouse/shelf_goods "t0,t1,...,t9"（每 2s + 每次货架状态变化）
"""

import random
import threading
import rospy
from std_msgs.msg import String
from gazebo_msgs.srv import (SpawnModel, SpawnModelRequest,
                              DeleteModel, DeleteModelRequest)
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

GOODS_NONE = 0
GOODS_A    = 1
GOODS_B    = 2
GOODS_C    = 3
GOODS_NAME = {0: '空', 1: 'A', 2: 'B', 3: 'C'}

N_SHELVES = 10

# Gazebo 货架中心坐标，对应 warehouse_cross_multi.world
SHELF_POSITIONS = [
    (-1.5,  6.5),
    ( 1.5,  6.5),
    (-8.5,  1.5),
    (-5.5,  1.5),
    (-8.5, -0.5),
    (-5.5, -0.5),
    ( 5.5,  1.5),
    ( 8.5,  1.5),
    ( 5.5, -0.5),
    ( 8.5, -0.5),
]
CARGO_Z       = 0.245
RESTOCK_DELAY = 80.0   # 取货后空置秒数。原20s太快→货架总是满的→QMIX调度无用武之地；
                       # 调到≈一轮搬运时长，让稳态约半数货架空着，还原训练时(空置40步)的缺货压力

GOODS_COLOR = {
    GOODS_A: (0.2, 0.9, 0.2),
    GOODS_B: (0.9, 0.9, 0.2),
    GOODS_C: (0.2, 0.5, 0.9),
}

N_DELIVER = 3                    # 投递点 A/B/C

N_ROBOTS = 4
CARRY_Z  = 0.45                  # 载货箱悬在车顶高度（高于激光面0.30→不被任何车激光误当障碍）

shelf_goods = [GOODS_NONE] * N_SHELVES
reserved    = [-1] * N_SHELVES   # 各货架被哪辆车预约（-1=空闲）
deliver_reserved = [-1] * N_DELIVER   # 各投递点被哪辆车占用（-1=空闲）
lock = threading.Lock()
shelf_goods_pub      = None
reserved_pub         = None
deliver_reserved_pub = None
cargo_marker_pub     = None

# 演示用：载货箱"骑"在车上（省略物理抓取，用视觉箱子还原搬运过程）
set_state_pub = None
robot_pose    = {}               # rid -> geometry_msgs/Pose（各车实时位姿）
carry_goods   = {}               # rid -> goods_type（仅在载货途中存在）
carry_lock    = threading.Lock()


def _cargo_sdf(goods_type):
    s = 0.15
    r, g, b = GOODS_COLOR[goods_type]
    return (
        '<?xml version="1.0"?>'
        '<sdf version="1.6"><model name="cargo"><static>true</static>'
        '<link name="link">'
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
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
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
            rospy.logwarn('[cargo_mgr] spawn cargo_shelf_%d 失败: %s',
                          shelf_idx, resp.status_message)
    except Exception as e:
        rospy.logwarn('[cargo_mgr] spawn 异常: %s', e)


def delete_cargo(shelf_idx):
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=2.0)
        srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        srv(DeleteModelRequest(model_name='cargo_shelf_{}'.format(shelf_idx)))
        rospy.loginfo('[cargo_mgr] cargo_shelf_%d 已删除', shelf_idx)
    except Exception as e:
        rospy.logwarn('[cargo_mgr] delete 异常: %s', e)


# ── 演示：载货箱骑在车上 ──────────────────────────────────────────────
def _carry_sdf(goods_type):
    """载货箱：非静态(才能被 set_model_state 搬动) + 纯视觉(无碰撞→不撞车、不被激光看见)
    + 关重力(两帧之间不下坠)。颜色与货架货物同色。"""
    s = 0.22
    r, g, b = GOODS_COLOR[goods_type]
    return (
        '<?xml version="1.0"?>'
        '<sdf version="1.6"><model name="carry"><static>false</static>'
        '<link name="link"><gravity>0</gravity>'
        '<inertial><mass>0.01</mass><inertia>'
        '<ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz>'
        '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
        '<visual name="vis"><geometry>'
        f'<box><size>{s} {s} {s}</size></box></geometry>'
        '<material>'
        f'<ambient>{r} {g} {b} 1</ambient>'
        f'<diffuse>{r} {g} {b} 1</diffuse>'
        '</material></visual>'
        '</link></model></sdf>'
    )


def spawn_carry(rid, goods_type):
    delete_carry(rid)   # 防止上一个残留
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
        srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        req = SpawnModelRequest()
        req.model_name      = 'carry_{}'.format(rid)
        req.model_xml       = _carry_sdf(goods_type)
        req.reference_frame = 'world'
        p = robot_pose.get(rid)
        x = p.position.x if p else 0.0
        y = p.position.y if p else 0.0
        req.initial_pose = Pose(position=Point(x=x, y=y, z=CARRY_Z),
                                orientation=Quaternion(w=1.0))
        srv(req)
    except Exception as e:
        rospy.logwarn('[cargo_mgr] spawn carry_%d 异常: %s', rid, e)


def delete_carry(rid):
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=2.0)
        srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        srv(DeleteModelRequest(model_name='carry_{}'.format(rid)))
    except Exception:
        pass


def _odom_cb(msg, rid):
    robot_pose[rid] = msg.pose.pose


def _carry_tick(_evt):
    """15Hz 把每个载货箱贴到对应车的实时位姿（跟着车跑、随车转向）。"""
    if set_state_pub is None:
        return
    with carry_lock:
        rids = list(carry_goods.keys())
    for rid in rids:
        p = robot_pose.get(rid)
        if p is None:
            continue
        ms = ModelState()
        ms.model_name      = 'carry_{}'.format(rid)
        ms.reference_frame = 'world'
        ms.pose.position.x = p.position.x
        ms.pose.position.y = p.position.y
        ms.pose.position.z = CARRY_Z
        ms.pose.orientation = p.orientation
        set_state_pub.publish(ms)


def _publish_markers():
    ma = MarkerArray()
    now = rospy.Time.now()
    with lock:
        goods_snapshot = list(shelf_goods)

    for i, (x, y) in enumerate(SHELF_POSITIONS):
        g = goods_snapshot[i]

        # 方块
        cube = Marker()
        cube.header.frame_id = 'map'
        cube.header.stamp    = now
        cube.ns   = 'cargo'
        cube.id   = i
        cube.type = Marker.CUBE

        # 文字
        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp    = now
        txt.ns   = 'cargo_label'
        txt.id   = i
        txt.type = Marker.TEXT_VIEW_FACING

        if g == GOODS_NONE:
            cube.action = Marker.DELETE
            txt.action  = Marker.DELETE
        else:
            r, gb, b = GOODS_COLOR[g]
            cube.action = Marker.ADD
            cube.pose.position.x = x
            cube.pose.position.y = y
            cube.pose.position.z = CARGO_Z
            cube.pose.orientation.w = 1.0
            cube.scale.x = 0.15
            cube.scale.y = 0.15
            cube.scale.z = 0.15
            cube.color.r = r
            cube.color.g = gb
            cube.color.b = b
            cube.color.a = 1.0
            cube.lifetime = rospy.Duration(0)

            txt.action = Marker.ADD
            txt.pose.position.x = x
            txt.pose.position.y = y
            txt.pose.position.z = CARGO_Z + 0.22
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.22
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = GOODS_NAME[g]
            txt.lifetime = rospy.Duration(0)

        ma.markers.append(cube)
        ma.markers.append(txt)

    cargo_marker_pub.publish(ma)


_test_done      = False
_test_start_t   = None
_pickup_log     = []   # (shelf_idx, robot_id, elapsed)


def _publish():
    global _test_done
    with lock:
        goods_snapshot    = list(shelf_goods)
        reserved_snapshot = list(reserved)
        deliver_snapshot  = list(deliver_reserved)
    msg_str = ','.join(str(g) for g in goods_snapshot)
    shelf_goods_pub.publish(String(data=msg_str))
    reserved_pub.publish(String(data=','.join(str(r) for r in reserved_snapshot)))
    deliver_reserved_pub.publish(String(data=','.join(str(r) for r in deliver_snapshot)))
    _publish_markers()

    if not RESTOCK_ENABLED and not _test_done:
        if all(g == GOODS_NONE for g in goods_snapshot):
            _test_done = True
            elapsed = (rospy.Time.now() - _test_start_t).to_sec() if _test_start_t else 0
            rospy.logwarn('=' * 55)
            rospy.logwarn('[测试完成] 全部 %d 个货架已取完，用时 %.1f 秒', N_SHELVES, elapsed)
            rospy.logwarn('取货记录：')
            for shelf, rid, t in _pickup_log:
                rospy.logwarn('  货架%-2d → robot_%d  (+%.1fs)', shelf, rid, t)
            rospy.logwarn('=' * 55)


def _do_restock(shelf_idx):
    new_goods = random.randint(1, 3)
    with lock:
        shelf_goods[shelf_idx] = new_goods
    spawn_cargo(shelf_idx, new_goods)
    rospy.loginfo('[cargo_mgr] 货架%d 补货 → %s', shelf_idx, GOODS_NAME[new_goods])
    _publish()


RESTOCK_ENABLED = True   # 由 ROS 参数 ~restock 覆盖


def pickup_cb(msg):
    """msg: "robot_id shelf_idx" """
    try:
        parts = msg.data.strip().split()
        robot_id  = int(parts[0])
        shelf_idx = int(parts[1])
    except Exception:
        rospy.logwarn('[cargo_mgr] pickup 消息格式错误: %s', msg.data)
        return

    with lock:
        if shelf_goods[shelf_idx] == GOODS_NONE:
            rospy.logwarn('[cargo_mgr] 货架%d 已空，忽略 (robot_%d)', shelf_idx, robot_id)
            return
        goods = shelf_goods[shelf_idx]    # 先记下货物类型，再清空（演示载货箱要用）
        shelf_goods[shelf_idx] = GOODS_NONE
        reserved[shelf_idx]    = -1   # 取走即释放预约

    elapsed = (rospy.Time.now() - _test_start_t).to_sec() if _test_start_t else 0
    _pickup_log.append((shelf_idx, robot_id, elapsed))
    if RESTOCK_ENABLED:
        rospy.loginfo('[cargo_mgr] robot_%d 取货架%d，启动 %.0fs 补货计时',
                      robot_id, shelf_idx, RESTOCK_DELAY)
    else:
        rospy.loginfo('[cargo_mgr] robot_%d 取货架%d（测试模式，不补货）',
                      robot_id, shelf_idx)
    delete_cargo(shelf_idx)
    # 演示：货箱"骑"到车上，跟着车一路开到投递区
    with carry_lock:
        carry_goods[robot_id] = goods
    spawn_carry(robot_id, goods)
    _publish()
    if RESTOCK_ENABLED:
        t = threading.Timer(RESTOCK_DELAY, _do_restock, args=[shelf_idx])
        t.daemon = True
        t.start()


def claim_cb(msg):
    """msg: "robot_id shelf_idx" —— 货架预约，唯一发号，先到先得"""
    try:
        parts = msg.data.strip().split()
        robot_id  = int(parts[0])
        shelf_idx = int(parts[1])
    except Exception:
        return
    if not (0 <= shelf_idx < N_SHELVES):
        return
    with lock:
        # 一车同时只占一个货架：先释放该车之前占的其它货架
        for k in range(N_SHELVES):
            if reserved[k] == robot_id and k != shelf_idx:
                reserved[k] = -1
        # 先到先得：货架有货且未被占 → 授予
        if shelf_goods[shelf_idx] != GOODS_NONE and reserved[shelf_idx] == -1:
            reserved[shelf_idx] = robot_id
    _publish()


def deliver_claim_cb(msg):
    """msg: "robot_id point_idx" —— 投递点预约，唯一发号，先到先得。
    point_idx 0/1/2=A/B/C；<0 表示释放该车占用的投递点。"""
    try:
        parts = msg.data.strip().split()
        robot_id  = int(parts[0])
        point_idx = int(parts[1])
    except Exception:
        return
    with lock:
        # 一车同时只占一个投递点：先释放该车之前占的其它点
        for k in range(N_DELIVER):
            if deliver_reserved[k] == robot_id and k != point_idx:
                deliver_reserved[k] = -1
        # 先到先得：点未被占 → 授予
        if 0 <= point_idx < N_DELIVER and deliver_reserved[point_idx] == -1:
            deliver_reserved[point_idx] = robot_id
    _publish()


def delivery_cb(msg):
    """msg: "robot_id goods_type" """
    try:
        parts = msg.data.strip().split()
        robot_id   = int(parts[0])
        goods_type = int(parts[1])
    except Exception:
        return
    # 投递完成 → 释放该车占的投递点（point = goods_type-1）
    point_idx = goods_type - 1
    with lock:
        if 0 <= point_idx < N_DELIVER and deliver_reserved[point_idx] == robot_id:
            deliver_reserved[point_idx] = -1
    # 演示：投递完成，车上的货箱消失
    with carry_lock:
        carry_goods.pop(robot_id, None)
    delete_carry(robot_id)
    _publish()
    rospy.loginfo('[cargo_mgr] robot_%d 投递 %s ✓', robot_id, GOODS_NAME.get(goods_type, '?'))


def main():
    global shelf_goods_pub, reserved_pub, deliver_reserved_pub, cargo_marker_pub
    global RESTOCK_ENABLED, set_state_pub
    rospy.init_node('cargo_manager', anonymous=False)

    RESTOCK_ENABLED = rospy.get_param('~restock', True)
    if not RESTOCK_ENABLED:
        rospy.logwarn('[cargo_mgr] 测试模式：取货后不补货')

    shelf_goods_pub  = rospy.Publisher('/warehouse/shelf_goods', String,
                                       queue_size=1, latch=True)
    reserved_pub     = rospy.Publisher('/warehouse/shelf_reserved', String,
                                       queue_size=1, latch=True)
    deliver_reserved_pub = rospy.Publisher('/warehouse/deliver_reserved', String,
                                           queue_size=1, latch=True)
    cargo_marker_pub = rospy.Publisher('/warehouse/cargo_markers', MarkerArray,
                                       queue_size=1, latch=True)
    rospy.Subscriber('/warehouse/pickup',        String, pickup_cb)
    rospy.Subscriber('/warehouse/delivery',      String, delivery_cb)
    rospy.Subscriber('/warehouse/claim',         String, claim_cb)
    rospy.Subscriber('/warehouse/deliver_claim', String, deliver_claim_cb)

    # 演示：载货箱骑在车上——订阅各车位姿 + set_model_state 话题 + 15Hz 跟随
    set_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    for i in range(N_ROBOTS):
        rospy.Subscriber('/robot_{}/odom'.format(i), Odometry,
                         lambda m, ii=i: _odom_cb(m, ii))
    rospy.Timer(rospy.Duration(1.0 / 15.0), _carry_tick)

    rospy.sleep(2.0)

    rospy.loginfo('[cargo_mgr] 初始化 %d 个货架...', N_SHELVES)
    for i in range(N_SHELVES):
        goods = random.randint(1, 3)
        with lock:
            shelf_goods[i] = goods
        spawn_cargo(i, goods)
        rospy.sleep(0.3)

    global _test_start_t
    _test_start_t = rospy.Time.now()
    _publish()
    rospy.loginfo('[cargo_mgr] 初始化完成')

    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        _publish()
        rate.sleep()


if __name__ == '__main__':
    main()
