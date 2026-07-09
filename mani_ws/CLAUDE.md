# CLAUDE.md — mani_ws Gazebo 联调指引

> 工作空间：`~/mani_ws`（启智 MANI 平台代码 + AGV-QMIX 阶段二 Gazebo 联调）
> 开发机：Ubuntu 20.04，ROS Noetic
> 实车：Ubuntu 18.04，ROS Melodic（Jetson NX）

---

## 当前进度

| 阶段 | 状态 |
|---|---|
| 阶段一：QMIX 训练 | ✅ 完成 |
| 单车 Gazebo 联调（test_wp.launch） | ✅ 完成，可正常运行 |
| 四车 Gazebo spawn（warehouse_multi.launch） | ✅ 完成（world 文件嵌入方案） |
| 四车 QMIX 推理接入（robot_agent.py） | ✅ 完成 |
| 四车货物管理（cargo_manager.py） | ✅ 完成 |
| 完整 QMIX 联调（warehouse_qmix.launch） | ✅ 完成，待实机测试 |

**最优 checkpoint**：`~/qmix_project/checkpoints_v5_attn/qmix_ep800.pt`  
**贪婪评估**：19.0 次/ep（100 episode 均值）

---

## 项目参数（对应训练环境，不能改动）

| 参数 | 值 |
|---|---|
| 车辆数 | 4 台 |
| 货架数 | 10 个（含 A/B/C 三类货物） |
| 投递区 | 3 个（A / B / C） |
| obs_dim | 87 |
| state_dim | 74 |
| n_actions | 13（货架 0-9 + 投递区 A/B/C） |

---

## 地图参数（20×16 十字形，格子 ≈ 0.5m×0.5m）

```
货架位置（行,列）→ Gazebo (x,y) = (列×0.5, -(行×0.5))：
  货架 0: (1,8)   货架 1: (1,11)
  货架 2: (6,1)   货架 3: (6,4)
  货架 4: (8,1)   货架 5: (8,4)
  货架 6: (6,15)  货架 7: (6,18)
  货架 8: (8,15)  货架 9: (8,18)

投递区：
  A: (13,7)   B: (13,9)   C: (13,11)

机器人起点（已在 world 文件中 hardcode）：
  robot_0: x=-1.5  y=-7.0  yaw=π/2（朝北）
  robot_1: x=-0.5  y=-7.0  yaw=π/2
  robot_2: x= 0.5  y=-7.0  yaw=π/2
  robot_3: x= 1.5  y=-7.0  yaw=π/2

航点映射（waterplus_map_tools 格式，WP名=字符串）：
  货架 0→WP"1"  货架 1→WP"2"  ... 货架 9→WP"10"
  投递区 A→WP"11"  B→WP"12"  C→WP"13"
```

---

## 关键文件总览

### URDF

| 文件 | 用途 |
|---|---|
| `src/warehouse_qmix/urdf/agv_nav.urdf.xacro` | 四车用简化模型（box+cylinder，无网格，轻量） |
| `src/wpb_mani/wpb_mani_description/urdf/wpb_mani.urdf.xacro` | 单车用完整模型（含真实网格+机械臂+物理控制器） |

**agv_nav.urdf.xacro 结构**（参数 `robot_ns`）：
- base_link：0.57×0.45×0.28m 蓝色长方体，高度中心在 z=0.14m
- laser link：半径 0.04m 黑色圆柱，挂在 (0.215, 0, 0.30)
- 插件：`libgazebo_ros_planar_move.so`（支持多机器人命名空间）
- 激光：360线，10Hz，射程 0.1~10m，发布到 `$(robot_ns)/scan_raw`

### Launch

| 文件 | 用途 |
|---|---|
| `launch/test_wp.launch` | **单车**完整导航+取货，使用 `wpb_mani.urdf.xacro` |
| `launch/warehouse_multi.launch` | **四车** Gazebo，机器人嵌入 world，含 AMCL+move_base |
| `launch/warehouse_qmix.launch` | **完整四车 QMIX**：包含 multi + wp_navi_server×4 + cargo_manager + robot_agent×4 |

### World 文件

| 文件 | 用途 |
|---|---|
| `src/wpb_mani/wpb_mani_simulator/worlds/warehouse_cross.world` | 基础世界（货架+投递区+墙），单车用 |
| `src/wpb_mani/wpb_mani_simulator/worlds/warehouse_cross_multi.world` | 四车世界，agv_nav SDF 已嵌入（888行） |

**重要**：`warehouse_cross_multi.world` 是由脚本生成的，每次修改 `agv_nav.urdf.xacro` 后必须重新生成。

### 脚本

| 文件 | 用途 |
|---|---|
| `scripts/test_wp.py` | 单车完整取货-投递循环（已验证可用） |
| `scripts/test_all_wp.py` | 遍历 WP1→WP13，验证航点连通性 |
| `scripts/shelf_marker_pub.py` | RViz 货架可视化标记 |
| `scripts/cargo_manager.py` | **全局货物管理**（spawn/delete/补货/广播） |
| `scripts/robot_agent.py` | **QMIX 推理代理**（每车一个实例，~robot_id 参数） |

### Nav 配置

| 文件 | 关键参数 |
|---|---|
| `nav_config/costmap_common.yaml` | `robot_radius=0.25`（不能改大，否则 GlobalPlanner 拒绝货架附近目标） |
| `nav_config/global_costmap.yaml` | Dijkstra 全局规划 |
| `nav_config/local_costmap.yaml` | 滚动窗口 4×4m，5Hz 更新 |

---

## 四车 Gazebo 方案说明（重要）

### 死锁根本原因

`gazebo_ros_api_plugin` 的 `worldStatsCallback`（~1000Hz）和 `spawn_urdf_model` 服务争同一把锁 `this->lock_`。  
第一台车的插件开始运行后，后续的 `spawn_model` 永远拿不到锁 → 进程挂死（`Sl` 状态）。

catkin_ws 2台车勉强能过是因为碰巧第2台在第1台插件启动前完成 spawn，4台必然失败。

### 解决方案：world 文件嵌入

不调用 `spawn_model` 服务，而是把机器人 SDF 直接写进 `.world` 文件：

```bash
# 重新生成 world 文件（修改 agv_nav.urdf.xacro 后执行）
source ~/mani_ws/devel/setup.bash
for i in 0 1 2 3; do
  rosrun xacro xacro ~/mani_ws/src/warehouse_qmix/urdf/agv_nav.urdf.xacro \
    robot_ns:=robot_$i > /tmp/agv_$i.urdf 2>/dev/null
  gz sdf -p /tmp/agv_$i.urdf 2>/dev/null | grep -v "^Warning\|^Error\|^\[" > /tmp/agv_$i.sdf
done
python3 /tmp/gen_world.py
```

`gen_world.py` 位于 `/tmp/gen_world.py`（重启后丢失，内容见附录）。

---

## QMIX 推理接口（Python）

```python
import sys, torch
sys.path.append('/home/jasper/qmix_project')
from qmix_network import AttentionAgentQNet

net = AttentionAgentQNet(obs_dim=87, n_actions=13, n_agents=4, n_shelves=10)
ckpt = torch.load('/home/jasper/qmix_project/checkpoints_v5_attn/qmix_ep800.pt',
                  map_location='cpu', weights_only=False)
net.load_state_dict(ckpt['agent_net'])
net.eval()

# 推理（carrying_type: 0=空, 1=A, 2=B, 3=C）
with torch.no_grad():
    q = net(obs_tensor)  # obs_tensor: [1, 87]
n_shelves = 10
if carrying_type != 0:
    action = int(q[0, n_shelves:].argmax()) + n_shelves   # 选投递区
else:
    action = int(q[0, :n_shelves].argmax())                # 选货架
```

---

## obs 构造方法（对应 warehouse_env._get_obs）

```python
# self_feat (19维)
#   位置归一化 (2)：[row/15, col/19]
#   携带状态 one-hot (4)：[1,0,0,0]=空, [0,1,0,0]=A, [0,0,1,0]=B, [0,0,0,1]=C
#   当前目标 one-hot (13)：动作 0-12 对应的 one-hot

# other_feats (18维)：其他 3 台车，每台 6 维
#   [row/15, col/19, carrying_one_hot(4)]

# shelf_feats (50维)：10 货架，每架 5 维
#   [goods_one_hot(3), row/15, col/19]
#   goods_one_hot: [1,0,0]=A, [0,1,0]=B, [0,0,1]=C

obs = np.concatenate([self_feat, other_feats_flat, shelf_feats_flat])  # shape: (87,)
```

---

## 话题规范

| 话题 | 类型 | 说明 |
|---|---|---|
| `/robot_i/carrying` | `std_msgs/Int32` | 0=空, 1=A, 2=B, 3=C（i=0..3） |
| `/warehouse/pickup` | `std_msgs/String` | "i shelf_idx"（触发取货状态更新） |
| `/warehouse/delivery` | `std_msgs/String` | "i goods_type"（触发补货） |
| `/warehouse/shelf_goods` | `std_msgs/String` | "t0,t1,...,t9"（广播货架当前货物类型） |
| `/robot_i/waterplus/navi_waypoint` | `std_msgs/String` | 发给各车的目标航点名 |
| `/robot_i/waterplus/navi_result` | `std_msgs/String` | 各车导航结果（"done" / "failed"） |

---

## 下一步：四车 QMIX 接入

需要新建以下文件：

### 1. `scripts/robot_agent.py`（每台车一个节点实例）
- 接收参数：`robot_id`（0-3）
- 订阅：`/robot_i/odom`，`/warehouse/shelf_goods`
- 构造 87维 obs → 调用 QMIX 网络 → 发布航点到 `/robot_i/waterplus/navi_waypoint`
- 到达后发布 `/warehouse/pickup` 或 `/warehouse/delivery`

### 2. `scripts/cargo_manager.py`（全局货物管理）
- 订阅 `/warehouse/pickup`、`/warehouse/delivery`
- 维护 10 个货架状态（A/B/C/空）
- 广播 `/warehouse/shelf_goods`
- 管理 Gazebo 货箱 spawn/delete + 20s 补货计时

### 3. `launch/warehouse_qmix.launch`
- 包含 `warehouse_multi.launch`
- 为 robot_0~3 各启动一个 `robot_agent.py` 节点（带 `robot_id` 参数）
- 启动 `cargo_manager.py`
- 为 robot_0~3 各启动 `wp_navi_server`（需 ns 隔离）

---

## 常用命令

```bash
# 激活工作空间
source ~/mani_ws/devel/setup.bash

# 编译
cd ~/mani_ws && catkin_make

# 脚本修改后同步到 devel（必须执行）
cp src/warehouse_qmix/scripts/cargo_manager.py devel/lib/warehouse_qmix/
cp src/warehouse_qmix/scripts/robot_agent.py   devel/lib/warehouse_qmix/

# 启动完整 QMIX 联调（四车 + QMIX 推理）
roslaunch warehouse_qmix warehouse_qmix.launch

# 启动单车测试
roslaunch warehouse_qmix test_wp.launch

# 启动四车（仅 Gazebo + 导航栈，无 QMIX）
roslaunch warehouse_qmix warehouse_multi.launch

# 监控话题
rostopic echo /warehouse/shelf_goods      # 货架状态
rostopic echo /robot_0/carrying           # robot_0 携带状态
rostopic hz /robot_0/scan_raw
```

---

## 已知坑（重要）

| 坑 | 说明 |
|---|---|
| `robot_radius=0.25m` | 不能改大，≥0.3m 时 GlobalPlanner 拒绝货架附近的目标点 |
| costmap `sensor_frame` 不加 ns 前缀 | 在 launch 里用 `<param>` 覆盖时写 `robot_i/laser`，在 yaml 里不写前缀 |
| `spawn_model` 服务死锁 | 4台车不能用 spawn_model，必须嵌入 world 文件 |
| link_attacher 死锁 | 不要用，取货改用 delete_model + 状态记录 |
| 脚本修改后必须 cp 到 devel | ROS 执行的是 devel 下的副本 |
| AMCL 需 remap static_map | `<remap from="static_map" to="/static_map"/>` |
| `agv_nav.urdf.xacro` 改动后 | 必须重新运行 gen_world.py 重新生成 warehouse_cross_multi.world |
| warehouse_multi.launch 无 wp_navi_server | 目前只有 Gazebo + AMCL + move_base，还没有航点导航服务器 |

---

## 附录：gen_world.py（重新生成 world 文件）

```python
#!/usr/bin/env python3
import re

BASE_WORLD = "/home/jasper/mani_ws/src/wpb_mani/wpb_mani_simulator/worlds/warehouse_cross.world"

robots = [
    ("robot_0", -1.5, -7.0, 1.5707963),
    ("robot_1", -0.5, -7.0, 1.5707963),
    ("robot_2",  0.5, -7.0, 1.5707963),
    ("robot_3",  1.5, -7.0, 1.5707963),
]

def extract_model_body(sdf_path):
    with open(sdf_path) as f:
        txt = f.read()
    m = re.search(r'<model[^>]*>(.*?)</model>', txt, re.DOTALL)
    if not m:
        raise ValueError(f"No <model> found in {sdf_path}")
    return m.group(1)

with open(BASE_WORLD) as f:
    world_txt = f.read()

inserts = []
for name, x, y, yaw in robots:
    idx = name.split("_")[1]
    body = extract_model_body(f"/tmp/agv_{idx}.sdf")
    body = re.sub(r'\s*<static>[^<]*</static>', '', body)
    model_xml = (
        f'\n    <model name="{name}">\n'
        f'      <pose>{x} {y} 0 0 0 {yaw}</pose>\n'
        f'      <static>false</static>'
        + body +
        f'    </model>\n'
    )
    inserts.append(model_xml)

insert_block = "".join(inserts)
new_world = world_txt.replace("</world>", insert_block + "</world>")

out_path = "/home/jasper/mani_ws/src/wpb_mani/wpb_mani_simulator/worlds/warehouse_cross_multi.world"
with open(out_path, "w") as f:
    f.write(new_world)

print(f"Written: {out_path}, Total lines: {new_world.count(chr(10))}")
```

---

## 参考文档

| 文档 | 用途 |
|---|---|
| `~/AGV_QMIX_对话总结.md` | 项目总进展，训练结果，已踩坑汇总 |
| `~/qmix_project/PLAN.md` | 阶段验收条件（阶段二见第二节） |
| `~/mani_ws/README_notes.md` | 启智 MANI 平台完整接口文档 |
| `~/catkin_ws/CLAUDE.md` | 旧版 2 台车实现（robot_agent.py / cargo_manager.py 逻辑参考） |
