# my_warehouse_qizhi_mani

多 AGV 仓库货物搬运系统 —— **QMIX 多智能体强化学习做任务分配 + ROS move_base 做导航**，
在 Gazebo 中四车协同取货/投递的完整仿真。基于启智 MANI 平台。

## 目录结构

```
.
├── mani_ws/                      # ROS (catkin) 工作空间
│   └── src/
│       ├── warehouse_qmix/       # ★ 本项目核心包：QMIX 推理代理、货物管理、四车 launch、导航配置
│       ├── wpb_mani/             # 启智 MANI 平台（机器人模型 / Gazebo world / 仿真器）
│       └── waterplus_map_tools/  # 航点管理与导航服务（wp_manager / wp_navi_server）
├── qmix_project/                 # QMIX 训练工程（纯 Python 2D 仿真训练 + 权重 checkpoint）
│   ├── train.py / warehouse_env.py / qmix_network.py
│   └── checkpoints_v5_attn/qmix_ep800.pt   # ★ 部署用最优权重
├── waypoints.xml                 # 航点定义（货架 / 投递区 / 进出闸口）
└── README.md
```

## 环境

- Ubuntu 20.04 + ROS Noetic
- Gazebo 11
- Python 3，PyTorch（仅 CPU 推理即可）

## 编译

```bash
cd mani_ws
catkin_make
source devel/setup.bash
```

> 航点文件默认从 `/home/<user>/waypoints.xml` 加载（见 `warehouse_qmix.launch`）。
> 把本仓库根目录的 `waypoints.xml` 放到该路径，或修改 launch 中的 `load` 参数。
> QMIX 权重路径在 `robot_agent.py` 顶部的 `CHECKPOINT` 常量，按实际位置修改。

## 运行

```bash
# 完整四车 QMIX 联调（Gazebo + 导航 + QMIX 推理 + 货物管理）
roslaunch warehouse_qmix warehouse_qmix.launch
```

## 核心组件（warehouse_qmix/scripts）

| 文件 | 作用 |
|---|---|
| `robot_agent.py` | 每车一个节点：实时 QMIX 推理选目标 + 导航状态机 + 车-车避障 |
| `cargo_manager.py` | 全局货物管理：货架发号防抢、20s 补货、Gazebo 货箱增删 |
| `neighbor_laser_pub.py` | 虚拟邻车雷达：把各车喂进彼此代价地图，让 DWA 能避开别的车 |

## 训练（qmix_project）

```bash
cd qmix_project
python3 train.py          # 训练
python3 evaluate.py       # 评估
```

最优 checkpoint：`checkpoints_v5_attn/qmix_ep800.pt`（贪婪评估 ~19 次投递/局）。
