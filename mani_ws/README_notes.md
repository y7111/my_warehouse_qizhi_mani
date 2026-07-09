# 启智 MANI 机器人开发参考文档

> **用途**：供 AI 辅助开发时快速定位接口、包结构和开发规范。本文基于官方实验指导书 V1.1.7、`mani_ws` 原版代码整理，是后续所有开发工作的信息基础。
>
> **工作空间**：`~/mani_ws`（原版代码，开发用）| `~/catkin_ws`（含自定义修改）
> **开发机环境**：Ubuntu 20.04，ROS Noetic
> **机器人本机环境**：Ubuntu 18.04，ROS Melodic（Jetson NX）

---

## 0. 快速命令速查

```bash
# 激活工作空间（每个新终端执行一次）
source ~/mani_ws/devel/setup.bash

# 查看机器人模型
roslaunch wpb_mani_description urdf.launch

# 启动建图仿真（场景1 + Gmapping）
roslaunch wpb_mani_simulator scene_1_gmapping.launch

# 新终端：键盘控制机器人
rosrun wpb_mani_simulator keyboard_control

# 保存地图
rosrun map_server map_saver -f ~/mani_ws/src/wpb_mani/wpb_mani_tutorials/maps/map

# 启动导航仿真
roslaunch wpb_mani_simulator navigation.launch

# 查看节点关系图
rqt_graph

# 查看话题数据流
rostopic echo /scan
rostopic hz /cmd_vel
```

---

## 1. 硬件规格（来自官方指导书）

### 1.1 整机参数

| 参数 | 值 |
|---|---|
| 整机重量 | 约 20kg（含抓取组件） |
| 最大承载 | 50kg |
| 工作环境 | 室内，地面承重 ≥ 20kg，坡度 ≤ 30° |
| 工作温度 | 15°C ~ 35°C |
| 防水等级 | 无防水，禁止接触液体 |

### 1.2 核心硬件

| 硬件 | 型号/规格 |
|---|---|
| 计算单元 | NVIDIA Jetson NX |
| 底盘 | 麦克纳姆轮全向移动底盘（4轮） |
| 激光雷达 | 思岚 RPLIDAR A2，测距 0.15m~12m，360°，角度分辨率 0.9°，10Hz |
| 深度相机 | Microsoft Azure Kinect（RGB + TOF深度） |
| 机械臂 | OpenManipulator-X，4自由度旋转关节 + 1个线性手爪 |
| IMU | MPU6050，内置于底层主控板 |
| 电池 | 7节 3500mAh 锂离子串联，输出 23.1V ~ 29.4V |

### 1.3 电控通信链路

```
Jetson NX ──USB──► MANI主控板 ──RS485──► 4个伺服电机模块（50ms周期）
                                └──RS485──► 机械臂控制板 ──UART-BUS──► OpenManipulator-X
         ──USB──► RPLidar A2
         ──USB──► Azure Kinect
         ──USB──► USB-HUB ──► 显示器
         ──HDMI──► 显示器
主控板内置 FTDI 芯片，USB → UART，波特率 115200
```

### 1.4 开关操作

| 开关 | 位置 | 功能 |
|---|---|---|
| 总电源开关 | 底盘尾部左侧（带指示灯） | 接通整机电源，启动 Ubuntu 系统 |
| 动力开关 | 底盘尾部右侧（红色蘑菇头） | 控制底盘电机和机械臂的电力，顺时针旋转弹起=通电，按下锁死=断电 |

> **紧急停止**：按下红色动力开关即可立即停止机器人运动，系统继续运行。

---

## 2. 软件环境

### 2.1 机器人本机（Jetson NX）

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 18.04（ARM架构） |
| ROS版本 | Melodic |
| 用户名 | robot |
| 初始密码 | 6 |
| 串口设备名 | `/dev/ftdi` |
| 推荐IDE | VS Code |

### 2.2 开发机（本机）

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 20.04 |
| ROS版本 | Noetic |
| 工作空间 | `~/mani_ws`（原版） / `~/catkin_ws`（自定义） |

### 2.3 推荐开发流程

```
开发机(仿真调试) ──USB/网络共享──► 机器人本机(实物运行)
先仿真验证逻辑 → 再部署到真机
```

### 2.4 出厂预装软件包

| 包名 | 功能 |
|---|---|
| `wpb_mani_bringup` | 基础功能（硬件驱动） |
| `wpb_mani_behaviors` | 行为服务（抓取/物体定位） |
| `wpb_mani_tutorials` | 应用例程（示例代码） |
| `wpb_local_planner` | 导航局部规划器 |
| `wpb_mani_description` | 模型描述（URDF） |
| `wpb_mani_moveit_config` | MoveIt! 参数配置 |
| `wpb_mani_simulator` | Gazebo 仿真配置 |

---

## 3. 功能模块清单

| 功能模块 | 关键技术 | 说明 |
|---|---|---|
| 底盘运动控制 | `geometry_msgs/Twist` → `/cmd_vel` | 全向移动，linear.x前后、linear.y左右平移、angular.z旋转，均可叠加 |
| 手柄遥控 | `joy` 包 → `/joy` | 左摇杆控制平移，右摇杆控制旋转，deadzone=0.12，设备 `/dev/input/js0` |
| 激光雷达 | RPLIDAR A2 → `/scan` | 360° 测距数组，含车体死角滤波器，供避障和建图使用 |
| SLAM 建图 | Gmapping | 激光雷达 + 里程计，构建二维栅格地图，`map_saver` 保存 |
| 自主导航 | AMCL + move_base | 全局路径规划（A*）+ 自定义局部规划器（`wpb_local_planner`） |
| 路点导航 | `waterplus_map_tools` | Rviz 插件标注路点，字符串话题触发，无需 actionlib |
| 2D 视觉特征提取 | OpenCV HSV | 订阅 `/rgb/image_raw`，BGR→HSV阈值分割，提取轮廓质心 |
| 2D 视觉目标跟踪 | OpenCV + P控制器 | 质心偏差驱动 `angular.z`，实现视觉对准闭环跟踪 |
| AR 二维码识别 | `ar_track_alvar` | 识别 AR 标签，输出标签相对机器人的六自由度位姿 |
| 3D 点云处理 | PCL | 订阅 `/points2`，PassThrough滤波 + RANSAC平面拟合 + 欧氏聚类 |
| 物体空间定位 | PCL 欧氏聚类 | 聚类后计算包围盒质心，输出 xyz 绝对坐标到 `/wpb_mani/boxes_3d` |
| 机械臂关节控制 | JointState 直控 | 向 `/wpb_mani/joint_ctrl` 发送各关节弧度，绕过 MoveIt! |
| MoveIt! 运动规划 | 正解/逆解/避障轨迹 | 正解：关节角→末端位姿；逆解：目标位姿→关节角；支持碰撞避障 |
| 移动抓取 | 全流程串联 | 导航→视觉定位→逆解→抓取，发布坐标到 `/wpb_mani/grab_box` 一键触发 |

---

## 4. 核心 ROS 话题接口

| 话题 | 消息类型 | 方向 | 用途 |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 发布 | **底盘速度控制**。`linear.x`前后(m/s)，`linear.y`左右(m/s)，`angular.z`旋转(rad/s)，其余分量置0 |
| `/odom` | `nav_msgs/Odometry` | 订阅 | 底盘里程计，当前位姿估计和速度反馈 |
| `/joy` | `sensor_msgs/Joy` | 订阅 | 手柄原始输入（摇杆值±1，按键0/1） |
| `/scan` | `sensor_msgs/LaserScan` | 订阅 | 雷达数据（经过死角滤波后），`ranges[]`数组，索引0=正前方，逆时针增大 |
| `/robot/scan_raw` | `sensor_msgs/LaserScan` | 订阅 | **仿真中雷达原始话题**（注意命名空间，非真机话题） |
| `/map` | `nav_msgs/OccupancyGrid` | 订阅 | Gmapping 或 map_server 发布的二维栅格地图 |
| `/rgb/image_raw` | `sensor_msgs/Image` | 订阅 | Azure Kinect 原始 RGB 图像，配合 `cv_bridge` 使用 |
| `/rgb/camera_info` | `sensor_msgs/CameraInfo` | 订阅 | 相机内参矩阵，用于像素坐标→空间坐标反投影 |
| `/ar_pose_marker` | `ar_track_alvar_msgs/AlvarMarkers` | 订阅 | AR标签识别结果，含每个 Marker 的 ID 和六自由度位姿 |
| `/points2` | `sensor_msgs/PointCloud2` | 订阅 | Azure Kinect 彩色三维点云，转为 PCL 格式处理 |
| `/joint_states` | `sensor_msgs/JointState` | 订阅 | 机械臂各关节当前弧度反馈 |
| `/wpb_mani/joint_ctrl` | `sensor_msgs/JointState` | 发布 | 机械臂底层直控：4个关节弧度 + 手爪间距(m) |
| `/waterplus/navi_waypoint` | `std_msgs/String` | 发布 | 触发路点导航，发送路点名称字符串（如 `"Goal1"`） |
| `/waterplus/navi_result` | `std_msgs/String` | 订阅 | 导航结果反馈，到达返回 `"done"`，失败返回错误码 |
| `/wpb_mani/plane_height` | `std_msgs/Float64` | 发布 | 设置 3D 检测基准平面高度（如桌面绝对高度，单位 m） |
| `/wpb_mani/boxes_3d` | `wpb_mani_behaviors/Coord` | 订阅 | 物体检测结果，按距离排序返回名称和 xyz 坐标数组 |
| `/wpb_mani/grab_box` | `geometry_msgs/Pose` | 发布 | 触发完整抓取行为，发送目标物三维坐标即可 |
| `/wpb_mani/grab_result` | `std_msgs/String` | 订阅 | 抓取完成反馈，返回 `"done"` |

### 4.1 速度控制详解

```
geometry_msgs/Twist:
  linear:
    x: 前进(+) / 后退(-)   单位 m/s
    y: 左平移(+) / 右平移(-) 单位 m/s  ← 麦轮全向底盘特有
    z: 0（对本机器人无意义）
  angular:
    x: 0
    y: 0
    z: 左转(+) / 右转(-)   单位 rad/s
```

---

## 5. 核心 Launch 文件速查

| Launch 文件 | 所在包 | 用途 |
|---|---|---|
| `urdf.launch` | `wpb_mani_description` | 仅加载机器人模型到 Rviz，验证URDF |
| `wpb_mani_gazebo.launch` | `wpb_mani_simulator` | 启动空白 Gazebo 仿真场景（无建图） |
| `scene_1_gmapping.launch` | `wpb_mani_simulator` | **建图仿真**：场景1 + Gmapping + Rviz（已修复雷达话题） |
| `navigation.launch` | `wpb_mani_simulator` | 导航仿真：AMCL + move_base + 加载地图 |
| `grab_box.launch` | `wpb_mani_simulator` | 抓取箱子仿真 |
| `moveit.launch` | `wpb_mani_simulator` | MoveIt! 机械臂仿真 |
| `gmapping.launch` | `wpb_mani_tutorials` | **真机 SLAM 建图** |
| `nav.launch` | `wpb_mani_tutorials` | 真机自主导航 |
| `mobile_manipulation.launch` | `wpb_mani_tutorials` | 真机移动抓取（启动所有依赖） |
| `mani_ctrl.launch` | `wpb_mani_tutorials` | 真机机械臂控制测试 |
| `add_waypoint_mani.launch` | `waterplus_map_tools` | Rviz 中交互式标注/编辑路点 |
| `wpb_mani_nav_test.launch` | `waterplus_map_tools` | 路点导航测试 |
| `normal.launch` | `wpb_mani_bringup` | 真机完整启动（底盘+雷达+Kinect） |
| `base_only.launch` | `wpb_mani_bringup` | 真机仅启动底盘驱动（不含传感器） |
| `base_lidar.launch` | `wpb_mani_bringup` | 真机底盘+激光雷达 |
| `js_ctrl.launch` | `wpb_mani_bringup` | 真机手柄遥控 |
| `kinect_test.launch` | `wpb_mani_bringup` | 真机 Azure Kinect 测试 |
| `calibrate_velocity.launch` | `wpb_mani_bringup` | 底盘速度校准 |
| `calibrate_kinect.launch` | `wpb_mani_bringup` | Kinect 视角校准 |
| `calibrate_gripper.launch` | `wpb_mani_bringup` | 手爪校准 |

---

## 6. 包结构与文件职责

```
mani_ws/src/
│
├── waterplus_map_tools/              # 路点导航中间件
│   ├── src/wp_manager.cpp            # 路点数据库（存储/查询/删除）
│   ├── src/wp_navi_server.cpp        # 导航服务端（接收路点名，驱动move_base）
│   ├── src/pose_navi_server.cpp      # 按坐标导航服务端
│   ├── src/wp_nav_test.cpp           # 导航测试节点
│   ├── srv/GetWaypointByName.srv     # 按名称查路点服务
│   ├── srv/AddNewWaypoint.srv        # 新增路点服务
│   └── msg/Waypoint.msg              # 路点消息（name + pose）
│
└── wpb_mani/
    │
    ├── wpb_mani_description/         # 机器人模型（只含描述，不含逻辑）
    │   ├── urdf/wpb_mani.urdf.xacro              # 完整机器人URDF主文件
    │   ├── urdf/open_manipulator_x.urdf.xacro    # OpenManipulator-X机械臂URDF
    │   ├── urdf/open_manipulator_x.gazebo.xacro  # 机械臂Gazebo仿真插件
    │   └── meshes/                               # 底盘/麦轮/机械臂各部件3D模型(.dae/.stl)
    │
    ├── wpb_mani_bringup/             # 真实硬件驱动层
    │   ├── src/WPB_Mani_driver.cpp   # 底层串口通信驱动（RS485/FTDI）
    │   ├── src/wpb_mani_lidar_filter.cpp  # 真机雷达死角滤波节点
    │   ├── src/wpb_mani_js_vel.cpp   # 手柄→cmd_vel 转换节点
    │   ├── config/wpb_mani.yaml      # 相机安装高度/倾角等底层参数
    │   └── scripts/install_for_melodic.sh  # 依赖安装脚本（真机用）
    │
    ├── wpb_mani_simulator/           # Gazebo仿真层
    │   ├── src/wpb_mani_plugin.cpp        # Gazebo底盘运动学插件
    │   ├── src/wpb_mani_sim_lidar_filter.cpp  # 仿真雷达滤波（sub_topic已修复）
    │   ├── src/keyboard_control.cpp       # 键盘控制节点（wasd+方向键）
    │   ├── src/wpb_mani_gripper.cpp       # 仿真手爪控制
    │   ├── worlds/scene_1.world           # 仿真场景1（带墙壁/桌子/箱子/储物桶）
    │   ├── models/                        # 桌子/箱子/储物桶/球等仿真模型
    │   └── config/wpb_mani_control.yaml   # 仿真控制器参数
    │
    ├── wpb_mani_moveit_config/       # 机械臂运动规划配置
    │   ├── config/kinematics.yaml         # KDL运动学求解器配置
    │   ├── config/joint_limits.yaml       # 4关节+手爪的角度限位
    │   ├── config/ompl_planning.yaml      # OMPL规划器参数
    │   ├── config/wpb_mani.srdf           # MoveIt!语义描述（规划组定义）
    │   └── launch/move_group.launch       # MoveIt! move_group 核心服务
    │
    ├── wpb_local_planner/            # 自定义局部路径规划器（替换DWA）
    │   ├── src/wpb_local_planner.cpp      # 规划器主逻辑（注册为move_base插件）
    │   └── src/CLidarAC.cpp              # 雷达避障控制核心（拦截/scan_filtered）
    │
    ├── wpb_mani_behaviors/           # 行为封装层（对外暴露简洁话题接口）
    │   ├── src/wpb_mani_boxes_3d.cpp     # 3D物体识别：PCL聚类→发布xyz坐标到/wpb_mani/boxes_3d
    │   └── src/wpb_mani_grab_box.cpp     # 抓取行为：接收坐标→MoveIt!逆解→执行抓取序列
    │
    └── wpb_mani_tutorials/           # 示例代码（二次开发起点）
        ├── src/wpb_mani_velocity_control.cpp      # 速度控制
        ├── src/wpb_mani_lidar_data.cpp            # 读取雷达 ranges[] 数组
        ├── src/wpb_mani_lidar_behavior.cpp        # 雷达触发避障状态机
        ├── src/wpb_mani_cruise.cpp                # 自主巡游
        ├── src/wpb_mani_joint_control.cpp         # 机械臂关节直控
        ├── src/wpb_mani_arm_forward_kinematics.cpp # 正运动学示例
        ├── src/wpb_mani_arm_inverse_kinematics.cpp # 逆运动学示例
        ├── src/wpb_mani_image_node.cpp            # 读取RGB图像
        ├── src/wpb_mani_cv_hsv.cpp                # OpenCV HSV颜色提取
        ├── src/wpb_mani_cv_follow.cpp             # 视觉跟踪闭环控制
        ├── src/wpb_mani_ar_track.cpp              # AR二维码识别
        ├── src/wpb_mani_pointcloud_node.cpp       # 读取点云
        ├── src/wpb_mani_pass_through.cpp          # PCL PassThrough滤波
        ├── src/wpb_mani_plane_detect.cpp          # PCL平面识别（RANSAC）
        ├── src/wpb_mani_object_detect.cpp         # 物体检测
        ├── src/wpb_mani_sphere_detect.cpp         # 球体检测
        ├── src/wpb_mani_cylinder_detect.cpp       # 圆柱体检测
        ├── src/wpb_mani_grab_demo.cpp             # 抓取完整流程
        ├── src/wpb_mani_grab_height.cpp           # 带高度判断的抓取
        ├── src/wpb_mani_waypoint_navigation.cpp   # 路点导航状态机示例
        ├── src/wpb_mani_mobile_manipulation.cpp   # 移动+抓取综合示例
        ├── src/wpb_mani_qr_demo.cpp               # QR码扫描
        └── maps/map.yaml                          # 预存导航地图
```

---

## 7. 系统架构分层

```
┌─────────────────────────────────────────────────┐
│         用户业务逻辑层（自定义包 / tutorials）      │
│  状态机 / 任务调度 / 业务流程控制                   │
└──────────────┬──────────────────┬────────────────┘
               │发布话题           │发布话题
               ▼                  ▼
┌──────────────────┐  ┌───────────────────────────┐
│ waterplus_map_   │  │    wpb_mani_behaviors      │
│ tools            │  │ boxes_3d（PCL聚类定位）    │
│ 路点导航中间件    │  │ grab_box（抓取行为封装）   │
└──────┬───────────┘  └──────────┬────────────────┘
       │依赖                      │依赖MoveIt!
       ▼                          ▼
┌──────────────────────────────────────────────────┐
│   规划层：wpb_local_planner + wpb_mani_moveit_config│
│   move_base（全局A*+自定义局部规划）+ MoveIt!       │
└──────────────────┬───────────────────────────────┘
                   │依赖
                   ▼
┌──────────────────────────────────────────────────┐
│   硬件/仿真驱动层                                   │
│   真机: wpb_mani_bringup（串口+雷达+Kinect驱动）    │
│   仿真: wpb_mani_simulator（Gazebo插件）            │
└──────────────────┬───────────────────────────────┘
                   │依赖
                   ▼
┌──────────────────────────────────────────────────┐
│   模型描述层：wpb_mani_description（URDF/XACRO）   │
└──────────────────────────────────────────────────┘
```

**开发原则**：业务代码只与行为封装层的简洁话题交互，不直接操作底层驱动。
- 导航：发布 `String` → `/waterplus/navi_waypoint`，监听 `/waterplus/navi_result`
- 抓取：发布 `Pose` → `/wpb_mani/grab_box`，监听 `/wpb_mani/grab_result`

---

## 8. 全部 23 个实验对照表（官方指导书）

| 编号 | 实验名称 | 核心 Launch / 命令 | 所属包 | 关键知识点 |
|---|---|---|---|---|
| 1 | 初识 ROS | `roslaunch wpb_mani_description urdf.launch` | `wpb_mani_description` | roslaunch/rosrun/rostopic/rqt_graph 基础操作 |
| 2 | 启智MANI运动控制 | `roslaunch wpb_mani_bringup js_ctrl.launch` | `wpb_mani_bringup` | 真机启动，手柄遥控，节点网络理解 |
| 3 | 运动控制Node编程 | `roslaunch wpb_mani_simulator wpb_mani_gazebo.launch` | 自建包 | 创建Package，编写发布 `/cmd_vel` 节点，catkin_make |
| 4 | 激光雷达数据获取 | `roslaunch wpb_mani_bringup base_lidar.launch` | 自建包 | 订阅 `/scan`，解析 `ranges[]` 数组，获取指定角度距离 |
| 5 | 激光雷达自主避障 | `roslaunch wpb_mani_bringup base_lidar.launch` | 自建包 | 状态机：检测前方最小距离，触发转向指令 |
| 6 | SLAM建图与导航仿真 | `roslaunch wpb_mani_simulator scene_1_gmapping.launch` | `wpb_mani_simulator` | Gmapping建图，map_saver保存，navigation.launch导航 |
| 7 | SLAM建图与导航真机 | `roslaunch wpb_mani_bringup normal.launch` + `gmapping.launch` | `wpb_mani_tutorials` | 真机建图全流程，手柄控制遍历环境 |
| 8 | 自主导航Node编程 | `roslaunch wpb_mani_tutorials nav.launch` | 自建包 | `actionlib::SimpleActionClient`，发送 `MoveBaseGoal` |
| 9 | MapTools插件使用 | `roslaunch waterplus_map_tools add_waypoint_mani.launch` | `waterplus_map_tools` | Rviz交互标注路点，字符串触发多点巡航 |
| 10 | 彩色相机数据获取 | `roslaunch wpb_mani_bringup kinect_test.launch` | 自建包 | 订阅 `/rgb/image_raw`，cv_bridge 转换，imshow显示 |
| 11 | 平面视觉特征提取 | `roslaunch wpb_mani_bringup kinect_test.launch` | 自建包 | BGR→HSV，inRange阈值，findContours，moments质心 |
| 12 | 平面视觉目标跟踪 | `roslaunch wpb_mani_bringup kinect_test.launch` | 自建包 | 质心X偏差→P控制器→`angular.z`，视觉闭环跟踪 |
| 13 | 二维码识别与定位 | `roslaunch wpb_mani_simulator ar_track.launch` | 自建包 | 订阅 `/ar_pose_marker`，获取标签ID和三维位姿 |
| 14 | 立体视觉数据获取 | `roslaunch wpb_mani_bringup kinect_test.launch` | 自建包 | 订阅 `/points2`，pcl::fromROSMsg，Rviz点云可视化 |
| 15 | 立体视觉平面识别 | `roslaunch wpb_mani_bringup kinect_test.launch` | 自建包 | PCL PassThrough滤波，SACSegmentation RANSAC平面拟合 |
| 16 | 物体空间定位 | `roslaunch wpb_mani_bringup kinect_test.launch` | `wpb_mani_behaviors` | EuclideanClusterExtraction聚类，计算质心xyz坐标 |
| 17 | 机械臂关节控制编程 | `roslaunch wpb_mani_simulator moveit.launch` | 自建包 | 发布 `/wpb_mani/joint_ctrl`，直接控制4关节+手爪 |
| 18 | 机械臂MoveIt!系统 | `roslaunch wpb_mani_tutorials moveit.launch` | `wpb_mani_tutorials` | MoveIt! 界面操作，规划组，碰撞检测，可视化轨迹 |
| 19 | 机械臂运动学正解 | `roslaunch wpb_mani_tutorials moveit.launch` | 自建包 | `move_group.getCurrentPose()`，关节角→末端位姿 |
| 20 | 机械臂运动学逆解 | `roslaunch wpb_mani_tutorials moveit.launch` | 自建包 | `setPoseTarget()`，目标位姿→关节角→执行轨迹 |
| 21 | 物品抓取Node编程 | `roslaunch wpb_mani_simulator grab_box.launch` | 自建包 | 订阅 `/wpb_mani/boxes_3d`，发布 `/wpb_mani/grab_box`，监听 result |
| 22 | 多点巡航编程 | `roslaunch wpb_mani_tutorials nav.launch` | 自建包 | 多路点状态机，监听 navi_result 后切换下一路点 |
| 23 | 移动抓取综合实训 | `roslaunch wpb_mani_tutorials mobile_manipulation.launch` | 自建包 | **终极实验**：导航→视觉定位→抓取→搬运全流程 |

---

## 9. 二次开发指南

### 9.1 新建自定义包（标准流程）

```bash
cd ~/mani_ws/src
catkin_create_pkg my_pkg roscpp geometry_msgs sensor_msgs std_msgs
# 在 my_pkg/src/ 下新建 .cpp 文件
# 在 CMakeLists.txt 末尾添加：
# add_executable(my_node src/my_node.cpp)
# add_dependencies(my_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(my_node ${catkin_LIBRARIES})
cd ~/mani_ws && catkin_make
```

### 9.2 导航任务状态机模板（推荐写法）

```cpp
// 不要直接用 actionlib，用 waterplus_map_tools 字符串接口
ros::Publisher wp_pub = n.advertise<std_msgs::String>("/waterplus/navi_waypoint", 1);

// 订阅导航结果
void naviResultCallback(const std_msgs::String::ConstPtr& msg) {
    if (msg->data == "done") {
        // 切换到下一个状态
    }
}

// 发送导航指令
std_msgs::String wp_msg;
wp_msg.data = "PickPoint";   // 与 MapTools 中标注的路点名称一致
wp_pub.publish(wp_msg);
```

### 9.3 视觉开发依赖

```xml
<!-- package.xml 中添加 -->
<depend>cv_bridge</depend>
<depend>image_transport</depend>
<depend>sensor_msgs</depend>

<!-- CMakeLists.txt 中 find_package 添加 -->
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport sensor_msgs OpenCV)
```

### 9.4 PCL 点云开发依赖

```xml
<!-- package.xml -->
<depend>pcl_ros</depend>
<depend>pcl_conversions</depend>

<!-- CMakeLists.txt -->
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(my_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
```

### 9.5 MoveIt! 开发依赖

```xml
<!-- package.xml -->
<depend>moveit_ros_planning_interface</depend>

<!-- CMakeLists.txt -->
find_package(catkin REQUIRED COMPONENTS moveit_ros_planning_interface)
```

---

## 10. 已知问题与修复记录

| 问题现象 | 根本原因 | 修复方案 | 影响文件 |
|---|---|---|---|
| Rviz 中 LaserScan 无数据，Gmapping 不建图，`/scan` 无消息 | 仿真雷达发布在 `/robot/scan_raw`，但 `wpb_mani_sim_lidar_filter` 节点默认订阅 `/scan_raw`（命名空间不一致） | 在 launch 文件的 lidar filter 节点中添加 `<param name="sub_topic" value="/robot/scan_raw"/>` | `mani_ws/src/wpb_mani/wpb_mani_simulator/launch/scene_1_gmapping.launch`（已修复） |

---

## 11. 附录（官方指导书附录内容）

| 附录 | 内容 |
|---|---|
| 附录一 | 开发机 ROS 环境安装（Ubuntu 18.04 + Melodic），参考官方指导书第301页 |
| 附录二 | VS Code 安装配置，参考官方指导书第303页 |
| 附录三 | 机器人系统恢复（Jetson NX 刷机），参考官方指导书第307页 |
| 附录四 | Azure Kinect 视角调整（修改 `wpb_mani.yaml` 中的相机安装角度参数） |
| 附录五 | 遥控手柄摇杆方向调整（修改 launch 文件中的 `axis_linear`/`axis_angular` 参数） |
