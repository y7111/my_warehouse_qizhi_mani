# RViz 多机器人导航修复说明

## 问题描述
在启动 `two_robots_nav.launch` 运行多智能体仓储系统时，RViz 会产生多个问题，主要包括：
- 成本地图显示异常
- TF 坐标变换错误
- 导航目标设置问题

## 修复内容

### 1. costmap_common_params.yaml
**问题**: 激光雷达主题使用绝对路径 `/scan`，在多机器人命名空间中会导致订阅失败
**修复**: 将 `topic: /scan` 改为 `topic: scan`，使其在命名空间下正确工作
```yaml
# 修改前
topic: /scan

# 修改后
topic: scan
```

### 2. global_costmap_params.yaml
**问题**: 缺少成本地图层定义，导致某些 RViz 显示问题
**修复**: 添加了明确的成本地图层配置
```yaml
plugins:
  - {name: static_layer, type: 'costmap_2d::StaticLayer'}
  - {name: obstacle_layer, type: 'costmap_2d::ObstacleLayer'}
  - {name: inflation_layer, type: 'costmap_2d::InflationLayer'}
```

### 3. local_costmap_params.yaml
**问题**: 缺少本地成本地图层定义
**修复**: 添加了本地成本地图层配置
```yaml
plugins:
  - {name: obstacle_layer, type: 'costmap_2d::ObstacleLayer'}
  - {name: inflation_layer, type: 'costmap_2d::InflationLayer'}
```

### 4. two_robots_nav.launch
**问题**: RViz 没有显示输出日志，难以调试
**修复**: 添加 `output="screen"` 参数
```xml
<!-- 修改前 -->
<node name="rviz" pkg="rviz" type="rviz" args="-d $(find wpb_mani_simulator)/rviz/nav.rviz" />

<!-- 修改后 -->
<node name="rviz" pkg="rviz" type="rviz" args="-d $(find wpb_mani_simulator)/rviz/nav.rviz" output="screen" />
```

### 5. nav.rviz
**问题**: SetGoal 工具只能为单一机器人发送导航目标
**修复**: 添加了两个 SetGoal 工具，分别用于 robot1 和 robot2
```yaml
# 原配置
- Class: rviz/SetGoal
  Topic: /move_base_simple/goal

# 修复后
- Class: rviz/SetGoal
  Topic: /robot1/move_base_simple/goal
- Class: rviz/SetGoal
  Topic: /robot2/move_base_simple/goal
```

## 修改的文件列表
1. `/home/jasper/catkin_ws/src/wpb_mani/wpb_mani_tutorials/nav_lidar/costmap_common_params.yaml`
2. `/home/jasper/catkin_ws/src/wpb_mani/wpb_mani_tutorials/nav_lidar/global_costmap_params.yaml`
3. `/home/jasper/catkin_ws/src/wpb_mani/wpb_mani_tutorials/nav_lidar/local_costmap_params.yaml`
4. `/home/jasper/catkin_ws/src/wpb_mani/wpb_mani_simulator/launch/two_robots_nav.launch`
5. `/home/jasper/catkin_ws/src/wpb_mani/wpb_mani_simulator/rviz/nav.rviz`

## 使用方法
```bash
cd /home/jasper/catkin_ws
source devel/setup.bash
roslaunch wpb_mani_simulator two_robots_nav.launch
```

## 预期结果
- 两个机器人在 Gazebo 中正常显示
- RViz 中正确显示两个机器人的模型、激光雷达数据和成本地图
- 可以通过 RViz 的 SetGoal 工具为两个机器人分别设置导航目标
- 没有关键的 TF 或成本地图错误

## 故障排除
如果还有问题，可以尝试：
1. 清理构建：`catkin_make clean && catkin_make`
2. 检查 ROS 主控节点：`roscore` 在单独的终端运行
3. 查看 RViz 的错误消息（Status 面板）
4. 检查 TF 树：`rosrun tf view_frames`
