# AGV QMIX 智能仓储系统 — 开发计划

> 最后同步：2026-05-19
> 目标：网络技术挑战赛 国一特等奖
> 比赛时间：2026年6月中旬（约4周）

---

## 项目定位

多AGV协同调度系统，技术路线：
- **高层决策**：QMIX + Attention 机制（任务分配）
- **低层导航**：A* 路径规划（避障导航）
- **仿真平台**：Gazebo + ROS1（Noetic 开发机 / Melodic 实车）
- **实车平台**：启智MANI，Jetson NX 边缘计算

**核心创新点**：在 AgentQNet 中嵌入 Multi-Head Attention，让每台机器人决策时显式感知所有货架状态，输出 Attention 权重可可视化。

---

## 当前代码状态

| 文件 | 状态 | 说明 |
|---|---|---|
| `warehouse_env.py` | 旧版，待替换 | 10×10蛇形走廊，2-3台车，4货架，A/B两类 |
| `qmix_network.py` | 旧版，待升级 | 普通MLP，无Attention |
| `train.py` | 旧版，待更新 | N_AGENTS=2，N_SHELVES=4，对应旧环境 |
| `checkpoints_v4/` | 旧版训练结果 | ep2500最优，新环境需重训 |
| `evaluate.py` | 待适配 | 需对应新环境接口 |

---

## 新地图设计（已确认）

**20×16 十字形 + 多走廊**

```
     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
R0:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R1:  ■  ■  ■  ■  ■  ■  ■  .  S  .  .  S  .  ■  ■  ■  ■  ■  ■  ■  ← 北臂货架
R2:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R3:  ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 北臂瓶颈（2通道）
R4:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R5:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
R6:  .  S  .  .  S  .  .  .  .  .  .  .  .  .  .  S  .  .  S  .   ← 西/东臂上层货架
R7:  .  .  ■  ■  ■  .  .  .  .  .  .  .  .  .  .  ■  ■  ■  .  .   ← 走廊隔墙
R8:  .  S  .  .  S  .  .  .  .  .  .  .  .  .  .  S  .  .  S  .   ← 西/东臂下层货架
R9:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
R10: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R11: ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 南臂瓶颈（2通道）
R12: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R13: ■  ■  ■  ■  ■  ■  ■  A  .  B  .  C  .  ■  ■  ■  ■  ■  ■  ■  ← 投递区
R14: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R15: ■  ■  ■  ■  ■  ■  ■  .  0  1  .  2  3  ■  ■  ■  ■  ■  ■  ■  ← 4台车起点
```

**关键参数：**
- 货架（10个）：(1,8) (1,11) (6,1) (6,4) (8,1) (8,4) (6,15) (6,18) (8,15) (8,18)
- 投递区：A=(13,7)  B=(13,9)  C=(13,11)
- 机器人起点：(15,8) (15,9) (15,11) (15,12)
- n_actions = 13（10货架 + 3投递区）
- obs_dim ≈ 86，state_dim ≈ 74

**设计意图：**
- 瓶颈 R3/R11 各只有 2 格通道 → 强制 4 车协商进出顺序
- 西/东臂走廊隔墙 R7 → 货架必须绕行，路径多样
- 3类货物 + 优先级订单 → 任务复杂度足够

---

## 新网络设计（待实现）

### AttentionAgentQNet

```
输入：自身状态(self_feat) + 其他车状态(other_feats) + 货架状态(shelf_feats)
  ↓
self_feat → Linear → Query (d_k)
shelf_feats → Linear → Key (d_k)
shelf_feats → Linear → Value (d_v)
  ↓
Multi-Head Attention → 货架上下文向量 context
  ↓
[self_feat || other_feats_pooled || context] → MLP → Q值 (n_actions=13)
```

**关键设计：**
- Attention 权重形状 `[n_heads, n_shelves]` → 训练时可输出可视化
- 保持参数共享（4台车用同一网络）
- QMixNet 结构不变，只替换 AgentQNet

---

## 阶段验收条件（硬性门槛）

> **规则：以下每一项必须全部达到，才允许进入下一阶段。未达到禁止推进。**

---

### 阶段一验收：格子世界训练完成

#### G1 — 环境正确性（代码层面）
- [x] `warehouse_env.py` 连续运行 **1000 episode** 不报错、不死锁、不卡死
- [x] A* 能为每台车规划出有效路径（所有货架/投递区均可到达，无孤立不可达格）
- [x] 碰撞处理正确：同一时刻不存在两台车占据同一格
- [x] 订单系统正确：A/B/C 三类货物与对应投递区匹配，错投不给奖励

#### G2 — 网络正确性（代码层面）
- [x] `AttentionAgentQNet` forward pass 输出形状正确 `(batch, 13)`，无 NaN/Inf
- [x] Attention 权重对 10 个货架做 softmax，每行之和 = 1.0（误差 < 1e-5）
- [x] 梯度能正常反传（`loss.backward()` 后所有参数 `.grad` 不为 None）

#### G3 — 训练收敛（定量指标）
- [x] Attention-QMIX 训练曲线**明显上升**，最后 200 episode 平均奖励 ≥ 随机策略的 **3 倍**
- [x] 每 episode 平均完成任务数（MAX_STEPS=300，4台车）**≥ 8 个**（贪婪评估 19.0 次）
- [x] 训练已收敛：最后 200 episode 的平均奖励 ≥ 全程最高 200 episode 均值的 **85%**

#### G4 — 对照实验（比赛说服力）
- [x] 同等训练轮数下，**Attention-QMIX 最终奖励 > 普通 MLP-QMIX**（高 6.5%，已接受）
- [x] 对比图已保存：reward_curve_compare.png（训练曲线 + 柱状对比图）

#### G5 — Attention 可解释性
- [x] 能输出某一 episode 的 attention 权重序列，肉眼可见不同时刻权重分布有变化（attention_heatmap.png，25 次投递，4 agent 热力图）

---

### 阶段二验收：Gazebo 联调完成（暂定，后续细化）
- [ ] 加载 checkpoint，4 台车在 Gazebo 中能完成至少一轮完整取货-投递流程
- [ ] RViz 中可看到 Attention 权重的可视化标记

---

## 开发任务清单

### 阶段一：环境重建（当前）
- [ ] **warehouse_env.py 重写** — 20×16十字地图，4台车，10货架，3类货物，动态订单队列
- [ ] **train.py 参数更新** — N_AGENTS=4，N_SHELVES=10，n_actions=13，obs_dim/state_dim更新
- [ ] **跑通训练循环** — 确认无 bug，reward 能上升

### 阶段二：Attention 创新（第1-2周）
- [ ] **AttentionAgentQNet 实现** — 替换 qmix_network.py 中的 AgentQNet
- [ ] **对照基线训练** — 同参数跑普通 MLP 版本，记录奖励曲线
- [ ] **Attention-QMIX 训练** — 跑到收敛，保存最优 checkpoint

### 阶段三：对照实验（第2-3周）
- [ ] **消融实验** — Attn-QMIX vs 普通QMIX vs 随机分配，各跑5次取均值
- [ ] **效率对比** — 完成任务数/episode，平均等待时间，瓶颈冲突率
- [ ] **Attention 可视化** — 输出每步 attention 热力图，制作动图

### 阶段四：Gazebo 联调（第3周）
- [ ] **新 Gazebo world** — 对应十字地图布局（在 catkin_ws 中创建）
- [ ] **robot_agent.py 适配** — 对应新的 4 台车 + 10 货架接口
- [ ] **cargo_manager.py 适配** — 3类货物 + 动态订单
- [ ] **端到端跑通** — 从训练 checkpoint 加载，Gazebo 中实际运行

### 阶段五：实车部署（第4周）
- [ ] **TorchScript 导出** — `torch.jit.script(model)` 导出推理模型
- [ ] **Jetson NX 推理节点** — ROS节点，订阅传感器，发布 waypoint 目标
- [ ] **实车调试** — 简化场景（2车+4货架）先跑通

---

## 对照实验设计（比赛必备）

| 方案 | 说明 | 预期结果 |
|---|---|---|
| 随机分配 | 随机选货架/投递区 | 基准最低 |
| 普通 QMIX | 当前 MLP 版本 | 中等 |
| **Attention-QMIX** | 本方案 | 最优 |

指标：
- 每 episode 完成任务数（越高越好）
- 平均任务完成时间（越低越好）
- 瓶颈处冲突次数（越低说明协同越好）

---

## 风险点

| 风险 | 概率 | 应对 |
|---|---|---|
| 4车十字地图训练不收敛 | 中 | 先跑2车版本验证环境，再扩展到4车 |
| Gazebo联调时间不够 | 中 | 优先保证仿真展示效果，实车作为加分项 |
| Attention 效果不显著 | 低 | 调整 n_heads、d_k，或改用门控注意力 |

---

## 文件结构

```
qmix_project/
├── PLAN.md              ← 本文件
├── warehouse_env.py     ← 待重写（十字地图版）
├── qmix_network.py      ← 待升级（加入Attention）
├── train.py             ← 待更新（新环境参数）
├── evaluate.py          ← 待适配
├── astar_demo.py        ← A* 验证脚本
├── checkpoints_v4/      ← 旧版训练结果（保留备用）
└── checkpoints_v5/      ← 新版训练结果（待创建）
```

---

## 参考接口

**waterplus_map_tools 导航：**
```
发布：/waterplus/navi_waypoint (geometry_msgs/Pose)
抓取：/wpb_mani/grab_box (std_msgs/String)
```

**实车账号：** robot / 6，串口 /dev/ftdi

**ROS版本：** 开发机 Noetic / 实车 Melodic（Jetson NX）
