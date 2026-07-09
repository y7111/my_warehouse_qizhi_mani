# AGV QMIX 项目对话总结

> 最后更新：2026-05-22（供新对话快速恢复上下文）

---

## 一、项目背景

- **比赛时间**：2026 年 6 月中旬
- **目标**：多 AGV 协同货物搬运系统，仿真训练后部署到实车（Jetson NX）边缘推理
- **硬件**：4 辆万向轮 AGV，配机械臂，Jetson NX，ROS1 Melodic（实车）/ Noetic（开发机）
- **场景**：20×16 十字形地图，10 个货架，A/B/C 三类货物，3 个投递区，4 台车协同搬运

---

## 二、整体架构（已确定）

```
QMIX（高层任务分配）—— AttentionAgentQNet，双路注意力，4台车共享参数
  决策：每辆车下一步去哪个货架 / 投递区（n_actions=13）
  触发：仅当某辆车完成当前任务时
        ↓
A* / move_base（低层路径规划）
  负责：具体逐格导航，绕行障碍
```

**训练流程：**

```
纯 Python 20×16 网格仿真（已完成 ✅）
        ↓ checkpoints_v5_attn/qmix_ep800.pt
Gazebo 仿真验证（接入 ROS） ← 当前阶段
        ↓ 部署
Jetson NX 实车推理
```

---

## 三、QMIX 训练（已完成 ✅）

### 3.1 最终超参数

| 参数 | 值 |
|---|---|
| N_AGENTS | 4 |
| N_SHELVES | 10 |
| obs_dim | 87 |
| state_dim | 74 |
| n_actions | 13（货架 0-9 + 投递区 A/B/C） |
| MAX_STEPS | 300 |
| N_EPISODES | 6000 |
| EPS_START/END/DECAY | 1.0 / 0.10 / 1500 |
| LR | 2e-4 |
| UPDATE_INTERVAL | 4 |

### 3.2 训练结果（贪婪评估，ε=0，100 episode 均值）

| 方案 | 平均送达次数 |
|---|---|
| 随机策略 | 4.3 |
| MLP-QMIX（ep1400） | 17.8 ± 0.51 |
| **Attention-QMIX（ep800）** | **19.0 ± 0.47** |

**最优 checkpoint**：`~/qmix_project/checkpoints_v5_attn/qmix_ep800.pt`

> 注意：训练奖励在 ep800 之后仍会继续上升（因为 ε 在减小），但贪婪策略质量（ε=0 评估）在 ep800 最优。ep1500+ 训练曲线更高是 ε 减少的噪声减少效果，不代表策略变好。

### 3.3 阶段一验收（G1-G5 全部通过 ✅）

| 验收项 | 状态 | 说明 |
|---|---|---|
| G1 环境正确性 | ✅ | 6000 episode 无报错，A* 路径有效，碰撞处理正确 |
| G2 网络正确性 | ✅ | forward pass 输出 (batch,13) 无 NaN，权重 softmax 正确 |
| G3 训练收敛 | ✅ | ep100: -40 → ep1500: 153，清晰上升曲线 |
| G4 对照实验 | ✅（接受） | Attention 比 MLP 高 6.5%（目标 10%，已接受） |
| G5 可解释性 | ✅ | attention_heatmap.png 证明权重动态变化，非均匀分布 |

---

## 四、代码文件状态

| 文件 | 位置 | 状态 |
|---|---|---|
| `warehouse_env.py` | `~/qmix_project/` | ✅ 完成（20×16 十字地图，4 车，10 货架，A/B/C） |
| `qmix_network.py` | `~/qmix_project/` | ✅ 完成（AttentionAgentQNet + MLPAgentQNet + QMixNet） |
| `train.py` | `~/qmix_project/` | ✅ 完成（v5，EPS_DECAY=1500，纯随机探索） |
| `checkpoints_v5_attn/` | `~/qmix_project/` | ✅ **最优：ep800.pt**（19.0 次/ep） |
| `reward_curve_compare.png` | `~/qmix_project/` | ✅ 训练曲线 + 对比柱状图 |
| `attention_heatmap.png` | `~/qmix_project/` | ✅ Attention 权重热力图（G5） |

---

## 五、新地图设计（20×16 十字形）

```
     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
R0:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R1:  ■  ■  ■  ■  ■  ■  ■  .  S  .  .  S  .  ■  ■  ■  ■  ■  ■  ■  ← 北臂货架
R3:  ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 瓶颈（2通道）
R6:  .  S  .  .  S  .  .  ...  .  .  .  S  .  .  S  .   ← 西/东臂上层货架
R8:  .  S  .  .  S  .  .  ...  .  .  .  S  .  .  S  .   ← 西/东臂下层货架
R11: ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 瓶颈（2通道）
R13: ■  ■  ■  ■  ■  ■  ■  A  .  B  .  C  .  ■  ...  ■  ← 投递区
R15: ■  ■  ■  ■  ■  ■  ■  .  0  1  .  2  3  ■  ...  ■  ← 4台车起点
```

**关键坐标：**

- 货架（10个）：(1,8)(1,11)(6,1)(6,4)(8,1)(8,4)(6,15)(6,18)(8,15)(8,18)
- 投递区：A=(13,7)  B=(13,9)  C=(13,11)
- 机器人起点：(15,8)(15,9)(15,11)(15,12)

---

## 六、QMIX 网络结构（AttentionAgentQNet）

```
obs 拆分（obs_dim=87）：
  self_feat   (19维)：自身位置(2) + 携带状态 one-hot(4) + 目标 one-hot(13)
  other_feats (18维)：其他 3 台车的位置(2)+携带(4) × 3
  shelf_feats (50维)：10 货架的货物类型 one-hot(3)+位置(2) × 10

双路注意力：
  货架注意力：self_feat → Q，shelf_feats → K/V → shelf_ctx (32维)
  车间注意力：self_feat → Q，other_feats → K/V → agent_ctx (32维)

货架头：[self_feat(19) || shelf_ctx(32) || agent_ctx(32)] → Q(0..9)
投递头：self_feat(19) only → Q(10..12)（隔离 attention 干扰）
```

---

## 七、已踩坑（勿重蹈）

| 坑 | 现象 | 结论 |
|---|---|---|
| EPS_DECAY=6000 太慢 | ep1100 时 ε=0.835，几乎全随机，看不到学习 | 改为 1500 |
| 空货架惩罚 -2.0 | 训练奖励 153→118 崩溃，贪婪评估 13.5（比随机差） | 删除，保持原奖励结构 |
| MAX_STEPS=600 | 比 300 差 5 倍效率，惩罚积累淹没稀疏正奖励 | 保持 300 |
| obs 引用污染（旧版） | Q 值全收敛到同一值（≈1.43），什么都没学到 | 已修复，存副本 |
| 梯度更新过频（旧版） | ep1000→ep2000 送达 9.64→2.96 灾难性遗忘 | UPDATE_INTERVAL=4 |
| link_attacher attach 货架 | ODE 物理线程挂起，_attach_srv 永不返回 | 删除模型，用 carrying 状态追踪 |

---

## 八、下一步：阶段二 Gazebo 联调

**目标**：加载 `checkpoints_v5_attn/qmix_ep800.pt`，4 台车在 Gazebo 中完成完整取货-投递闭环。

**待完成：**

- [ ] 新建 Gazebo world（对应 20×16 十字地图，10 货架，3 投递区）
- [ ] `robot_agent.py` 适配（4 台车 + 10 货架 + A/B/C 三类货物）
- [ ] `cargo_manager.py` 适配（3 类货物 + 动态订单）
- [ ] 端到端联调 launch 文件
- [ ] （可选）RViz 中 Attention 权重可视化标记

**工作空间**：`~/mani_ws/`（见 `~/mani_ws/CLAUDE.md`）

---

## 九、新对话时告诉 Claude 的内容

> 把以下内容发给新对话的 Claude：

```
我在做一个多 AGV 仓储 QMIX 调度项目，6月中旬比赛。
请先读以下文件了解当前进展：
- ~/AGV_QMIX_对话总结.md（所有上下文，重点看第三节训练结果和第八节待做事项）
- ~/qmix_project/PLAN.md（阶段验收条件，阶段一已全部完成）
- ~/mani_ws/CLAUDE.md（Gazebo 工作空间指引）

当前任务：阶段二 Gazebo 联调。
最优模型：~/qmix_project/checkpoints_v5_attn/qmix_ep800.pt
读完后告诉我你的理解，然后我们开始。
```
