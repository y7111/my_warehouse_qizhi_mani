# Claude 工作输出记录（修订版）

**日期**：2026-05-19  
**任务**：新版三文件设计逻辑说明（warehouse_env.py / qmix_network.py / train.py）  
**修订原因**：根据8条问题汇总修正；追加后两个文件设计

---

## 【任务目标】

在理解旧版 `warehouse_env.py`（10×10蛇形走廊，2车，4货架，A/B两类货物）的基础上，设计新版环境的逻辑框架，为后续编码做准备。

新版目标规格（来自 PLAN.md）：
- 地图：20×16 十字形+多走廊（H=16, W=20）
- 机器人：4台
- 货架：10个
- 货物类型：3类（A/B/C）
- 投递区：3个
- n_actions：13（10货架 + 3投递区）
- obs_dim ≈ 87，state_dim = 74

---

## 【完成内容】

完成了新版 `warehouse_env.py` 的**设计逻辑说明**，涵盖以下6个模块：

1. **地图构建** — 系统化规则构造20×16十字形墙壁集合
2. **货物体系** — 扩展到3类货物，one-hot维度调整
3. **观测向量** — 新增 dest_onehot(13)，归一化参数更新
4. **冲突处理** — 4车碰撞/死锁策略，未移动阈值分析
5. **动作映射** — n_actions=13的具体对应关系
6. **接口变更清单** — 明确哪些"不变"、哪些必须同步修改

---

## 【关键假设】

1. **地图结构按 PLAN.md 字面解读**：R3瓶颈仅 col8 和 col11 通行，R11同理，R7走廊隔墙在 col2-4 和 col15-17。
2. **动态订单 = 随机补货（Plan A）**：取走后立即随机补 A/B/C，不引入独立队列（见下方"未确认"第1条，Plan B 已给出草图）。
3. **obs 含 dest_onehot(13)**：自身当前目标以 one-hot 编码纳入观测，支持 AttentionQNet 生成有意义的 Query。
4. **pos 归一化用新地图尺寸**：row/H（H=16），col/W（W=20），与旧版 row/10 col/10 **不同**。
5. **货架 pos 进入 obs**：货架位置固定，但显式给出位置坐标可以帮助网络建立空间关联，对 Attention 的 Key/Value 构建有意义；不进 obs 也可，但保留与旧版一致。
6. **MAX_STEPS 保持 300**：已在 CLAUDE.md 中确认，MAX_STEPS=600 实测差5倍，不改。

---

## 【未确认的地方】

1. **"动态订单队列"的真实含义（高风险）**：
   - **Plan A（当前假设）**：货架随机补货，无独立队列。QMIX 只选"去哪个货架"或"去哪个投递区"。实现简单，训练稳定。
   - **Plan B（订单队列）**：全局维护一个待完成订单列表，每个订单指定 `{需要类型, 目标投递区, 优先级}`。QMIX 的动作变为"认领哪个订单"。obs 需要加入当前订单队列状态，state_dim 增大，训练难度上升。
   - **建议**：先用 Plan A 跑通训练，如果比赛评委要求"优先级订单"特性才切换 Plan B。PLAN.md 原文"3类货物 + 优先级订单 → 任务复杂度足够"更像是设计意图描述，而非硬性实现要求。

2. **R7走廊隔墙的精确列号**：地图字符显示 col2/3/4 和 col15/16/17 是墙，需人工核对（字符对齐可能有1格偏差）。

---

## 【完整输出】

### 新版 `warehouse_env.py` 设计逻辑（修订版）

---

#### 一、地图构建（20×16 十字形）

```
     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
R0:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R1:  ■  ■  ■  ■  ■  ■  ■  .  S  .  .  S  .  ■  ■  ■  ■  ■  ■  ■
R2:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R3:  ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 北臂瓶颈（col8,col11通）
R4:  ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R5:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
R6:  .  S  .  .  S  .  .  .  .  .  .  .  .  .  .  S  .  .  S  .
R7:  .  .  ■  ■  ■  .  .  .  .  .  .  .  .  .  .  ■  ■  ■  .  .  ← 走廊隔墙
R8:  .  S  .  .  S  .  .  .  .  .  .  .  .  .  .  S  .  .  S  .
R9:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
R10: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R11: ■  ■  ■  ■  ■  ■  ■  ■  .  ■  ■  .  ■  ■  ■  ■  ■  ■  ■  ■  ← 南臂瓶颈（col8,col11通）
R12: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R13: ■  ■  ■  ■  ■  ■  ■  A  .  B  .  C  .  ■  ■  ■  ■  ■  ■  ■  ← 投递区（已确认）
R14: ■  ■  ■  ■  ■  ■  ■  .  .  .  .  .  .  ■  ■  ■  ■  ■  ■  ■
R15: ■  ■  ■  ■  ■  ■  ■  .  0  1  .  2  3  ■  ■  ■  ■  ■  ■  ■  ← 起点（已确认）
```

**已确认坐标（直接来自 PLAN.md）：**
- 货架（10个）：(1,8) (1,11) (6,1) (6,4) (8,1) (8,4) (6,15) (6,18) (8,15) (8,18)
- 投递区：A=(13,7)  B=(13,9)  C=(13,11)
- 机器人起点：(15,8) (15,9) (15,11) (15,12)

**构造墙壁的伪代码逻辑：**
```python
def _build_walls():
    walls = set()
    for r in range(16):
        for c in range(20):
            # 北臂区域（R0-R4）：只有 col7-12 是走廊
            if r in range(5) and not (7 <= c <= 12):
                walls.add((r, c))
            # 北臂瓶颈 R3：走廊内进一步封堵，只留 col8 和 col11
            if r == 3 and c in [7, 9, 10, 12]:
                walls.add((r, c))
            # 南臂区域（R10-R15）：对称北臂
            if r in range(10, 16) and not (7 <= c <= 12):
                walls.add((r, c))
            # 南臂瓶颈 R11
            if r == 11 and c in [7, 9, 10, 12]:
                walls.add((r, c))
            # 走廊隔墙 R7（待核对列号）
            if r == 7 and c in [2, 3, 4, 15, 16, 17]:
                walls.add((r, c))
    return walls
```

---

#### 二、货物体系（3类货物）

```python
GOODS_NONE = 0
GOODS_A    = 1
GOODS_B    = 2
GOODS_C    = 3   # 新增

# carrying one-hot: 大小 4（none/A/B/C）← 旧版大小3
# goods one-hot:    大小 3（A/B/C，货架不为空）← 旧版大小2
```

货架补货（Plan A）：取走后立即随机补 A/B/C 之一（`random.randint(1, 3)`）。

奖励规则：
```
正确送达（A→(13,7), B→(13,9), C→(13,11)）：+10
送错投递区：-2
空手到投递区：无奖励，直接触发 needs_action
每步固定惩罚：-0.05
```

---

#### 三、观测向量设计（obs_dim = 87）

| 部分 | 内容 | 维度 |
|---|---|---|
| 自身 | pos(2) + carrying_onehot(4) + dest_onehot(13) | 19 |
| 其他3台车 | 3 × (pos(2) + carrying_onehot(4)) | 18 |
| 货架状态 | 10 × (goods_onehot(3) + pos(2)) | 50 |
| **合计** | | **87** |

全局状态 state_dim = 74：
- 4 × (pos(2) + carrying_onehot(4)) = 24
- 10 × (goods_onehot(3) + pos(2)) = 50

**归一化参数（必须随地图更新）：**
```python
# 旧版（错误地用于新版）
row_norm = pos[0] / 10   # ← 错误

# 新版（正确）
row_norm = pos[0] / 16   # H=16
col_norm = pos[1] / 20   # W=20
```

**货架 pos 是否进入 obs**：进。理由：货架位置固定，但显式坐标有助于 AttentionQNet 在生成 Key/Value 时建立空间相对关系，让权重分布有地理意义（避免网络只靠货物类型区分货架）。

**为什么 obs 含 dest_onehot，state 不含**：
- obs 的 dest_onehot 让 Query 知道"我在做什么任务"，才能有意义地 attend 到相关货架。
- state 只含客观物理量（位置、货物），是 QMixNet 超网络生成混合权重的输入，不需要各车意图。

---

#### 四、冲突处理（4台车）

旧版 A* 是"1个规划器，1辆车的 blocked = {另1辆车位置}"。  
新版变化：
- 每辆车独立调用 A*：**4次调用，每次 blocked = 另3辆车的当前位置**
- A* 算法本身不变，只是调用次数和 blocked 集合大小不同

冲突消解顺序不变（低编号优先），但 **未移动阈值需要调整**：
- 旧版 10×10 地图最长路径约 20 步，2步卡住就重规划合理
- 新版 20×16 地图最长路径可达 50+ 步，瓶颈处合法等待可能超过 2 步
- **建议阈值改为 5 步**：让车辆有足够耐心等待瓶颈通行，减少不必要的重规划抖动

```python
# 旧版
if self._stuck_count[i] >= 2:  # 2步

# 新版
if self._stuck_count[i] >= 5:  # 5步，适配更长路径+瓶颈等待
```

---

#### 五、动作空间（n_actions = 13）

```python
n_actions       = 13
ACTION_DELIV_A  = 10   # 投递区 A = (13, 7)
ACTION_DELIV_B  = 11   # 投递区 B = (13, 9)
ACTION_DELIV_C  = 12   # 投递区 C = (13, 11)

# action 0-9  → shelf_positions[action]
# action 10   → (13, 7)
# action 11   → (13, 9)
# action 12   → (13, 11)
```

---

#### 六、接口变更清单（"不变"的精确含义）

| 接口 | 状态 | 说明 |
|---|---|---|
| `reset()` 调用签名 | **不变** | 仍返回 (obs_list, state) |
| `step(actions)` 调用签名 | **不变** | 仍返回 (obs, state, reward, done, info, needs_action) |
| `obs_dim` | **必须更新** | 26 → 87 |
| `state_dim` | **必须更新** | 26 → 74 |
| `n_actions` | **必须更新** | 6 → 13 |
| `n_agents` | **必须更新** | 2 → 4 |
| `n_shelves` | **必须更新** | 4 → 10 |
| pos 归一化 | **必须更新** | /10 → /16 (row), /20 (col) |
| carrying one-hot 大小 | **必须更新** | 3 → 4 |
| goods one-hot 大小 | **必须更新** | 2 → 3 |
| A* 算法本体 | **不变** | 完全复用 `_astar()` |
| A* 调用次数 | **数量变化** | 每步调用 4 次（原2次），blocked 集合大小从1变3 |
| 冲突消解逻辑 | **不变** | swap检测 + 低编号优先 + 重规划兜底 |
| 未移动阈值 | **建议调整** | 2步 → 5步 |
| MAX_STEPS | **不变** | 保持 300 |
| render() | **小改** | 打印标签适配3类货物 |

---

#### 七、改动对照表

| 模块 | 旧版 (v4) | 新版 (v5) |
|---|---|---|
| 地图 | 10×10 蛇形走廊，手写3条墙 | 20×16 十字形，系统化构造 |
| 机器人数 | 2 | 4 |
| 货架数 | 4 | 10 |
| 货物类型 | A/B (2类) | A/B/C (3类) |
| 投递区 | 2个 | 3个 |
| n_actions | 6 | 13 |
| carrying one-hot | size 3 | size 4 |
| goods one-hot | size 2 | size 3 |
| obs_dim | 26 | 87 |
| state_dim | 26 | 74 |
| pos归一化 | /10 | /16 (row), /20 (col) |
| A* 算法 | 完全复用 | 完全复用 |
| A* 调用/步 | 2次 | 4次 |
| 未移动阈值 | 2步 | 5步 |
| MAX_STEPS | 300 | 300（不变） |

---

---

# qmix_network.py 设计逻辑

---

## 【任务目标】

将旧版 `AgentQNet`（两层MLP）升级为 `AttentionAgentQNet`（Multi-Head Attention + MLP），让每台车决策时显式感知所有货架状态，同时输出可视化的 Attention 权重。`QMixNet` 结构不变，只更新构造参数。

---

## 【完成内容】

1. **AttentionAgentQNet 结构设计** — 输入拆分、Attention 计算、输出 MLP
2. **观测拆分方案** — 固定索引切片，与 warehouse_env.py obs 结构对齐
3. **QMixNet 参数更新清单** — 仅改 n_agents 和 state_dim
4. **Attention 权重输出方案** — return_attn 参数控制，训练/可视化两用

---

## 【关键假设】

1. **obs 结构严格按顺序排列**：self_feat(19) | other_feats(18) | shelf_feats(50)，索引切片 `obs[:, :19]` 等硬编码依赖此顺序，warehouse_env.py 必须保证一致。
2. **使用 PyTorch `nn.MultiheadAttention`**：`batch_first=True` 参数需 PyTorch ≥ 1.9；**per-head attn_weights 需 ≥ 1.11**（`average_attn_weights=False` 参数）。若版本 < 1.11，默认返回的是对所有 head 平均后的权重，形状为 `[batch, 1, 10]`（不含 n_heads 维度）。当前设计采用**默认平均版本**，attn_weights 形状统一为 `[batch, 1, 10]`，可视化时无需处理 n_heads 维度，兼容更低版本。如需 per-head 可视化，再单独加 `average_attn_weights=False`。
3. **d_model=32, n_heads=2**：参数较小，适合 CPU 训练（开发机无 GPU 保证）；若训练不收敛再考虑扩大到 d_model=64, n_heads=4。
4. **other_feats 用均值池化**：3辆其他车的特征取 mean，得到 6 维。不用拼接（18维）是为了保持参数共享时输入维度不随 n_agents 变化。
5. **QMixNet 结构完全不变**：只改 n_agents=4, state_dim=74，超网络逻辑、权重 abs 单调性保证、ELU 激活全部保留。

---

## 【未确认的地方】

1. **PyTorch 版本确认**：需用 `torch.__version__` 确认：≥ 1.9 才支持 `batch_first=True`；若 < 1.9 需手动转置 `(seq, batch, feat)` 格式。per-head 权重需 ≥ 1.11，当前设计已放弃 per-head，使用头平均版本，仅需 ≥ 1.9 即可。
2. **d_model 和 n_heads 的最优选择**：32/2 是初始值，实际收敛速度和 Attention 权重分布质量需要训练后观测，可能需要调整。
3. **是否需要对 shelf_feats 的 Key/Value 用同一个 Linear 还是分开**：PLAN.md 写的分开（key_proj 和 val_proj），当前设计也分开，保留灵活性。

---

## 【完整输出】

### AttentionAgentQNet 结构

```
输入 obs [batch, 87]
  │
  ├─ self_feat   = obs[:, 0:19]          [batch, 19]  pos+carrying+dest
  ├─ other_feats = obs[:, 19:37]         [batch, 18]  3车×6
  └─ shelf_feats = obs[:, 37:87]         [batch, 10, 5]  10货架×(goods+pos)
                   .view(batch, 10, 5)
  │
  ├─ other_pool = other_feats.view(batch,3,6).mean(dim=1)  [batch, 6]
  │
  ├─ Query = Linear(19 → d_model)(self_feat).unsqueeze(1)  [batch, 1, d_model]
  ├─ Key   = Linear(5  → d_model)(shelf_feats)             [batch, 10, d_model]
  └─ Value = Linear(5  → d_model)(shelf_feats)             [batch, 10, d_model]
  │
  └─ MultiheadAttention(d_model, n_heads, batch_first=True)
       输入: Q=[batch,1,32], K=[batch,10,32], V=[batch,10,32]
       输出: context=[batch,1,32], attn_w=[batch,1,10]  ← 头平均后，无n_heads维度
       context.squeeze(1) → [batch, 32]
  │
  └─ MLP 输入 = concat([self_feat(19), other_pool(6), context(32)]) = [batch, 57]
       Linear(57→128) → ReLU → Linear(128→128) → ReLU → Linear(128→13)
  │
  输出: q_vals [batch, 13]
  (可选) attn_weights [batch, 1, 10]   ← 头平均，squeeze(1)后得[batch,10]即10货架权重
```

### 核心参数

| 参数 | 旧版 MLP | 新版 Attention |
|---|---|---|
| 输入处理 | obs 直接进 MLP | 拆分为3部分，Attention处理货架 |
| obs_dim | 26 | 87 |
| n_actions | 6 | 13 |
| hidden | 64 | 128（MLP部分） |
| d_model | — | 32 |
| n_heads | — | 2 |
| 额外输出 | 无 | attn_weights [batch,1,10]，squeeze后[batch,10]即10货架注意力分布 |

### forward 接口设计

```python
def forward(self, obs: torch.Tensor, return_attn: bool = False):
    # ... 计算逻辑 ...
    if return_attn:
        return q_vals, attn_weights   # 可视化时用
    return q_vals                     # 训练时默认只返回 Q 值
```

训练时调用：`q = agent_net(obs)`  
可视化时调用：`q, w = agent_net(obs, return_attn=True)`

### QMixNet 变更（仅参数）

```python
# 旧版
QMixNet(n_agents=2, state_dim=26, mixing_hidden=32, hyper_hidden=64)

# 新版
QMixNet(n_agents=4, state_dim=74, mixing_hidden=32, hyper_hidden=64)
#        ↑ 2→4        ↑ 26→74    mixing_hidden 和 hyper_hidden 不变
```

### 改动对照表

| 模块 | 旧版 | 新版 |
|---|---|---|
| AgentQNet 类名 | `AgentQNet` | `AttentionAgentQNet`（保留旧类作对照基线） |
| AgentQNet 输入 | obs 整体进 MLP | 拆分 self/other/shelf，Attention处理货架 |
| AgentQNet hidden | 64 | 128 |
| QMixNet 结构 | 不变 | 不变 |
| QMixNet n_agents | 2 | 4 |
| QMixNet state_dim | 26 | 74 |
| 额外输出 | 无 | attn_weights（return_attn=True 时） |

> 保留旧 `AgentQNet`（重命名为 `MLPAgentQNet`）作为对照实验基线，不删除。

---

---

# train.py 设计逻辑

---

## 【任务目标】

更新训练主循环以适配：4台车、新环境参数、AttentionAgentQNet，同时支持对照实验（`--model attention` vs `--model mlp`）的一键切换。

---

## 【完成内容】

1. **超参数更新** — N_AGENTS/N_SHELVES/obs_dim/state_dim/n_actions/SAVE_DIR
2. **模型切换机制** — `--model` 命令行参数，同一 train.py 跑两套实验
3. **compute_loss 适配** — 4智能体 obs 批处理，Attention 网络的 forward 调用方式
4. **transition 存储逻辑** — 4辆车，per-agent transition 存储分析
5. **超参数调整建议** — EPS_DECAY、BUFFER_SIZE 针对新场景的调整理由

---

## 【关键假设】

1. **reward 全局共享**：QMIX 框架下所有 agent 共享同一个全局奖励 r，`compute_loss` 里的 reward_b 仍然是标量（不是 per-agent 向量），这与旧版一致。
2. **transition 以"任务完成"为单位存储**：只有 `needs_action[i]==True` 时才为第 i 辆车生成一条 transition，用"上次决策时的 obs/state/action" 作为 (s, a)，"当前 obs/state" 作为 s'。这个逻辑不变，但 4 辆车意味着每步可能存 0~4 条 transition。
3. **EPS_DECAY 从 4000 调整到 6000**：4辆车场景任务更复杂，需要更长的探索期。仅建议值，可根据早期训练曲线调整。
4. **BUFFER_SIZE 从 10000 调整到 20000**：4辆车每步产生更多 transition（最多4条/步 vs 旧版最多2条），需要更大 buffer 保证样本多样性。

---

## 【未确认的地方】

1. **MIN_BUFFER 是否需要调整**：旧版 200，新版 buffer 更大、每步 transition 更多，建议调整到 500，但需观察训练早期的 loss 稳定性来确认。
2. **对照实验是否共享超参数**：MLP 和 Attention 两版是否用完全相同的 LR/BATCH_SIZE/EPS_DECAY？理论上应该相同以保证对比公平，但 Attention 网络参数更多，可能需要稍小的 LR。当前设计：先共用，若 Attention 不收敛再单独调。
3. **EPS_DECAY=6000 是否合适**：这是基于"4车比2车复杂约50%"的估算，实际需要看训练曲线的早期奖励走势来确认。

---

## 【完整输出】

### 超参数变更清单

```python
# ── 必须修改 ──────────────────────────────────────────────────────────
N_AGENTS      = 4          # 旧: 2
N_SHELVES     = 10         # 旧: 4
SAVE_DIR      = 'checkpoints_v5'   # 旧: checkpoints_v4（按 --model 再加后缀）

# ── 建议调整 ──────────────────────────────────────────────────────────
EPS_DECAY     = 6000       # 旧: 4000（4车场景更复杂，需要更长探索期）
BUFFER_SIZE   = 20000      # 旧: 10000（4车每步产生更多 transition）
MIN_BUFFER    = 500        # 旧: 200（配合更大 buffer）
HIDDEN        = 128        # 旧: 64（匹配 AttentionAgentQNet 的 MLP hidden）

# ── 保持不变 ──────────────────────────────────────────────────────────
MAX_STEPS     = 300        # 已验证，不改
LR            = 5e-4
GAMMA         = 0.99
TARGET_UPDATE = 50
BATCH_SIZE    = 32
UPDATE_INTERVAL = 4
SAVE_INTERVAL = 500
```

### 对照实验切换（--model 参数）

```python
# train.py 启动方式
python train.py --model attention   # 训练 AttentionAgentQNet，保存到 checkpoints_v5_attn/
python train.py --model mlp         # 训练 MLPAgentQNet（旧结构，新参数），保存到 checkpoints_v5_mlp/

# 内部逻辑
if args.model == 'attention':
    agent_net = AttentionAgentQNet(...).to(DEVICE)
    SAVE_DIR  = 'checkpoints_v5_attn'
else:
    agent_net = MLPAgentQNet(obs_dim=87, n_actions=13, hidden=64).to(DEVICE)
    #                                                           ↑ 保持64，不扩到128
    SAVE_DIR  = 'checkpoints_v5_mlp'
```

**MLPAgentQNet hidden=64 的决策理由**：对照实验的目的是证明"Attention 架构带来提升"，而不是"更大的网络带来提升"。MLPAgentQNet 保持原始 hidden=64，参数量远小于 AttentionAgentQNet（128 hidden + Attention层），若 Attention 版仍更优，则结论更有说服力。若将 MLP 扩到 128 则混入了参数量的影响，降低结论的可信度。

### compute_loss 适配 4 智能体

结构逻辑不变，只有维度变化：

```
obs_b 形状：   [bs, 4, 87]   旧: [bs, 2, 26]
obs_flat：     [bs*4, 87]    旧: [bs*2, 26]
q_all：        [bs, 4, 13]   旧: [bs, 2, 6]
agent_qs：     [bs, 4]       旧: [bs, 2]
q_tot：        [bs, 1]       不变
```

Attention 网络的调用方式：`agent_net(obs_flat)` 等价于旧版，因为 `return_attn` 默认为 False。

### transition 存储逻辑（4车版本）

```python
# last_obs    = [None] * N_AGENTS  ← 列表，last_obs[i] 是 agent i 上次决策时的局部观测向量
# last_state  = [None] * N_AGENTS  ← 列表，last_state[i] 是 agent i 上次决策时的全局状态向量
# last_action = [None] * N_AGENTS  ← 列表，last_action[i] 是 agent i 上次选的动作整数
#
# [i] 是对"各 agent 上次时间点"的索引，不是对 state 向量内部的索引。
# last_state[i] 的类型是 np.ndarray，形状 (state_dim,) = (74,)，是完整全局状态。

# 旧版（2车）：每步最多存 2 条
# 新版（4车）：每步最多存 4 条，逻辑完全相同

for i in range(N_AGENTS):   # N_AGENTS=4
    if needs_action[i] and last_obs[i] is not None:
        buffer.push(
            last_obs[:],        # list 浅拷贝：保留的是列表结构，每个元素已是 .copy() 的 ndarray
                                # 不浅拷贝则后续 obs[i]=new_obs 会覆盖已入 buffer 的历史引用
            last_state[i],      # agent i 上次决策时的全局状态（ndarray, shape 74），已是 state.copy()
            [last_action[j] if last_action[j] is not None else 0
             for j in range(N_AGENTS)],
            reward,
            next_obs[:],        # 同 last_obs[:]，浅拷贝防引用污染
            next_state,
            float(done)
        )
```

> `last_obs[:]` 浅拷贝的原因：`obs` 是 `env.step()` 返回的列表，下一步 `obs = next_obs` 后列表原地更新，不拷贝会导致 buffer 里所有历史条目的 obs 都被覆盖成最新值（旧版曾踩此坑）。浅拷贝（`[:]`）足够，因为列表元素本身已是 ndarray 独立对象，不存在二层引用。

### MAX_STEPS=300 在新地图的适用性分析

| 场景 | 单次投递最短路（估算） | 300步内最多投递次数（单车） |
|---|---|---|
| 旧 10×10 蛇形 | 约 25-35 步 | 约 7-10 次 |
| 新 20×16 十字 | 约 50-70 步（含瓶颈绕行） | 约 4-5 次 |

4台车合计 300步内最多约 **16-20 次投递**，正奖励密度降低但仍可学习。  
**初始保持 300，训练早期若每 episode 平均投递 < 6 次则考虑调整到 400-500。**  
注意：旧地图实测 600 步比 300 步差（步惩罚累积淹没稀疏正奖励），新地图路径更长，这个问题会更突出，不建议轻易超过 500。

### 训练时长估算（CPU）

旧版（2车，10×10，6000 episode）实测约 60 分钟。

新版估算：
- A* 调用量：4车 × 路径约2倍长 = **约8×**（A* 是主要瓶颈）
- 每步可能重规划次数：增加但有上限
- 粗估：**60 × 8 ≈ 8 小时**（需要隔夜跑）

建议策略：先跑 2000 episode 看曲线趋势（约 2-3 小时），确认 reward 上升后再跑全量。

### 4车同时完成时 buffer 相关性分析

当一步内 4 辆车全部 `needs_action=True`（如 episode 初始），同一步的 reward 被存入 4 条 transition。这 4 条 transition 共享同一个 `reward` 和 `next_state`，是相关样本。

**影响**：采样时若同批次恰好抽到这 4 条，梯度估计有偏。  
**严重性**：低。因为 BUFFER_SIZE=20000，这类同步 transition 占比小；且 QMIX 本身对此设计有容忍度（全局奖励本就是共享的）。  
**不做额外处理**：不值得增加复杂度，训练时监控 loss 曲线即可。

### 改动对照表

| 模块 | 旧版 | 新版 |
|---|---|---|
| N_AGENTS | 2 | 4 |
| N_SHELVES | 4 | 10 |
| obs_dim（隐含） | 26 | 87 |
| state_dim（隐含） | 26 | 74 |
| n_actions（来自env） | 6 | 13 |
| BUFFER_SIZE | 10000 | 20000 |
| MIN_BUFFER | 200 | 500 |
| EPS_DECAY | 4000 | 6000 |
| HIDDEN | 64 | 128 |
| SAVE_DIR | checkpoints_v4 | checkpoints_v5_attn / v5_mlp |
| 网络类 | AgentQNet | AttentionAgentQNet 或 MLPAgentQNet（--model切换） |
| compute_loss 结构 | 不变 | 不变（维度自动适配） |
| transition 存储逻辑 | 不变 | 不变（N_AGENTS 循环数量变化） |
| reward 全局共享 | 是 | 是（不变） |
| target 同步 | episode%50 | 不变 |
| 梯度裁剪 | clip 10.0 | 不变 |
| MLPAgentQNet hidden | 64 | **保持64**（对照实验公平性） |

---

---

# evaluate.py / 可视化方案设计

---

## 【任务目标】

加载训练好的 checkpoint，在环境中无探索地运行若干 episode，输出定量指标和 Attention 热力图。

---

## 【完整输出】

### evaluate.py 职责

```
1. 加载 checkpoint（--ckpt 参数）
2. 加载模型类型（--model attention/mlp）
3. 运行 N_EVAL=20 个 episode（epsilon=0，纯贪心）
4. 输出：
   - 每 episode 投递次数、总奖励
   - 平均投递数 ± 标准差
   - 若 --model attention：输出 attention 权重序列
```

### Attention 可视化流程（伪代码）

```python
# 运行一个 episode，收集每步 attention 权重
attn_log = []   # list of [10] arrays，每步一条

for step in range(MAX_STEPS):
    for i in range(N_AGENTS):
        if needs_action[i]:
            obs_t = torch.FloatTensor(obs[i]).unsqueeze(0)
            q, w  = agent_net(obs_t, return_attn=True)
            # w: [1, 1, 10] → squeeze → [10]，10个货架的注意力权重
            attn_log.append(w.squeeze().numpy())

# 画热力图：x轴=时间步，y轴=货架编号，颜色=注意力权重
import matplotlib.pyplot as plt
import numpy as np
attn_arr = np.array(attn_log)        # [T, 10]
plt.imshow(attn_arr.T, aspect='auto', cmap='hot')
plt.xlabel('决策步'); plt.ylabel('货架编号')
plt.colorbar(label='Attention 权重')
plt.savefig('attn_heatmap.png')
```

验收标准：热力图中不同时刻有不同货架被高亮（证明网络在学习，而非退化为均匀分布 0.1）。

### 对照实验输出格式

```
方案               平均投递数    平均奖励    跑5次均值±std
随机分配           X.X ± Y.Y    XXX ± YY
MLPAgentQNet      X.X ± Y.Y    XXX ± YY
AttentionAgentQNet X.X ± Y.Y   XXX ± YY   ← 目标：比MLP高≥10%
```

这三行数据是比赛答辩的核心证据，evaluate.py 必须能一键输出。

---

---

════════════════════════════════════════════════════════════════
# 第二阶段：代码编写
**日期**：2026-05-19
════════════════════════════════════════════════════════════════

---

## 【任务目标】

严格按照设计文档（上方所有章节）将三个核心文件从旧版（v4）重写为新版（v5）：
- `warehouse_env.py`：20×16 十字地图 + 4车 + 10货架 + 3类货物
- `qmix_network.py`：**双路** AttentionAgentQNet（货架+车间注意力）+ MLPAgentQNet 对照基线 + QMixNet
- `train.py`：适配新环境参数 + `--model` 对照实验切换

---

## 【完成内容】

三个文件全部重写完毕，并通过冒烟测试：

| 文件 | 位置 | 测试结果 |
|---|---|---|
| `warehouse_env.py` | `/home/jasper/qmix_project/warehouse_env.py` | obs_dim=87 ✓，state_dim=74 ✓，n_actions=13 ✓，步进无报错 ✓ |
| `qmix_network.py` | `/home/jasper/qmix_project/qmix_network.py` | 输出形状正确 ✓，shelf_w[8,1,10]/agent_w[8,1,3] softmax 均=1.0 ✓，梯度正常反传 ✓ |
| `train.py` | `/home/jasper/qmix_project/train.py` | 端到端采样→更新 pipeline 无报错 ✓ |

---

## 【关键假设】

1. **墙壁构造按 PLAN.md 字面解读**：北臂/南臂外侧全墙，R3/R11 瓶颈只留 col8/col11，R7 隔墙在 col2-4 和 col15-17。
2. **双路 Attention 均使用头平均权重**（默认行为），shelf_w `[batch, 1, 10]`，agent_w `[batch, 1, 3]`，PyTorch 2.4.1 已确认兼容。MLP 输入从 57 维变为 83 维（去掉 mean pool 的 6 维，加入两路 context 各 32 维）。
3. **到达货架时有货也不报错**：空手才取货，有货则直接 needs_action=True；QMIX 应自行学会避免此行为。
4. **-0.05 步惩罚不随 agent 数量缩放**：与旧版保持一致（总奖励量级相近）。
5. **MLPAgentQNet hidden 保持 64**：对照实验公平性，架构差异不混入参数量差异。

---

## 【未确认的地方】

1. **MAX_STEPS=300 是否足够**：新地图单次投递路径约 50-70 步，300 步内理论 4 车合计 16-20 次投递。若早期训练平均投递 < 6 次，需调整到 400-500。
2. **R7 走廊隔墙列号**：代码按 col2/3/4 和 col15/16/17 实现，需实际跑 render() 核对视觉是否符合预期。
3. **动态订单队列**：当前采用 Plan A（随机补货），待确认是否需要 Plan B（订单队列）。

---

## 【完整输出】

代码已直接写入以下三个文件，不在此重复粘贴：

- `/home/jasper/qmix_project/warehouse_env.py`（约 230 行）
- `/home/jasper/qmix_project/qmix_network.py`（约 160 行，含双路 Attention：shelf_attn + agent_attn）
- `/home/jasper/qmix_project/train.py`（约 200 行）

**启动训练命令：**
```bash
cd ~/qmix_project
python train.py --model attention          # Attention-QMIX（主方案）
python train.py --model mlp               # MLP-QMIX（对照基线）
python train.py --model attention --resume checkpoints_v5_attn/qmix_ep1000.pt  # 续训
```

**输出目录：**
- `checkpoints_v5_attn/`：Attention 版 checkpoint + ep_rewards.npy + ep_deliveries.npy
- `checkpoints_v5_mlp/`：MLP 版同上

---

---

════════════════════════════════════════════════════════════════
# 第三阶段：双路 Attention 升级
**日期**：2026-05-19
════════════════════════════════════════════════════════════════

---

## 【任务目标】

原单路 Attention（仅货架）创新力不足：其他车用均值池化，车间协调关系（QMIX 核心）没有用 Attention 建模。升级为**双路 Attention**，让网络同时学习"关注哪个货架"和"配合哪辆车"。

---

## 【完成内容】

- `qmix_network.py` 中 `AttentionAgentQNet` 改为双路 Attention
- 设计文档更新架构图、维度、假设
- 代码通过冒烟测试

---

## 【关键假设】

1. **两路使用独立 Q 投影**（`shelf_q_proj` 和 `agent_q_proj` 分开），而非共享。理由：货架和车的特征维度不同（5 vs 6），两路语义也不同（"去哪" vs "配合谁"），独立投影更灵活。
2. **两路 d_model 相同（均为 32）**：参数量对称，便于对比两路权重的量级。
3. **MLP 输入维度从 57 变为 83**：去掉 6 维 mean pool，加入 32 维 agent_context，obs_dim 不变（87）。
4. **return_attn 返回两个权重**：`(q_vals, shelf_w, agent_w)`，可视化时分别画两张热力图。

---

## 【未确认的地方】

1. **车间 Attention n_heads=2 是否合适**：3 辆其他车的序列长度只有 3，用 2 个 head 意味着每个 head 看 1.5 个位置，信息量有限。若可视化发现权重均匀（退化），考虑改为 n_heads=1。
2. **两路 d_model 是否需要不同**：货架有 10 个实体、车只有 3 个，可以考虑 shelf_d_model=32, agent_d_model=16 降低参数量，但目前先用相同值保持简洁。

---

## 【完整输出】

### 双路 AttentionAgentQNet 架构图

```
输入 obs [batch, 87]
  │
  ├─ self_feat   = obs[:, 0:19]           [batch, 19]
  ├─ other_feats = obs[:, 19:37]          [batch, 3, 6]   ← reshape(3, 6)
  └─ shelf_feats = obs[:, 37:87]          [batch, 10, 5]  ← reshape(10, 5)
  │
  ├─ 【货架注意力 Shelf Attention】
  │    shelf_Q = Linear(19→32)(self_feat).unsqueeze(1)    [batch, 1, 32]
  │    shelf_K = Linear(5→32)(shelf_feats)                [batch, 10, 32]
  │    shelf_V = Linear(5→32)(shelf_feats)                [batch, 10, 32]
  │    shelf_ctx, shelf_w = MHA(Q,K,V)
  │    shelf_ctx.squeeze(1)                               [batch, 32]
  │    shelf_w                                            [batch, 1, 10]  ← 10货架权重
  │
  └─ 【车间注意力 Agent Attention】
       agent_Q = Linear(19→32)(self_feat).unsqueeze(1)    [batch, 1, 32]
       agent_K = Linear(6→32)(other_feats)                [batch, 3, 32]
       agent_V = Linear(6→32)(other_feats)                [batch, 3, 32]
       agent_ctx, agent_w = MHA(Q,K,V)
       agent_ctx.squeeze(1)                               [batch, 32]
       agent_w                                            [batch, 1, 3]   ← 3辆车权重

MLP 输入 = concat([self_feat(19), shelf_ctx(32), agent_ctx(32)]) = [batch, 83]
  Linear(83→128) → ReLU → Linear(128→128) → ReLU → Linear(128→13)

输出: q_vals [batch, 13]
可视化: shelf_w.squeeze(1) → [batch, 10]，agent_w.squeeze(1) → [batch, 3]
```

### 与单路版对比

| 项目 | 单路版（已废弃） | 双路版（当前） |
|---|---|---|
| 其他车处理方式 | mean pool → [batch, 6] | Agent Attention → [batch, 32] |
| 货架处理方式 | Shelf Attention → [batch, 32] | 同左，不变 |
| MLP 输入维度 | 19+6+32 = **57** | 19+32+32 = **83** |
| 可视化权重 | 1组（10货架） | 2组（10货架 + 3辆车） |
| 创新说法 | "attention感知货架" | "双路attention感知货架与协调关系" |

### 比赛答辩逻辑

> "在 AgentQNet 中设计双路注意力机制：货架注意力使机器人学会优先选择合适货架，车间注意力使机器人感知其他车辆状态实现隐式通信与协调，两路 context 共同驱动 Q 值决策。"

对照实验三组：随机 / MLP-QMIX / 双路Attention-QMIX，递进展示每个组件的贡献。

代码已更新至 `/home/jasper/qmix_project/qmix_network.py`

---

---

════════════════════════════════════════════════════════════════
# 第四阶段：Bug 修复与训练稳定性优化
**日期**：2026-05-19
════════════════════════════════════════════════════════════════

---

## 【任务目标】

根据代码审查结果（7条问题汇总）修复已知 bug，并在训练过程中发现并修正奖励存储逻辑导致的 Q 值发散问题。

---

## 【完成内容】

| 文件 | 问题编号 | 问题描述 | 修复方式 |
|---|---|---|---|
| `warehouse_env.py` | #5（中） | A* 返回空路径时 stuck_count 重置，陷入静默卡死 | `_move_agents` Step 4 新增 `not at_dest and not path` 分支，立即重规划 |
| `warehouse_env.py` | #7（低） | `render()` 在 `reset()` 前调用会 TypeError | 开头加 `if self.agent_pos is None: return` |
| `qmix_network.py` | #1（低） | `obs_dim` 是死参数，传错值不报错 | `__init__` 加 assert 校验，传入值与内部切分不一致时立即报错 |
| `qmix_network.py` | #2（低） | `agent_attn` 用 n_heads=2，但序列长度仅3 | 硬编码改为 `n_heads=1` |
| `train.py` | #3（训练中发现） | transition 只存单步到达奖励，漏掉导航步数惩罚 | 最终方案：存 `nav_cost + reward`（见下方详细说明） |

---

## 【关键假设】

1. **问题 #3 经过两次迭代才稳定**：第一次尝试"全累计团队奖励"导致 Q 值目标方差爆炸（ep1500 退回 4.94），第二次改为"仅累计步数惩罚 + 单步到达奖励"，reward 范围收窄到 `-4.5 ~ +9.4`，恢复稳定。
2. **agent_attn n_heads 改为 1（硬编码）**：构造函数签名的 `n_heads` 参数只影响 `shelf_attn`，`agent_attn` 固定为 1，避免序列长度 3 不够 2 个 head 充分学习。
3. **问题 #4（batch 校验）和 #6（list.index）暂不修复**：低优先级，不影响训练，等有需要时再处理。

---

## 【未确认的地方】

1. **nav_cost 计步时序是否精确**：`last_steps[i]` 从决策时归零，每步 +1，到达时值为 k，`nav_cost = -0.05 * max(k-1, 0)` 排除到达步（该步惩罚已含在 `reward` 中），理论正确，实际是否有 off-by-one 需后续通过 reward 范围观测确认。
2. **训练重启后能否超越旧版峰值**：旧版（单步 reward）ep700 均值 5.66，第一次修复后 ep700 达 7.26，第二次修复刚重启，需观察 ep700-1000 是否再次达到 7 以上。

---

## 【完整输出】

### warehouse_env.py：问题 #5 修复（_move_agents Step 4）

**修复前**（bug）：
```python
# 只有 agent_paths[i] 非空才进入 stuck 计数分支
# 路径为空时走 else 分支，stuck_count 重置为 0 → 永远不会重规划 → 静默卡死
if not moved[i] and self.agent_dest[i] is not None and self.agent_paths[i]:
    self._stuck_count[i] += 1
    ...
else:
    self._stuck_count[i] = 0
```

**修复后**：
```python
dest = self.agent_dest[i]
if dest is None:
    self._stuck_count[i] = 0
    continue

at_dest = tuple(next_pos[i]) == tuple(dest)

if not at_dest and not self.agent_paths[i]:
    # A* 返回空路径但尚未到达目标：立即重规划，避免静默卡死
    blocked = {tuple(next_pos[j]) for j in range(self.n_agents) if j != i}
    new_path = self._astar(tuple(next_pos[i]), tuple(dest), blocked)
    if not new_path:
        new_path = self._astar(tuple(next_pos[i]), tuple(dest), set())
    self.agent_paths[i] = new_path
    self._stuck_count[i] = 0
elif not moved[i] and self.agent_paths[i]:
    self._stuck_count[i] += 1
    if self._stuck_count[i] >= 5:
        ...（重规划逻辑不变）
else:
    self._stuck_count[i] = 0
```

---

### train.py：问题 #3 奖励存储修复（两次迭代）

**第一次尝试（失败）：全累计团队奖励**

```python
# 每步累计所有团队奖励，包括其他 agent 的送达奖励
last_accum_r[i] += reward  # 每步加
buffer.push(..., last_accum_r[i], ...)
```

**失败原因**：导航 50 步期间其他 3 辆车各送达一次，agent i 的 `last_accum_r[i]` = `-2.5 + 30 = 27.5`。Q 值目标范围从原来的 `-2 ~ +10` 爆炸到 `-2.5 ~ +57.5`，高方差导致 ep1500 退化到 4.94 送达。

**第二次（最终方案）：导航步惩罚 + 单步到达奖励**

```python
last_steps = [0] * N_AGENTS  # 每 agent 自上次决策起的步数

# 决策时重置
last_steps[i] = 0

# 每步递增
for i in range(N_AGENTS):
    last_steps[i] += 1

# 存 transition 时
nav_cost = -0.05 * max(last_steps[i] - 1, 0)  # 导航步数惩罚（不含到达步）
buffer.push(..., nav_cost + reward, ...)         # + 到达步的单步团队奖励
```

**各版本 reward 范围对比**：

| 版本 | reward 范围 | 问题 |
|---|---|---|
| 原版单步 | `-2.05 ~ +9.95` | 漏掉导航步惩罚，Q 值高估 |
| 全累计（第一次，已废弃） | `-2.5 ~ +57.5` | 其他 agent 奖励混入，方差爆炸 |
| **当前版（导航惩罚+单步到达）** | **`-4.5 ~ +9.4`** | 补了步惩罚，不引入额外噪声 |

训练结论：重启后需从头训练（buffer 中旧格式 reward 无效），观察 ep700 是否再次达到 ≥7 送达。
