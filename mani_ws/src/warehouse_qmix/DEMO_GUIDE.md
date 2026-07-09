# 多AGV仓库调度系统 — 演示速查清单

> 答辩/录屏照此操作。每个演示**单独跑**,跑前先 `source ~/mani_ws/devel/setup.bash`。
> 凡是带真实模型(mesh)的,启动前先 `export GAZEBO_MODEL_DATABASE_URI=""` 禁 fuel 联网,加载快。

---

## 演示一:四车 QMIX 协同调度(核心)

**讲点**:多智能体强化学习实时推理,4 车协同取货-投递、自动避障、缺货调度。

```bash
# 终端1 — 主仿真(四车 + QMIX 推理 + 货物管理 + 导航)
source ~/mani_ws/devel/setup.bash
roslaunch warehouse_qmix warehouse_qmix.launch
```

```bash
# 终端2 — Web 实时监控大屏
source ~/mani_ws/devel/setup.bash
roslaunch warehouse_qmix dashboard.launch
# 浏览器打开 http://localhost:8080
```

**看什么**:Gazebo 里 4 车并行作业;大屏上实时显示各车任务、货架库存、累计吞吐曲线。

---

## 演示二:单车完整作业闭环(真实机械臂抓取)

**讲点**:完整 wpb_mani 真车,导航→机械臂抓取→搬运→投递 全流程闭环。

```bash
export GAZEBO_MODEL_DATABASE_URI=""
source ~/mani_ws/devel/setup.bash
roslaunch warehouse_qmix single_grab.launch
```

**看什么**:车导航到货架→横向对中→伸臂→夹爪抓起货箱→抬臂搬运→投递区下探放下。

---

## 演示三:对比实验(技术深度,跑数据/出图)

**讲点**:QMIX vs 随机/轮转/就近贪心,4 缺货档 × 各 100 局。QMIX 冲突少 ~30%、稳定性翻倍。

```bash
cd ~/qmix_project
python3 experiments/run_compare.py --episodes 100 --restock 20 40 80 120   # 出表
python3 experiments/plot_compare.py                                        # 出图
# 图:experiments/metrics_vs_scarcity.png, throughput_box.png
```

---

## 答辩开场顺序建议
1. 先放**演示二**视频(真实机械臂作业)——抓眼球、证明"真能做"
2. 再开**演示一**(四车协同 + 大屏)——展示规模与智能
3. 用**演示三**的图表收尾——用数据证明 QMIX 的价值

> ⚠️ 四车真实模型(hybrid_*)因本机算力不足未采用,属探索性尝试,不进演示。
