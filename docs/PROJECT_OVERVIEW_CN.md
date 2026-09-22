# Feeding Coordination：5–10 分钟项目概览

> 截至 2026-09-22。本文是入口；需要复现、解释数值口径或继续研究时，请读 [`HANDOFF_FULL_HISTORY_CN.md`](HANDOFF_FULL_HISTORY_CN.md) 和对应阶段的产物/contract。

## 1. 项目到底研究什么？

人的进食示范里，同一叉子轨迹可以由不同的手臂姿态完成。只让七自由度 Kinova Gen3 追踪叉子末端，机器人肘部可能任意转动或突然换分支。本项目研究这种**冗余协调（redundancy coordination）**：在固定工具任务下，人类 Stereo-SEW 冗余角 `ψ(t)` 如何随工具轨迹（tool trajectory）`U(t)` 演化，再用已有、冻结的 Exact-SEW 映射为机器人关节 `q(t)`。项目不重新发明 SEW/IK，也不把人体肘部三维位置直接当机器人硬目标。

后来补上「上下文 → 叉子工具轨迹」工程层，形成完整**模拟**链条。当前能说的是 *algorithmic feasibility*，不能说已经标定叉尖、测得真实嘴部、完成物理喂食或安全验证。

## 2. 一开始怎么想，为什么改了？

最初希望利用**完整 planned utensil trajectory** 的未来信息，超过局部工具状态对 `ψ` 的预测；这对应 PCRC/full-plan 主线。这个假设值得测，但不能当结果。实际路线如下：

```text
human-to-robot retargeting（已有 SEW/Exact-SEW 工作）
   ↓ 将求解器冻结为依赖
human redundancy coordination 学习
   ↓ 最初假设未来完整 plan 有新增信息
Phase 3 → Phase 3.1：去除 ψ₀ 环形 OOD 后，full plan 无可靠增益
   ↓
19D StrongLocal（局部、Δψ）+ task-equivalent robot study
   ↓ Robot R2/R3：B2/B3 很有竞争力；HumanGT/RobotSmooth 是 tradeoff
G0 Contextual ProMP 工具生成 → G1 完整模拟链 → G1V 展示
```

另一条重要纠错链：起初因缺少手到工具标定，担心没有可用工具位姿；Phase 1.5 发现 OptiTrack **直接跟踪 fork rigid body**，可据此定义 `U`，但**不是已知叉尖**。Phase 2 在虚拟 `P→U=I` 下只有 35.79% exact replay；A/B/C/D 诊断发现主因是 raw fork 与旧 canonical orientation 固定约 90° 的目标语义差，而不是 solver、mount 或 adapter 退化。Robot R0 修正模拟朝向后 HumanGT replay 为 96.41%；这仍不是物理标定或真实可行率。

## 3. 当前完整系统长什么样？

```text
阶段 + plate 位置 + derived mouth proxy + 初始 tracked fork pose
                       ↓
           上下文 ProMP（Contextual ProMP）
             [Retrieval = 工程对照]
                       ↓
             101 点 fork U(t)
                       ↓
          19D StrongLocal → Δψ(t)
                       ↓ 加测得的初始 ψ₀
                      ψ(t)
                       ↓
         冻结 Exact-SEW → Gen3 q(t)
                       ↓
          模拟评估 / G1V 只读展示
```

在固定 measured U 的科学实验里，比较 B0/B1/B2/B3、StrongLocal、H*、HumanGT 与 RobotSmooth。HumanGT 是离线 oracle/reference，使用人的 measured `ψ`；RobotSmooth 是以机器人连续性为中心的因果启发式对照，不是人体预测器。G1 的 MeasuredUReference/StrongLocal 则使用 measured U 与**预测**的 StrongLocal `ψ`，不等于 HumanGT。

## 4. 各阶段一眼看懂

| 阶段 | 当时要回答什么 | 方法 | 实际观察与决定 |
| --- | --- | --- | --- |
| Phase 0 | 如何可靠复用旧机器人求解器？ | 锁定 Exact-SEW commit、薄 stateful adapter/契约测试 | 冻结求解层，不再把 retargeting 当论文 novelty |
| Phase 1 | 人体数据、坐标、`ψ` 是否可信？ | 7 takes、S/E/W、B-frame、Stereo-SEW、缺失掩码 | 127 bites、254 段、33,999 帧；发现工具标定语义待查 |
| Phase 1.5 | 是否真没有工具 pose？ | exact Motive/NatNet frame join，直接跟踪 `fork` | 33,981 valid pose（99.947%）；fork rigid body 可作 `U`，叉尖未标定 |
| Phase 2 | measured `U,ψ` 能否在固定机器人上 replay？ | 当时虚拟 `P→U=I`、每 run 新 solver | 11,529/32,216＝35.79%；不能解释为物理不可行 |
| Phase 2 diagnostic | 旧 replay 为何与 Phase 2 冲突？ | A/B/C/D 位置/朝向因子比较 | raw/canonical 朝向约差 90° 是主因，位置变化次之；排除 mount/solver/adapter |
| Phase 3 | history/future/full plan 是否比 local 多信息？ | M1–M6、完整 take 留一、parent-bite 统计 | 原 M3 异常差，history 相对它似乎有益；进入 3.1 审计 |
| Phase 3.1 | 是 local 不够，还是特征有错？ | L0–L4、ψ₀ OOD 检查、H*/F*/P* gate | 七个 fold 均选 L3；raw ψ₀ 线性特征在 ±π 出错；停止 PCRC |
| Phase 3.2 | 最终人类模型与简单 baseline？ | 19D L3 StrongLocal、预测 `Δψ`、OOF；另存部署模型 | 比 hold `ψ₀` RMSE 降 38.5%；对简单 pose baseline 只有温和/指标依赖优势 |
| Robot R0 | 目标语义修正后如何公平比较？ | 同一 virtual U、canonical 90°、B0–B3/H*/HumanGT/StrongLocal | HumanGT 96.41% exact frame success；仅模拟几何，非物理标定 |
| Robot R1 | q 是否连续，跳变从何来？ | `>0.5 rad` violation、clean interval、因果 RobotSmooth | near-π 跳变与分支变化相关；RobotSmooth 3 次 violation vs StrongLocal 30 次 |
| Robot R2 | whole-arm / motion 真的谁更好？ | q、robot elbow/arm plane、速度/jerk、Jacobian、paired support | StrongLocal 不可靠整体优于 B2/B3；HumanGT vs RobotSmooth 有 tradeoff；碰撞/retiming 不可宣称 |
| Robot R3 | 换初始 `ψ₀` 还能稳定吗？ | `−0.25/0/+0.25 rad`，同初始目标条件 | 一些结论稳定，但 velocity、margin、feasibility 等排序会变 |
| G0 | 能从当前上下文生成工具轨迹吗？ | Retrieval vs 12-basis Contextual ProMP、LOTO | ProMP 工具 pose 误差更低；生成数值成功不等于真实喂食 |
| G1 | 工具误差会传到 ψ/q 吗？ | generated U → fold-specific StrongLocal → Exact-SEW | ProMP 比 Retrieval 在 tool/ψ/q 与完整 pipeline 成功率上更好 |
| G1V | 如何让人看见差异与失败？ | stored-q read-only MuJoCo/ψ panel/视频 | 已实现展示；当前工作区的旧 manifest 与部分视频文件存在性不一致，交付时须核对 |

## 5. 最值得记住的研究发现

1. `U(t)` 有直接跟踪的 fork rigid-body 数据，但 fork tip/food point 还没有物理标定。
2. raw fork frame 与旧 canonical hand frame 约差 90°；Phase 2 的 35.79% 不能叫真实 feeding feasibility。
3. 人的 `ψ` 确实有系统演化；仅持有初始 `ψ₀` 不够。
4. 原始 M3 的坏表现主要是把环形 `ψ₀` 当普通线性特征导致 OOD，trial_0014 特别明显。
5. 修正后局部 L3 足够有竞争力；history、future、full-plan/PCRC **没有可靠新增收益**。
6. StrongLocal 的 `ψ` RMSE 比 hold `ψ₀` 降 38.5%，但不在所有指标/机器人质量上胜过 pose-only 或 B2/B3。
7. near-π `q` 跳变与 Exact-SEW 离散 branch 变化相关，不等价于工具或 `ψ` 输入跳变。
8. RobotSmooth 的连续性最好，但相对 HumanGT 牺牲 human consistency / 某些 coverage；没有单一全局赢家。
9. 初始 `ψ₀` 偏移能改变运动、余量和可行性排序；部署起始冗余选择仍未解决。
10. G0 LOTO：ProMP vs Retrieval 工具位置 RMS `0.06793 vs 0.08676 m`，姿态 RMS `0.32435 vs 0.41916 rad`。
11. G1 在 246 条 eligible 记录上，ProMP vs Retrieval 的 StrongLocal 完整 pipeline 成功 `224/246=91.06% vs 201/246=81.71%`；工具误差优势传播到 ψ/q，但这仍是模拟链。

这些比较的统计单位以 parent bite 为主；每帧样本不能被当作彼此独立的显著性样本。G0 用 254 记录，G1 因 8 个初始 ψ 无效仅用 246，二者均按完整 take 做外层留出。具体置信区间、条件与支持集见详细 handoff 和 `outputs/*/summary.json`。

## 6. 哪些路线已停止？哪些仍是 provisional？

**已停止作为当前主线：**PCRC/full-plan residual、history/future summary 主方法、raw `ψ₀` 线性特征、新 SEW/IK、把 identity `P→U` 当真实几何。Diffusion/Transformer/CVAE 在目前样本与基线证据下没有引入依据。不要把停止写成「从未研究过」。

**尚未物理成立：**mouth proxy 是 derived/untrusted，不是真实测得嘴部；叉尖与 `P→U` 平移、真实 Gen3 base、plate/recipient workspace 未标定；G0/G1 timing 是 provisional，缺权威速度/加速度限值与 retiming；当前碰撞模型未验证；recipient identity、真实机器人执行、人体/环境安全及 feeding outcome 均未建立。虚拟 simulation geometry 不等于 physical calibration。

**当前状态：**算法可行性研究和受控模拟比较已完成到 G1；G1V 展示工具可读 saved q 且不跑 IK。物理精度验证未完成，真实机器人执行未完成。本次审计时旧 G1V manifest 所列五个 MP4/六张图在未提交工作树中标为删除，当前另有三路/playlist 文件；演示产物交付前应按实际目录与 manifest 核对，不能只引用清单。

## 7. 下一步：不要从旧 PCRC 计划接着做

建议下一主线是 **Physical Calibration Accuracy Validation**，依次为：Gen3 base / OptiTrack / tracked-fork 外参（P0）→ fork rigid body 到 physical fork-tip/food point（P1）→ held-out 标定精度验证（P2）→ 真实 plate/recipient/workspace 摆位（P3）→ 无人体接触的 Gen3 执行（P4）→ mannequin/几何代理评估（P5）→ 论文主张和复现材料冻结。每一步需独立的测量、设备与安全条件，不能由当前模拟成功率自动授权实机动作。

## 8. 如果半年后回来，只记住这十句话

1. 本研究核心是**固定工具任务下人的冗余协调**，不是重做 IK。
2. `U` 是 tracked fork rigid body，**不是**已标定叉尖。
3. `ψ` 来自 measured shoulder/elbow/wrist，人体肩运动保留。
4. Phase 2 的 35.79% 属于历史 identity 工具目标；90° 朝向语义修正至关重要。
5. 最终人类模型是 19D L3 StrongLocal，学 `Δψ`，`ψ₀` 仅作重建 offset。
6. PCRC/full-plan 被 Phase 3.1 的实证 gate 停止，不要默默复活。
7. StrongLocal 对 hold 有益，但 B2/B3 在机器人层面同样强。
8. RobotSmooth 是追求连续性的工程对照；HumanGT 是离线人的真值参考，二者有 tradeoff。
9. Contextual ProMP 比 Retrieval 的完整**模拟**链更好；mouth proxy、初始 `ψ₀`、时间与几何仍是部署缺口。
10. 下一步做物理标定与 held-out 精度验证，再谈实机、碰撞、安全或 feeding success。
